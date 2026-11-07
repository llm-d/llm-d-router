/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package threadidentity

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/cespare/xxhash/v2"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

// PluginType is the canonical type name used to register the plugin.
const PluginType = "thread-identity"

// ThreadIdentityKey is the request-attribute key carrying the DERIVED thread id,
// for the readers that want the thread by name (the group scheduler). The queue key
// travels on a different attribute: see agentidentity.AgentIdentityKey, below.
var ThreadIdentityKey = fwkplugin.NewDataKey("thread-identity", "")

// ThreadArrivalKey is the request-attribute key carrying the ARRIVAL RECORD: which
// of the three outcomes this request was (continued / forked / new) and its turn
// number on its thread. It is a pure OUTPUT — the scheduler reads it to tell a
// thread's first turn from a continuation (a newcomer from an incumbent); nothing
// here reads it back, so publishing it cannot change the derivation.
var ThreadArrivalKey = fwkplugin.NewDataKey("thread-arrival", "")

// ThreadArrival is the published arrival record.
type ThreadArrival struct {
	// Outcome is the arrival outcome: OutcomeMatched (continued its thread),
	// OutcomeForked (started a sibling of a busy tip) or OutcomeNew.
	Outcome Outcome `json:"outcome"`
	// Turn is the arrival's 1-based turn number on its thread: 1 for a thread's
	// first request, n for the n-th continuation.
	Turn int `json:"turn"`
}

// ThreadBlocksKey is the request-attribute key carrying the RAW per-prompt prefix
// hashes the identity was derived from: one PromptBlocks per prompt, exactly as the
// prefix scorer hashes them. The scheduler's costs are these blocks, so it reads them
// here instead of re-tokenising the request at dispatch time: one render and one
// hash per request, shared by the identity and the admission decisions.
var ThreadBlocksKey = fwkplugin.NewDataKey("thread-blocks", "")

// PromptBlocks is one prompt's block keys in the prefix scorer's hash space — the
// prompt's tokens at the engine block size, its trailing block partial — and its
// token count.
//
// READ-ONLY once published. The scheduler shares these slices across passes and
// goroutines without a copy, so nothing may append to, sort or otherwise write Keys
// after RequestHeader returns: that would be a data race the compiler cannot see and
// a chain two decisions would disagree about.
type PromptBlocks struct {
	// Keys holds the prompt's block hashes in order, the trailing partial block
	// included.
	Keys []prefixhash.BlockHash
	// Tokens is the prompt's token count.
	Tokens int
}

// threadMarkerKey is the request-attribute key carrying the request's marker —
// the handle all three hooks use to move this request between its thread's
// pending and in-flight sets. It equals the request id when the client supplied
// one and a per-request synthetic id otherwise.
var threadMarkerKey = fwkplugin.NewDataKey("thread-marker", "")

// defaultBlockSizeTokens is the identity block length a config that omits it gets.
const (
	// defaultBlockSizeTokens is the identity block length — the prefix scorer's
	// B, one shared knob, not an identity-specific size.
	defaultBlockSizeTokens = 64
	// defaultMarkerTTLSeconds bounds how long an outstanding marker keeps its
	// thread busy when the request never reports an end of stream (a request
	// rejected in admission control never gets one). It is the CLIENT's request
	// timeout, not the flow queue's 5-minute idle GC: 16% of a0's requests waited
	// over 600s for admission, so a 600s horizon would expire markers of requests
	// that are still legitimately queued. Pruning errs toward continuation.
	//
	// It MUST sit above the client's own timeout, or the marker of a request that is
	// still running expires and its thread takes a second request as a different chain:
	// the campaign harness runs a 9000s timeout and every pin sets this to 9300s, so
	// 9300 is the default and a config that forgets the key is safe.
	defaultMarkerTTLSeconds = 9300
	// defaultSweepSeconds is the cap-enforcement tick.
	defaultSweepSeconds = 60
	// defaultMaxThreads is the table cap (the only eviction path).
	defaultMaxThreads = 100000
)

// Parameters is the plugin's configuration block.
type Parameters struct {
	// BlockSizeTokens is the identity block length in tokens — the prefix
	// scorer's block size. Zero selects the default (64).
	BlockSizeTokens int `json:"blockSizeTokens,omitempty"`
	// TokenProducer configures the render backend that supplies the request's
	// real tokens, in the token-producer plugin's own parameter language. It is
	// REQUIRED: the identity is a token-space key, so a plugin without this block
	// has no identity to derive and refuses to start (the estimate backend it
	// would otherwise build is a byte-space key — the retired v1 behaviour).
	TokenProducer json.RawMessage `json:"tokenProducer,omitempty"`
	// MarkerTTLSeconds bounds how long an outstanding marker (arrival or in
	// flight) that never settles keeps its thread busy. It must be at least the
	// client's request timeout (the default, 9300s, is the campaign harness's 9000s
	// plus headroom). There is no thread TTL: threads are retained (no idle GC — a
	// dormant client that returns must find its thread).
	MarkerTTLSeconds float64 `json:"markerTtlSeconds,omitempty"`
	// QueuedTTLSeconds is the deprecated name of MarkerTTLSeconds, accepted so an
	// older arm's config still parses.
	QueuedTTLSeconds float64 `json:"queuedTtlSeconds,omitempty"`
	// SweepSeconds is the cap-enforcement tick.
	SweepSeconds float64 `json:"sweepSeconds,omitempty"`
	// MaxThreads caps the thread table: the only eviction path, idlest
	// non-outstanding threads first.
	MaxThreads int `json:"maxThreads,omitempty"`
	// ForkOnBusy is the continuation rule's busy half: extend a busy tip → fork a
	// new thread. Default true (the design's rule). False makes every match a
	// continuation — a measurement arm only.
	ForkOnBusy *bool `json:"forkOnBusy,omitempty"`
	// StickyBusyGuard is H1: a request that extends a tip AND begins with the
	// tip's assistant reply is a proven continuation even when the thread looks
	// busy. Default true.
	StickyBusyGuard *bool `json:"stickyBusyGuard,omitempty"`
	// RetryRule is S2: a request identical to a tip whose thread has nothing in
	// flight (only arrival markers) is a client retry, not a sibling. Default true.
	RetryRule *bool `json:"retryRule,omitempty"`
	// LogRequests emits one JSON line per request on stdout:
	//
	//	thread-identity: request {"seq":…,"scope":…,"derived":…,"outcome":…}
	//
	// Default FALSE — it is an INSTRUMENT for scoring a shadow arm whose derived
	// id reaches no span. It changes no state and no decision.
	LogRequests bool `json:"logRequests,omitempty"`
	// LogHeaders adds the client's own identity headers to that line, for an
	// UNSCOPED arm where no session header is sent and the census cannot
	// attribute a thread to a client session.
	LogHeaders bool `json:"logHeaders,omitempty"`
	// LogHeaderNames extends the logHeaders whitelist. A pure observer: no header
	// here is read by the identity, the scope or any decision.
	LogHeaderNames []string `json:"logHeaderNames,omitempty"`
}

// loggedHeaderNames is the built-in logHeaders whitelist: the client's
// correlation id (the client's own session ground truth) and its request id.
var loggedHeaderNames = []string{"x-correlation-id", "x-request-id"}

// loggedHeaderList returns the effective whitelist: built-ins plus extras,
// lowercased and de-duplicated (request headers are stored lowercased).
func loggedHeaderList(extra []string) []string {
	out := append([]string(nil), loggedHeaderNames...)
	seen := make(map[string]struct{}, len(out)+len(extra))
	for _, n := range out {
		seen[n] = struct{}{}
	}
	for _, n := range extra {
		n = strings.ToLower(strings.TrimSpace(n))
		if n == "" {
			continue
		}
		if _, dup := seen[n]; dup {
			continue
		}
		seen[n] = struct{}{}
		out = append(out, n)
	}
	return out
}

// maxLoggedHeaderLen truncates a copied header value so one malformed client
// cannot blow up a log line.
const maxLoggedHeaderLen = 128

// validate resolves defaults in place and rejects impossible values.
func (p *Parameters) validate() error {
	if p.BlockSizeTokens == 0 {
		p.BlockSizeTokens = defaultBlockSizeTokens
	}
	if p.BlockSizeTokens < 0 {
		return fmt.Errorf("blockSizeTokens must be > 0, got %d", p.BlockSizeTokens)
	}
	if len(p.TokenProducer) == 0 {
		return fmt.Errorf("tokenProducer is required: the identity is a token-space key and must be rendered (modelName/vllm)")
	}
	if p.MarkerTTLSeconds == 0 {
		p.MarkerTTLSeconds = p.QueuedTTLSeconds
	}
	if p.MarkerTTLSeconds == 0 {
		p.MarkerTTLSeconds = defaultMarkerTTLSeconds
	}
	if p.MarkerTTLSeconds < 0 {
		return fmt.Errorf("markerTtlSeconds must be >= 0, got %v", p.MarkerTTLSeconds)
	}
	if p.SweepSeconds == 0 {
		p.SweepSeconds = defaultSweepSeconds
	}
	if p.SweepSeconds <= 0 {
		return fmt.Errorf("sweepSeconds must be > 0, got %v", p.SweepSeconds)
	}
	if p.MaxThreads == 0 {
		p.MaxThreads = defaultMaxThreads
	}
	if p.MaxThreads < 0 {
		return fmt.Errorf("maxThreads must be > 0, got %d", p.MaxThreads)
	}
	return nil
}

var (
	_ fwkrc.RequestHeaderProcessor = &Plugin{}
	_ fwkrc.PreRequest             = &Plugin{}
	_ fwkrc.ResponseBodyProcessor  = &Plugin{}
	_ fwkplugin.ProducerPlugin     = &Plugin{}
	_ fwkplugin.StateDumper        = &Plugin{}
	_ fwkdl.Registrant             = &Plugin{}
)

// Plugin derives thread identity from the request body and owns the thread table.
// One instance serves every hook: the busy/tip state is a single mutex-guarded
// struct rather than a cross-plugin channel.
type Plugin struct {
	typedName fwkplugin.TypedName

	idx     *threadIndex
	metrics *threadMetrics
	// handle is the runtime's plugin set (nil in a standalone build): the flow-key ownership
	// check reads the sibling plugins' declarations from it at startup.
	handle fwkplugin.Handle
	// tokens renders the request's real tokens: the upstream token-producer
	// plugin, driven as a library through its exported Plugin API; tokenize is
	// the call site, so a test can substitute a deterministic renderer.
	tokens    *tokenizer.Plugin
	tokenize  func(ctx context.Context, request *fwksched.InferenceRequest) error
	blockSize int

	markerTTL      time.Duration
	sweep          time.Duration
	forkOnBusy     bool
	sticky         bool
	retry          bool
	logRequests    bool
	logHeaders     bool
	logHeadersList []string
	// now is the clock seam: the stage-0c replay drives recorded arrival /
	// dispatch / settle times through the real hooks.
	now func() time.Time

	// logger carries the startup echo (default level) and the per-request instrument
	// (V(DEBUG), off by default).
	logger logr.Logger

	anonSeq atomic.Uint64
	// logSeq orders the LogRequests instrument's lines in arrival order.
	logSeq atomic.Uint64
}

// Factory builds the thread-identity plugin.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{}
	if rawParameters != nil {
		// STRICT: an unknown parameter is a configuration error, not something to ignore. The
		// identity's surface shrank when scoping was removed (no sessionHeaders), and a stale
		// key must fail the startup loudly instead of silently doing nothing.
		rawParameters.DisallowUnknownFields()
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", PluginType, err)
	}

	ctx := context.Background()
	if handle != nil {
		ctx = handle.Context()
	}
	typedName := fwkplugin.TypedName{Type: PluginType, Name: name}
	// The token producer is the upstream plugin itself, built through its exported
	// factory so its configuration language, render backend, warmup and endpoint
	// discovery are the ones every other token consumer gets. A nil handle (a
	// standalone build in a test) still works: the shim below only supplies the
	// context the factory reads.
	runtimeHandle := handle
	if handle == nil {
		handle = backgroundHandle{}
	}
	built, err := tokenizer.PluginFactory(name, json.NewDecoder(bytes.NewReader(params.TokenProducer)), handle)
	if err != nil {
		return nil, fmt.Errorf("failed to build the token producer of the '%s' plugin: %w", PluginType, err)
	}
	producer, ok := built.(*tokenizer.Plugin)
	if !ok {
		return nil, fmt.Errorf("the token producer of the '%s' plugin built a %T", PluginType, built)
	}
	// FAIL CLOSED: a non-rendering backend leaves the identity in byte space, which
	// is the retired v1 key. A missing block is caught in validate(); this catches a
	// block that parses but selects no renderer.
	if producer.Backend() != tokenizer.BackendVLLM {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: tokenProducer selected the %q backend — "+
			"identity is a token-space key, so the block must render (modelName/vllm)", PluginType, producer.Backend())
	}

	p := &Plugin{
		typedName: typedName,
		handle:    runtimeHandle,
		tokens:    producer,
		tokenize: func(ctx context.Context, request *fwksched.InferenceRequest) error {
			// nil endpoints: the render looks the request's own body up, it never
			// scores candidates — the same call the director's data-producer phase
			// makes at dispatch, so both find the same one-render-per-request skip.
			return producer.Produce(ctx, request, nil)
		},
		blockSize:      params.BlockSizeTokens,
		markerTTL:      time.Duration(params.MarkerTTLSeconds * float64(time.Second)),
		sweep:          time.Duration(params.SweepSeconds * float64(time.Second)),
		forkOnBusy:     params.ForkOnBusy == nil || *params.ForkOnBusy,
		sticky:         params.StickyBusyGuard == nil || *params.StickyBusyGuard,
		retry:          params.RetryRule == nil || *params.RetryRule,
		logRequests:    params.LogRequests,
		logHeaders:     params.LogRequests && params.LogHeaders,
		logHeadersList: loggedHeaderList(params.LogHeaderNames),
		logger:         log.FromContext(ctx).WithName(PluginType + "/" + name),
		now:            time.Now,
	}
	p.idx = newThreadIndex(params.BlockSizeTokens, p.markerTTL, params.MaxThreads)
	p.metrics = newThreadMetrics()

	// The effective config is echoed at the DEFAULT level, always: a run's arm —
	// including the render backend — must be verifiable from its own log. Rendered as
	// key=value pairs, not JSON: a logger escapes quotes inside a message.
	p.logger.V(logutil.DEFAULT).Info(PluginType + ": effective config " + paramsEcho(params))

	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := p.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", PluginType, err)
			}
		}
		if p.idx.maxThreads > 0 {
			go p.runSweeper(handle.Context())
		}
	}
	return p, nil
}

// paramsEcho renders the parameters as sorted key=value pairs (the startup echo's form).
func paramsEcho(params Parameters) string {
	raw, err := json.Marshal(params)
	if err != nil {
		return ""
	}
	fields := map[string]any{}
	if err := json.Unmarshal(raw, &fields); err != nil {
		return string(raw)
	}
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%v", k, fields[k]))
	}
	return strings.Join(parts, " ")
}

// BlockSize is the identity block length. It is the prefix scorer's B, so a
// consumer that charges the published blocks must be configured with the same
// value; the scheduler's startup gate compares the two.
func (p *Plugin) BlockSize() int { return p.blockSize }

// TypedName returns the type and name of the plugin.
func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// Produces declares the attributes written at the request-header hook. The thread id is
// published twice: under its own key, and under the flow-key attribute
// (agentidentity.AgentIdentityKey) — the one attribute the director turns into the request's
// flow key. The key carries no producer name by design ("any plugin able to resolve an agent
// identity is an acceptable source"), and the director's derivation runs on a request this
// hook can only reach through its attribute store: the scope hands a RequestHeader processor a
// copy whose struct fields are not shared, so a fairness id written to the field would be lost.
func (p *Plugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		agentidentity.AgentIdentityKey: "",
		ThreadIdentityKey:              "",
		ThreadArrivalKey:               ThreadArrival{},
		ThreadBlocksKey:                []PromptBlocks{},
		threadMarkerKey:                "",
	}
}

// RegisterDependencies forwards the render backend's endpoint discovery, so a
// discovery-backed renderer sees the endpoint lifecycle like the token-producer
// plugin's does.
func (p *Plugin) RegisterDependencies(r fwkdl.Registrar) error {
	if err := p.refuseForeignFlowKeyUsers(); err != nil {
		return err
	}
	return p.tokens.RegisterDependencies(r)
}

// refuseForeignFlowKeyUsers is the fail-closed ownership check for the flow-key attribute. This
// plugin publishes its thread id on agentidentity.AgentIdentityKey because that is the one
// attribute the director turns into a request's flow key, so it must be the attribute's ONLY
// owner: no other configured plugin may produce it (a second writer overwrites the queue key,
// by configuration order, with no error from the framework) or consume it (a reader of that
// key would take thread ids for session ids). It reads the siblings' DECLARATIONS, not their
// types, so it covers agent-identity, thunderagent and sessionstate today and any later
// reader or writer. A plugin that reads the key without declaring it cannot be seen here — the
// README lists that as a drawback.
func (p *Plugin) refuseForeignFlowKeyUsers() error {
	if p.handle == nil {
		return nil // standalone build: there are no siblings to ask
	}
	var conflicts []string
	for _, other := range p.handle.GetAllPlugins() {
		if other == nil || other.TypedName() == p.typedName {
			continue
		}
		if producer, ok := other.(fwkplugin.ProducerPlugin); ok {
			if _, declared := producer.Produces()[agentidentity.AgentIdentityKey]; declared {
				conflicts = append(conflicts, other.TypedName().String()+" produces it")
			}
		}
		if consumer, ok := other.(fwkplugin.ConsumerPlugin); ok {
			deps := consumer.Consumes()
			_, required := deps.Required[agentidentity.AgentIdentityKey]
			_, optional := deps.Optional[agentidentity.AgentIdentityKey]
			if required || optional {
				conflicts = append(conflicts, other.TypedName().String()+" consumes it")
			}
		}
	}
	if len(conflicts) == 0 {
		return nil
	}
	sort.Strings(conflicts)
	return fmt.Errorf("the '%s' plugin owns the flow-key attribute '%s': the director turns that one attribute into every request's flow key, "+
		"so there is exactly one owner. Remove the conflicting plugin(s) from the configuration — a second writer would silently overwrite the "+
		"queue key by configuration order, and a reader would take thread ids for session ids: %s",
		PluginType, agentidentity.AgentIdentityKey, strings.Join(conflicts, "; "))
}

// RequestHeader derives the thread id before admission control and publishes it as the
// request's flow key — the flow-control queue key must exist before the request enqueues,
// and this hook runs after body parsing but before Admit. The director reads the flow-key
// attribute after the header processors and keys the request's queue on it; an explicit
// fairness header still wins, and a tenant header composes with it ("tenant/thread") when
// the pool turns composition on.
//
// It renders the request's tokens HERE, on arrival, so the dispatch-time token
// producer finds them already present and skips: one render per request, shared
// by identity and every prefix scorer.
//
// It also publishes the thread, the session (when the client sent one) and the
// arrival record. It never fails the request: a request the render cannot
// describe gets its own thread (counted) and the request proceeds.
func (p *Plugin) RequestHeader(ctx context.Context, request *fwksched.InferenceRequest) error {
	if request == nil || request.Body == nil {
		return nil
	}
	// An explicit x-llm-d-inference-fairness-id header is the tenant (the director's derivation
	// gives it precedence, or composes it with the flow-key attribute). It is counted, because a
	// harness that sets it overrides the thread as the queue key with no other visible signal.
	explicit := request.FairnessID != ""

	blocks, c, ok := p.chainFor(ctx, request)
	if len(blocks) > 0 {
		request.PutAttribute(ThreadBlocksKey, blocks)
	}
	roles := messageRoles(request.Body)
	marker := p.markerID(request)

	res := p.idx.Resolve(p.now(), Arrival{Chain: c, Roles: roles, Marker: marker},
		p.forkOnBusy, p.sticky, p.retry)
	p.metrics.observeResolve(res)
	if !ok {
		// A render failure leaves the request without an identity: it gets its own
		// thread (never a continuation) and the counter names the failure.
		p.metrics.observeRenderFailure()
	}

	request.PutAttribute(ThreadIdentityKey, res.ID)
	// ALWAYS published, header or not: with no header the thread is the flow key; with a
	// tenant header and composition on it is the second half of "tenant/thread" (skipping it
	// here would collapse each tenant into one queue); with composition off the director keys
	// the tenant as sent and never reads it.
	request.PutAttribute(agentidentity.AgentIdentityKey, res.ID)
	request.PutAttribute(ThreadArrivalKey, ThreadArrival{Outcome: res.Outcome, Turn: res.Turn})
	request.PutAttribute(threadMarkerKey, marker)
	if explicit {
		p.metrics.observeExplicitOverride()
	}
	p.logRequest(request, res, explicit, c)
	return nil
}

// chainFor renders the request's tokens and hashes them into its identity chain.
// ok is false when the request has no identity: a render error, or a body that
// renders to no tokens.
//
// The render runs under the REQUEST's context and carries no time constant of its
// own: it is bounded by the backend's own render timeout (the token producer's
// `timeout`) and by the client abandoning the request. A separate, shorter bound
// here would turn an arrival burst or a loaded sidecar into a false NEW thread —
// an identity we mint because we gave up, not because the content was new.
func (p *Plugin) chainFor(ctx context.Context, request *fwksched.InferenceRequest) ([]PromptBlocks, chain, bool) {
	start := p.now()
	err := p.tokenize(ctx, request)
	p.metrics.observeRender(p.now().Sub(start))
	if err != nil {
		return nil, chain{}, false
	}

	start = p.now()
	blocks := promptBlocks(ctx, request, p.blockSize)
	c := identityChain(blocks, p.blockSize)
	p.metrics.observeChain(p.now().Sub(start))
	if len(c.blocks) == 0 {
		return nil, chain{}, false
	}
	if prompts := tokenizedPrompts(request); prompts > 1 {
		p.metrics.addMultiPrompt(prompts)
	}
	return blocks, c, true
}

// promptBlocks hashes a tokenized request into its raw per-prompt block keys — one
// PromptBlocks per prompt, each prompt seeded independently. These are the blocks
// the scheduler charges, published as they are (see ThreadBlocksKey).
func promptBlocks(ctx context.Context, request *fwksched.InferenceRequest, blockSize int) []PromptBlocks {
	hashes, promptTokens := prefixhash.GetBlockHashesWithPromptTokens(ctx, request, blockSize, maxChainBlocks)
	if len(hashes) == 0 {
		return nil
	}
	blocks := make([]PromptBlocks, len(hashes))
	for i, prompt := range hashes {
		blocks[i] = PromptBlocks{Keys: prompt, Tokens: promptTokens[i]}
	}
	return blocks
}

// identityChain hashes a tokenized request into its identity chain, in the prefix
// scorer's own block-hash space (model + cache salt seeded, chained). ALL prompts
// count: their chains are concatenated in order.
//
// MULTI-PROMPT RE-CHAIN (F1). prefixhash seeds each prompt independently from the
// model and cache salt and chains within the prompt, which is right for the prefix
// scorer: its keys must match the engine's, and the engine hashes each sequence
// (each prompt) from its own start. But the identity chain is ONE flat chain over
// the concatenation, so an independently seeded prompt would put the same block
// keys in every request carrying that prompt: tips [A,C] and [B,C] with |A|=|B|
// would share the keys of C and look like a continuation of one another — a false
// continuation, the direction that hurts here (it merges two conversations' queue
// keys). So beyond the FIRST prompt the chain CONTINUES: every block of a later
// prompt is re-chained onto the previous block's id, id_i = H(block_i, id_{i-1}),
// exactly the chaining prefixhash already applies inside a prompt. A single-prompt
// request — every prompt of the WEKA and agentic corpora as measured — is
// untouched: it keeps prefixhash's hashes verbatim.
func identityChain(blocks []PromptBlocks, blockSize int) chain {
	if len(blocks) == 0 {
		return chain{}
	}
	total := 0
	for _, prompt := range blocks {
		total += len(prompt.Keys)
	}
	c := chain{blocks: make([]prefixhash.BlockHash, 0, total)}
	for i, prompt := range blocks {
		for _, h := range prompt.Keys {
			if i > 0 {
				h = rechainBlock(h, c.blocks[len(c.blocks)-1])
			}
			c.blocks = append(c.blocks, h)
		}
		c.tokens += prompt.Tokens
	}
	c.fullBlocks = len(c.blocks)
	if last := blocks[len(blocks)-1].Tokens; last%blockSize != 0 {
		// The last prompt ends mid-block: that block is the tip's partial block.
		c.fullBlocks--
	}
	return c
}

// rechainBlock continues a block chain across a prompt boundary: the block's own
// hash, chained onto the previous block's id, in the same 16-byte serialization
// prefixhash uses between two blocks of one prompt.
func rechainBlock(h, prev prefixhash.BlockHash) prefixhash.BlockHash {
	var buf [16]byte
	prefixhash.PutBlockHash(buf[:8], h)
	prefixhash.PutBlockHash(buf[8:], prev)
	return prefixhash.BlockHash(xxhash.Sum64(buf[:]))
}

// tokenizedPrompts is the number of prompts the render produced.
func tokenizedPrompts(request *fwksched.InferenceRequest) int {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return 0
	}
	return len(request.Body.TokenizedRequest.Prompts)
}

// logRequest emits the per-request instrument line when Parameters.LogRequests is
// on (off by default): one write per line, in arrival order, carrying the client
// scope, the derived thread, the outcome and the identity size.
func (p *Plugin) logRequest(request *fwksched.InferenceRequest, res ResolveResult, explicit bool, c chain) {
	if !p.logRequests {
		return
	}
	// One line per request is measurement detail: V(DEBUG), so it is off unless asked for.
	l := p.logger.V(logutil.DEBUG)
	if !l.Enabled() {
		return
	}
	// STRUCTURED fields, not a JSON blob: the logger escapes an embedded object, which
	// no parser can read. The keys are the instrument's contract.
	kv := []any{
		"seq", p.logSeq.Add(1),
		"ts", p.now().UTC().Format(time.RFC3339Nano),
		"derived", res.ID,
		"outcome", string(res.Outcome),
		"turn", res.Turn,
		"explicit", explicit,
		"blocks", c.fullBlocks,
		"tokens", c.tokens,
		"marker", markerOf(request),
	}
	if res.Retry {
		kv = append(kv, "retry", true)
	}
	if res.StickyBusy {
		kv = append(kv, "sticky", true)
	}
	if hdrs := p.loggedHeaders(request); len(hdrs) > 0 {
		for k, v := range hdrs {
			kv = append(kv, "header_"+k, v)
		}
	}
	l.Info(PluginType+": request", kv...)
}

// loggedHeaders copies the whitelisted client headers from the request (lowercase
// keys, as the handler stores them; a case-insensitive scan covers a client that
// framed them otherwise). Pure read — no attribute, no id, no decision.
func (p *Plugin) loggedHeaders(request *fwksched.InferenceRequest) map[string]string {
	if !p.logHeaders || request == nil || len(request.Headers) == 0 {
		return nil
	}
	out := make(map[string]string, len(p.logHeadersList))
	for _, name := range p.logHeadersList {
		v := request.Headers[name]
		if v == "" {
			for k, cand := range request.Headers {
				if strings.EqualFold(k, name) {
					v = cand
					break
				}
			}
		}
		if v == "" {
			continue
		}
		if len(v) > maxLoggedHeaderLen {
			v = v[:maxLoggedHeaderLen]
		}
		out[name] = v
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// PreRequest records the DISPATCH transition, moving this request's marker from
// its thread's pending set to its in-flight set. The thread is already busy from
// ARRIVAL, so nothing about identity depends on this hook; it names WHICH requests
// are running.
func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	if request == nil {
		return nil
	}
	if id, ok := fwksched.ReadRequestAttribute[string](request, ThreadIdentityKey); ok && id != "" {
		p.idx.Dispatch(id, markerOf(request), p.now())
	}
	return nil
}

// ResponseBody settles the thread at the end of the response lifecycle. The
// director forces this call on abort/disconnect for every dispatched request, so
// the busy flag cannot wedge.
func (p *Plugin) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	if id, ok := fwksched.ReadRequestAttribute[string](request, ThreadIdentityKey); ok && id != "" {
		p.idx.Settle(id, markerOf(request), p.now())
	}
}

// Produce implements DataProducer ONLY to receive RequestAbandoned: the framework has the event but
// delivers it to data producers, so the plugin has to be one to hear it. It writes nothing, and it
// is not a step in the derivation.
func (p *Plugin) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

// RequestAbandoned releases the request's marker. A request that leaves the path before a pod serves
// it never reaches the end of a response, so nothing else would release its marker — and while the
// marker stands the thread's tip is BUSY, so the client's next turn (which extends the same tip) is
// forced to FORK: a new thread id, a new flow queue, a new seat, for a turn that is the same
// conversation. This releases exactly what the end of stream releases, and nothing more.
func (p *Plugin) RequestAbandoned(_ context.Context, request *fwksched.InferenceRequest) {
	if request == nil {
		return
	}
	if id, ok := fwksched.ReadRequestAttribute[string](request, ThreadIdentityKey); ok && id != "" {
		// RELEASE, not Settle: this request was never dispatched, so its marker is still PENDING and
		// an end-of-stream release (in flight only) would leave it standing.
		p.idx.Release(id, markerOf(request), p.now())
	}
}

// DumpState reports the thread table.
func (p *Plugin) DumpState() (json.RawMessage, error) {
	snap := p.idx.Snapshot(time.Now())
	p.metrics.setRecords(snap.Threads)
	return json.Marshal(snap)
}

// runSweeper is the background sweep that drops markers whose TTL has passed.
func (p *Plugin) runSweeper(ctx context.Context) {
	ticker := time.NewTicker(p.sweep)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.sweepOnce(time.Now())
		}
	}
}

// sweepOnce enforces the table cap and refreshes the records gauge. There is no
// idle GC (see threadIndex.EnforceCap): a dormant client that returns must find
// its thread, or it is minted a fresh id and re-enters as a newcomer.
func (p *Plugin) sweepOnce(now time.Time) {
	if removed := p.idx.EnforceCap(now); removed > 0 {
		p.metrics.addEvicted(removed)
	}
	p.metrics.setRecords(p.idx.Len())
}

// markerID returns the request's marker handle for the thread's outstanding set:
// its request id when the client or framework supplied one, a per-request
// synthetic id otherwise.
func (p *Plugin) markerID(request *fwksched.InferenceRequest) string {
	if id := strings.TrimSpace(request.RequestID); id != "" {
		return id
	}
	return fmt.Sprintf("anon-%d", p.anonSeq.Add(1))
}

// markerOf reads the marker an earlier hook published, falling back to the
// request id so the operations stay no-ops rather than corrupting another
// request's marker.
func markerOf(request *fwksched.InferenceRequest) string {
	if marker, ok := fwksched.ReadRequestAttribute[string](request, threadMarkerKey); ok && marker != "" {
		return marker
	}
	return request.RequestID
}

// messageRoles returns the request's message roles in wire order. Body shapes
// without a message array return nil: H1 needs message structure and cannot fire
// for them.
func messageRoles(body *fwkrequest.InferenceRequestBody) []string {
	if body == nil {
		return nil
	}
	switch {
	case body.ChatCompletions != nil:
		msgs := body.ChatCompletions.Messages
		if len(msgs) == 0 {
			return nil
		}
		roles := make([]string, len(msgs))
		for i, m := range msgs {
			roles[i] = m.Role
		}
		return roles
	case body.Messages != nil:
		msgs := body.Messages.Messages
		if len(msgs) == 0 {
			return nil
		}
		roles := make([]string, len(msgs))
		for i, m := range msgs {
			roles[i] = m.Role
		}
		return roles
	}
	return nil
}

// backgroundHandle is the nil-handle fallback for a plugin built without a runtime handle (a
// standalone test build): the tokenizer factory reads the context and the metrics recorder, so
// both answer without a runtime, and every other method is a no-op. A plugin built WITH a handle
// uses the real one.
type backgroundHandle struct{}

// Context is the process context the tokenizer factory and the warmup read.
func (backgroundHandle) Context() context.Context { return context.Background() }

// Metrics reports no recorder: a standalone build registers no series.
func (backgroundHandle) Metrics() fwkplugin.MetricsRecorder { return nil }

// Plugin reports no plugin under any name: there is no runtime plugin set.
func (backgroundHandle) Plugin(string) fwkplugin.Plugin { return nil }

// AddPlugin ignores the addition: a standalone build has no plugin set to maintain.
func (backgroundHandle) AddPlugin(string, fwkplugin.Plugin) {}

// GetAllPlugins reports none: there is no runtime plugin set.
func (backgroundHandle) GetAllPlugins() []fwkplugin.Plugin { return nil }

// GetAllPluginsWithNames reports none: there is no runtime plugin set.
func (backgroundHandle) GetAllPluginsWithNames() map[string]fwkplugin.Plugin { return nil }

// PodList reports no pods: a standalone build has no datastore behind it.
func (backgroundHandle) PodList() []types.NamespacedName { return nil }

// CrossReplicaSyncer reports none: a standalone build has no cross-replica sync.
func (backgroundHandle) CrossReplicaSyncer() fwkplugin.Plugin { return nil }

// SetCrossReplicaSyncer ignores the assignment: there is no runtime to hold it.
func (backgroundHandle) SetCrossReplicaSyncer(fwkplugin.Plugin) {}

// RefreshMetricsInterval reports zero: a standalone build polls nothing.
func (backgroundHandle) RefreshMetricsInterval() time.Duration { return 0 }
