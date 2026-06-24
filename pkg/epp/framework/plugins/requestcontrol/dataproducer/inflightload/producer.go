/*
Copyright 2026 The Kubernetes Authors.

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

package inflightload

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	inflightloadconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload/constants"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

const (
	InFlightLoadProducerType = inflightloadconstants.InFlightLoadProducerType
	profilePrefill           = "prefill"
	maxDebugDumpEndpoints    = 100
)

// Config controls optional behaviors of InFlightLoadProducer.
type Config struct {
	// AddEstimatedOutputTokens controls whether estimated output tokens are added to
	// the in-flight token counter. Defaults to false.
	AddEstimatedOutputTokens bool `json:"addEstimatedOutputTokens"`
	// OutputRatio is the estimated output-to-input token ratio used when
	// AddEstimatedOutputTokens is true: estimated output = round(inputTokens * OutputRatio).
	// Must be non-negative. Unset defaults to DefaultOutputRatio.
	OutputRatio *float64 `json:"outputRatio,omitempty"`
	// MaxEstimatedOutputTokens optionally caps the estimated output tokens added per
	// request when AddEstimatedOutputTokens is true, regardless of input length or
	// the client-requested output cap. Must be non-negative. Unset means no cap.
	MaxEstimatedOutputTokens *int64 `json:"maxEstimatedOutputTokens,omitempty"`
	// PrefixMatchInfoProducerName selects which prefix-cache producer's
	// PrefixCacheMatchInfo to read for the cached-prefix discount. Empty defaults
	// to the approximate-prefix producer; set it to a precise-prefix-cache
	// producer's instance name to discount against precise cache state instead.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName,omitempty"`
}

func defaultConfig() Config {
	return Config{AddEstimatedOutputTokens: false}
}

func InFlightLoadProducerFactory(name string, decoder *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("handle is nil")
	}
	ctx := handle.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	cfg := defaultConfig()
	if decoder != nil {
		if err := decoder.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to decode inflight-load-producer parameters: %w", err)
		}
	}

	outputRatio := DefaultOutputRatio
	if cfg.OutputRatio != nil {
		if *cfg.OutputRatio < 0 {
			return nil, fmt.Errorf("outputRatio must be non-negative, got %v", *cfg.OutputRatio)
		}
		outputRatio = *cfg.OutputRatio
	}

	if cfg.MaxEstimatedOutputTokens != nil && *cfg.MaxEstimatedOutputTokens < 0 {
		return nil, fmt.Errorf("maxEstimatedOutputTokens must be non-negative, got %v", *cfg.MaxEstimatedOutputTokens)
	}

	return &InFlightLoadProducer{
		typedName:                fwkplugin.TypedName{Type: InFlightLoadProducerType, Name: name},
		requestTracker:           newConcurrencyTracker(),
		tokenTracker:             newConcurrencyTracker(),
		tokenEstimator:           NewSimpleTokenEstimatorWithConfig(outputRatio, cfg.MaxEstimatedOutputTokens),
		addEstimatedOutputTokens: cfg.AddEstimatedOutputTokens,
		dk:                       attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(name),
		prefixMatchInfoDK:        attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(cfg.PrefixMatchInfoProducerName),
		uncachedRequestTokensDk:  attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(name),
		PluginState:              fwkplugin.NewPluginState(ctx),
	}, nil
}

var (
	_ requestcontrol.PreRequest            = &InFlightLoadProducer{}
	_ requestcontrol.ResponseBodyProcessor = &InFlightLoadProducer{}
	_ requestcontrol.DataProducer          = &InFlightLoadProducer{}
	_ datalayer.EndpointExtractor          = (*InFlightLoadProducer)(nil)
	_ datalayer.Registrant                 = &InFlightLoadProducer{}
	_ fwkplugin.ConsumerPlugin             = &InFlightLoadProducer{}
	_ fwkplugin.StateDumper                = &InFlightLoadProducer{}
)

type InFlightLoadProducer struct {
	typedName                fwkplugin.TypedName
	requestTracker           *concurrencyTracker
	tokenTracker             *concurrencyTracker
	tokenEstimator           TokenEstimator
	addEstimatedOutputTokens bool
	PluginState              *fwkplugin.PluginState
	dk                       fwkplugin.DataKey
	prefixMatchInfoDK        fwkplugin.DataKey
	uncachedRequestTokensDk  fwkplugin.DataKey
	registeredEndpoints      sync.Map // key: string (NamespacedName), value: datalayer.Endpoint
}

// addedTokensEntry tracks a request's contribution to the global token and
// request counters. OnEvicted rolls back the contribution exactly once,
// whether triggered by explicit release at end-of-stream or by the janitor's
// TTL reaper. The fields are atomic so releaseTokensEarly and OnEvicted
// can race safely: whichever swaps first does the decrement, the other
// sees 0 and is a no-op.
type addedTokensEntry struct {
	endpointID     string
	tokens         atomic.Int64
	tokenTracker   *concurrencyTracker
	requestTracker *concurrencyTracker
	requests       atomic.Int32
}

var _ fwkplugin.EvictableStateData = (*addedTokensEntry)(nil)

// Clone returns a distinct copy of the entry with the current atomic values.
// The tracker references remain shared, but the cloned state object itself is
// independent so later mutation or eviction of the clone does not alias the
// original entry.
func (e *addedTokensEntry) Clone() fwkplugin.StateData {
	if e == nil {
		return nil
	}
	clone := &addedTokensEntry{
		endpointID:     e.endpointID,
		tokenTracker:   e.tokenTracker,
		requestTracker: e.requestTracker,
	}
	clone.tokens.Store(e.tokens.Load())
	clone.requests.Store(e.requests.Load())
	return clone
}

// addIfPresent applies delta only when the endpoint is still tracked.
// This avoids recreating a deleted endpoint with a negative in-flight count
// during delayed eviction cleanup.
func (t *concurrencyTracker) addIfPresent(endpointID string, delta int64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	counter, ok := t.counts[endpointID]
	if !ok {
		return
	}
	counter.Add(delta)
}

// decIfPresent decrements the endpoint only when it is still tracked.
func (t *concurrencyTracker) decIfPresent(endpointID string) {
	t.addIfPresent(endpointID, -1)
}

func (e *addedTokensEntry) OnEvicted(_ string, _ fwkplugin.StateKey) {
	if t := e.tokens.Swap(0); t != 0 {
		e.tokenTracker.addIfPresent(e.endpointID, -t)
	}
	if e.requests.Swap(0) != 0 {
		e.requestTracker.decIfPresent(e.endpointID)
	}
}

type inFlightLoadState struct {
	Endpoints      []endpointInFlightLoadState `json:"endpoints"`
	TotalEndpoints int                         `json:"totalEndpoints"`
	MaxEndpoints   int                         `json:"maxEndpoints"`
	Truncated      bool                        `json:"truncated"`
}

type endpointInFlightLoadState struct {
	Endpoint string `json:"endpoint"`
	Requests int64  `json:"requests"`
	Tokens   int64  `json:"tokens"`
}

func (p *InFlightLoadProducer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// DumpState implements [fwkplugin.StateDumper] and exposes per-endpoint
// in-flight request and token counts for the /debug/plugins/state endpoint.
//
// The request and token tracker maps are snapshotted under separate read
// locks, so the returned per-endpoint Requests and Tokens values are not
// guaranteed to correspond to the same instant in time and the endpoint set
// itself may change between the two snapshots. This is acceptable for a
// debug endpoint, where best-effort visibility is preferred over coordinating
// a single global lock that would contend with the hot path.
//
// The endpoint list is capped to the busiest endpoints to keep the debug
// payload bounded when a deployment has a large endpoint set.
func (p *InFlightLoadProducer) DumpState() (json.RawMessage, error) {
	state := p.snapshotState()
	return json.Marshal(state)
}

func (p *InFlightLoadProducer) snapshotState() inFlightLoadState {
	requestCounts := map[string]int64{}
	if p.requestTracker != nil {
		requestCounts = p.requestTracker.snapshot()
	}

	tokenCounts := map[string]int64{}
	if p.tokenTracker != nil {
		tokenCounts = p.tokenTracker.snapshot()
	}

	endpointSet := make(map[string]struct{}, len(requestCounts)+len(tokenCounts))
	for endpointID := range requestCounts {
		endpointSet[endpointID] = struct{}{}
	}
	for endpointID := range tokenCounts {
		endpointSet[endpointID] = struct{}{}
	}

	endpointIDs := make([]string, 0, len(endpointSet))
	for endpointID := range endpointSet {
		endpointIDs = append(endpointIDs, endpointID)
	}
	sort.Strings(endpointIDs)

	state := inFlightLoadState{
		Endpoints:      make([]endpointInFlightLoadState, 0, len(endpointIDs)),
		TotalEndpoints: len(endpointIDs),
		MaxEndpoints:   maxDebugDumpEndpoints,
	}
	for _, endpointID := range endpointIDs {
		state.Endpoints = append(state.Endpoints, endpointInFlightLoadState{
			Endpoint: endpointID,
			Requests: requestCounts[endpointID],
			Tokens:   tokenCounts[endpointID],
		})
	}

	sort.SliceStable(state.Endpoints, func(i, j int) bool {
		iLoad := state.Endpoints[i].Requests + state.Endpoints[i].Tokens
		jLoad := state.Endpoints[j].Requests + state.Endpoints[j].Tokens
		if iLoad != jLoad {
			return iLoad > jLoad
		}
		return state.Endpoints[i].Endpoint < state.Endpoints[j].Endpoint
	})
	if len(state.Endpoints) > maxDebugDumpEndpoints {
		state.Endpoints = state.Endpoints[:maxDebugDumpEndpoints]
		state.Truncated = true
	}

	return state
}

// RegisterDependencies declares that this plugin needs an endpoint-notification-source to track
// endpoint lifecycle events. The source is auto-created if not already in the config.
func (p *InFlightLoadProducer) RegisterDependencies(r datalayer.Registrar) error {
	return r.Register(datalayer.PendingRegistration{
		Owner:         p.TypedName(),
		SourceType:    sourcenotifications.EndpointNotificationSourceType,
		Extractor:     p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// Extract handles endpoint lifecycle events to manage dynamic attributes.
func (p *InFlightLoadProducer) Extract(ctx context.Context, event datalayer.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}

	id := event.Endpoint.GetMetadata().NamespacedName.String()

	switch event.Type {
	case datalayer.EventDelete:
		// This guard assumes the datalayer delivers the same Endpoint pointer for
		// delete as was used for the preceding add. If the datalayer ever
		// reconstructs endpoint objects on delete, this check would need to use a
		// generation counter instead of pointer identity.
		if registered, ok := p.registeredEndpoints.Load(id); ok && registered != event.Endpoint {
			log.FromContext(ctx).V(logutil.DEFAULT).Info("Ignoring stale delete for replaced endpoint", "endpoint", id)
			break
		}
		p.registeredEndpoints.Delete(id)
		p.DeleteEndpoint(id)
		log.FromContext(ctx).V(logutil.DEFAULT).Info("Cleaned up in-flight load for deleted endpoint", "endpoint", id)
	case datalayer.EventAddOrUpdate:
		p.registeredEndpoints.Store(id, event.Endpoint)
		event.Endpoint.GetAttributes().Put(p.dk.String(), &datalayer.DynamicAttribute{
			Get: func() datalayer.Cloneable {
				return &attrconcurrency.InFlightLoad{
					Tokens:   p.GetTokens(id),
					Requests: p.GetRequests(id),
				}
			},
		})
		log.FromContext(ctx).V(logutil.DEFAULT).Info("Injected dynamic attribute into endpoint", "key", p.dk.String(), "endpoint", id)
	}
	return nil
}

func (p *InFlightLoadProducer) Produce(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	var inputTokens int64
	if request != nil {
		inputTokens = p.tokenEstimator.EstimateInput(request)
	}

	for _, e := range endpoints {
		if e == nil || e.GetMetadata() == nil {
			continue
		}
		if request != nil {
			tokens := p.estimateRequestTokens(e, request, inputTokens)
			e.Put(p.uncachedRequestTokensDk.String(), &attrconcurrency.UncachedRequestTokens{
				Tokens: tokens,
			})
		}
	}
	return nil
}

func (p *InFlightLoadProducer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) {
	if result == nil || len(result.ProfileResults) == 0 {
		return
	}

	if request == nil {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("Skipping in-flight load tracking: request is nil")
		return
	}

	if request.RequestID == "" {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("Skipping in-flight load tracking: missing RequestID")
		return
	}

	if p.PluginState == nil {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("Skipping in-flight load tracking: PluginState is nil", "requestID", request.RequestID)
		return
	}

	inputTokens := p.tokenEstimator.EstimateInput(request)

	for profileName, profileResult := range result.ProfileResults {
		if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
			continue
		}
		// Only track the first endpoint (the primary target), as requested by reviewers.
		endpoint := profileResult.TargetEndpoints[0]
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		eid := endpoint.GetMetadata().NamespacedName.String()
		p.requestTracker.inc(eid)

		// Compute the uncached prompt portion this endpoint must actually compute.
		// Prefer the prefix producer's view (real tokens) when available so the
		// match-length and the input length are in the same units; fall back to
		// the (estimated) input tokens otherwise.
		tokens := p.estimateRequestTokens(endpoint, request, inputTokens)

		p.tokenTracker.add(eid, tokens)

		entry := &addedTokensEntry{
			endpointID:     eid,
			tokenTracker:   p.tokenTracker,
			requestTracker: p.requestTracker,
		}
		entry.tokens.Store(tokens)
		entry.requests.Store(1)
		p.PluginState.Write(
			request.RequestID,
			fwkplugin.StateKey(addedTokensKey(eid, profileName)),
			entry,
		)
	}
}

func (p *InFlightLoadProducer) estimateRequestTokens(endpoint fwksched.Endpoint, request *fwksched.InferenceRequest, inputTokens int64) int64 {
	adjustedInput := uncachedInputTokens(endpoint, inputTokens, p.prefixMatchInfoDK.String())
	tokens := adjustedInput
	if p.addEstimatedOutputTokens {
		var maxOutputTokens *int64
		if request != nil && request.Body != nil {
			maxOutputTokens = request.Body.MaxOutputTokens
		}
		// Output tokens are based on the full input, not the cached portion.
		tokens += p.tokenEstimator.EstimateOutput(inputTokens, maxOutputTokens)
	}
	return tokens
}

func (p *InFlightLoadProducer) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	resp *requestcontrol.Response,
	_ *datalayer.EndpointMetadata,
) {
	if request == nil || resp == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}

	result := request.SchedulingResult
	if result == nil {
		return
	}

	// When output tokens are excluded, the in-flight token estimate represents only
	// the prompt cost, which is consumed by prefill. As soon as the first chunk
	// arrives (StartOfStream), prefill is done across all profiles, so free the
	// token counters for every targeted endpoint regardless of profile name.
	// Request counters are still released on EndOfStream below via PluginState.Delete.
	if !p.addEstimatedOutputTokens && resp.StartOfStream {
		for profileName, profileResult := range result.ProfileResults {
			if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
				continue
			}
			endpoint := profileResult.TargetEndpoints[0]
			if endpoint == nil || endpoint.GetMetadata() == nil {
				continue
			}
			p.releaseTokensEarly(endpoint, request, profileName)
		}
	}

	// Early prefill release (on first chunk). Frees the primary profile's
	// prefill contribution as soon as prefill completes, while other profiles'
	// entries remain until EndOfStream.
	if p.addEstimatedOutputTokens && resp.StartOfStream {
		if prefillResult, ok := result.ProfileResults[profilePrefill]; ok && len(prefillResult.TargetEndpoints) > 0 {
			endpoint := prefillResult.TargetEndpoints[0]
			if endpoint != nil && endpoint.GetMetadata() != nil {
				p.release(endpoint, request, profilePrefill)
			}
		}
	}

	// Full cleanup on completion vs. lifetime extension on an intermediate chunk.
	// PluginState.Delete iterates remaining entries via per-key LoadAndDelete,
	// firing OnEvicted at most once per entry; entries already released at
	// StartOfStream are gracefully no-op'd (LoadAndDelete miss / atomic Swap-to-0).
	if resp.EndOfStream {
		p.PluginState.Delete(request.RequestID)
	} else {
		p.PluginState.Touch(request.RequestID)
	}
}

// release surgically deletes a single profile's entry from PluginState,
// triggering OnEvicted to roll back that profile's counter contribution.
// Used at StartOfStream when a single profile needs to be released ahead of
// the EndOfStream bulk Delete.
func (p *InFlightLoadProducer) release(endpoint fwksched.Endpoint, request *fwksched.InferenceRequest, profileName string) {
	if endpoint == nil || request == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}
	meta := endpoint.GetMetadata()
	if meta == nil {
		return
	}
	eid := meta.NamespacedName.String()
	key := fwkplugin.StateKey(addedTokensKey(eid, profileName))

	// DeleteKey triggers OnEvicted, which decrements the counters exactly once.
	// If the janitor already reaped the request, this is a no-op.
	p.PluginState.DeleteKey(request.RequestID, key)
}

// releaseTokensEarly frees only the token portion of a profile's entry
// (request counter stays held), used at StartOfStream for the
// addEstimatedOutputTokens=false path where prefill completion frees tokens
// but the request remains in-flight until EndOfStream.
func (p *InFlightLoadProducer) releaseTokensEarly(endpoint fwksched.Endpoint, request *fwksched.InferenceRequest, profileName string) {
	if endpoint == nil || request == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}
	meta := endpoint.GetMetadata()
	if meta == nil {
		return
	}
	eid := meta.NamespacedName.String()

	key := fwkplugin.StateKey(addedTokensKey(eid, profileName))
	if entry, err := fwkplugin.ReadPluginStateKey[*addedTokensEntry](p.PluginState, request.RequestID, key); err == nil {
		if t := entry.tokens.Swap(0); t != 0 {
			entry.tokenTracker.addIfPresent(entry.endpointID, -t)
		}
	}
}

func addedTokensKey(endpointID, profileName string) string {
	return endpointID + "|" + profileName + "|added"
}

// uncachedInputTokens returns the prompt tokens this endpoint must actually compute,
// excluding any prefix already cached on it.
//
// When the configured prefix producer (approximate or precise) has populated
// PrefixCacheMatchInfo on the endpoint under prefixMatchInfoKey, the matched and
// total block counts are in real (tokenized) units, so we use them directly:
// uncached = (TotalBlocks - MatchBlocks) * BlockSizeTokens. For very long prompts
// where the prefix index is capped (MaxPrefixTokensToMatch), any tail beyond the
// cap is added back from the (estimated) inputTokens so the full prompt cost is
// still reflected.
//
// When the attribute is missing, we fall back to the estimated inputTokens.
func uncachedInputTokens(endpoint fwksched.Endpoint, inputTokens int64, prefixMatchInfoKey string) int64 {
	if endpoint == nil {
		return nonNeg(inputTokens)
	}
	raw, ok := endpoint.Get(prefixMatchInfoKey)
	if !ok {
		return nonNeg(inputTokens)
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info == nil || info.BlockSizeTokens() <= 0 {
		return nonNeg(inputTokens)
	}

	blockSize := int64(info.BlockSizeTokens())
	matched := int64(info.MatchBlocks()) * blockSize
	indexed := int64(info.TotalBlocks()) * blockSize

	uncachedIndexed := indexed - matched
	if uncachedIndexed < 0 {
		uncachedIndexed = 0
	}

	// Tail beyond the indexed portion (e.g., when MaxPrefixTokensToMatch caps total).
	tail := inputTokens - indexed
	if tail < 0 {
		tail = 0
	}

	return uncachedIndexed + tail
}

func nonNeg(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}

func (p *InFlightLoadProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.dk:                      attrconcurrency.InFlightLoad{},
		p.uncachedRequestTokensDk: attrconcurrency.UncachedRequestTokens{},
	}
}

// Consumes declares TokenizedPrompt as required so the data-layer DAG orders a
// token-producer ahead of this producer and auto-creates one when none is
// configured; without it the input-token estimate silently reads zero.
// PrefixCacheMatchInfo is optional — used to discount the already-cached prompt
// prefix from the prefix producer selected by prefixMatchInfoProducerName
// (approximate by default, or a precise-prefix-cache producer).
func (p *InFlightLoadProducer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			tokenproducer.TokenizedPromptDataKey: fwksched.TokenizedPrompt{},
		},
		Optional: map[fwkplugin.DataKey]any{
			p.prefixMatchInfoDK: attrprefix.PrefixCacheMatchInfo{},
		},
	}
}

// DeleteEndpoint removes an endpoint from the concurrency trackers to prevent memory leaks.
// This matches the design of the previous saturation detector and is called by the
// ExtractNotification hook to ensure deterministic cleanup of stateful data.
func (p *InFlightLoadProducer) DeleteEndpoint(endpointID string) {
	p.requestTracker.delete(endpointID)
	p.tokenTracker.delete(endpointID)
}

func (p *InFlightLoadProducer) GetTokens(eid string) int64 {
	return p.tokenTracker.get(eid)
}

func (p *InFlightLoadProducer) GetRequests(eid string) int64 {
	return p.requestTracker.get(eid)
}

// concurrencyTracker manages thread-safe counters for inflight requests.
type concurrencyTracker struct {
	mu     sync.RWMutex
	counts map[string]*atomic.Int64
}

func newConcurrencyTracker() *concurrencyTracker {
	return &concurrencyTracker{
		counts: make(map[string]*atomic.Int64),
	}
}

func (t *concurrencyTracker) get(endpointID string) int64 {
	t.mu.RLock()
	counter, exists := t.counts[endpointID]
	t.mu.RUnlock()

	if !exists {
		return 0
	}
	return counter.Load()
}

func (t *concurrencyTracker) snapshot() map[string]int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	result := make(map[string]int64, len(t.counts))
	for endpointID, counter := range t.counts {
		result[endpointID] = counter.Load()
	}
	return result
}

func (t *concurrencyTracker) inc(endpointID string) {
	t.add(endpointID, 1)
}

func (t *concurrencyTracker) add(endpointID string, delta int64) {
	t.mu.RLock()
	counter, exists := t.counts[endpointID]
	t.mu.RUnlock()

	if exists {
		counter.Add(delta)
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	if counter, exists = t.counts[endpointID]; exists {
		counter.Add(delta)
		return
	}

	counter = &atomic.Int64{}
	counter.Store(delta)
	t.counts[endpointID] = counter
}

func (t *concurrencyTracker) delete(endpointID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.counts, endpointID)
}
