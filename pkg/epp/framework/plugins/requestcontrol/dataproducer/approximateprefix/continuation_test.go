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

package approximateprefix

import (
	"cmp"
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	eppdatalayer "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

const (
	contBlockSize  = 4
	contPrompt     = "The quick brown fox jumps over the lazy dog. " // 45 bytes, not 4-aligned
	contCompletion = "It was a sunny afternoon in the park."
)

// completionsRequest returns a /v1/completions request tokenized the way the
// estimate backend tokenizes a single string prompt.
func completionsRequest(prompt string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "m",
		Body: &fwkrh.InferenceRequestBody{
			Completions:      &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
			TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: fwkrh.PackBytes([]byte(prompt))}}},
		},
	}
}

type continuationFixture struct {
	p               *dataProducer
	decode, prefill fwksched.Endpoint
}

// newContinuationFixture validates the producer's dependencies against a
// token-producer built from tokenProducerConfig, as startup does. An empty
// tokenProducerConfig skips validation.
func newContinuationFixture(t *testing.T, tokenProducerConfig string) *continuationFixture {
	t.Helper()
	return newContinuationFixtureWithCapacity(t, tokenProducerConfig, defaultLRUCapacityPerServer)
}

func newContinuationFixtureWithCapacity(t *testing.T, tokenProducerConfig string, lruCapacity int) *continuationFixture {
	t.Helper()
	disableMinBlockSizeClamp(t)
	handle := testHandle()
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config{
		BlockSizeTokens:        contBlockSize,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   lruCapacity,
	}, handle)
	require.NoError(t, err)
	if tokenProducerConfig != "" {
		tp, err := tokenproducer.PluginFactory(tokenproducer.PluginType, plugin.StrictDecoder(json.RawMessage(tokenProducerConfig)), handle)
		require.NoError(t, err)
		reg, err := fwkdl.BuildRegistry([]plugin.Plugin{tp, p})
		require.NoError(t, err)
		require.NoError(t, reg.ValidateConsumer(p))
	}
	return &continuationFixture{
		p:       p,
		decode:  fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "decode", Namespace: "default"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes()),
		prefill: fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "prefill", Namespace: "default"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes()),
	}
}

// serve schedules req on the decode and prefill endpoints and ends its
// response with the given completion and termination cause.
func (f *continuationFixture) serve(t *testing.T, req *fwksched.InferenceRequest, completion *fwkrh.Completion, cause requestcontrol.TerminationCause) {
	t.Helper()
	f.schedule(t, req)
	f.endStream(t, req, completion, cause)
}

// schedule runs Produce and PreRequest for req on the decode and prefill endpoints.
func (f *continuationFixture) schedule(t *testing.T, req *fwksched.InferenceRequest) {
	t.Helper()
	ctx := context.Background()
	require.NoError(t, f.p.Produce(ctx, req, []fwksched.Endpoint{f.decode, f.prefill}))
	require.NoError(t, f.p.PreRequest(ctx, req, &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode":                          {TargetEndpoints: []fwksched.Endpoint{f.decode}},
			experimentalDefaultPrefillProfile: {TargetEndpoints: []fwksched.Endpoint{f.prefill}},
		},
	}))
}

// endStream delivers the end-of-stream response body for req.
func (f *continuationFixture) endStream(t *testing.T, req *fwksched.InferenceRequest, completion *fwkrh.Completion, cause requestcontrol.TerminationCause) {
	t.Helper()
	ctx := context.Background()
	f.p.ResponseBody(ctx, req, &requestcontrol.Response{
		RequestID:        req.RequestID,
		StartOfStream:    true,
		EndOfStream:      true,
		TerminationCause: cause,
		Completion:       completion,
	}, f.decode.GetMetadata())
	f.p.wg.Wait()
	_, tracked := f.p.pluginState.LastAccessTime(req.RequestID)
	assert.False(t, tracked, "plugin state released at end of stream")
}

// nextTurn returns the matched blocks of the decode and prefill endpoints for
// a request whose prompt is contPrompt followed by contCompletion.
func (f *continuationFixture) nextTurn(t *testing.T) (decode, prefill int) {
	t.Helper()
	req := completionsRequest(contPrompt + contCompletion)
	require.NoError(t, f.p.Produce(context.Background(), req, []fwksched.Endpoint{f.decode, f.prefill}))
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType)
	match := func(ep fwksched.Endpoint) int {
		info, ok := ep.Get(key)
		require.True(t, ok)
		return info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks()
	}
	return match(f.decode), match(f.prefill)
}

// nextTurnBlocks returns the number of blocks of contPrompt followed by contCompletion.
func nextTurnBlocks() int {
	req := completionsRequest(contPrompt + contCompletion)
	return len(prefixhash.GetBlockHashes(context.Background(), req, contBlockSize, defaultMaxPrefixBlocks)[0])
}

// fullPromptBlocks is the number of contPrompt blocks a request starting with
// contPrompt followed by more text shares with contPrompt itself.
var fullPromptBlocks = len(contPrompt) / (fwkrh.BytesPerToken * contBlockSize)

// estimateTokenProducer selects the zero-config estimate backend.
const estimateTokenProducer = `{}`

func TestContinuationIndexedForServingEndpoint(t *testing.T) {
	f := newContinuationFixture(t, estimateTokenProducer)
	f.serve(t, completionsRequest(contPrompt), &fwkrh.Completion{Text: contCompletion, FinishReason: "stop"}, requestcontrol.TerminationCauseNatural)

	decode, prefill := f.nextTurn(t)
	assert.Equal(t, nextTurnBlocks(), decode, "decode endpoint credited for prompt and completion")
	assert.Equal(t, fullPromptBlocks, prefill, "prefill endpoint credited for the prompt only")
}

// TestContinuationNotIndexedAfterRequestAbandoned covers a request the director
// abandons after this plugin's PreRequest, as when a later PreRequest plugin fails.
func TestContinuationNotIndexedAfterRequestAbandoned(t *testing.T) {
	f := newContinuationFixture(t, estimateTokenProducer)
	req := completionsRequest(contPrompt)
	f.schedule(t, req)
	_, tracked := f.p.pluginState.LastAccessTime(req.RequestID)
	require.True(t, tracked, "continuation state held after PreRequest")

	f.p.RequestAbandoned(context.Background(), req)
	_, tracked = f.p.pluginState.LastAccessTime(req.RequestID)
	require.False(t, tracked, "continuation state released by RequestAbandoned")

	f.endStream(t, req, &fwkrh.Completion{Text: contCompletion, FinishReason: "stop"}, requestcontrol.TerminationCauseNatural)
	decode, prefill := f.nextTurn(t)
	assert.Equal(t, prefill, decode, "decode endpoint gets no continuation credit")
}

// TestContinuationKeepsLeadingBlocksWithinCapacity checks that a continuation
// longer than the endpoint's LRU leaves a run from block 0, which the indexer's
// prefix match requires.
func TestContinuationKeepsLeadingBlocksWithinCapacity(t *testing.T) {
	const capacity = 3
	require.Greater(t, nextTurnBlocks(), capacity)
	f := newContinuationFixtureWithCapacity(t, estimateTokenProducer, capacity)
	f.serve(t, completionsRequest(contPrompt), &fwkrh.Completion{Text: contCompletion, FinishReason: "stop"}, requestcontrol.TerminationCauseNatural)

	decode, prefill := f.nextTurn(t)
	assert.Equal(t, capacity, decode, "decode endpoint keeps the leading blocks")
	assert.Equal(t, fullPromptBlocks, prefill, "prefill endpoint is not evicted by the continuation")
}

// gatedIndexer holds the first Add until release is closed and records the
// length of each Add in the order the inserts are applied.
type gatedIndexer struct {
	indexerInterface
	waiting, release chan struct{}
	passed           atomic.Bool
	mu               sync.Mutex
	applied          []int
}

func (g *gatedIndexer) Add(hashes []blockHash, s server) {
	if g.passed.CompareAndSwap(false, true) {
		close(g.waiting)
		<-g.release
	}
	g.indexerInterface.Add(hashes, s)
	g.mu.Lock()
	g.applied = append(g.applied, len(hashes))
	g.mu.Unlock()
}

func (g *gatedIndexer) appliedLens() []int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]int(nil), g.applied...)
}

// TestContinuationIndexedAfterLatePromptInsert holds PreRequest's prompt
// insert until the response has ended. At capacity, a prompt insert applied
// after the continuation would evict the continuation's first completion block
// with the prompt's zero-padded last block.
func TestContinuationIndexedAfterLatePromptInsert(t *testing.T) {
	const capacity = 3
	f := newContinuationFixtureWithCapacity(t, estimateTokenProducer, capacity)
	gate := &gatedIndexer{indexerInterface: f.p.indexerInst, waiting: make(chan struct{}), release: make(chan struct{})}
	f.p.indexerInst = gate
	req := completionsRequest(contPrompt)

	ctx := context.Background()
	require.NoError(t, f.p.Produce(ctx, req, []fwksched.Endpoint{f.decode, f.prefill}))
	require.NoError(t, f.p.PreRequest(ctx, req, &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode":                          {TargetEndpoints: []fwksched.Endpoint{f.decode}},
			experimentalDefaultPrefillProfile: {TargetEndpoints: []fwksched.Endpoint{f.prefill}},
		},
	}))
	<-gate.waiting
	f.p.ResponseBody(ctx, req, &requestcontrol.Response{
		RequestID:        req.RequestID,
		StartOfStream:    true,
		EndOfStream:      true,
		TerminationCause: requestcontrol.TerminationCauseNatural,
		Completion:       &fwkrh.Completion{Text: contCompletion, FinishReason: "stop"},
	}, f.decode.GetMetadata())
	// Give a continuation insert that does not wait for the prompt time to land.
	time.Sleep(50 * time.Millisecond)
	close(gate.release)
	f.p.wg.Wait()

	full := nextTurnBlocks()
	applied := gate.appliedLens()
	require.NotEmpty(t, applied)
	assert.Equal(t, full, applied[len(applied)-1], "continuation inserted after the prompt, applied %v", applied)
	decode, _ := f.nextTurn(t)
	assert.Equal(t, capacity, decode, "decode endpoint keeps the leading continuation blocks")
}

// TestContinuationBackendRecordedAtStartup runs the startup sequence: missing
// producers are created, then dependencies are validated.
func TestContinuationBackendRecordedAtStartup(t *testing.T) {
	tests := []struct {
		name          string
		tokenProducer string
		want          bool
	}{
		{name: "auto-created token-producer", want: true},
		{name: "configured estimate token-producer", tokenProducer: estimateTokenProducer, want: true},
		{name: "configured vllm token-producer", tokenProducer: `{"modelName":"test-model"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handle := testHandle()
			p, err := ApproxPrefixCacheFactory(ApproxPrefixCachePluginType, nil, handle)
			require.NoError(t, err)
			handle.AddPlugin(p.TypedName().Name, p)
			if tt.tokenProducer != "" {
				tp, err := tokenproducer.PluginFactory(tokenproducer.PluginType, plugin.StrictDecoder(json.RawMessage(tt.tokenProducer)), handle)
				require.NoError(t, err)
				handle.AddPlugin(tokenproducer.PluginType, tp)
			}
			require.NoError(t, eppdatalayer.CreateMissingDataProducers(handle.Context(),
				map[string]string{tokenproducer.TokenizedPromptDataKey.String(): tokenproducer.PluginType},
				map[string]plugin.FactoryFunc{tokenproducer.PluginType: tokenproducer.PluginFactory}, handle))
			producer := p.(*dataProducer)
			assert.False(t, producer.estimateTokens.Load(), "not recorded before dependency validation")

			_, err = eppdatalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins())
			require.NoError(t, err)
			assert.Equal(t, tt.want, producer.estimateTokens.Load())
		})
	}
}

func TestContinuationNotIndexed(t *testing.T) {
	completion := &fwkrh.Completion{Text: contCompletion, FinishReason: "stop"}
	tests := []struct {
		name          string
		tokenProducer string
		unvalidated   bool
		mutate        func(*fwksched.InferenceRequest)
		completion    *fwkrh.Completion
		cause         requestcontrol.TerminationCause
	}{
		{name: "no completion text (error status, oversized body, several choices)", cause: requestcontrol.TerminationCauseNatural},
		{name: "client disconnect", completion: completion, cause: requestcontrol.TerminationCauseClientDisconnect},
		{name: "error termination", completion: completion, cause: requestcontrol.TerminationCauseError},
		{
			name:       "echo enabled",
			mutate:     func(r *fwksched.InferenceRequest) { r.Body.Completions.Echo = json.RawMessage("true") },
			completion: completion, cause: requestcontrol.TerminationCauseNatural,
		},
		{
			name:       "streaming request",
			mutate:     func(r *fwksched.InferenceRequest) { r.Body.Stream = true },
			completion: completion, cause: requestcontrol.TerminationCauseNatural,
		},
		{
			name:          "vllm token-producer backend",
			tokenProducer: `{"modelName":"test-model"}`,
			completion:    completion, cause: requestcontrol.TerminationCauseNatural,
		},
		{
			name:        "token-producer backend not validated",
			unvalidated: true,
			completion:  completion, cause: requestcontrol.TerminationCauseNatural,
		},
		{
			name: "token IDs not the estimate encoding of the prompt",
			mutate: func(r *fwksched.InferenceRequest) {
				r.Body.TokenizedRequest.Prompts[0].TokenIDs = []uint32{1, 2, 3, 4, 5, 6, 7, 8}
			},
			completion: completion, cause: requestcontrol.TerminationCauseNatural,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenProducer := cmp.Or(tt.tokenProducer, estimateTokenProducer)
			if tt.unvalidated {
				tokenProducer = ""
			}
			f := newContinuationFixture(t, tokenProducer)
			req := completionsRequest(contPrompt)
			if tt.mutate != nil {
				tt.mutate(req)
			}
			f.serve(t, req, tt.completion, tt.cause)

			decode, prefill := f.nextTurn(t)
			assert.Equal(t, prefill, decode, "decode endpoint gets no continuation credit")
		})
	}
}

// TestEstimateContinuationKeys shows why continuation keys are computed from
// the concatenated bytes: the prompt's own keys end in a zero-padded
// pseudo-token and a partial block that the next turn does not produce.
func TestEstimateContinuationKeys(t *testing.T) {
	keys := func(text string) []prefixhash.BlockHash {
		return prefixhash.GetBlockHashes(context.Background(), completionsRequest(text), contBlockSize, defaultMaxPrefixBlocks)[0]
	}
	promptKeys, nextTurn := keys(contPrompt), keys(contPrompt+contCompletion)
	shared := 0
	for shared < min(len(promptKeys), len(nextTurn)) && promptKeys[shared] == nextTurn[shared] {
		shared++
	}
	assert.Equal(t, fullPromptBlocks, shared)
}
