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

package preciseprefixcache

import (
	"encoding/json"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/test/utils"
)

// installRealTokenProducer pre-registers a vllm-backed token-producer in the
// handle so preciseproducer.PluginFactory's startup-time check for an
// engine-aligned tokenizer (#1471) is satisfied. Tests that exercise the
// producer factory in isolation must call this first.
func installRealTokenProducer(t *testing.T, handle fwkplugin.Handle) {
	t.Helper()
	raw := json.RawMessage(`{"modelName":"test-model"}`)
	tp, err := tokenproducer.PluginFactory(tokenproducer.PluginType, fwkplugin.StrictDecoder(raw), handle)
	require.NoError(t, err)
	handle.AddPlugin(tp.TypedName().Name, tp)
}

// In self-host mode the plugin satisfies Scorer, DataProducer, PreRequest,
// and EndpointExtractor.
func TestPluginFactory_SelfHostInterfaces(t *testing.T) {
	handle := fwkplugin.NewEppHandle(utils.NewTestContext(t), nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	plg, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(`{}`)), handle)
	require.NoError(t, err)

	_, ok := plg.(scheduling.Scorer)
	assert.True(t, ok, "plugin must be a Scorer")
	_, ok = plg.(requestcontrol.DataProducer)
	assert.True(t, ok, "plugin must be a DataProducer")
	_, ok = plg.(requestcontrol.PreRequest)
	assert.True(t, ok, "plugin must be a PreRequest")
	_, ok = plg.(fwkdl.EndpointExtractor)
	assert.True(t, ok, "plugin must be an EndpointExtractor")
}

// The inner scorer's Consumes set must include every key the inner producer
// Produces, so the data-layer DAG links them.
func TestPluginFactory_InnerScorerConsumesProducerKeys(t *testing.T) {
	handle := fwkplugin.NewEppHandle(utils.NewTestContext(t), nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	plg, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(`{}`)), handle)
	require.NoError(t, err)
	p := plg.(*Plugin)

	produces := p.Produces()
	require.Len(t, produces, 1)
	consumes := p.Consumes()
	for k := range produces {
		_, inRequired := consumes.Required[k]
		_, inOptional := consumes.Optional[k]
		assert.True(t, inRequired || inOptional, "Consumes must include produced key %s", k.String())
	}
}

// With a precise-prefix-cache-producer pre-registered, the factory returns
// a Scorer-only plugin pointed at it.
func TestPluginFactory_DefersToExistingProducer(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
	installRealTokenProducer(t, handle)

	existing, err := preciseproducer.PluginFactory("my-precise", nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(existing.TypedName().Name, existing)

	plg, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(`{"speculativeIndexing":true}`)), handle)
	require.NoError(t, err)

	_, isScorer := plg.(scheduling.Scorer)
	assert.True(t, isScorer)
	_, isProducer := plg.(requestcontrol.DataProducer)
	assert.False(t, isProducer, "defer-mode plugin must not be a DataProducer")
}

// Two precise-prefix-cache-producer instances leave the legacy plugin unable
// to choose; the factory errors instead of picking one non-deterministically.
func TestPluginFactory_RejectsMultipleExistingProducers(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
	installRealTokenProducer(t, handle)

	first, err := preciseproducer.PluginFactory("first", nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(first.TypedName().Name, first)

	second, err := preciseproducer.PluginFactory("second", nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(second.TypedName().Name, second)

	_, err = PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(`{}`)), handle)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "multiple precise-prefix-cache-producer instances")
}

// stubPromptTokenizer captures the inputs Tokenize was called with so
// tests can verify the legacy path routed the prompt through the pool.
type stubPromptTokenizer struct {
	tokens   []uint32
	lastRReq *tokenizerTypes.RenderChatRequest
	lastRaw  string
	calls    int
}

func (s *stubPromptTokenizer) Tokenize(rr *tokenizerTypes.RenderChatRequest, prompt string) ([]uint32, *tokenization.MultiModalFeatures) {
	s.calls++
	s.lastRReq = rr
	s.lastRaw = prompt
	return s.tokens, nil
}

// With a wrapper-owned tokenizer pool, Consumes must drop the inner
// producer's TokenizedPrompt dependency — the wrapper supplies tokens
// itself and no upstream token-producer is required.
func TestLegacyProducer_ConsumesDropsTokenizedPromptWhenPoolSet(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
	installRealTokenProducer(t, handle)

	inner, err := preciseproducer.PluginFactory("inner", nil, handle)
	require.NoError(t, err)

	lp := &legacyProducer{
		Producer:      inner.(*preciseproducer.Producer),
		tokenizerPool: &stubPromptTokenizer{},
	}
	assert.Empty(t, lp.Consumes())

	lpNoPool := &legacyProducer{Producer: inner.(*preciseproducer.Producer)}
	assert.NotEmpty(t, lpNoPool.Consumes(), "without a pool, Consumes must keep TokenizedPrompt")
}

// When a completions prompt arrives without TokenizedPrompt and the pool
// is set, Produce must route the prompt through the pool and stash the
// resulting tokens on request.Body.TokenizedPrompt.
func TestLegacyProducer_TokenizesCompletionPromptViaPool(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
	installRealTokenProducer(t, handle)

	inner, err := preciseproducer.PluginFactory("inner", nil, handle)
	require.NoError(t, err)

	stub := &stubPromptTokenizer{tokens: []uint32{1, 2, 3}}
	lp := &legacyProducer{
		Producer:      inner.(*preciseproducer.Producer),
		tokenizerPool: stub,
	}

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt:    fwkrh.Prompt{Raw: "hello world"},
				CacheSalt: "leg-salt",
			},
		},
	}
	require.NoError(t, lp.Produce(ctx, req, nil))

	assert.Equal(t, 1, stub.calls)
	assert.Equal(t, "hello world", stub.lastRaw)
	require.NotNil(t, req.Body.TokenizedPrompt)
	assert.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedPrompt.PerPromptTokens[0])
	// Wrapper-owned tokenization must still carry the cache salt so precise
	// keys stay isolated on this path.
	assert.Equal(t, "leg-salt", req.Body.TokenizedPrompt.CacheSalt)
}

// End-to-end: tokens from the pool flow into the embedded producer, get
// hashed into block keys, and a PrefixCacheMatchInfo is written to each
// endpoint. With an empty index the match count is 0, but the totalBlocks
// denominator must reflect the tokens the wrapper supplied. Guards against
// regressions in the pool → tokens → block-key → attribute pipeline that
// the deleted UDS integration tests used to cover.
func TestLegacyProducer_TokensFlowToEndpointAttribute(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	// Small block size for a predictable totalBlocks: 16 tokens / 4 = 4 blocks.
	installRealTokenProducer(t, handle)

	const blockSize = 4
	rawCfg := json.RawMessage(`{"tokenProcessorConfig":{"blockSize":4}}`)
	inner, err := preciseproducer.PluginFactory("inner", fwkplugin.StrictDecoder(rawCfg), handle)
	require.NoError(t, err)

	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	lp := &legacyProducer{
		Producer:      inner.(*preciseproducer.Producer),
		tokenizerPool: &stubPromptTokenizer{tokens: tokens},
	}

	req := &scheduling.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
		},
	}
	endpoint := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: "pod1"},
		Address:        "10.0.0.1",
		Port:           "8000",
	}, nil, nil)

	require.NoError(t, lp.Produce(ctx, req, []scheduling.Endpoint{endpoint}))

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("inner").String()
	raw, ok := endpoint.Get(key)
	require.True(t, ok, "endpoint should have PrefixCacheMatchInfo set")
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	require.True(t, ok)
	assert.Equal(t, len(tokens)/blockSize, info.TotalBlocks(),
		"totalBlocks must reflect tokens produced by the wrapper-owned pool")
	assert.Equal(t, 0, info.MatchBlocks(),
		"empty index → no matches")
}

// Pre-existing TokenizedPrompt must skip the pool entirely, so the new-path
// token-producer pipeline isn't shadowed by the legacy pool.
func TestLegacyProducer_KeepsExistingTokenizedPrompt(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
	installRealTokenProducer(t, handle)

	inner, err := preciseproducer.PluginFactory("inner", nil, handle)
	require.NoError(t, err)

	stub := &stubPromptTokenizer{tokens: []uint32{9, 9, 9}}
	lp := &legacyProducer{
		Producer:      inner.(*preciseproducer.Producer),
		tokenizerPool: stub,
	}

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{{5, 5, 5}}},
			Completions:     &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "should not tokenize"}},
		},
	}
	require.NoError(t, lp.Produce(ctx, req, nil))

	assert.Equal(t, 0, stub.calls, "pool must not be called when tokens already present")
	assert.Equal(t, []uint32{5, 5, 5}, req.Body.TokenizedPrompt.PerPromptTokens[0])
}
