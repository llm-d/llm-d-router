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
	"context"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"k8s.io/apimachinery/pkg/util/sets"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

// setupSpanRecorder installs an in-memory span recorder as the global tracer
// provider and returns it, restoring the previous provider on cleanup.
func setupSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })
	return recorder
}

func spanByName(t *testing.T, recorder *tracetest.SpanRecorder, name string) sdktrace.ReadOnlySpan {
	t.Helper()
	for _, s := range recorder.Ended() {
		if s.Name() == name {
			return s
		}
	}
	t.Fatalf("no %s span recorded", name)
	return nil
}

func spanAttrs(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	attrs := make(map[attribute.Key]attribute.Value)
	for _, kv := range span.Attributes() {
		attrs[kv.Key] = kv.Value
	}
	return attrs
}

// mmSpanRequest returns a single-prompt request whose two images span all
// four blocks of the prompt.
func mmSpanRequest(id string) *scheduling.InferenceRequest {
	tokens := make([]uint32, 4*testBlockSize)
	for i := range tokens {
		tokens[i] = uint32(i)
	}
	return &scheduling.InferenceRequest{
		RequestID:   id,
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{
					TokenIDs: tokens,
					MultiModalFeatures: []fwkrh.MultiModalFeature{
						{Modality: fwkrh.ModalityImage, Hash: "img-a", Offset: 2, Length: 20},
						{Modality: fwkrh.ModalityImage, Hash: "img-b", Offset: 32, Length: 20},
					},
				}},
			},
		},
	}
}

// The producer span carries the request's MM block total and the maximum MM
// match across endpoints, so a trace shows what MM reuse existed before the
// scheduler chose.
func TestProduce_EmitsMMBlockAttributes(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := utils.NewTestContext(t)

	keys := []kvblock.BlockHash{0xAA, 0xBB, 0xCC, 0xDD}
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return keys, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			return map[string]kvcache.PodMatch{
				"10.0.0.1:8080": {WeightedScore: 2, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
				"10.0.0.2:8080": {WeightedScore: 5, MatchedBlocks: 5, BlocksByTier: map[string]int{"gpu": 5}},
			}, nil
		},
	}
	p := newProducerWithIndexer(ctx, idx)

	require.NoError(t, p.Produce(ctx, mmSpanRequest("req-mm-span"), freshEndpoints()))

	attrs := spanAttrs(spanByName(t, recorder, "produce_precise_prefix_cache"))
	assert.Equal(t, int64(4), attrs[mmTotalBlocksKey].AsInt64(),
		"both images together span all four blocks")
	assert.Equal(t, int64(4), attrs[mmMatchedBlocksKey].AsInt64(),
		"the five-block pod holds every MM block, the two-block pod only two")
}

// Text-only requests carry no MM attribution, so neither MM block attribute is
// emitted: absent never reads as a zero match.
func TestProduce_TextOnlyOmitsMMBlockAttributes(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := utils.NewTestContext(t)

	keys := []kvblock.BlockHash{0xAA, 0xBB, 0xCC, 0xDD}
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return keys, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			return map[string]kvcache.PodMatch{
				"10.0.0.1:8080": {WeightedScore: 2, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
			}, nil
		},
	}
	p := newProducerWithIndexer(ctx, idx)

	require.NoError(t, p.Produce(ctx, tokenizedRequest("req-text-span", 4*testBlockSize), freshEndpoints()))

	attrs := spanAttrs(spanByName(t, recorder, "produce_precise_prefix_cache"))
	assert.NotContains(t, attrs, mmMatchedBlocksKey)
	assert.NotContains(t, attrs, mmTotalBlocksKey)
}

// Each prompt's MM blocks count toward the request total, so a multi-prompt
// request's total sums across prompts.
func TestProduce_MultiPromptMMTotalBlocksSums(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := utils.NewTestContext(t)

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return []kvblock.BlockHash{0x1, 0x2}, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			return map[string]kvcache.PodMatch{
				"10.0.0.1:8080": {WeightedScore: 2, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
			}, nil
		},
	}
	p := newProducerWithIndexer(ctx, idx)

	prompt := func() fwkrh.PromptTokens {
		return fwkrh.PromptTokens{
			TokenIDs:           make([]uint32, 2*testBlockSize),
			MultiModalFeatures: []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "img", Offset: 0, Length: testBlockSize}},
		}
	}
	req := &scheduling.InferenceRequest{
		RequestID: "req-mm-two-prompts",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{prompt(), prompt()}},
		},
	}

	require.NoError(t, p.Produce(ctx, req, freshEndpoints()))

	attrs := spanAttrs(spanByName(t, recorder, "produce_precise_prefix_cache"))
	assert.Equal(t, int64(2), attrs[mmTotalBlocksKey].AsInt64(),
		"each prompt's image spans one block")
	assert.Equal(t, int64(2), attrs[mmMatchedBlocksKey].AsInt64(),
		"the two-block match covers both prompts' images")
}

// An MM request that matched nothing emits a zero match rather than omitting
// the attributes, so zero stays distinguishable from untracked.
func TestProduce_ZeroMMMatchEmitsAttributes(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := utils.NewTestContext(t)

	keys := []kvblock.BlockHash{0xAA, 0xBB, 0xCC, 0xDD}
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return keys, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			return map[string]kvcache.PodMatch{}, nil
		},
	}
	p := newProducerWithIndexer(ctx, idx)

	require.NoError(t, p.Produce(ctx, mmSpanRequest("req-mm-zero"), freshEndpoints()))

	attrs := spanAttrs(spanByName(t, recorder, "produce_precise_prefix_cache"))
	assert.Equal(t, int64(0), attrs[mmMatchedBlocksKey].AsInt64(),
		"no endpoint holds any of the request's blocks")
	assert.Equal(t, int64(4), attrs[mmTotalBlocksKey].AsInt64())
}

// The MM block total is request-wide: a prompt shorter than one block
// produces no keys and cannot match, but its MM content still counts, so the
// produce and PreRequest spans report the same total for one request.
func TestProduce_SubBlockPromptCountsInMMTotal(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := utils.NewTestContext(t)

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, tokens []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			if len(tokens) < testBlockSize {
				return nil, nil
			}
			return []kvblock.BlockHash{0x1, 0x2}, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			return map[string]kvcache.PodMatch{
				"10.0.0.1:8080": {WeightedScore: 2, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
			}, nil
		},
	}
	p := newProducerWithIndexer(ctx, idx)

	// Prompt A fills two blocks with its image in block 0; prompt B holds
	// five tokens with its image, too few for any block key.
	req := &scheduling.InferenceRequest{
		RequestID: "req-mm-sub-block",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{
					{
						TokenIDs:           make([]uint32, 2*testBlockSize),
						MultiModalFeatures: []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "img-a", Offset: 0, Length: testBlockSize}},
					},
					{
						TokenIDs:           make([]uint32, 5),
						MultiModalFeatures: []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "img-b", Offset: 0, Length: 5}},
					},
				},
			},
		},
	}

	require.NoError(t, p.Produce(ctx, req, freshEndpoints()))

	attrs := spanAttrs(spanByName(t, recorder, "produce_precise_prefix_cache"))
	assert.Equal(t, int64(2), attrs[mmTotalBlocksKey].AsInt64(),
		"prompt B's image spans a block index even though the prompt produces no keys")
	assert.Equal(t, int64(1), attrs[mmMatchedBlocksKey].AsInt64(),
		"only prompt A's image can sit inside the matched prefix")
}

// The PreRequest total comes from the request's features, so a multi-prompt
// request's total sums across prompts on the PreRequest span as well.
func TestPreRequest_MultiPromptMMTotalBlocksSums(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := utils.NewTestContext(t)

	const name = "precise-mm-span-multi"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})
	p.blockSizeTokens = testBlockSize

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 2, testBlockSize).
		WithCachedBlockCount(2).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 2, MatchTokens: 2 * testBlockSize}))

	prompt := func() fwkrh.PromptTokens {
		return fwkrh.PromptTokens{
			TokenIDs:           make([]uint32, 2*testBlockSize),
			MultiModalFeatures: []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "img", Offset: 0, Length: testBlockSize}},
		}
	}
	req := &scheduling.InferenceRequest{
		RequestID: "req-mm-two-prompts-prerequest",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{prompt(), prompt()}},
		},
	}

	_ = p.PreRequest(ctx, req, primaryOnly("decode", endpoint))

	attrs := spanAttrs(spanByName(t, recorder, "pre_request_precise_prefix_cache"))
	assert.Equal(t, int64(2), attrs[mmTotalBlocksKey].AsInt64(),
		"each prompt's image spans one block")
	assert.Equal(t, int64(2), attrs[mmMatchedBlocksKey].AsInt64())
}

// PreRequest carries the chosen endpoint's MM match, so the gap between the
// producer span's maximum and this value shows MM reuse the routing decision
// left behind.
func TestPreRequest_EmitsChosenEndpointMMBlocks(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := utils.NewTestContext(t)

	const name = "precise-mm-span"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})
	p.blockSizeTokens = testBlockSize

	chosen, warmer := freshEndpoints()[0], freshEndpoints()[1]
	chosen.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(3, 8, testBlockSize).
		WithCachedBlockCount(3).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 2, MatchTokens: 2 * testBlockSize}))
	warmer.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(7, 8, testBlockSize).
		WithCachedBlockCount(7).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 7, MatchTokens: 7 * testBlockSize}))

	_ = p.PreRequest(ctx, mmRequest("req-mm-chosen", 8*testBlockSize),
		primaryWithScored("decode", chosen, chosen, warmer))

	attrs := spanAttrs(spanByName(t, recorder, "pre_request_precise_prefix_cache"))
	assert.Equal(t, int64(2), attrs[mmMatchedBlocksKey].AsInt64(),
		"the chosen endpoint's two MM blocks, not its three prompt blocks or the warmer endpoint's seven")
	assert.Equal(t, int64(8), attrs[mmTotalBlocksKey].AsInt64(),
		"the image spans the full eight-block prompt")
}

// A request whose chosen endpoint's match info carries no MM attribution
// emits neither attribute, so absent never reads as a zero match.
func TestPreRequest_NoMMAttributionOmitsMMBlockAttributes(t *testing.T) {
	recorder := setupSpanRecorder(t)
	ctx := utils.NewTestContext(t)

	const name = "precise-mm-span-none"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})
	p.blockSizeTokens = testBlockSize

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 8, testBlockSize).WithCachedBlockCount(2))

	_ = p.PreRequest(ctx, mmRequest("req-mm-none", 8*testBlockSize), primaryOnly("default", endpoint))

	attrs := spanAttrs(spanByName(t, recorder, "pre_request_precise_prefix_cache"))
	assert.NotContains(t, attrs, mmMatchedBlocksKey)
	assert.NotContains(t, attrs, mmTotalBlocksKey)
}
