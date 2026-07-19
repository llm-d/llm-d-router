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
	"encoding/json"
	"fmt"
	"testing"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

type fakeKVCacheIndexer struct {
	computeFromTokens func(ctx context.Context, tokens []uint32, model string, extra []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error)
	index             *fakeKVBlockIndex
}

func (f *fakeKVCacheIndexer) ComputeBlockKeysFromTokens(ctx context.Context, tokens []uint32, model string, extra []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
	if f.computeFromTokens != nil {
		return f.computeFromTokens(ctx, tokens, model, extra)
	}
	return []kvblock.BlockHash{}, nil
}

func (f *fakeKVCacheIndexer) KVBlockIndex() kvblock.Index { return f.index }

type fakeKVBlockIndex struct {
	lookup  func(ctx context.Context, keys []kvblock.BlockHash, podSet sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error)
	addFn   func(ctx context.Context, prevKeys, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error
	clearFn func(ctx context.Context, podIdentifier string) error
}

func (f *fakeKVBlockIndex) Lookup(ctx context.Context, keys []kvblock.BlockHash, podSet sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
	if f.lookup != nil {
		return f.lookup(ctx, keys, podSet)
	}
	return map[kvblock.BlockHash][]kvblock.PodEntry{}, nil
}

func (f *fakeKVBlockIndex) Add(ctx context.Context, prevKeys, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
	if f.addFn != nil {
		return f.addFn(ctx, prevKeys, keys, entries)
	}
	return nil
}

func (f *fakeKVBlockIndex) Evict(_ context.Context, _ kvblock.BlockHash, _ kvblock.KeyType, _ []kvblock.PodEntry) error {
	return nil
}

func (f *fakeKVBlockIndex) GetRequestKey(_ context.Context, _ kvblock.BlockHash) (kvblock.BlockHash, error) {
	return kvblock.EmptyBlockHash, nil
}

func (f *fakeKVBlockIndex) Clear(ctx context.Context, podIdentifier string) error {
	if f.clearFn != nil {
		return f.clearFn(ctx, podIdentifier)
	}
	return nil
}

type fakeKVBlockScorer struct {
	score func(ctx context.Context, keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error)
}

func (f *fakeKVBlockScorer) Strategy() kvcache.KVScoringStrategy {
	return kvcache.LongestPrefixMatch
}

func (f *fakeKVBlockScorer) Score(ctx context.Context, keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error) {
	if f.score != nil {
		return f.score(ctx, keys, keyToPods)
	}
	return map[string]float64{}, nil
}

var testEndpoints = []scheduling.Endpoint{
	scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: "pod-a"},
			Address:        "10.0.0.1",
			Port:           "8080",
		}, nil, nil),
	scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: "pod-b"},
			Address:        "10.0.0.2",
			Port:           "8080",
		}, nil, nil),
}

const testBlockSize = 16

func freshEndpoints() []scheduling.Endpoint {
	return []scheduling.Endpoint{
		scheduling.NewEndpoint(
			&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-a"},
				Address:        "10.0.0.1",
				Port:           "8080",
			}, nil, nil),
		scheduling.NewEndpoint(
			&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-b"},
				Address:        "10.0.0.2",
				Port:           "8080",
			}, nil, nil),
	}
}

func newProducerWithIndexer(ctx context.Context, idx kvCacheIndexer, scorer kvcache.KVBlockScorer) *Producer {
	return &Producer{
		typedName:       plugin.TypedName{Type: PluginType, Name: "test"},
		kvCacheIndexer:  idx,
		kvBlockScorer:   scorer,
		kvEventsConfig:  &kvevents.Config{},
		dk:              attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test"),
		pluginState:     plugin.NewPluginState(ctx),
		blockSizeTokens: testBlockSize,
	}
}

// Tokens present → Produce hashes and writes per-endpoint match info.
func TestProduce_UsesTokenizedPrompt(t *testing.T) {
	ctx := utils.NewTestContext(t)

	tokens := []uint32{10, 20, 30, 40, 50}
	wantKey := kvblock.BlockHash(0xCAFE)

	var capturedTokens []uint32

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, ts []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			capturedTokens = ts
			return []kvblock.BlockHash{wantKey}, nil
		},
		index: &fakeKVBlockIndex{
			lookup: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
				return map[kvblock.BlockHash][]kvblock.PodEntry{
					wantKey: {{PodIdentifier: "10.0.0.1:8080"}},
				}, nil
			},
		},
	}
	scorer := &fakeKVBlockScorer{
		score: func(_ context.Context, _ []kvblock.BlockHash, _ map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error) {
			return map[string]float64{"10.0.0.1:8080": 1.0}, nil
		},
	}

	p := newProducerWithIndexer(ctx, idx, scorer)

	req := &scheduling.InferenceRequest{
		RequestID:   "req-1",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokens}},
		},
	}

	require.NoError(t, p.Produce(ctx, req, testEndpoints))
	require.Equal(t, tokens, capturedTokens)

	raw, ok := testEndpoints[0].Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test").String())
	require.True(t, ok)
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	require.True(t, ok)
	assert.Equal(t, 1, info.MatchBlocks())
	assert.Equal(t, 1, info.TotalBlocks())
	assert.Equal(t, 16, info.BlockSizeTokens())

	raw2, ok := testEndpoints[1].Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test").String())
	require.True(t, ok)
	info2 := raw2.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 0, info2.MatchBlocks())
	assert.Equal(t, 1, info2.TotalBlocks())
}

// No tokens → no-op (no prompt-string fallback).
func TestProduce_NoTokens_NoOp(t *testing.T) {
	ctx := utils.NewTestContext(t)
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			t.Fatalf("ComputeBlockKeysFromTokens must not be called when no tokens")
			return nil, nil
		},
		index: &fakeKVBlockIndex{},
	}
	p := newProducerWithIndexer(ctx, idx, &fakeKVBlockScorer{})

	req := &scheduling.InferenceRequest{
		RequestID:   "req-2",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "no tokens here"}},
		},
	}
	require.NoError(t, p.Produce(ctx, req, testEndpoints))
}

// Empty TokenIDs → no-op.
func TestProduce_EmptyTokenizedPrompt_NoOp(t *testing.T) {
	ctx := utils.NewTestContext(t)
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			t.Fatalf("ComputeBlockKeysFromTokens must not be called for empty TokenIDs")
			return nil, nil
		},
		index: &fakeKVBlockIndex{},
	}
	p := newProducerWithIndexer(ctx, idx, &fakeKVBlockScorer{})

	req := &scheduling.InferenceRequest{
		RequestID:   "req-3",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions:     &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "p"}},
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{{}}},
		},
	}
	require.NoError(t, p.Produce(ctx, req, testEndpoints))
}

func TestProduce_MultiPromptEmptyBlockKeys_NoOp(t *testing.T) {
	ctx := utils.NewTestContext(t)
	endpoints := freshEndpoints()

	promptA := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	promptB := []uint32{9, 10, 11, 12, 13, 14, 15, 16}

	var computeCalls [][]uint32
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, ts []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			computeCalls = append(computeCalls, append([]uint32{}, ts...))
			if len(ts) >= testBlockSize {
				return []kvblock.BlockHash{0xABCD}, nil
			}
			return nil, nil
		},
		index: &fakeKVBlockIndex{
			lookup: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
				t.Fatalf("Lookup must not be called when no prompt produces block keys")
				return nil, assert.AnError
			},
		},
	}
	scorer := &fakeKVBlockScorer{
		score: func(_ context.Context, _ []kvblock.BlockHash, _ map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error) {
			t.Fatalf("Score must not be called when no prompt produces block keys")
			return nil, assert.AnError
		},
	}

	p := newProducerWithIndexer(ctx, idx, scorer)
	req := &scheduling.InferenceRequest{
		RequestID:   "req-multi-empty",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{promptA, promptB},
			},
		},
	}

	require.NoError(t, p.Produce(ctx, req, endpoints))
	require.Equal(t, [][]uint32{promptA, promptB}, computeCalls)

	_, ok := endpoints[0].Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test").String())
	assert.False(t, ok)
}

func TestProduce_MultiPromptSkipsEmptyPromptKeys(t *testing.T) {
	ctx := utils.NewTestContext(t)
	endpoints := freshEndpoints()

	shortPrompt := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	fullPrompt := []uint32{20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
	wantKey := kvblock.BlockHash(0xCAFE)

	var computeCalls [][]uint32
	var lookupCalls [][]kvblock.BlockHash
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, ts []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			computeCalls = append(computeCalls, append([]uint32{}, ts...))
			switch {
			case len(ts) == len(shortPrompt):
				return nil, nil
			case len(ts) == len(fullPrompt):
				return []kvblock.BlockHash{wantKey}, nil
			default:
				t.Fatalf("unexpected flat token lookup with %d tokens", len(ts))
				return nil, nil
			}
		},
		index: &fakeKVBlockIndex{
			lookup: func(_ context.Context, keys []kvblock.BlockHash, _ sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
				require.NotEmpty(t, keys)
				lookupCalls = append(lookupCalls, append([]kvblock.BlockHash{}, keys...))
				return map[kvblock.BlockHash][]kvblock.PodEntry{
					wantKey: {{PodIdentifier: "10.0.0.1:8080"}},
				}, nil
			},
		},
	}
	scorer := &fakeKVBlockScorer{
		score: func(_ context.Context, keys []kvblock.BlockHash, _ map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error) {
			require.Equal(t, []kvblock.BlockHash{wantKey}, keys)
			return map[string]float64{"10.0.0.1:8080": 1.0}, nil
		},
	}

	p := newProducerWithIndexer(ctx, idx, scorer)
	req := &scheduling.InferenceRequest{
		RequestID:   "req-multi-mixed",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{shortPrompt, fullPrompt},
			},
		},
	}

	require.NoError(t, p.Produce(ctx, req, endpoints))
	require.Equal(t, [][]uint32{shortPrompt, fullPrompt}, computeCalls)
	require.Equal(t, [][]kvblock.BlockHash{{wantKey}}, lookupCalls)

	raw, ok := endpoints[0].Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test").String())
	require.True(t, ok)
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	require.True(t, ok)
	assert.Equal(t, 1, info.MatchBlocks())
	assert.Equal(t, 1, info.TotalBlocks())

	raw, ok = endpoints[1].Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test").String())
	require.True(t, ok)
	info, ok = raw.(*attrprefix.PrefixCacheMatchInfo)
	require.True(t, ok)
	assert.Equal(t, 0, info.MatchBlocks())
	assert.Equal(t, 1, info.TotalBlocks())
}

// Per-tier contiguous counts are attached per endpoint and summed across
// prompts; endpoints without tier data get a non-nil empty map.
func TestProduce_WritesCachedBlocksByTier(t *testing.T) {
	ctx := utils.NewTestContext(t)
	endpoints := freshEndpoints()

	promptA := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	promptB := []uint32{20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
	keysA := []kvblock.BlockHash{0xA1, 0xA2}
	keysB := []kvblock.BlockHash{0xB1}

	gpuA := kvblock.PodEntry{PodIdentifier: "10.0.0.1:8080", DeviceTier: "gpu"}
	cpuA := kvblock.PodEntry{PodIdentifier: "10.0.0.1:8080", DeviceTier: "cpu"}

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, ts []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			if len(ts) == len(promptA) {
				return keysA, nil
			}
			return keysB, nil
		},
		index: &fakeKVBlockIndex{
			lookup: func(_ context.Context, keys []kvblock.BlockHash, _ sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
				if keys[0] == keysA[0] {
					// Prompt A: block 0 on gpu+cpu, block 1 on gpu only.
					return map[kvblock.BlockHash][]kvblock.PodEntry{
						keysA[0]: {gpuA, cpuA},
						keysA[1]: {gpuA},
					}, nil
				}
				// Prompt B: single block on gpu+cpu.
				return map[kvblock.BlockHash][]kvblock.PodEntry{
					keysB[0]: {gpuA, cpuA},
				}, nil
			},
		},
	}
	scorer := &fakeKVBlockScorer{
		score: func(_ context.Context, _ []kvblock.BlockHash, _ map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error) {
			return map[string]float64{"10.0.0.1:8080": 1.0}, nil
		},
	}

	p := newProducerWithIndexer(ctx, idx, scorer)
	req := &scheduling.InferenceRequest{
		RequestID:   "req-by-tier",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{promptA, promptB},
			},
		},
	}

	require.NoError(t, p.Produce(ctx, req, endpoints))

	raw, ok := endpoints[0].Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test").String())
	require.True(t, ok)
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	require.True(t, ok)
	assert.Equal(t, 3, info.CachedBlockCount())
	// gpu: 2 (prompt A) + 1 (prompt B); cpu: 1 (prompt A) + 1 (prompt B).
	assert.Equal(t, map[string]int{"gpu": 3, "cpu": 2}, info.CachedBlocksByTier())

	raw, ok = endpoints[1].Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test").String())
	require.True(t, ok)
	info, ok = raw.(*attrprefix.PrefixCacheMatchInfo)
	require.True(t, ok)
	assert.Equal(t, 0, info.CachedBlockCount())
	assert.NotNil(t, info.CachedBlocksByTier())
	assert.Empty(t, info.CachedBlocksByTier())
}

// Multimodal features flow through to ComputeBlockKeysFromTokens.
func TestProduce_PassesMMExtraFeatures(t *testing.T) {
	ctx := utils.NewTestContext(t)

	tokens := make([]uint32, 16)
	for i := range tokens {
		tokens[i] = uint32(i)
	}
	var captured []*kvblock.BlockExtraFeatures

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, extra []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			captured = extra
			return []kvblock.BlockHash{0xAA}, nil
		},
		index: &fakeKVBlockIndex{
			lookup: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
				return map[kvblock.BlockHash][]kvblock.PodEntry{}, nil
			},
		},
	}
	scorer := &fakeKVBlockScorer{}

	p := newProducerWithIndexer(ctx, idx, scorer)

	req := &scheduling.InferenceRequest{
		RequestID:   "req-mm",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{tokens},
				MultiModalFeatures: []fwkrh.MultiModalFeature{
					{Modality: fwkrh.ModalityImage, Hash: "abc", Offset: 2, Length: 4},
				},
			},
		},
	}

	require.NoError(t, p.Produce(ctx, req, testEndpoints))
	require.NotNil(t, captured)
}

// Cache salt is folded into the first block's extra keys, isolating salted
// prompts. Mirrors engine-side ingestion, which folds vLLM's block-0 cache_salt
// into the same per-block hash list.
func TestProduce_FoldsCacheSalt(t *testing.T) {
	ctx := utils.NewTestContext(t)

	tokens := make([]uint32, 16)
	for i := range tokens {
		tokens[i] = uint32(i)
	}

	tests := []struct {
		name string
		mm   []fwkrh.MultiModalFeature
		want []kvblock.MMHash
	}{
		{
			name: "salt only",
			want: []kvblock.MMHash{{Hash: "s3cr3t"}},
		},
		{
			name: "salt appended after mm hash",
			mm:   []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "abc", Offset: 2, Length: 4}},
			want: []kvblock.MMHash{{Hash: "abc"}, {Hash: "s3cr3t"}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var captured []*kvblock.BlockExtraFeatures
			idx := &fakeKVCacheIndexer{
				computeFromTokens: func(_ context.Context, _ []uint32, _ string, extra []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
					captured = extra
					return []kvblock.BlockHash{0xAA}, nil
				},
				index: &fakeKVBlockIndex{},
			}
			p := newProducerWithIndexer(ctx, idx, &fakeKVBlockScorer{})

			req := &scheduling.InferenceRequest{
				RequestID:   "req-salt",
				TargetModel: "test-model",
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &fwkrh.TokenizedPrompt{
						PerPromptTokens:    [][]uint32{tokens},
						MultiModalFeatures: tc.mm,
						CacheSalt:          "s3cr3t",
					},
				},
			}

			require.NoError(t, p.Produce(ctx, req, testEndpoints))
			require.Len(t, captured, 1)
			require.NotNil(t, captured[0])
			require.Equal(t, tc.want, captured[0].MMHashes)
		})
	}
}

// No salt → extra features stay untouched (nil for a text-only prompt).
func TestProduce_NoCacheSalt_NoExtraFeatures(t *testing.T) {
	ctx := utils.NewTestContext(t)

	tokens := make([]uint32, 16)
	for i := range tokens {
		tokens[i] = uint32(i)
	}
	captured := []*kvblock.BlockExtraFeatures{{}} // sentinel; expect overwrite to nil
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, extra []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			captured = extra
			return []kvblock.BlockHash{0xAA}, nil
		},
		index: &fakeKVBlockIndex{},
	}
	p := newProducerWithIndexer(ctx, idx, &fakeKVBlockScorer{})

	req := &scheduling.InferenceRequest{
		RequestID:   "req-nosalt",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokens}},
		},
	}

	require.NoError(t, p.Produce(ctx, req, testEndpoints))
	require.Nil(t, captured)
}

// nil request / empty body → don't touch the indexer.
func TestProduce_NoOpPaths(t *testing.T) {
	ctx := utils.NewTestContext(t)
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			t.Fatalf("ComputeBlockKeysFromTokens must not be called")
			return nil, nil
		},
		index: &fakeKVBlockIndex{},
	}
	p := newProducerWithIndexer(ctx, idx, &fakeKVBlockScorer{})

	require.NoError(t, p.Produce(ctx, &scheduling.InferenceRequest{RequestID: "x"}, testEndpoints))
	require.NoError(t, p.Produce(ctx, &scheduling.InferenceRequest{RequestID: "x", Body: &fwkrh.InferenceRequestBody{}}, testEndpoints))
}

// Tokens-only: reject legacy tokenizersPoolConfig at factory time.
func TestPluginFactory_RejectsTokenizersPoolConfig(t *testing.T) {
	handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
	raw := json.RawMessage(`{"indexerConfig":{"tokenizersPoolConfig":{"modelName":"x"}}}`)

	_, err := PluginFactory("test", plugin.StrictDecoder(raw), handle)
	require.Error(t, err)
	require.Contains(t, err.Error(), "tokenizersPoolConfig is not supported")
}

// Key built from string literals so an upstream rename trips the test.
func TestProduces_DeclaresPrefixCacheMatchInfo(t *testing.T) {
	p := &Producer{
		typedName: plugin.TypedName{Type: PluginType, Name: "x"},
		dk:        attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("x"),
	}
	expected := plugin.NewDataKey("PrefixCacheMatchInfoDataKey", "approx-prefix-cache-producer").
		WithNonEmptyProducerName("x")
	_, ok := p.Produces()[expected]
	require.True(t, ok)
}

func TestConsumes_DeclaresTokenizedPrompt(t *testing.T) {
	p := &Producer{typedName: plugin.TypedName{Type: PluginType, Name: "x"}}
	expected := plugin.NewDataKey("TokenizedPrompt", "token-producer")
	_, ok := p.Consumes().Required[expected]
	require.True(t, ok)
}

// New() must resolve blockSize via tokenProcessor.BlockSize(). Both the
// canonical blockSizeTokens field and the deprecated blockSize field flow
// to the per-endpoint MatchInfo; the latter via kvblock's promotion rule.
func TestNew_BlockSizeFlowsViaTokenProcessor(t *testing.T) {
	cases := []struct {
		name string
		cfg  *kvblock.TokenProcessorConfig
		want int
	}{
		//nolint:staticcheck // SA1019: exercising backward-compat path on purpose.
		{name: "blockSize only (deprecated)", cfg: &kvblock.TokenProcessorConfig{BlockSize: 64}, want: 64},
		{name: "blockSizeTokens only", cfg: &kvblock.TokenProcessorConfig{BlockSizeTokens: 64}, want: 64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Cancel on test exit so the indexer.Run + kvevents pool workers
			// New() launches don't outlive the subtest.
			ctx, cancel := context.WithCancel(utils.NewTestContext(t))
			t.Cleanup(cancel)

			idxCfg, err := kvcache.NewDefaultConfig()
			require.NoError(t, err)
			cfg := PluginConfig{
				TokenProcessorConfig: tc.cfg,
				IndexerConfig:        idxCfg,
				KVEventsConfig:       kvevents.DefaultConfig(),
			}
			name := "t-" + tc.name
			p, err := New(ctx, name, cfg)
			require.NoError(t, err)

			tokens := make([]uint32, 64)
			for i := range tokens {
				tokens[i] = uint32(i + 1)
			}
			endpoint := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-x"},
				Address:        "10.0.0.9",
				Port:           "8080",
			}, nil, nil)
			req := &scheduling.InferenceRequest{
				RequestID:   "r",
				TargetModel: "m",
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokens}},
				},
			}
			require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{endpoint}))

			raw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name).String())
			require.True(t, ok)
			info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
			require.True(t, ok, "expected *PrefixCacheMatchInfo, got %T", raw)
			assert.Equal(t, tc.want, info.BlockSizeTokens())
			assert.Equal(t, 1, info.TotalBlocks())
		})
	}
}

type fakeSubscriberManager struct {
	ids       []string
	endpoints []string
}

func (f *fakeSubscriberManager) EnsureSubscriber(_ context.Context, _, _, _ string, _ bool) error {
	return nil
}
func (f *fakeSubscriberManager) RemoveSubscriber(_ context.Context, _ string) {}
func (f *fakeSubscriberManager) GetActiveSubscribers() ([]string, []string) {
	return f.ids, f.endpoints
}
func (f *fakeSubscriberManager) Shutdown(_ context.Context) {}

func TestDumpState(t *testing.T) {
	// Start() is intentionally not called: NoTTL entries never expire, and the
	// enumeration helpers work on an unstarted cache.
	specCache := ttlcache.New[string, *speculativeEntries]()
	specCache.Set("req-2", &speculativeEntries{}, ttlcache.NoTTL)
	specCache.Set("req-1", &speculativeEntries{}, ttlcache.NoTTL)

	p := &Producer{
		typedName:          plugin.TypedName{Type: PluginType, Name: "x"},
		subscribersManager: &fakeSubscriberManager{ids: []string{"ns/pod-b", "ns/pod-a"}},
		speculativeCache:   specCache,
		speculativeEnabled: true,
		blockSizeTokens:    64,
	}

	payload, err := p.DumpState()
	require.NoError(t, err)
	// Subscriber pod identities and live request ids are enumerated for debugging.
	assert.Contains(t, string(payload), "ns/pod-a")

	var state precisePrefixState
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Equal(t, precisePrefixState{
		Subscribers:             []string{"ns/pod-a", "ns/pod-b"},
		TotalSubscribers:        2,
		MaxSubscribers:          maxDumpSubscribers,
		SpeculativeIndexing:     true,
		SpeculativeEntries:      []string{"req-1", "req-2"},
		TotalSpeculativeEntries: 2,
		MaxSpeculativeEntries:   maxDumpSpeculativeEntries,
		BlockSizeTokens:         64,
	}, state)
}

func TestDumpStateEmpty(t *testing.T) {
	p := &Producer{}

	payload, err := p.DumpState()
	require.NoError(t, err)
	assert.True(t, json.Valid(payload))
	// Empty lists serialize as [] not null, matching the documented response shape.
	assert.Contains(t, string(payload), `"subscribers":[]`)
	assert.Contains(t, string(payload), `"speculativeEntries":[]`)

	var state precisePrefixState
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Equal(t, precisePrefixState{
		Subscribers:           []string{},
		SpeculativeEntries:    []string{},
		MaxSubscribers:        maxDumpSubscribers,
		MaxSpeculativeEntries: maxDumpSpeculativeEntries,
	}, state)
}

func TestDumpStateCaps(t *testing.T) {
	specCache := ttlcache.New[string, *speculativeEntries]()
	ids := make([]string, 0, maxDumpSubscribers+5)
	for i := 0; i < maxDumpSubscribers+5; i++ {
		ids = append(ids, fmt.Sprintf("ns/pod-%03d", i))
	}
	for i := 0; i < maxDumpSpeculativeEntries+7; i++ {
		specCache.Set(fmt.Sprintf("req-%04d", i), &speculativeEntries{}, ttlcache.NoTTL)
	}

	p := &Producer{
		subscribersManager: &fakeSubscriberManager{ids: ids},
		speculativeCache:   specCache,
	}

	payload, err := p.DumpState()
	require.NoError(t, err)

	var state precisePrefixState
	require.NoError(t, json.Unmarshal(payload, &state))
	// Lists are capped, but the totals report the full counts so a consumer can
	// tell the dump is partial from totalX > maxX.
	assert.Len(t, state.Subscribers, maxDumpSubscribers)
	assert.Equal(t, maxDumpSubscribers+5, state.TotalSubscribers)
	assert.Len(t, state.SpeculativeEntries, maxDumpSpeculativeEntries)
	assert.Equal(t, maxDumpSpeculativeEntries+7, state.TotalSpeculativeEntries)
}
