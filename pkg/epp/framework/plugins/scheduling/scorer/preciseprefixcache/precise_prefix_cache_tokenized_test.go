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
	"hash/fnv"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

type mockKVCacheIndexer struct {
	getPodScoresFunc             func(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string, podIdentifiers []string) (map[string]float64, error)
	scoreTokensFunc              func(ctx context.Context, tokens []uint32, modelName string, podIdentifiers []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error)
	computeBlockKeysFromTokensFn func(ctx context.Context, tokens []uint32, modelName string, extraFeatures []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error)
}

func (m *mockKVCacheIndexer) GetPodScores(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string, podIdentifiers []string) (map[string]float64, error) {
	if m.getPodScoresFunc != nil {
		return m.getPodScoresFunc(ctx, renderReq, prompt, modelName, podIdentifiers)
	}
	return map[string]float64{}, nil
}

func (m *mockKVCacheIndexer) ScoreTokens(ctx context.Context, tokens []uint32, modelName string, podIdentifiers []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
	if m.scoreTokensFunc != nil {
		return m.scoreTokensFunc(ctx, tokens, modelName, podIdentifiers, extraFeatures)
	}
	return map[string]float64{}, nil
}

func (m *mockKVCacheIndexer) ComputeBlockKeys(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string) ([]kvblock.BlockHash, error) {
	return nil, nil
}

func (m *mockKVCacheIndexer) ComputeBlockKeysFromTokens(ctx context.Context, tokens []uint32, modelName string, extraFeatures []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
	if m.computeBlockKeysFromTokensFn != nil {
		return m.computeBlockKeysFromTokensFn(ctx, tokens, modelName, extraFeatures)
	}
	return nil, nil
}

func (m *mockKVCacheIndexer) KVBlockIndex() kvblock.Index {
	return nil
}

var testEndpoints = []scheduling.Endpoint{
	scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: "pod-a"},
			Address:        "10.0.0.1",
			Port:           "8080",
		},
		nil, nil,
	),
	scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: "pod-b"},
			Address:        "10.0.0.2",
			Port:           "8080",
		},
		nil, nil,
	),
}

func TestScorer_UsesTokenizedPrompt(t *testing.T) {
	ctx := utils.NewTestContext(t)
	tokenIDs := []uint32{10, 20, 30, 40, 50}
	var capturedTokens []uint32
	var capturedModel string

	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, tokens []uint32, modelName string, _ []string, _ []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				capturedTokens = tokens
				capturedModel = modelName
				return map[string]float64{"10.0.0.1:8080": 1.0}, nil
			},
		},
	}

	request := &scheduling.InferenceRequest{
		RequestID:   "test-tokenized",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: tokenIDs},
		},
	}

	scorer.Score(ctx, request, testEndpoints)

	require.Equal(t, tokenIDs, capturedTokens)
	require.Equal(t, "test-model", capturedModel)
}

func TestScorer_PassesExtraFeaturesToScoreTokens(t *testing.T) {
	ctx := utils.NewTestContext(t)
	tokenIDs := []uint32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}
	var capturedExtraFeatures []*kvblock.BlockExtraFeatures

	scorer := &Scorer{
		typedName:       plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig:  &kvevents.Config{},
		pluginState:     plugin.NewPluginState(ctx),
		blockSizeTokens: 16,
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, _ []uint32, _ string, _ []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				capturedExtraFeatures = extraFeatures
				return map[string]float64{"10.0.0.1:8080": 1.0}, nil
			},
		},
	}

	request := &scheduling.InferenceRequest{
		RequestID:   "test-mm",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				TokenIDs: tokenIDs,
				MultiModalFeatures: []fwkrh.MultiModalFeature{
					{Modality: fwkrh.ModalityImage, Hash: "abc123", Offset: 2, Length: 4},
				},
			},
		},
	}

	scorer.Score(ctx, request, testEndpoints)

	require.NotNil(t, capturedExtraFeatures, "extraFeatures should be passed to ScoreTokens when MMFeatures present")
}

func TestScorer_NilExtraFeaturesForTextOnly(t *testing.T) {
	ctx := utils.NewTestContext(t)
	tokenIDs := []uint32{10, 20, 30, 40, 50}
	var capturedExtraFeatures []*kvblock.BlockExtraFeatures
	called := false

	scorer := &Scorer{
		typedName:       plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig:  &kvevents.Config{},
		pluginState:     plugin.NewPluginState(ctx),
		blockSizeTokens: 16,
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, _ []uint32, _ string, _ []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				called = true
				capturedExtraFeatures = extraFeatures
				return map[string]float64{"10.0.0.1:8080": 1.0}, nil
			},
		},
	}

	request := &scheduling.InferenceRequest{
		RequestID:   "test-text-only",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: tokenIDs},
		},
	}

	scorer.Score(ctx, request, testEndpoints)

	require.True(t, called, "ScoreTokens should have been called")
	assert.Nil(t, capturedExtraFeatures, "extraFeatures should be nil for text-only requests")
}

func TestScorer_SkipsTokenizedPromptWhenEmpty(t *testing.T) {
	ctx := utils.NewTestContext(t)
	fromTokensCalled := false

	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, _ []uint32, _ string, _ []string, _ []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				fromTokensCalled = true
				return map[string]float64{}, nil
			},
			getPodScoresFunc: func(_ context.Context, _ *types.RenderChatRequest, _ string, _ string, _ []string) (map[string]float64, error) {
				return map[string]float64{}, nil
			},
		},
	}

	request := &scheduling.InferenceRequest{
		RequestID:   "test-skip-empty",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions:     &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: []uint32{}},
		},
	}

	scorer.Score(ctx, request, testEndpoints)
	assert.False(t, fromTokensCalled, "ScoreTokens should not be called with empty TokenIDs")
}

// TestScorer_GenerateFallback_UsesTokenIDs covers the getScores fallback path
// when no TokenizedPrompt is set: a Generate body should drive ScoreTokens.
func TestScorer_GenerateFallback_UsesTokenIDs(t *testing.T) {
	ctx := utils.NewTestContext(t)
	tokenIDs := []uint32{10, 20, 30, 40, 50}
	var capturedTokens []uint32
	var capturedModel string
	var capturedExtraFeatures []*kvblock.BlockExtraFeatures

	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, tokens []uint32, modelName string, _ []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				capturedTokens = tokens
				capturedModel = modelName
				capturedExtraFeatures = extraFeatures
				return map[string]float64{"10.0.0.1:8080": 1.0}, nil
			},
		},
	}

	request := &scheduling.InferenceRequest{
		RequestID:   "test-generate-fallback",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Generate: &fwkrh.GenerateRequest{TokenIDs: tokenIDs},
		},
	}

	scorer.Score(ctx, request, testEndpoints)

	assert.Equal(t, tokenIDs, capturedTokens)
	assert.Equal(t, "test-model", capturedModel)
	assert.Nil(t, capturedExtraFeatures, "extraFeatures should be nil for text-only generate request")
}

// TestScorer_GenerateFallback_PassesFeaturesToScoreTokens locks in that the
// getScores fallback honors Body.Generate.Features so two requests with the
// same token_ids but different image hashes route distinctly.
func TestScorer_GenerateFallback_PassesFeaturesToScoreTokens(t *testing.T) {
	ctx := utils.NewTestContext(t)
	tokenIDs := []uint32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}
	var capturedExtraFeatures []*kvblock.BlockExtraFeatures

	scorer := &Scorer{
		typedName:       plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig:  &kvevents.Config{},
		pluginState:     plugin.NewPluginState(ctx),
		blockSizeTokens: 16,
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, _ []uint32, _ string, _ []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				capturedExtraFeatures = extraFeatures
				return map[string]float64{"10.0.0.1:8080": 1.0}, nil
			},
		},
	}

	request := &scheduling.InferenceRequest{
		RequestID:   "test-generate-mm",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Generate: &fwkrh.GenerateRequest{
				TokenIDs: tokenIDs,
				Features: &tokenization.MultiModalFeatures{
					MMHashes: map[string][]string{"image": {"abc123hash"}},
					MMPlaceholders: map[string][]kvblock.PlaceholderRange{
						"image": {{Offset: 2, Length: 4}},
					},
				},
			},
		},
	}

	scorer.Score(ctx, request, testEndpoints)

	require.NotNil(t, capturedExtraFeatures, "extraFeatures should be passed when Generate.Features is present")
}

// TestScorer_ComputeBlockKeys_GenerateFallback exercises computeBlockKeys
// directly and locks in that Body.Generate.Features flows through to
// ComputeBlockKeysFromTokens via extraFeatures.
func TestScorer_ComputeBlockKeys_GenerateFallback(t *testing.T) {
	tokenIDs := []uint32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}

	tests := []struct {
		name            string
		body            *fwkrh.InferenceRequestBody
		wantExtraNonNil bool
		wantTokenIDs    []uint32
	}{
		{
			name: "text-only generate request — extraFeatures nil",
			body: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{TokenIDs: tokenIDs},
			},
			wantExtraNonNil: false,
			wantTokenIDs:    tokenIDs,
		},
		{
			name: "generate request with multimodal Features — extraFeatures non-nil",
			body: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{
					TokenIDs: tokenIDs,
					Features: &tokenization.MultiModalFeatures{
						MMHashes: map[string][]string{"image": {"abc123hash"}},
						MMPlaceholders: map[string][]kvblock.PlaceholderRange{
							"image": {{Offset: 2, Length: 4}},
						},
					},
				},
			},
			wantExtraNonNil: true,
			wantTokenIDs:    tokenIDs,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			var capturedTokens []uint32
			var capturedExtra []*kvblock.BlockExtraFeatures

			scorer := &Scorer{
				typedName:       plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
				blockSizeTokens: 16,
				kvCacheIndexer: &mockKVCacheIndexer{
					computeBlockKeysFromTokensFn: func(_ context.Context, tokens []uint32, _ string, extra []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
						capturedTokens = tokens
						capturedExtra = extra
						return nil, nil
					},
				},
			}

			request := &scheduling.InferenceRequest{
				RequestID:   "test-compute-block-keys",
				TargetModel: "test-model",
				Body:        tc.body,
			}

			_, err := scorer.computeBlockKeys(ctx, request)
			require.NoError(t, err)
			assert.Equal(t, tc.wantTokenIDs, capturedTokens)
			if tc.wantExtraNonNil {
				assert.NotNil(t, capturedExtra, "extraFeatures should be non-nil when Features present")
			} else {
				assert.Nil(t, capturedExtra, "extraFeatures should be nil for text-only request")
			}
		})
	}
}

// TestScorer_ComputeBlockKeys_GenerateMMHashesAffectKeys exercises the
// regression case where Generate.Features.MMHashes is silently dropped while
// extraFeatures is still emitted as a non-nil slice. The mock derives block
// keys deterministically from the per-block MMHashes, so two requests sharing
// TokenIDs but differing in MMHashes must produce different block keys; an
// empty/identical extraFeatures payload would collapse them.
func TestScorer_ComputeBlockKeys_GenerateMMHashesAffectKeys(t *testing.T) {
	tokenIDs := []uint32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}

	keysFromHash := func(hash string) []kvblock.BlockHash {
		ctx := utils.NewTestContext(t)
		var captured []*kvblock.BlockExtraFeatures

		scorer := &Scorer{
			typedName:       plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
			blockSizeTokens: 16,
			kvCacheIndexer: &mockKVCacheIndexer{
				computeBlockKeysFromTokensFn: func(_ context.Context, tokens []uint32, _ string, extra []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
					captured = extra
					keys := make([]kvblock.BlockHash, len(tokens)/16)
					for i := range keys {
						h := fnv.New64a()
						_, _ = h.Write([]byte{byte(tokens[i*16])})
						if i < len(extra) && extra[i] != nil {
							for _, mm := range extra[i].MMHashes {
								_, _ = h.Write([]byte(mm.Hash))
							}
						}
						keys[i] = kvblock.BlockHash(h.Sum64())
					}
					return keys, nil
				},
			},
		}

		request := &scheduling.InferenceRequest{
			RequestID:   "test-mm-hash-affects-keys",
			TargetModel: "test-model",
			Body: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{
					TokenIDs: tokenIDs,
					Features: &tokenization.MultiModalFeatures{
						MMHashes: map[string][]string{"image": {hash}},
						MMPlaceholders: map[string][]kvblock.PlaceholderRange{
							"image": {{Offset: 2, Length: 4}},
						},
					},
				},
			},
		}

		keys, err := scorer.computeBlockKeys(ctx, request)
		require.NoError(t, err)
		require.NotEmpty(t, captured, "extraFeatures must reach the indexer when Features are present")
		return keys
	}

	keysA := keysFromHash("abc123hash")
	keysB := keysFromHash("xyz789hash")

	require.Equal(t, len(keysA), len(keysB), "key counts should match for identical TokenIDs")
	assert.NotEqual(t, keysA, keysB, "different MMHashes must produce different block keys")
}
