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

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
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
	getPodScoresFunc func(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string, podIdentifiers []string) (map[string]float64, error)
	scoreTokensFunc  func(ctx context.Context, tokens []uint32, modelName string, podIdentifiers []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error)
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

	scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)

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

	scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)

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

	scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)

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

	scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)
	assert.False(t, fromTokensCalled, "ScoreTokens should not be called with empty TokenIDs")
}
