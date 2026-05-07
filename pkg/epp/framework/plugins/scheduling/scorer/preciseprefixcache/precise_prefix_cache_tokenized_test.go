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
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

type mockKVCacheIndexer struct {
	getPodScoresFunc func(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string, podIdentifiers []string) (map[string]float64, error)
	scoreTokensFunc  func(ctx context.Context, tokens []uint32, modelName string, podIdentifiers []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error)
	index            kvblock.Index
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
	return m.index
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

func TestScorer_MultiPromptTokenizedIndependentScoring(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// Two prompts: tokens for "hello" and "world"
	prompt1Tokens := []uint32{10, 20, 30}
	prompt2Tokens := []uint32{40, 50, 60}

	var scoreTokensCalls [][]uint32
	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, tokens []uint32, _ string, _ []string, _ []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				tokensCopy := make([]uint32, len(tokens))
				copy(tokensCopy, tokens)
				scoreTokensCalls = append(scoreTokensCalls, tokensCopy)

				// Pod A has "hello" cached, Pod B has "world" cached
				if tokens[0] == 10 {
					return map[string]float64{
						"10.0.0.1:8080": 1.0,
						"10.0.0.2:8080": 0.0,
					}, nil
				}
				return map[string]float64{
					"10.0.0.1:8080": 0.0,
					"10.0.0.2:8080": 1.0,
				}, nil
			},
		},
	}

	allFlat := append(append([]uint32{}, prompt1Tokens...), prompt2Tokens...)
	request := &scheduling.InferenceRequest{
		RequestID:   "test-multi-prompt",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Strings: []string{"hello", "world"}},
			},
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				TokenIDs:        allFlat,
				PerPromptTokens: [][]uint32{prompt1Tokens, prompt2Tokens},
			},
		},
	}

	scores := scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)

	// ScoreTokens should have been called once per prompt
	require.Len(t, scoreTokensCalls, 2, "ScoreTokens should be called once per prompt")
	assert.Equal(t, prompt1Tokens, scoreTokensCalls[0])
	assert.Equal(t, prompt2Tokens, scoreTokensCalls[1])

	// Both pods should have score 1.0 (sum of independent scores)
	gotByAddress := make(map[string]float64)
	for ep, score := range scores {
		if m := ep.GetMetadata(); m != nil {
			gotByAddress[m.Address+":"+m.Port] = score
		}
	}
	assert.Equal(t, 1.0, gotByAddress["10.0.0.1:8080"], "pod-a: 1.0 from prompt1 + 0.0 from prompt2")
	assert.Equal(t, 1.0, gotByAddress["10.0.0.2:8080"], "pod-b: 0.0 from prompt1 + 1.0 from prompt2")
}

func TestScorer_InvalidPerPromptTokensUsesFlatScoring(t *testing.T) {
	ctx := utils.NewTestContext(t)

	tokenIDs := []uint32{10, 20, 30, 40}
	var scoreTokensCalls [][]uint32
	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, tokens []uint32, _ string, _ []string, _ []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				scoreTokensCalls = append(scoreTokensCalls, append([]uint32{}, tokens...))
				return map[string]float64{"10.0.0.1:8080": 1.0}, nil
			},
		},
	}

	request := &scheduling.InferenceRequest{
		RequestID:   "test-invalid-per-prompt",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hello"}}},
			},
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				TokenIDs:        tokenIDs,
				PerPromptTokens: [][]uint32{{10, 20}, {30, 40}},
			},
		},
	}

	scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)

	require.Len(t, scoreTokensCalls, 1)
	assert.Equal(t, tokenIDs, scoreTokensCalls[0])
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

func TestAggregatePromptScores(t *testing.T) {
	assert.Equal(t,
		map[string]float64{"pod-a": 1.25, "pod-b": 2.0},
		aggregatePromptScores([]map[string]float64{
			{"pod-a": 1.0, "pod-b": 0.5},
			{"pod-a": 0.25, "pod-b": 1.5},
		}))
}

type recordedAddCall struct {
	engineKeys  []kvblock.BlockHash
	requestKeys []kvblock.BlockHash
	entries     []kvblock.PodEntry
}

type recordingIndex struct {
	addCalls []recordedAddCall
}

func (r *recordingIndex) Lookup(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
	return map[kvblock.BlockHash][]kvblock.PodEntry{}, nil
}

func (r *recordingIndex) Add(_ context.Context, engineKeys, requestKeys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
	r.addCalls = append(r.addCalls, recordedAddCall{
		engineKeys:  cloneBlockHashes(engineKeys),
		requestKeys: cloneBlockHashes(requestKeys),
		entries:     append([]kvblock.PodEntry{}, entries...),
	})
	return nil
}

func cloneBlockHashes(in []kvblock.BlockHash) []kvblock.BlockHash {
	if in == nil {
		return nil
	}
	return append([]kvblock.BlockHash{}, in...)
}

func (r *recordingIndex) Evict(_ context.Context, _ kvblock.BlockHash, _ kvblock.KeyType, _ []kvblock.PodEntry) error {
	return nil
}

func (r *recordingIndex) GetRequestKey(_ context.Context, _ kvblock.BlockHash) (kvblock.BlockHash, error) {
	return kvblock.EmptyBlockHash, nil
}

func newPreRequestTestScorer(ctx context.Context, index kvblock.Index) *Scorer {
	return &Scorer{
		typedName:          plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig:     &kvevents.Config{},
		pluginState:        plugin.NewPluginState(ctx),
		kvCacheIndexer:     &mockKVCacheIndexer{index: index},
		speculativeEnabled: true,
		speculativeCache:   ttlcache.New[string, *speculativeEntries](),
		speculativeTTL:     time.Second,
	}
}

func TestPreRequest_AddsSpeculativeEntriesPerPrompt(t *testing.T) {
	ctx := utils.NewTestContext(t)
	index := &recordingIndex{}
	scorer := newPreRequestTestScorer(ctx, index)
	request := &scheduling.InferenceRequest{RequestID: "test-pre-request"}
	scorer.pluginState.Write(request.RequestID, stateKey, &precisePluginState{
		blockKeys: [][]kvblock.BlockHash{{11, 12}, {21, 22}},
		scores:    map[string]float64{},
	})

	scorer.PreRequest(ctx, request, &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{testEndpoints[0]}},
		},
	})

	require.Len(t, index.addCalls, 2)
	assert.Nil(t, index.addCalls[0].engineKeys)
	assert.Equal(t, []kvblock.BlockHash{11, 12}, index.addCalls[0].requestKeys)
	assert.Equal(t, []kvblock.BlockHash{21, 22}, index.addCalls[1].requestKeys)
	assert.Equal(t, "10.0.0.1:8080", index.addCalls[0].entries[0].PodIdentifier)
	assert.True(t, index.addCalls[0].entries[0].Speculative)

	item := scorer.speculativeCache.Get(request.RequestID)
	require.NotNil(t, item)
	assert.Equal(t, []kvblock.BlockHash{11, 12, 21, 22}, item.Value().blockKeys)
}

func TestPreRequest_AddsPrefillSpeculativeEntriesPerPrompt(t *testing.T) {
	ctx := utils.NewTestContext(t)
	index := &recordingIndex{}
	scorer := newPreRequestTestScorer(ctx, index)
	request := &scheduling.InferenceRequest{RequestID: "test-pre-request-prefill"}
	scorer.pluginState.Write(request.RequestID, stateKey, &precisePluginState{
		blockKeys: [][]kvblock.BlockHash{{11, 12}, {21, 22}},
		scores:    map[string]float64{},
	})

	scorer.PreRequest(ctx, request, &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default":                  {TargetEndpoints: []scheduling.Endpoint{testEndpoints[0]}},
			experimentalPrefillProfile: {TargetEndpoints: []scheduling.Endpoint{testEndpoints[1]}},
		},
	})

	require.Len(t, index.addCalls, 4)
	assert.Equal(t, []kvblock.BlockHash{11, 12}, index.addCalls[0].requestKeys)
	assert.Equal(t, []kvblock.BlockHash{21, 22}, index.addCalls[1].requestKeys)
	assert.Equal(t, []kvblock.BlockHash{11, 12}, index.addCalls[2].requestKeys)
	assert.Equal(t, []kvblock.BlockHash{21, 22}, index.addCalls[3].requestKeys)
	assert.Equal(t, "10.0.0.1:8080", index.addCalls[0].entries[0].PodIdentifier)
	assert.Equal(t, "10.0.0.2:8080", index.addCalls[2].entries[0].PodIdentifier)
}
