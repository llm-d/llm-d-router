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
	"fmt"
	"testing"

	"github.com/go-logr/logr"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	datagraph "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixmetrics"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

func TestAdmissionPreparationDefersStateAndReusesMatchingSnapshot(t *testing.T) {
	queries := 0
	match := 1
	indexer := &fakeKVCacheIndexer{
		computeFromTokens: func(context.Context, []uint32, string, []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return []kvblock.BlockHash{123}, nil
		},
		matchBlockKeys: func(context.Context, []kvblock.BlockHash, sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			queries++
			return map[string]kvcache.PodMatch{"10.0.0.1:8080": {WeightedScore: float64(match), MatchedBlocks: match}}, nil
		},
	}
	p := newProducerWithIndexer(t.Context(), indexer)
	p.speculativeEnabled = true
	request := &scheduling.InferenceRequest{RequestID: "queued", Body: &fwkrh.InferenceRequestBody{
		TokenizedRequest: fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, testBlockSize)}),
	}}
	endpoints := freshEndpoints()
	require.NoError(t, p.PrepareForAdmission(t.Context(), request, endpoints))
	_, err := plugin.ReadPluginStateKey[*blockKeysState](p.pluginState, request.RequestID, blockKeysStateKey)
	require.Error(t, err)
	info, _ := endpoints[0].Get(p.dk)
	require.Equal(t, 1, info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())

	match = 0
	require.NoError(t, p.PrepareForAdmission(t.Context(), request, endpoints))
	require.Equal(t, 2, queries)
	_, err = plugin.ReadPluginStateKey[*blockKeysState](p.pluginState, request.RequestID, blockKeysStateKey)
	require.Error(t, err)
	match = 1
	datagraph.RegisterScopeSpecs([]plugin.Plugin{p})
	scoped, violations := datagraph.ScopeRequest(logr.Discard(), requestcontrol.DataProducerExtensionPoint, p, request)
	require.NoError(t, p.Produce(t.Context(), scoped, endpoints))
	require.NoError(t, violations.Write())
	require.Equal(t, 2, queries)
	info, _ = endpoints[0].Get(p.dk)
	require.Zero(t, info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())
	state, err := plugin.ReadPluginStateKey[*blockKeysState](p.pluginState, request.RequestID, blockKeysStateKey)
	require.NoError(t, err)
	require.Equal(t, [][]kvblock.BlockHash{{123}}, state.perPromptKeys)

	prepared, ok := scheduling.ReadRequestAttribute[*admissionPrefix](request, p.admissionKey())
	require.True(t, ok)
	cloned := prepared.Clone().(*admissionPrefix)
	prepared.perPromptKeys[0][0]++
	require.Equal(t, kvblock.BlockHash(123), cloned.perPromptKeys[0][0])
}

func TestAdmissionPreparationCancellationPublishesNoState(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	indexer := &fakeKVCacheIndexer{
		computeFromTokens: func(context.Context, []uint32, string, []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return []kvblock.BlockHash{123}, nil
		},
		matchBlockKeys: func(context.Context, []kvblock.BlockHash, sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			cancel()
			return map[string]kvcache.PodMatch{"10.0.0.1:8080": {WeightedScore: 1}}, nil
		},
	}
	p := newProducerWithIndexer(t.Context(), indexer)
	p.speculativeEnabled = true
	request := &scheduling.InferenceRequest{RequestID: "cancelled", Body: &fwkrh.InferenceRequestBody{
		TokenizedRequest: fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, testBlockSize)}),
	}}
	endpoints := freshEndpoints()
	require.ErrorIs(t, p.PrepareForAdmission(ctx, request, endpoints), context.Canceled)
	_, ok := request.GetAttribute(p.admissionKey())
	require.False(t, ok)
	for _, endpoint := range endpoints {
		_, ok = endpoint.Get(p.dk)
		require.False(t, ok)
	}
	_, err := plugin.ReadPluginStateKey[*blockKeysState](p.pluginState, request.RequestID, blockKeysStateKey)
	require.Error(t, err)
	_, err = plugin.ReadPluginStateKey[*bestAvailableState](p.pluginState, request.RequestID, bestAvailableStateKey)
	require.Error(t, err)
}

func TestAdmissionPredictionUsesPreparedMatchesAndProduceCandidates(t *testing.T) {
	const name = "precise-admission-predictions"
	const promptBlocks = 8
	queries := 0
	var matches []int
	indexer := &fakeKVCacheIndexer{
		computeFromTokens: func(context.Context, []uint32, string, []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return []kvblock.BlockHash{1, 2, 3, 4, 5, 6, 7, 8}, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, candidates sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			queries++
			result := make(map[string]kvcache.PodMatch, len(matches))
			for i, blocks := range matches {
				id := fmt.Sprintf("10.0.0.%d:8080", i+1)
				if candidates.Has(id) {
					result[id] = kvcache.PodMatch{
						WeightedScore: float64(blocks) / 2,
						MatchedBlocks: blocks,
						BlocksByTier:  map[string]int{"cpu": blocks},
					}
				}
			}
			return result, nil
		},
	}
	prefixmetrics.Register()
	p := newProducerForProduceAndPreRequest(t.Context(), name, indexer)
	require.False(t, p.speculativeEnabled)
	endpoints := make([]scheduling.Endpoint, 4)
	for i := range endpoints {
		endpoints[i] = scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: fmt.Sprintf("pod-%d", i)},
			Address: fmt.Sprintf("10.0.0.%d", i+1), Port: "8080",
		}, nil, nil)
	}
	request := tokenizedRequest("queued-predictions", promptBlocks*testBlockSize)
	metrics := []string{predictedCachedTokensMetric, bestPredictedMetric, bestAvailableMetric, promptTokensMetric}
	before := make(map[string]*dto.Histogram, len(metrics))
	for _, metric := range metrics {
		before[metric] = sharedPrefixHistogram(t, metric, name, prefixmetrics.RoleDecode)
	}
	datagraph.RegisterScopeSpecs([]plugin.Plugin{p})
	for _, refreshedMatches := range [][]int{{2, 3, 5, 7}, {1, 2, 4, 6}} {
		matches = refreshedMatches
		scoped, scopedEndpoints, violations := datagraph.ScopeInvocation(logr.Discard(), requestcontrol.AdmissionDataProducerExtensionPoint, p, request, endpoints)
		require.NoError(t, p.PrepareForAdmission(t.Context(), scoped, scopedEndpoints))
		require.NoError(t, violations.Write())
		_, err := plugin.ReadPluginStateKey[*bestAvailableState](p.pluginState, request.RequestID, bestAvailableStateKey)
		require.Error(t, err)
		_, err = plugin.ReadPluginStateKey[*blockKeysState](p.pluginState, request.RequestID, blockKeysStateKey)
		require.Error(t, err)
		for _, metric := range metrics {
			require.Equal(t, before[metric].GetSampleCount(), sharedPrefixHistogram(t, metric, name, prefixmetrics.RoleDecode).GetSampleCount())
		}
	}
	require.Equal(t, 2, queries)
	prepared, ok := scheduling.ReadRequestAttribute[*admissionPrefix](request, p.admissionKey())
	require.True(t, ok)
	snapshot := prepared.Clone()
	matches = nil

	// The producer's candidate boundary excludes pod-3; the picker later excludes pod-2.
	produced := make([]scheduling.Endpoint, 3)
	for i := range produced {
		produced[i] = scheduling.NewEndpoint(endpoints[i].GetMetadata(), nil, nil)
	}
	scoped, scopedEndpoints, violations := datagraph.ScopeInvocation(logr.Discard(), requestcontrol.DataProducerExtensionPoint, p, request, produced)
	require.NoError(t, p.Produce(t.Context(), scoped, scopedEndpoints))
	require.NoError(t, violations.Write())
	require.Equal(t, 2, queries, "Produce must use the admitted match snapshot")
	require.Equal(t, snapshot, prepared, "publishing must not mutate the admission snapshot")
	state, err := plugin.ReadPluginStateKey[*bestAvailableState](p.pluginState, request.RequestID, bestAvailableStateKey)
	require.NoError(t, err)
	require.Equal(t, 4*testBlockSize, state.cachedTokens)
	for i, blocks := range []int{1, 2, 4} {
		info, ok := produced[i].Get(p.dk)
		require.True(t, ok)
		require.Equal(t, blocks, info.(*attrprefix.PrefixCacheMatchInfo).CachedBlockCount())
	}
	for _, metric := range metrics {
		require.Equal(t, before[metric].GetSampleCount(), sharedPrefixHistogram(t, metric, name, prefixmetrics.RoleDecode).GetSampleCount())
	}
	require.NoError(t, p.PreRequest(t.Context(), request, primaryWithScored("decode", produced[0], produced[0], produced[1])))
	want := []float64{testBlockSize, 2 * testBlockSize, 4 * testBlockSize, promptBlocks * testBlockSize}
	for i, metric := range metrics {
		after := sharedPrefixHistogram(t, metric, name, prefixmetrics.RoleDecode)
		require.Equal(t, before[metric].GetSampleCount()+1, after.GetSampleCount())
		require.Equal(t, before[metric].GetSampleSum()+want[i], after.GetSampleSum())
	}
	_, err = plugin.ReadPluginStateKey[*bestAvailableState](p.pluginState, request.RequestID, bestAvailableStateKey)
	require.Error(t, err)
}

func TestAdmissionMMPredictionUsesPreparedSnapshot(t *testing.T) {
	for _, tc := range []struct {
		name       string
		prefill    bool
		textOnly   bool
		miss       bool
		shortMM    bool
		narrow     bool
		wantBlocks int
		wantTokens int
	}{
		{name: "decode", wantBlocks: 3, wantTokens: 25},
		{name: "prefill", prefill: true, wantBlocks: 1, wantTokens: 14},
		{name: "miss", miss: true},
		{name: "text", textOnly: true},
		{name: "short-mm-prompt", shortMM: true, wantBlocks: 3, wantTokens: 25},
		{name: "narrowed-candidates", narrow: true, wantBlocks: 1, wantTokens: 14},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := setupSpanRecorder(t)
			name := "precise-admission-mm-" + tc.name
			computations, queries := 0, 0
			var matches [2][2]int
			indexer := &fakeKVCacheIndexer{
				computeFromTokens: func(_ context.Context, tokens []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
					computations++
					keys := make([]kvblock.BlockHash, len(tokens)/testBlockSize)
					for i := range keys {
						keys[i] = kvblock.BlockHash(tokens[0]) + kvblock.BlockHash(i)
					}
					return keys, nil
				},
				matchBlockKeys: func(_ context.Context, keys []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
					queries++
					prompt := 0
					if keys[0] == 100 {
						prompt = 1
					}
					result := make(map[string]kvcache.PodMatch, len(matches))
					for pod, counts := range matches {
						blocks := counts[prompt]
						result[fmt.Sprintf("10.0.0.%d:8080", pod+1)] = kvcache.PodMatch{
							WeightedScore: float64(blocks) / 2, MatchedBlocks: blocks,
							BlocksByTier: map[string]int{"cpu": blocks},
						}
					}
					return result, nil
				},
			}
			request := tokenizedRequest(name, 4*testBlockSize)
			request.Body.TokenizedRequest.Prompts = append(request.Body.TokenizedRequest.Prompts,
				fwkrh.PromptTokens{TokenIDs: make([]uint32, 2*testBlockSize)})
			request.Body.TokenizedRequest.Prompts[1].TokenIDs[0] = 100
			if !tc.textOnly {
				request.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrh.MultiModalFeature{
					{Modality: fwkrh.ModalityImage, Hash: "a", Offset: 2, Length: 20},
					{Modality: fwkrh.ModalityImage, Hash: "b", Offset: 32, Length: 20},
				}
				request.Body.TokenizedRequest.Prompts[1].MultiModalFeatures = []fwkrh.MultiModalFeature{
					{Modality: fwkrh.ModalityAudio, Hash: "c", Offset: 5, Length: 5},
				}
			}
			wantPromptTokens, wantMMBlocks := 45, 5
			if tc.shortMM {
				request.Body.TokenizedRequest.Prompts = append(request.Body.TokenizedRequest.Prompts, fwkrh.PromptTokens{
					TokenIDs: make([]uint32, testBlockSize/2),
					MultiModalFeatures: []fwkrh.MultiModalFeature{
						{Modality: fwkrh.ModalityAudio, Hash: "short", Offset: 1, Length: 3},
					},
				})
				wantPromptTokens += 3
				wantMMBlocks++
			}
			prefixmetrics.Register()
			p := newProducerForProduceAndPreRequest(t.Context(), name, indexer)
			datagraph.RegisterScopeSpecs([]plugin.Plugin{p})
			roles := []string{prefixmetrics.RoleDecode, prefixmetrics.RolePrefill}
			metrics := []string{mmPredictedCachedTokensMetric, mmPromptTokensMetric}
			predictionMetrics := []string{predictedCachedTokensMetric, bestPredictedMetric, bestAvailableMetric, promptTokensMetric}
			before := make(map[[2]string]*dto.Histogram)
			for _, role := range roles {
				for _, metric := range append(metrics, predictionMetrics...) {
					before[[2]string{role, metric}] = sharedPrefixHistogram(t, metric, name, role)
				}
			}
			checkMetrics := func(recordedMMRole, recordedRole string) {
				t.Helper()
				for _, role := range roles {
					for i, metric := range metrics {
						previous := before[[2]string{role, metric}]
						count, sum := previous.GetSampleCount(), previous.GetSampleSum()
						if role == recordedMMRole {
							count++
							sum += []float64{float64(tc.wantTokens), float64(wantPromptTokens)}[i]
						}
						after := sharedPrefixHistogram(t, metric, name, role)
						require.Equal(t, count, after.GetSampleCount(), "%s/%s", role, metric)
						require.Equal(t, sum, after.GetSampleSum(), "%s/%s", role, metric)
					}
					for _, metric := range predictionMetrics {
						count := before[[2]string{role, metric}].GetSampleCount()
						if role == recordedRole {
							count++
						}
						require.Equal(t, count, sharedPrefixHistogram(t, metric, name, role).GetSampleCount(), "%s/%s", role, metric)
					}
				}
			}
			endpoints := freshEndpoints()
			refreshed := [2][2]int{{2, 1}, {1, 0}}
			if tc.miss {
				refreshed = [2][2]int{}
			}
			for _, snapshot := range [][2][2]int{{{4, 2}, {4, 2}}, refreshed} {
				matches = snapshot
				scoped, scopedEndpoints, violations := datagraph.ScopeInvocation(logr.Discard(), requestcontrol.AdmissionDataProducerExtensionPoint, p, request, endpoints)
				require.NoError(t, p.PrepareForAdmission(t.Context(), scoped, scopedEndpoints))
				require.NoError(t, violations.Write())
				checkMetrics("", "")
			}
			require.Empty(t, recorder.Ended(), "admission retries must not record producer spans")
			matches = [2][2]int{{4, 2}, {4, 2}}
			produced := freshEndpoints()
			if tc.narrow {
				produced = produced[1:]
			}
			scoped, scopedEndpoints, violations := datagraph.ScopeInvocation(logr.Discard(), requestcontrol.DataProducerExtensionPoint, p, request, produced)
			require.NoError(t, p.Produce(t.Context(), scoped, scopedEndpoints))
			require.NoError(t, violations.Write())
			wantComputations := 4
			if tc.shortMM {
				wantComputations = 6
			}
			require.Equal(t, wantComputations, computations, "Produce must reuse the admitted block keys")
			require.Equal(t, 4, queries, "Produce must reuse the final admission match")
			checkMetrics("", "")
			require.Len(t, recorder.Ended(), 1)
			attrs := spanAttrs(spanByName(t, recorder, "produce_precise_prefix_cache"))
			wantMaxMatch := int64(1)
			if tc.miss || tc.narrow {
				wantMaxMatch = 0
			}
			require.Equal(t, wantMaxMatch, attrs[semconv.LLMDEPPProducerMaxMatchBlocksKey].AsInt64())
			if tc.textOnly {
				require.NotContains(t, attrs, mmMatchedBlocksKey)
				require.NotContains(t, attrs, mmTotalBlocksKey)
			} else {
				wantMaxMMMatch := int64(3)
				if tc.miss {
					wantMaxMMMatch = 0
				}
				if tc.narrow {
					wantMaxMMMatch = 1
				}
				require.Equal(t, wantMaxMMMatch, attrs[mmMatchedBlocksKey].AsInt64())
				require.Equal(t, int64(wantMMBlocks), attrs[mmTotalBlocksKey].AsInt64())
			}

			selected, role := produced[0], prefixmetrics.RoleDecode
			result := primaryOnly("decode", produced[0])
			if tc.prefill {
				selected, role = produced[1], prefixmetrics.RolePrefill
				result.ProfileResults[experimentalPrefillProfile] = &scheduling.ProfileRunResult{TargetEndpoints: []scheduling.Endpoint{selected}}
			}
			info, ok := p.matchInfo(selected)
			require.True(t, ok)
			predictionRole := role
			if tc.textOnly {
				require.Nil(t, info.MM())
				role = ""
			} else {
				require.NotNil(t, info.MM())
				require.Equal(t, tc.wantBlocks, info.MM().MatchBlocks)
				require.Equal(t, tc.wantTokens, info.MM().MatchTokens)
				prepared, ok := scheduling.ReadRequestAttribute[*admissionPrefix](request, p.admissionKey())
				require.True(t, ok)
				cloned := prepared.Clone().(*admissionPrefix)
				id := selected.GetMetadata().ID
				cloned.results[id].MM().MatchTokens++
				require.Equal(t, tc.wantTokens, prepared.results[id].MM().MatchTokens)
			}
			require.NoError(t, p.PreRequest(t.Context(), request, result))
			checkMetrics(role, predictionRole)
			wantModality := "audio,image"
			if tc.textOnly {
				wantModality = "none"
			}
			for _, metric := range predictionMetrics {
				require.Equal(t, wantModality, sharedPrefixModality(t, metric, name))
			}
		})
	}
}
