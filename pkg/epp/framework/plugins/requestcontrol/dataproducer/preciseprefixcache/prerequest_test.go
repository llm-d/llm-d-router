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
	"sync"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/observability/multimodal"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixmetrics"
	"github.com/llm-d/llm-d-router/test/utils"
)

// addCall captures one fakeKVBlockIndex.Add invocation.
type addCall struct {
	keys    []kvblock.BlockHash
	entries []kvblock.PodEntry
}

// evictCall captures one fakeKVBlockIndex.Evict invocation.
type evictCall struct {
	keyType kvblock.KeyType
	keys    []kvblock.BlockHash
	entries []kvblock.PodEntry
}

func newProducerForPreRequest(ctx context.Context, speculativeEnabled bool, idx *fakeKVBlockIndex) *Producer {
	return newNamedProducerForPreRequest(ctx, "test", speculativeEnabled, idx)
}

func newNamedProducerForPreRequest(ctx context.Context, name string, speculativeEnabled bool, idx *fakeKVBlockIndex) *Producer {
	cache := ttlcache.New[string, *speculativeEntries](
		ttlcache.WithTTL[string, *speculativeEntries](time.Minute),
	)
	return &Producer{
		typedName:          plugin.TypedName{Type: PluginType, Name: name},
		kvCacheIndexer:     &fakeKVCacheIndexer{index: idx},
		dk:                 attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		speculativeCache:   cache,
		speculativeTTL:     time.Minute,
		speculativeEnabled: speculativeEnabled,
		pluginState:        plugin.NewPluginState(ctx),
	}
}

func primaryOnly(name string, endpoint scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: name,
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			name: {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

// speculativeEnabled=true with populated block keys: index.Add called once
// with the primary pod identifier, and a speculative cache entry is created.
func TestPreRequest_SeedsSpeculativeForPrimary(t *testing.T) {
	ctx := utils.NewTestContext(t)

	var calls []addCall
	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			calls = append(calls, addCall{keys: keys, entries: entries})
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	blockKeys := []kvblock.BlockHash{0xAA, 0xBB}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-1"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: [][]kvblock.BlockHash{blockKeys}})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	require.Len(t, calls, 1)
	assert.Equal(t, blockKeys, calls[0].keys)
	require.Len(t, calls[0].entries, 1)
	assert.Equal(t, "10.0.0.1:8080", calls[0].entries[0].PodIdentifier)
	assert.True(t, calls[0].entries[0].Speculative)

	cached := p.speculativeCache.Get(req.RequestID)
	require.NotNil(t, cached)
	assert.Equal(t, [][]kvblock.BlockHash{blockKeys}, cached.Value().perPromptKeys)
	require.Len(t, cached.Value().podEntries, 1)
	assert.Equal(t, "10.0.0.1:8080", cached.Value().podEntries[0].PodIdentifier)
}

// speculativeEnabled=true with empty blockKeys: PreRequest must not call
// index.Add and must not create a cache entry.
func TestPreRequest_EmptyBlockKeys_NoAdd(t *testing.T) {
	ctx := utils.NewTestContext(t)

	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, _ []kvblock.BlockHash, _ []kvblock.PodEntry) error {
			t.Fatalf("index.Add must not be called when blockKeys are empty")
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	req := &scheduling.InferenceRequest{RequestID: "req-pre-empty"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: nil})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	assert.Nil(t, p.speculativeCache.Get(req.RequestID))
}

// P/D prefill profile: index.Add called twice (primary + prefill), and the
// cache entry tracks both pod identifiers.
func TestPreRequest_PrefillProfile_SeedsBoth(t *testing.T) {
	ctx := utils.NewTestContext(t)

	var calls []addCall
	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			calls = append(calls, addCall{keys: keys, entries: entries})
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	blockKeys := []kvblock.BlockHash{0xCC}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-pd"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: [][]kvblock.BlockHash{blockKeys}})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":                   {TargetEndpoints: []scheduling.Endpoint{testEndpoints[0]}},
			experimentalPrefillProfile: {TargetEndpoints: []scheduling.Endpoint{testEndpoints[1]}},
		},
	}
	_ = p.PreRequest(ctx, req, result)

	require.Len(t, calls, 2)
	assert.Equal(t, "10.0.0.1:8080", calls[0].entries[0].PodIdentifier)
	assert.Equal(t, "10.0.0.2:8080", calls[1].entries[0].PodIdentifier)

	cached := p.speculativeCache.Get(req.RequestID)
	require.NotNil(t, cached)
	require.Len(t, cached.Value().podEntries, 2)
	assert.Equal(t, "10.0.0.1:8080", cached.Value().podEntries[0].PodIdentifier)
	assert.Equal(t, "10.0.0.2:8080", cached.Value().podEntries[1].PodIdentifier)
}

// speculativeEnabled=false: early return — no index writes, no cache entry,
// and PluginState is left untouched.
func TestPreRequest_SpeculativeDisabled_NoOp(t *testing.T) {
	ctx := utils.NewTestContext(t)

	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, _ []kvblock.BlockHash, _ []kvblock.PodEntry) error {
			t.Fatalf("index.Add must not be called when speculative indexing is disabled")
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, false, idx)

	req := &scheduling.InferenceRequest{RequestID: "req-pre-off"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey,
		&blockKeysState{perPromptKeys: [][]kvblock.BlockHash{{0xDD}}})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	assert.Nil(t, p.speculativeCache.Get(req.RequestID))
}

// The predicted token count comes from the chosen endpoint's unweighted cached
// block count, not the tier-weighted match score, and is reported with
// speculative indexing off alongside the prompt tokens it is measured against.
func TestPreRequest_RecordsPrediction(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-predicted-records"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	// Weighted score 2.5 against 4 cached blocks: the token count must follow
	// the cached count.
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 8, testBlockSize).
		WithCachedBlockCount(4))

	// A non-default profile name proves the lookup follows PrimaryProfileName.
	beforePredicted := sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, promptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, tokenizedRequest("req-predicted", 8*testBlockSize),
		primaryOnly("decode", endpoint))

	assert.Equal(t, beforePredicted+float64(4*testBlockSize), sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(8*testBlockSize), sharedPrefixHistogram(t, promptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// A disaggregated request's cached-token count comes back from the prefiller,
// so the prediction reads the prefill endpoint's match info and is labelled
// with the prefill role.
func TestPreRequest_PD_RecordsPrefillPrediction(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-predicted-pd"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoints := freshEndpoints()
	decode, prefill := endpoints[0], endpoints[1]
	decode.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(8, 8, testBlockSize).WithCachedBlockCount(8))
	prefill.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(3, 8, testBlockSize).WithCachedBlockCount(3))

	beforePrefill := sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RolePrefill).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, promptTokensMetric, name, prefixmetrics.RolePrefill).GetSampleSum()
	beforeDecode := sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount()
	_ = p.PreRequest(ctx, tokenizedRequest("req-pd", 8*testBlockSize), &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":                   {TargetEndpoints: []scheduling.Endpoint{decode}},
			experimentalPrefillProfile: {TargetEndpoints: []scheduling.Endpoint{prefill}},
		},
	})

	assert.Equal(t, beforePrefill+float64(3*testBlockSize),
		sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RolePrefill).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(8*testBlockSize),
		sharedPrefixHistogram(t, promptTokensMetric, name, prefixmetrics.RolePrefill).GetSampleSum())
	assert.Equal(t, beforeDecode,
		sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount())
}

// A request carrying multimodal content records its prediction once, under
// the comma-joined sorted list of the modalities it carries, leaving the
// text-only series untouched.
func TestPreRequest_RecordsPredictionModality(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-predicted-modality"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 8, testBlockSize).
		WithCachedBlockCount(4))

	req := tokenizedRequest("req-predicted-mm", 8*testBlockSize)
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrh.MultiModalFeature{
		{Modality: fwkrh.ModalityImage, Hash: "img"},
		{Modality: "audio", Hash: "aud"},
	}

	const modality = "audio,image"
	// histogram reads the plugin's decode series for one modality label value.
	histogram := func(metricName, modality string) *dto.Histogram {
		t.Helper()
		families, err := ctrlmetrics.Registry.Gather()
		require.NoError(t, err)
		for _, family := range families {
			if family.GetName() != metricName {
				continue
			}
			for _, metric := range family.GetMetric() {
				labels := map[string]string{}
				for _, label := range metric.GetLabel() {
					labels[label.GetName()] = label.GetValue()
				}
				if labels["plugin_name"] == name && labels["endpoint_role"] == prefixmetrics.RoleDecode && labels["modality"] == modality {
					return metric.GetHistogram()
				}
			}
		}
		return nil
	}
	beforePredicted := histogram(predictedCachedTokensMetric, modality).GetSampleSum()
	beforePrompt := histogram(promptTokensMetric, modality).GetSampleSum()
	beforeNone := histogram(predictedCachedTokensMetric, mmobs.ModalityNone).GetSampleCount()
	_ = p.PreRequest(ctx, req, primaryOnly("default", endpoint))

	assert.Equal(t, beforePredicted+float64(4*testBlockSize), histogram(predictedCachedTokensMetric, modality).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(8*testBlockSize), histogram(promptTokensMetric, modality).GetSampleSum())
	assert.Equal(t, beforeNone, histogram(predictedCachedTokensMetric, mmobs.ModalityNone).GetSampleCount(),
		"a multimodal request must not record in the text-only series")
}

// Under P/D the best the picker could have chosen comes from the prefill
// profile's candidates, the same profile the selected prediction follows, so a
// warmer decode endpoint cannot push it above what prefill routing could reach.
func TestPreRequest_PD_BestPredictedFollowsPrefillProfile(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-best-pd"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoints := freshEndpoints()
	decode, prefill := endpoints[0], endpoints[1]
	decode.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(8, 8, testBlockSize).WithCachedBlockCount(8))
	prefill.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(3, 8, testBlockSize).WithCachedBlockCount(3))

	before := sharedPrefixHistogram(t, bestPredictedMetric, name, prefixmetrics.RolePrefill).GetSampleSum()
	_ = p.PreRequest(ctx, tokenizedRequest("req-best-pd", 8*testBlockSize), &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode": {
				TargetEndpoints:  []scheduling.Endpoint{decode},
				ScoredCandidates: []scheduling.ScoredEndpoint{{Endpoint: decode}},
			},
			experimentalPrefillProfile: {
				TargetEndpoints:  []scheduling.Endpoint{prefill},
				ScoredCandidates: []scheduling.ScoredEndpoint{{Endpoint: prefill}},
			},
		},
	})

	assert.Equal(t, before+float64(3*testBlockSize),
		sharedPrefixHistogram(t, bestPredictedMetric, name, prefixmetrics.RolePrefill).GetSampleSum())
}

// The two maxima carry the modalities the request holds, so the reuse routing
// and filtering left behind can be split between multimodal and text-only
// traffic.
func TestPreRequest_MaximaCarryModality(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-best-modality"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 8, testBlockSize).WithCachedBlockCount(2))
	req := tokenizedRequest("req-best-modality", 8*testBlockSize)
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrh.MultiModalFeature{
		{Modality: fwkrh.ModalityImage, Hash: "img"},
		{Modality: "audio", Hash: "aud"},
	}
	_ = p.PreRequest(ctx, req, primaryOnly("default", endpoint))

	assert.Equal(t, "audio,image", sharedPrefixModality(t, bestPredictedMetric, name))
	assert.Equal(t, "audio,image", sharedPrefixModality(t, bestAvailableMetric, name))
}

// An endpoint the producer never published match info for is not observed:
// a zero would be indistinguishable from a real zero-hit prediction.
func TestPreRequest_NoMatchInfo_RecordsNothing(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-predicted-absent"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	before := sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount()
	_ = p.PreRequest(ctx, tokenizedRequest("req-no-info", testBlockSize),
		primaryOnly("default", freshEndpoints()[0]))
	assert.Equal(t, before, sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount())

	// No endpoint at all is equally a no-op.
	_ = p.PreRequest(ctx, tokenizedRequest("req-no-endpoint", testBlockSize),
		&scheduling.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults:     map[string]*scheduling.ProfileRunResult{"default": {}},
		})
	assert.Equal(t, before, sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount())
}

// The best the picker could have chosen spans every scored candidate, so a
// request routed away from the warmer endpoint reports the hit it passed up.
func TestPreRequest_BestPredictedSpansScoredCandidates(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-best-scored"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoints := freshEndpoints()
	chosen, warmer := endpoints[0], endpoints[1]
	chosen.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(1, 8, testBlockSize).WithCachedBlockCount(1))
	warmer.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(6, 8, testBlockSize).WithCachedBlockCount(6))

	beforeSelected := sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	beforeBest := sharedPrefixHistogram(t, bestPredictedMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, tokenizedRequest("req-best-scored", 8*testBlockSize),
		primaryWithScored("default", chosen, chosen, warmer))

	assert.Equal(t, beforeSelected+float64(1*testBlockSize),
		sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, beforeBest+float64(6*testBlockSize),
		sharedPrefixHistogram(t, bestPredictedMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// A candidate dropped by a filter never reaches the picker. Produce records the
// reuse it held, so it counts as available without counting as a hit the picker
// could have taken.
func TestPreRequest_BestAvailableComesFromProduce(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-best-available"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(1, 8, testBlockSize).WithCachedBlockCount(1))
	req := tokenizedRequest("req-best-available", 8*testBlockSize)
	p.pluginState.Write(req.RequestID, bestAvailableStateKey, &bestAvailableState{cachedTokens: 7 * testBlockSize})

	beforeBest := sharedPrefixHistogram(t, bestPredictedMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	beforeAvailable := sharedPrefixHistogram(t, bestAvailableMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, req, primaryWithScored("default", endpoint, endpoint))

	assert.Equal(t, beforeBest+float64(1*testBlockSize),
		sharedPrefixHistogram(t, bestPredictedMetric, name, prefixmetrics.RoleDecode).GetSampleSum(),
		"the picker only scored the endpoint holding one block")
	assert.Equal(t, beforeAvailable+float64(7*testBlockSize),
		sharedPrefixHistogram(t, bestAvailableMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// Produce computes the pre-filter maximum by iterating candidates and
// converting blocks to tokens, and PreRequest reads it back out of plugin
// state. Running both extension points keeps a regression in that iteration or
// conversion from passing while the metric is stubbed into state.
func TestProduceThenPreRequest_RecordsBothMaxima(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-produce-to-prerequest"
	const chosenBlocks, warmerBlocks, promptBlocks = 2, 5, 8

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			keys := make([]kvblock.BlockHash, promptBlocks)
			for i := range keys {
				keys[i] = kvblock.BlockHash(i + 1)
			}
			return keys, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			return map[string]kvcache.PodMatch{
				"10.0.0.1:8080": {
					WeightedScore: chosenBlocks, MatchedBlocks: chosenBlocks,
					BlocksByTier: map[string]int{"gpu": chosenBlocks},
				},
				"10.0.0.2:8080": {
					WeightedScore: warmerBlocks, MatchedBlocks: warmerBlocks,
					BlocksByTier: map[string]int{"gpu": warmerBlocks},
				},
			}, nil
		},
	}
	p := newProducerForProduceAndPreRequest(ctx, name, idx)

	endpoints := freshEndpoints()
	chosen := endpoints[0]
	req := tokenizedRequest("req-produce-prerequest", promptBlocks*testBlockSize)
	req.TargetModel = "test-model"
	require.NoError(t, p.Produce(ctx, req, endpoints))

	// endpoints[1] holds the longer prefix, and no filter let it reach the picker.
	_ = p.PreRequest(ctx, req, primaryWithScored("default", chosen, chosen))

	assert.Equal(t, float64(chosenBlocks*testBlockSize),
		sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, float64(chosenBlocks*testBlockSize),
		sharedPrefixHistogram(t, bestPredictedMetric, name, prefixmetrics.RoleDecode).GetSampleSum(),
		"only the chosen endpoint reached the picker")
	assert.Equal(t, float64(warmerBlocks*testBlockSize),
		sharedPrefixHistogram(t, bestAvailableMetric, name, prefixmetrics.RoleDecode).GetSampleSum(),
		"Produce saw the warmer candidate before filtering")
	assert.Equal(t, float64(promptBlocks*testBlockSize),
		sharedPrefixHistogram(t, promptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

type countingGetEndpoint struct {
	scheduling.Endpoint
	gets int
}

func (e *countingGetEndpoint) Get(key plugin.DataKey) (datalayer.Cloneable, bool) {
	e.gets++
	return e.Endpoint.Get(key)
}

func TestBestAvailableState_Clone(t *testing.T) {
	id := freshEndpoints()[0].GetMetadata().ID
	state := &bestAvailableState{
		cachedTokens:          7 * testBlockSize,
		predictedCachedTokens: map[datalayer.ID]int{id: testBlockSize},
	}
	cloned, ok := state.Clone().(*bestAvailableState)
	require.True(t, ok)
	require.Equal(t, state, cloned)

	cloned.predictedCachedTokens[id] = 0
	assert.Equal(t, testBlockSize, state.predictedCachedTokens[id])
}

func TestProduceThenPreRequest_ReadsOnlySelectedMatchInfo(t *testing.T) {
	for _, scenario := range []string{"decode", "prefill", "no-scored-candidates", "missing-selected-match"} {
		t.Run(scenario, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			prefixmetrics.Register()

			name := "precise-prediction-reads-" + scenario
			const chosenBlocks, scoredBlocks, filteredBlocks, promptBlocks = 1, 5, 7, 8
			idx := &fakeKVCacheIndexer{
				computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
					return make([]kvblock.BlockHash, promptBlocks), nil
				},
				matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
					return map[string]kvcache.PodMatch{
						"10.0.0.1:8080": {MatchedBlocks: chosenBlocks, BlocksByTier: map[string]int{"gpu": chosenBlocks}},
						"10.0.0.2:8080": {MatchedBlocks: scoredBlocks, BlocksByTier: map[string]int{"gpu": scoredBlocks}},
						"10.0.0.3:8080": {MatchedBlocks: filteredBlocks, BlocksByTier: map[string]int{"gpu": filteredBlocks}},
					}, nil
				},
			}
			p := newProducerForProduceAndPreRequest(ctx, name, idx)
			endpoints := freshEndpoints()
			filteredMeta := *endpoints[1].GetMetadata()
			filteredMeta.ID.Namespace = "other"
			filteredMeta.Address = "10.0.0.3"
			endpoints = append(endpoints, scheduling.NewEndpoint(&filteredMeta, nil, nil))
			req := tokenizedRequest("req-prediction-reads", promptBlocks*testBlockSize)
			req.TargetModel = "test-model"
			require.NoError(t, p.Produce(ctx, req, endpoints))

			chosen := &countingGetEndpoint{Endpoint: endpoints[0]}
			// The scheduler can copy endpoints while preserving their identities.
			scored := &countingGetEndpoint{Endpoint: scheduling.NewEndpoint(
				endpoints[1].GetMetadata(), nil, endpoints[1].Clone())}
			result := primaryWithScored("decode", chosen, chosen, scored, scheduling.NewEndpoint(nil, nil, nil))
			role := prefixmetrics.RoleDecode
			wantBest, wantAvailable := scoredBlocks, filteredBlocks
			switch scenario {
			case "prefill":
				result.ProfileResults[experimentalPrefillProfile] = result.ProfileResults["decode"]
				result.ProfileResults["decode"] = &scheduling.ProfileRunResult{
					TargetEndpoints: []scheduling.Endpoint{endpoints[2]},
				}
				role = prefixmetrics.RolePrefill
			case "no-scored-candidates":
				result = primaryOnly("decode", chosen)
				wantBest, wantAvailable = chosenBlocks, chosenBlocks
			case "missing-selected-match":
				chosen.Endpoint = scheduling.NewEndpoint(endpoints[0].GetMetadata(), nil, nil)
			}

			before := sharedPrefixHistogram(t, predictedCachedTokensMetric, name, role).GetSampleCount()
			require.NoError(t, p.PreRequest(ctx, req, result))
			assert.Equal(t, 1, chosen.gets)
			assert.Zero(t, scored.gets, "candidate predictions must come from Produce state")
			_, err := p.pluginState.Read(req.RequestID, bestAvailableStateKey)
			require.ErrorIs(t, err, plugin.ErrNotFound)

			if scenario == "missing-selected-match" {
				for _, metric := range []string{predictedCachedTokensMetric, bestPredictedMetric, bestAvailableMetric} {
					assert.Equal(t, before, sharedPrefixHistogram(t, metric, name, role).GetSampleCount())
				}
				return
			}
			assert.Equal(t, float64(chosenBlocks*testBlockSize),
				sharedPrefixHistogram(t, predictedCachedTokensMetric, name, role).GetSampleSum())
			assert.Equal(t, float64(wantBest*testBlockSize),
				sharedPrefixHistogram(t, bestPredictedMetric, name, role).GetSampleSum())
			assert.Equal(t, float64(wantAvailable*testBlockSize),
				sharedPrefixHistogram(t, bestAvailableMetric, name, role).GetSampleSum())
		})
	}
}

// A profile handler that rebuilds the result from its targets alone, such as
// the data-parallel one, leaves no scored candidates. Both maxima then fall
// back to the chosen endpoint, so a routing miss toward a warmer candidate is
// not reported as reuse lost to filtering.
func TestProduceThenPreRequest_NoScoredCandidates_BothMaximaFallBack(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-produce-no-scored"
	const chosenBlocks, warmerBlocks, promptBlocks = 1, 6, 8

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			keys := make([]kvblock.BlockHash, promptBlocks)
			for i := range keys {
				keys[i] = kvblock.BlockHash(i + 1)
			}
			return keys, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			return map[string]kvcache.PodMatch{
				"10.0.0.1:8080": {
					WeightedScore: chosenBlocks, MatchedBlocks: chosenBlocks,
					BlocksByTier: map[string]int{"gpu": chosenBlocks},
				},
				"10.0.0.2:8080": {
					WeightedScore: warmerBlocks, MatchedBlocks: warmerBlocks,
					BlocksByTier: map[string]int{"gpu": warmerBlocks},
				},
			}, nil
		},
	}
	p := newProducerForProduceAndPreRequest(ctx, name, idx)

	endpoints := freshEndpoints()
	req := tokenizedRequest("req-produce-no-scored", promptBlocks*testBlockSize)
	req.TargetModel = "test-model"
	require.NoError(t, p.Produce(ctx, req, endpoints))
	_ = p.PreRequest(ctx, req, primaryOnly("default", endpoints[0]))

	selected := float64(chosenBlocks * testBlockSize)
	assert.Equal(t, selected,
		sharedPrefixHistogram(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, selected,
		sharedPrefixHistogram(t, bestPredictedMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, selected,
		sharedPrefixHistogram(t, bestAvailableMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// newProducerForProduceAndPreRequest builds a producer that can run both
// extension points, so a prediction can be followed from the candidate match
// through to the recorded metric.
func newProducerForProduceAndPreRequest(ctx context.Context, name string, idx kvCacheIndexer) *Producer {
	return &Producer{
		typedName:       plugin.TypedName{Type: PluginType, Name: name},
		kvCacheIndexer:  idx,
		dk:              attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		pluginState:     plugin.NewPluginState(ctx),
		blockSizeTokens: testBlockSize,
	}
}

// primaryWithScored selects target and reports scored as the candidates that
// reached the picker. A candidate the scheduler filtered out is left out.
func primaryWithScored(name string, target scheduling.Endpoint, scored ...scheduling.Endpoint) *scheduling.SchedulingResult {
	candidates := make([]scheduling.ScoredEndpoint, 0, len(scored))
	for _, endpoint := range scored {
		candidates = append(candidates, scheduling.ScoredEndpoint{Endpoint: endpoint})
	}
	return &scheduling.SchedulingResult{
		PrimaryProfileName: name,
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			name: {TargetEndpoints: []scheduling.Endpoint{target}, ScoredCandidates: candidates},
		},
	}
}

func tokenizedRequest(id string, tokenCount int) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID: id,
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, tokenCount)}},
			},
		},
	}
}

// mmRequest wraps tokenizedRequest with one image feature spanning the
// prompt's placeholder tokens.
func mmRequest(id string, featureTokens int) *scheduling.InferenceRequest {
	req := tokenizedRequest(id, featureTokens)
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrh.MultiModalFeature{
		{Modality: fwkrh.ModalityImage, Hash: "img", Offset: 0, Length: featureTokens},
	}
	return req
}

// A request whose match info carries multimodal attribution records its
// multimodal prediction alongside the prompt-level pair: the matched
// multimodal blocks in tokens, measured against the request's multimodal
// token total.
func TestPreRequest_RecordsMMPrediction(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-mm-predicted-records"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(5, 8, testBlockSize).
		WithCachedBlockCount(5).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 5, MatchTokens: 5 * testBlockSize}))

	beforePredicted := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, mmRequest("req-mm", 8*testBlockSize), primaryOnly("decode", endpoint))

	assert.Equal(t, beforePredicted+float64(5*testBlockSize),
		sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(8*testBlockSize),
		sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// A zero multimodal match is a real observation: the endpoint held none of the
// multimodal blocks, and the request still contributes its multimodal tokens
// to the denominator.
func TestPreRequest_RecordsMMPrediction_ZeroMatch(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-mm-predicted-zero"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(0, 4, testBlockSize).
		WithCachedBlockCount(0).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 0, MatchTokens: 0}))

	beforePredicted := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount()
	beforePrompt := sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, mmRequest("req-mm-zero", 4*testBlockSize), primaryOnly("decode", endpoint))

	assert.Equal(t, beforePredicted+1,
		sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount())
	assert.Equal(t, beforePrompt+float64(4*testBlockSize),
		sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// Match info without multimodal attribution covers a text-only request, and
// attribution on a request without multimodal tokens has nothing to measure
// against. Neither records in the mm pair, so a zero observation keeps
// meaning a multimodal request matched no blocks.
func TestPreRequest_NoMMAttribution_RecordsNothing(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-mm-predicted-absent"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 8, testBlockSize).
		WithCachedBlockCount(4))

	before := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount()
	_ = p.PreRequest(ctx, mmRequest("req-mm-no-attribution", 8*testBlockSize), primaryOnly("decode", endpoint))
	assert.Equal(t, before, sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount())

	// Attribution present but the request carries no multimodal tokens: the
	// pair would hold a zero-over-zero observation.
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 8, testBlockSize).
		WithCachedBlockCount(4).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 2, MatchTokens: 2 * testBlockSize}))
	_ = p.PreRequest(ctx, tokenizedRequest("req-text-only", 8*testBlockSize), primaryOnly("decode", endpoint))
	assert.Equal(t, before, sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount())
}

// A feature that starts or ends mid-block contributes only the tokens it
// holds: one matched block covering a 5-token feature records the feature's 5
// tokens, not the matched block's token multiple.
func TestPreRequest_RecordsMMPrediction_MidBlockFeature(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-mm-predicted-mid-block"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(1, 1, testBlockSize).
		WithCachedBlockCount(1).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 1, MatchTokens: 5}))

	beforePredicted := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, mmRequest("req-mm-mid-block", 5), primaryOnly("decode", endpoint))

	assert.Equal(t, beforePredicted+float64(5),
		sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum(),
		"a matched block of %d tokens must record the feature's 5 tokens", testBlockSize)
}

// Two images with only the first matched: the pair records the first image's
// tokens. The matched MM blocks hold text at the image's edges, and a
// block-granular count would pick that text up (32 of a 40-token MM total).
func TestPreRequest_RecordsMMPrediction_TwoImagesFirstMatched(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-mm-predicted-two-images"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 4, testBlockSize).
		WithCachedBlockCount(2).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 2, MatchTokens: 20}))

	req := tokenizedRequest("req-mm-two-images", 4*testBlockSize)
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrh.MultiModalFeature{
		{Modality: fwkrh.ModalityImage, Hash: "img-a", Offset: 2, Length: 20},
		{Modality: fwkrh.ModalityImage, Hash: "img-b", Offset: 32, Length: 20},
	}

	beforePredicted := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, req, primaryOnly("decode", endpoint))

	assert.Equal(t, beforePredicted+float64(20),
		sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(40),
		sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// The denominator sums multimodal feature tokens across prompts, and the
// numerator reads the matched-block total the producer attached across
// prompts.
func TestPreRequest_RecordsMMPrediction_MultiPrompt(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-mm-predicted-multi"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(7, 11, testBlockSize).
		WithCachedBlockCount(7).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 7, MatchTokens: 7 * testBlockSize}))

	req := &scheduling.InferenceRequest{
		RequestID: "req-mm-multi",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{
					{
						TokenIDs: make([]uint32, 4*testBlockSize),
						MultiModalFeatures: []fwkrh.MultiModalFeature{
							{Modality: fwkrh.ModalityImage, Hash: "img-a", Offset: 0, Length: 4 * testBlockSize},
						},
					},
					{
						TokenIDs: make([]uint32, 3*testBlockSize),
						MultiModalFeatures: []fwkrh.MultiModalFeature{
							{Modality: fwkrh.ModalityImage, Hash: "img-b", Offset: 0, Length: 3 * testBlockSize},
						},
					},
				},
			},
		},
	}

	beforePredicted := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, req, primaryOnly("decode", endpoint))

	assert.Equal(t, beforePredicted+float64(7*testBlockSize),
		sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(7*testBlockSize),
		sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// A prompt carrying several multimodal items contributes every feature's
// tokens to the denominator, not just the first item's.
func TestPreRequest_RecordsMMPrediction_MultipleFeaturesInPrompt(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-mm-predicted-features"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(7, 7, testBlockSize).
		WithCachedBlockCount(7).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 7, MatchTokens: 7 * testBlockSize}))

	req := tokenizedRequest("req-mm-features", 7*testBlockSize)
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrh.MultiModalFeature{
		{Modality: fwkrh.ModalityImage, Hash: "img-a", Offset: 0, Length: 4 * testBlockSize},
		{Modality: fwkrh.ModalityImage, Hash: "img-b", Offset: 4 * testBlockSize, Length: 3 * testBlockSize},
	}

	beforePredicted := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum()
	_ = p.PreRequest(ctx, req, primaryOnly("decode", endpoint))

	assert.Equal(t, beforePredicted+float64(7*testBlockSize),
		sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(7*testBlockSize),
		sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RoleDecode).GetSampleSum())
}

// A disaggregated request's multimodal prediction follows the prefill
// endpoint's match info and the prefill role, as the prompt-level pair does.
func TestPreRequest_PD_RecordsMMPrediction(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-mm-predicted-pd"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoints := freshEndpoints()
	decode, prefill := endpoints[0], endpoints[1]
	decode.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(8, 8, testBlockSize).
		WithCachedBlockCount(8).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 8, MatchTokens: 8 * testBlockSize}))
	prefill.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(3, 8, testBlockSize).
		WithCachedBlockCount(3).
		WithMM(attrprefix.MMMatchInfo{MatchBlocks: 3, MatchTokens: 3 * testBlockSize}))

	beforePrefill := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RolePrefill).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RolePrefill).GetSampleSum()
	beforeDecode := sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount()
	_ = p.PreRequest(ctx, mmRequest("req-mm-pd", 8*testBlockSize), &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":                   {TargetEndpoints: []scheduling.Endpoint{decode}},
			experimentalPrefillProfile: {TargetEndpoints: []scheduling.Endpoint{prefill}},
		},
	})

	assert.Equal(t, beforePrefill+float64(3*testBlockSize),
		sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RolePrefill).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(8*testBlockSize),
		sharedPrefixHistogram(t, mmPromptTokensMetric, name, prefixmetrics.RolePrefill).GetSampleSum())
	assert.Equal(t, beforeDecode,
		sharedPrefixHistogram(t, mmPredictedCachedTokensMetric, name, prefixmetrics.RoleDecode).GetSampleCount(),
		"the multimodal prediction must follow the prefill endpoint, not the decode one")
}

const (
	predictedCachedTokensMetric   = "llm_d_epp_prefix_predicted_cached_tokens"      //nolint:gosec // G101: metric name, not a credential
	bestPredictedMetric           = "llm_d_epp_prefix_best_predicted_cached_tokens" //nolint:gosec // G101: metric name, not a credential
	bestAvailableMetric           = "llm_d_epp_prefix_best_available_cached_tokens" //nolint:gosec // G101: metric name, not a credential
	promptTokensMetric            = "llm_d_epp_prefix_prompt_tokens"                //nolint:gosec // G101: metric name, not a credential
	mmPredictedCachedTokensMetric = "llm_d_epp_prefix_mm_predicted_cached_tokens"   //nolint:gosec // G101: metric name, not a credential
	mmPromptTokensMetric          = "llm_d_epp_prefix_mm_prompt_tokens"             //nolint:gosec // G101: metric name, not a credential
)

// sharedPrefixHistogram reads a shared prefix metric out of the registry it is
// registered against, since those metrics live in another package. A metric
// that has not been observed yet reads as nil, whose accessors return zero.
func sharedPrefixHistogram(t *testing.T, metricName, pluginName, role string) *dto.Histogram {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != metricName {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["plugin_name"] == pluginName && labels["endpoint_role"] == role {
				return metric.GetHistogram()
			}
		}
	}
	return nil
}

// sharedPrefixModality returns the modality label of the plugin's only series
// of a shared prefix metric.
func sharedPrefixModality(t *testing.T, metricName, pluginName string) string {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	var modalities []string
	for _, family := range families {
		if family.GetName() != metricName {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["plugin_name"] == pluginName {
				modalities = append(modalities, labels["modality"])
			}
		}
	}
	require.Len(t, modalities, 1)
	return modalities[0]
}

// TTL expiry of a speculative entry evicts every key of every prompt in the
// request, with RequestKey semantics and the seeded pod entries.
func TestSpeculativeTTLExpiry_EvictsAllPromptKeys(t *testing.T) {
	ctx, cancel := context.WithCancel(utils.NewTestContext(t))
	t.Cleanup(cancel)

	var mu sync.Mutex
	var evictions []evictCall
	idx := &fakeKVBlockIndex{
		evictFn: func(_ context.Context, keyType kvblock.KeyType, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			mu.Lock()
			defer mu.Unlock()
			evictions = append(evictions, evictCall{keyType: keyType, keys: keys, entries: entries})
			return nil
		},
	}

	cache, ttl, err := buildSpeculativeCache(ctx, PluginConfig{
		SpeculativeIndexing: true,
		SpeculativeTTL:      "20ms",
	}, idx)
	require.NoError(t, err)
	require.Equal(t, 20*time.Millisecond, ttl)

	perPrompt := [][]kvblock.BlockHash{{0xA1, 0xA2}, {0xB1}}
	pods := []kvblock.PodEntry{{PodIdentifier: "10.0.0.1:8080", Speculative: true}}
	cache.Set("req-ttl", &speculativeEntries{perPromptKeys: perPrompt, podEntries: pods}, ttl)

	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(evictions) == 1
	}, 2*time.Second, time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, evictions, 1)
	assert.Equal(t, []kvblock.BlockHash{0xA1, 0xA2, 0xB1}, evictions[0].keys)
	assert.Equal(t, kvblock.RequestKey, evictions[0].keyType)
	assert.Equal(t, pods, evictions[0].entries)
}
