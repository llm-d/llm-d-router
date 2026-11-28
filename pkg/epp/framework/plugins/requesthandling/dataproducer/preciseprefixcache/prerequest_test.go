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
	"bytes"
	"context"
	"fmt"
	"testing"
	"testing/synctest"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/common/observability/multimodal"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixmetrics"
	"github.com/llm-d/llm-d-router/test/utils"
)

func newProducerForPreRequest(ctx context.Context, speculativeEnabled bool, idx *fakeKVBlockIndex) *Producer {
	return newNamedProducerForPreRequest(ctx, "test", speculativeEnabled, idx)
}

func newNamedProducerForPreRequest(ctx context.Context, name string, speculativeEnabled bool, idx *fakeKVBlockIndex) *Producer {
	p := &Producer{
		typedName:      plugin.TypedName{Type: PluginType, Name: name},
		kvCacheIndexer: &fakeKVCacheIndexer{index: idx},
		dk:             attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		pluginState:    plugin.NewPluginState(ctx),
	}
	if speculativeEnabled {
		p.speculative = kvcache.NewSpeculativeIndex(time.Minute)
	}
	return p
}

// writeRejectingIndex fails the test on any write to the KV-block index.
func writeRejectingIndex(t *testing.T) *fakeKVBlockIndex {
	t.Helper()
	return &fakeKVBlockIndex{
		addFn: func(context.Context, []kvblock.BlockHash, []kvblock.BlockHash, []kvblock.PodEntry) error {
			t.Error("unexpected Add to the KV-block index")
			return nil
		},
		evictFn: func(context.Context, kvblock.KeyType, []kvblock.BlockHash, []kvblock.PodEntry) error {
			t.Error("unexpected Evict from the KV-block index")
			return nil
		},
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

// newSpeculativeTTLProducer returns a Producer whose hints live for ttl and are
// reclaimed in the background, as in New.
func newSpeculativeTTLProducer(ctx context.Context, ttl time.Duration) *Producer {
	p := newNamedProducerForPreRequest(ctx, "speculative-ttl", false, &fakeKVBlockIndex{})
	p.speculative = kvcache.NewSpeculativeIndex(ttl)
	go p.speculative.Run(ctx)
	return p
}

// seedSpeculative runs PreRequest for request id with keys routed to endpoint.
// It reports failures with assert, so it is safe to call from any goroutine.
func seedSpeculative(ctx context.Context, t *testing.T, p *Producer, id string,
	keys []kvblock.BlockHash, endpoint scheduling.Endpoint,
) {
	t.Helper()
	p.pluginState.Write(id, blockKeysStateKey, &blockKeysState{perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{
		{BlockSize: testBlockSize}: {keys},
	}})
	assert.NoError(t, p.PreRequest(ctx, &scheduling.InferenceRequest{RequestID: id},
		primaryOnly("default", endpoint)))
}

// hintedKeys returns how many leading keys are hinted for endpoint.
func hintedKeys(p *Producer, keys []kvblock.BlockHash, endpoint scheduling.Endpoint) int {
	return p.speculative.Match(keys, nil)[endpointPodID(endpoint)]
}

// endpointPodID returns the speculative pod identifier PreRequest uses for endpoint.
func endpointPodID(endpoint scheduling.Endpoint) string {
	md := endpoint.GetMetadata()
	return md.Address + ":" + md.Port
}

// With speculative indexing on, PreRequest hints the request's block keys for
// the primary endpoint and writes nothing to the KV-block index.
func TestPreRequest_SeedsSpeculativeForPrimary(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newProducerForPreRequest(ctx, true, writeRejectingIndex(t))

	blockKeys := []kvblock.BlockHash{0xAA, 0xBB}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-1"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
		perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{{BlockSize: testBlockSize}: {blockKeys}},
	})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	assert.Equal(t, map[string]int{"10.0.0.1:8080": len(blockKeys)}, p.speculative.Match(blockKeys, nil))
}

// Empty block keys leave no hint.
func TestPreRequest_EmptyBlockKeys_NoHint(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newProducerForPreRequest(ctx, true, &fakeKVBlockIndex{})

	req := &scheduling.InferenceRequest{RequestID: "req-pre-empty"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	assert.Zero(t, p.speculative.Len())
}

// seedDecodeWhenDisaggregated hints both the primary and the prefill endpoint.
func TestPreRequest_PrefillProfile_SeedDecode_SeedsBoth(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newProducerForPreRequest(ctx, true, &fakeKVBlockIndex{})
	p.seedDecode = true

	blockKeys := []kvblock.BlockHash{0xCC}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-pd"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
		perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{{BlockSize: testBlockSize}: {blockKeys}},
	})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		PrefillProfileName: "prefill",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{testEndpoints[0]}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{testEndpoints[1]}},
		},
	}
	_ = p.PreRequest(ctx, req, result)

	assert.Equal(t, map[string]int{"10.0.0.1:8080": 1, "10.0.0.2:8080": 1}, p.speculative.Match(blockKeys, nil))
}

// By default a disaggregated request seeds the prefill endpoint only.
func TestPreRequest_PrefillProfile_SeedsPrefillOnly(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newProducerForPreRequest(ctx, true, &fakeKVBlockIndex{})

	blockKeys := []kvblock.BlockHash{0xCC}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-pd-prefill-only"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
		perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{{BlockSize: testBlockSize}: {blockKeys}},
	})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{testEndpoints[0]}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{testEndpoints[1]}},
		},
		PrefillProfileName: "prefill",
	}
	_ = p.PreRequest(ctx, req, result)

	assert.Equal(t, map[string]int{"10.0.0.2:8080": 1}, p.speculative.Match(blockKeys, nil))

	// A decode-only request still seeds its decode endpoint.
	decodeKeys := []kvblock.BlockHash{0xDD}
	decodeOnly := &scheduling.InferenceRequest{RequestID: "req-pre-decode-only"}
	p.pluginState.Write(decodeOnly.RequestID, blockKeysStateKey, &blockKeysState{
		perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{{BlockSize: testBlockSize}: {decodeKeys}},
	})
	_ = p.PreRequest(ctx, decodeOnly, primaryOnly("decode", testEndpoints[0]))
	assert.Equal(t, map[string]int{"10.0.0.1:8080": 1}, p.speculative.Match(decodeKeys, nil))
}

// With speculative indexing off, PreRequest records no hint.
func TestPreRequest_SpeculativeDisabled_NoOp(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newProducerForPreRequest(ctx, false, &fakeKVBlockIndex{})

	req := &scheduling.InferenceRequest{RequestID: "req-pre-off"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey,
		&blockKeysState{perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{
			{BlockSize: testBlockSize}: {{0xDD}},
		}})

	require.NoError(t, p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0])))
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
		PrefillProfileName: "prefill",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{decode}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{prefill}},
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
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrequest.MultiModalFeature{
		{Modality: reqcommon.ModalityImage, Hash: "img"},
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
		PrefillProfileName: "prefill",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode": {
				TargetEndpoints:  []scheduling.Endpoint{decode},
				ScoredCandidates: []scheduling.ScoredEndpoint{{Endpoint: decode}},
			},
			"prefill": {
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
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrequest.MultiModalFeature{
		{Modality: reqcommon.ModalityImage, Hash: "img"},
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
		gpuCachedTokens:       map[datalayer.ID]int{id: testBlockSize},
	}
	cloned, ok := state.Clone().(*bestAvailableState)
	require.True(t, ok)
	require.Equal(t, state, cloned)

	cloned.predictedCachedTokens[id] = 0
	cloned.gpuCachedTokens[id] = 0
	assert.Equal(t, testBlockSize, state.predictedCachedTokens[id])
	assert.Equal(t, testBlockSize, state.gpuCachedTokens[id])
}

// Produce captures each candidate's GPU-tier and confirmed chains before the
// filters run, and PreRequest splits the chosen endpoint's GPU miss across
// them. The chosen endpoint holds 2 blocks in GPU memory and 6 in any tier, a
// scored peer holds 3 in GPU memory, a filtered peer holds 4, and another
// filtered peer holds 7 in its CPU tier only.
func TestProduceThenPreRequest_AttributesGPUMiss(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-gpu-miss"
	const promptBlocks = 8

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			keys := make([]kvblock.BlockHash, promptBlocks)
			for i := range keys {
				keys[i] = kvblock.BlockHash(i + 1)
			}
			return keys, nil
		},
		matchBlockKeys: func(_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string]) (map[string]kvcache.PodMatch, error) {
			podMatch := func(gpu, cpu, chain int) kvcache.PodMatch {
				return kvcache.PodMatch{
					WeightedScore: float64(chain), MatchedBlocks: chain, ConfirmedBlocks: chain,
					BlocksByTier: map[string]int{kvcache.GPUTier: gpu, "cpu": cpu},
				}
			}
			return map[string]kvcache.PodMatch{
				"10.0.0.1:8080": podMatch(2, 6, 6),
				"10.0.0.2:8080": podMatch(3, 3, 3),
				"10.0.0.3:8080": podMatch(4, 4, 4),
				"10.0.0.4:8080": podMatch(0, 7, 7),
			}, nil
		},
	}
	p := newProducerForProduceAndPreRequest(ctx, name, idx)

	endpoints := append(freshEndpoints(),
		scheduling.NewEndpoint(&datalayer.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-c"}, Address: "10.0.0.3", Port: "8080"}, nil, nil),
		scheduling.NewEndpoint(&datalayer.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-d"}, Address: "10.0.0.4", Port: "8080"}, nil, nil),
	)
	req := tokenizedRequest("req-gpu-miss", promptBlocks*testBlockSize)
	req.TargetModel = "test-model"
	require.NoError(t, p.Produce(ctx, req, endpoints))

	_ = p.PreRequest(ctx, req, primaryWithScored("default", endpoints[0], endpoints[0], endpoints[1]))

	for reason, blocks := range map[string]int{
		prefixmetrics.GPUMissFiltered:      1,
		prefixmetrics.GPUMissNotPicked:     1,
		prefixmetrics.GPUMissLowerTier:     2,
		prefixmetrics.GPUMissPeerLowerTier: 1,
		prefixmetrics.GPUMissUncached:      1,
	} {
		assert.Equal(t, float64(blocks*testBlockSize), gpuMissTokens(t, name, reason), reason)
	}
}

// With no state from Produce the scored maxima are unknown, so the chosen
// endpoint stands in for them and the miss reads as a lower-tier and uncached
// split on that endpoint alone.
func TestPreRequest_GPUMissWithoutProduceState(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-gpu-miss-no-state"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(4, 8, testBlockSize).
		WithCachedBlockCount(4).
		WithCachedBlocksByTier(map[string]int{kvcache.GPUTier: 1, "cpu": 4}))
	_ = p.PreRequest(ctx, tokenizedRequest("req-gpu-miss-no-state", 8*testBlockSize), primaryWithScored("default", endpoint, endpoint))

	assert.Equal(t, float64(3*testBlockSize), gpuMissTokens(t, name, prefixmetrics.GPUMissLowerTier))
	assert.Equal(t, float64(4*testBlockSize), gpuMissTokens(t, name, prefixmetrics.GPUMissUncached))
	assert.Zero(t, gpuMissTokens(t, name, prefixmetrics.GPUMissFiltered))
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
				result.PrefillProfileName = "prefill"
				result.ProfileResults["prefill"] = result.ProfileResults["decode"]
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
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: make([]uint32, tokenCount)}},
			},
		},
	}
}

// mmRequest wraps tokenizedRequest with one image feature spanning the
// prompt's placeholder tokens.
func mmRequest(id string, featureTokens int) *scheduling.InferenceRequest {
	req := tokenizedRequest(id, featureTokens)
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrequest.MultiModalFeature{
		{Modality: reqcommon.ModalityImage, Hash: "img", Offset: 0, Length: featureTokens},
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
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrequest.MultiModalFeature{
		{Modality: reqcommon.ModalityImage, Hash: "img-a", Offset: 2, Length: 20},
		{Modality: reqcommon.ModalityImage, Hash: "img-b", Offset: 32, Length: 20},
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
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{
					{
						TokenIDs: make([]uint32, 4*testBlockSize),
						MultiModalFeatures: []fwkrequest.MultiModalFeature{
							{Modality: reqcommon.ModalityImage, Hash: "img-a", Offset: 0, Length: 4 * testBlockSize},
						},
					},
					{
						TokenIDs: make([]uint32, 3*testBlockSize),
						MultiModalFeatures: []fwkrequest.MultiModalFeature{
							{Modality: reqcommon.ModalityImage, Hash: "img-b", Offset: 0, Length: 3 * testBlockSize},
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
	req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrequest.MultiModalFeature{
		{Modality: reqcommon.ModalityImage, Hash: "img-a", Offset: 0, Length: 4 * testBlockSize},
		{Modality: reqcommon.ModalityImage, Hash: "img-b", Offset: 4 * testBlockSize, Length: 3 * testBlockSize},
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
		PrefillProfileName: "prefill",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{decode}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{prefill}},
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

func TestParseSpeculativeTTL(t *testing.T) {
	for raw, want := range map[string]time.Duration{
		"":     defaultSpeculativeTTL,
		"0s":   defaultSpeculativeTTL,
		"-1s":  defaultSpeculativeTTL,
		"20ms": 20 * time.Millisecond,
	} {
		got, err := parseSpeculativeTTL(raw)
		require.NoError(t, err, "raw %q", raw)
		assert.Equal(t, want, got, "raw %q", raw)
	}
	_, err := parseSpeculativeTTL("soon")
	require.Error(t, err)
}

func TestBlockKeysStateClone(t *testing.T) {
	plain := kvblock.CacheFormat{BlockSize: 2}
	bigram := kvblock.CacheFormat{Bigram: true, BlockSize: 4}
	state := &blockKeysState{perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{
		plain: {{1, 2}}, bigram: {{3}},
	}}
	cloned, ok := state.Clone().(*blockKeysState)
	require.True(t, ok)
	state.perFormatKeys[plain][0][0] = 99
	delete(state.perFormatKeys, bigram)
	require.Equal(t, [][]kvblock.BlockHash{{1, 2}}, cloned.perFormatKeys[plain])
	require.Equal(t, [][]kvblock.BlockHash{{3}}, cloned.perFormatKeys[bigram])
}

// Requests that route the same prefix to the same pod share one hint. An older
// request's TTL must not expire it while a newer request that renewed it is
// still live.
func TestSpeculativeTTLExpiry_KeepsHintOfLiveRequest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(utils.NewTestContext(t))
		defer func() {
			cancel()
			synctest.Wait()
		}()

		p := newSpeculativeTTLProducer(ctx, 200*time.Millisecond)

		keys := []kvblock.BlockHash{0x1, 0x2, 0x3, 0x4}
		pod := testEndpoints[0]

		seedSpeculative(ctx, t, p, "A", keys, pod) // expires at 200ms
		time.Sleep(120 * time.Millisecond)
		seedSpeculative(ctx, t, p, "B", keys, pod) // expires at 320ms
		time.Sleep(90 * time.Millisecond)
		assert.Equal(t, len(keys), hintedKeys(p, keys, pod), "A expired while B is live")

		seedSpeculative(ctx, t, p, "C", keys, pod) // expires at 410ms
		time.Sleep(120 * time.Millisecond)
		assert.Equal(t, len(keys), hintedKeys(p, keys, pod), "B expired while C is live")

		time.Sleep(100 * time.Millisecond)
		assert.Zero(t, hintedKeys(p, keys, pod), "the last live request expired")
	})
}

// Hints expire per (block, pod): the same blocks routed to two pods stay hinted
// for each pod until that pod's own last request expires.
func TestSpeculativeTTLExpiry_SameBlocksOnTwoPods(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(utils.NewTestContext(t))
		defer func() {
			cancel()
			synctest.Wait()
		}()

		p := newSpeculativeTTLProducer(ctx, 200*time.Millisecond)

		keys := []kvblock.BlockHash{0x1, 0x2, 0x3, 0x4}
		podA, podB := testEndpoints[0], testEndpoints[1]
		chain := func(pod scheduling.Endpoint) int { return hintedKeys(p, keys, pod) }

		seedSpeculative(ctx, t, p, "A1", keys, podA) // expires at 200ms
		time.Sleep(120 * time.Millisecond)
		seedSpeculative(ctx, t, p, "B", keys, podB) // expires at 320ms
		time.Sleep(30 * time.Millisecond)
		seedSpeculative(ctx, t, p, "A2", keys, podA) // expires at 350ms

		time.Sleep(60 * time.Millisecond) // 210ms
		assert.Equal(t, len(keys), chain(podA), "A1 expired while A2 is live")
		assert.Equal(t, len(keys), chain(podB), "A1 expired, B is live")

		time.Sleep(120 * time.Millisecond) // 330ms
		assert.Equal(t, len(keys), chain(podA), "B expired while A2 is live")
		assert.Zero(t, chain(podB), "B, the last request on pod B, expired")

		time.Sleep(30 * time.Millisecond) // 360ms
		assert.Zero(t, chain(podA), "A2, the last request on pod A, expired")

		time.Sleep(50 * time.Millisecond) // 410ms, past the second rotation at 400ms
		synctest.Wait()
		assert.Zero(t, p.speculative.Len(), "the expired hints are reclaimed")
	})
}

// gpuMissTokens returns the plugin's llm_d_epp_prefix_gpu_miss_tokens_total for
// reason, or 0 when the series does not exist.
func gpuMissTokens(t *testing.T, pluginName, reason string) float64 {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != "llm_d_epp_prefix_gpu_miss_tokens_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["plugin_name"] == pluginName && labels["reason"] == reason {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
}

// endpointWithMetrics returns an endpoint with the given address and metrics.
func endpointWithMetrics(name, address string, metrics *datalayer.Metrics) scheduling.Endpoint {
	return scheduling.NewEndpoint(&datalayer.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Name: name},
		Address: address,
		Port:    "8080",
	}, metrics, nil)
}

// keyRange returns n distinct block keys starting at first.
func keyRange(first, n int) []kvblock.BlockHash {
	keys := make([]kvblock.BlockHash, n)
	for i := range keys {
		keys[i] = kvblock.BlockHash(first + i)
	}
	return keys
}

// Speculative keys stop at the last whole engine block the endpoint reports,
// since the engine cannot reuse a partial block. Without a reported engine
// block size every key is seeded.
func TestPreRequest_SpeculativeKeysStopAtEngineBlock(t *testing.T) {
	cases := []struct {
		name    string
		metrics *datalayer.Metrics
		prompts [][]kvblock.BlockHash
		want    []int // leading keys hinted, per prompt
	}{
		{
			name:    "prompt shorter than one engine block",
			metrics: &datalayer.Metrics{CacheBlockSize: 4 * testBlockSize},
			prompts: [][]kvblock.BlockHash{keyRange(1, 3)},
			want:    []int{0},
		},
		{
			name:    "partial engine block at the tail",
			metrics: &datalayer.Metrics{CacheBlockSize: 4 * testBlockSize},
			prompts: [][]kvblock.BlockHash{keyRange(1, 6)},
			want:    []int{4},
		},
		{
			name:    "whole engine blocks",
			metrics: &datalayer.Metrics{CacheBlockSize: 4 * testBlockSize},
			prompts: [][]kvblock.BlockHash{keyRange(1, 8)},
			want:    []int{8},
		},
		{
			name:    "engine block not a multiple of the router block",
			metrics: &datalayer.Metrics{CacheBlockSize: 40},
			prompts: [][]kvblock.BlockHash{keyRange(1, 4)},
			want:    []int{2},
		},
		{
			name:    "engine block smaller than the router block",
			metrics: &datalayer.Metrics{CacheBlockSize: testBlockSize / 2},
			prompts: [][]kvblock.BlockHash{keyRange(1, 3)},
			want:    []int{3},
		},
		{
			name:    "engine block size not scraped yet",
			metrics: &datalayer.Metrics{},
			prompts: [][]kvblock.BlockHash{keyRange(1, 3)},
			want:    []int{3},
		},
		{
			name:    "no metrics",
			prompts: [][]kvblock.BlockHash{keyRange(1, 3)},
			want:    []int{3},
		},
		{
			name:    "each prompt stops at its own engine block",
			metrics: &datalayer.Metrics{CacheBlockSize: 4 * testBlockSize},
			prompts: [][]kvblock.BlockHash{keyRange(1, 6), keyRange(100, 3)},
			want:    []int{4, 0},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			p := newProducerForPreRequest(ctx, true, &fakeKVBlockIndex{})
			endpoint := endpointWithMetrics("pod-a", "10.0.0.1", tc.metrics)
			req := &scheduling.InferenceRequest{RequestID: "req-engine-block"}
			p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
				perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{{BlockSize: testBlockSize}: tc.prompts},
			})

			require.NoError(t, p.PreRequest(ctx, req, primaryOnly("default", endpoint)))

			got := make([]int, len(tc.prompts))
			for i, prompt := range tc.prompts {
				got[i] = p.speculative.Match(prompt, nil)["10.0.0.1:8080"]
			}
			assert.Equal(t, tc.want, got)
		})
	}
}

// P/D targets share one format's keys, but each stops at its own engine block.
func TestPreRequest_PrefillProfile_EngineBlockPerTarget(t *testing.T) {
	ctx := utils.NewTestContext(t)

	p := newProducerForPreRequest(ctx, true, &fakeKVBlockIndex{})
	p.seedDecode = true

	keys := keyRange(1, 6)
	req := &scheduling.InferenceRequest{RequestID: "req-pd-engine-block"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
		perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{{BlockSize: testBlockSize}: {keys}},
	})
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		PrefillProfileName: "prefill",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode": {TargetEndpoints: []scheduling.Endpoint{
				endpointWithMetrics("pod-a", "10.0.0.1", &datalayer.Metrics{CacheBlockSize: 4 * testBlockSize}),
			}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{
				endpointWithMetrics("pod-b", "10.0.0.2", nil),
			}},
		},
	}

	require.NoError(t, p.PreRequest(ctx, req, result))

	assert.Equal(t, map[string]int{"10.0.0.1:8080": 4, "10.0.0.2:8080": len(keys)}, p.speculative.Match(keys, nil))
}

// The per-target trace log shows how many speculative keys the engine block
// cut, so a target with every key dropped is distinguishable from speculative
// indexing being off.
func TestPreRequest_LogsSpeculativeKeysDroppedAtEngineBlock(t *testing.T) {
	var buf bytes.Buffer
	ctx := log.IntoContext(t.Context(), logging.NewTestLoggerWithWriter(&buf))
	p := newProducerForPreRequest(ctx, true, &fakeKVBlockIndex{})
	endpoint := endpointWithMetrics("pod-a", "10.0.0.1", &datalayer.Metrics{CacheBlockSize: 4 * testBlockSize})
	req := &scheduling.InferenceRequest{RequestID: "req-engine-block-log"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
		perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{{BlockSize: testBlockSize}: {keyRange(1, 6), keyRange(10, 3)}},
	})

	require.NoError(t, p.PreRequest(ctx, req, primaryOnly("default", endpoint)))

	out := buf.String()
	assert.Contains(t, out, `"pod": "10.0.0.1:8080"`)
	assert.Contains(t, out, `"engineBlockSize": 64`)
	assert.Contains(t, out, `"keptKeys": 4`)
	assert.Contains(t, out, `"droppedKeys": 5`)
}

// A repeat of a routed prompt matches only the tokens the engine can reuse:
// whole engine blocks of the endpoint that served it.
func TestProduceThenPreRequest_SpeculativeMatchStopsAtEngineBlock(t *testing.T) {
	const name = "engine-block"
	cases := []struct {
		promptTokens int
		wantBlocks   int
	}{
		{promptTokens: 3 * testBlockSize, wantBlocks: 0},
		{promptTokens: 6*testBlockSize + 4, wantBlocks: 4},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%d tokens", tc.promptTokens), func(t *testing.T) {
			ctx, cancel := context.WithCancel(utils.NewTestContext(t))
			t.Cleanup(cancel)

			idxCfg, err := kvcache.NewDefaultConfig()
			require.NoError(t, err)
			p, err := New(ctx, name, PluginConfig{
				TokenProcessorConfig: &kvblock.TokenProcessorConfig{BlockSizeTokens: testBlockSize},
				IndexerConfig:        idxCfg,
				KVEventsConfig:       kvevents.DefaultConfig(),
				SpeculativeIndexing:  true,
				SpeculativeTTL:       "1m",
			})
			require.NoError(t, err)

			tokens := make([]uint32, tc.promptTokens)
			for i := range tokens {
				tokens[i] = uint32(i + 1)
			}
			request := func(id string) *scheduling.InferenceRequest {
				return &scheduling.InferenceRequest{
					RequestID:   id,
					TargetModel: "m",
					Body: &fwkrequest.InferenceRequestBody{TokenizedRequest: &fwkrequest.TokenizedRequest{
						Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens}},
					}},
				}
			}
			metrics := &datalayer.Metrics{CacheBlockSize: 4 * testBlockSize}

			first := endpointWithMetrics("pod-a", "10.0.0.1", metrics)
			routed := request("routed")
			require.NoError(t, p.Produce(ctx, routed, []scheduling.Endpoint{first}))
			require.NoError(t, p.PreRequest(ctx, routed, primaryOnly("default", first)))

			repeat := endpointWithMetrics("pod-a", "10.0.0.1", metrics)
			require.NoError(t, p.Produce(ctx, request("repeat"), []scheduling.Endpoint{repeat}))
			raw, ok := repeat.Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name))
			require.True(t, ok)
			info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
			require.True(t, ok)
			assert.Equal(t, tc.wantBlocks, info.CachedBlockCount())
		})
	}
}

func TestPreRequest_RankEndpointSeedsItsOwnRank(t *testing.T) {
	ctx := utils.NewTestContext(t)

	p := newProducerForPreRequest(ctx, true, &fakeKVBlockIndex{})
	p.kvEventsConfig = &kvevents.Config{DiscoverPods: true, PodDiscoveryConfig: kvevents.DefaultPodReconcilerConfig()}

	req := &scheduling.InferenceRequest{RequestID: "req-pre-rank"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perFormatKeys: map[kvblock.CacheFormat][][]kvblock.BlockHash{
		{BlockSize: testBlockSize}: {{0xAA}},
	}})

	_ = p.PreRequest(ctx, req, primaryOnly("default", rankEndpoints()[1]))

	assert.Equal(t, map[string]int{"10.0.0.1:8000/rank-1": 1}, p.speculative.Match([]kvblock.BlockHash{0xAA}, nil))
}
