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
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/observability/multimodal"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixmetrics"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry))
}

func TestRecordPrefixCacheMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recordPrefixCacheSize("test-plugin", "test-type", 4096)
	recordPrefixCacheMatch("test-plugin", "test-type", 10, 20)
	recordPrefixCacheMatch("test-plugin", "test-type", 0, 0)

	require.Equal(t, float64(4096), testutil.ToFloat64(llmdPrefixCacheSize.WithLabelValues("test-plugin", "test-type")))

	hitRatio, err := getHistogram(llmdPrefixCacheHitRatio, "test-plugin", "test-type")
	require.NoError(t, err)
	require.Equal(t, uint64(1), hitRatio.GetSampleCount())
	require.Equal(t, 0.5, hitRatio.GetSampleSum())

	hitLength, err := getHistogram(llmdPrefixCacheHitLength, "test-plugin", "test-type")
	require.NoError(t, err)
	require.Equal(t, uint64(2), hitLength.GetSampleCount())
	require.Equal(t, float64(10), hitLength.GetSampleSum())
}

func getHistogram(histogram *prometheus.HistogramVec, labelValues ...string) (*dto.Histogram, error) {
	metric, err := histogram.GetMetricWithLabelValues(labelValues...)
	if err != nil {
		return nil, err
	}
	dtoMetric := &dto.Metric{}
	if err := metric.(prometheus.Histogram).Write(dtoMetric); err != nil {
		return nil, err
	}
	return dtoMetric.GetHistogram(), nil
}

func resetMetrics() {
	llmdPrefixCacheSize.Reset()
	llmdPrefixCacheHitRatio.Reset()
	llmdPrefixCacheHitLength.Reset()
}

// PreRequest reports the prefix hit for the chosen endpoint in tokens, together
// with the prompt tokens it was measured against.
func TestPreRequestRecordsPrediction(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-records"
	p := producerForPrediction(t, name, 2)
	endpoints, result := endpointAndResult()

	// Seed the indexer: nothing is cached yet, so the prediction is zero while
	// the prompt still lands in the denominator.
	tokens := []uint32{1, 2, 3, 4}
	runPrediction(t, p, "seed", tokens, endpoints, result)
	require.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode))
	require.Equal(t, float64(len(tokens)), metricSum(t, promptTokensMetric, name, prefixmetrics.RoleDecode))

	// The same prompt now matches every block on the endpoint that was chosen.
	runPrediction(t, p, "repeat", tokens, endpoints, result)
	assert.Equal(t, float64(len(tokens)), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode))
	assert.Equal(t, float64(2*len(tokens)), metricSum(t, promptTokensMetric, name, prefixmetrics.RoleDecode))
}

// A request carrying multimodal content records its prediction once, under
// the comma-joined sorted list of the modalities it carries, leaving the
// text-only series untouched.
func TestPreRequestRecordsPredictionModality(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-modality"
	p := producerForPrediction(t, name, 2)
	endpoints, result := endpointAndResult()

	tokens := []uint32{1, 2, 3, 4}
	body := tokenizedBody(tokens)
	body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrh.MultiModalFeature{
		{Modality: fwkrh.ModalityImage, Hash: "img"},
	}
	image := string(fwkrh.ModalityImage)

	// sum reads the producer's decode series for one modality label value.
	sum := func(metricName, modality string) float64 {
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
					return metric.GetHistogram().GetSampleSum()
				}
			}
		}
		return 0
	}

	runPredictionWithBody(t, p, "seed", body, endpoints, result)
	assert.Equal(t, float64(0), sum(predictedCachedTokensMetric, image))
	assert.Equal(t, float64(len(tokens)), sum(promptTokensMetric, image))

	runPredictionWithBody(t, p, "repeat", body, endpoints, result)
	assert.Equal(t, float64(len(tokens)), sum(predictedCachedTokensMetric, image))
	assert.Equal(t, float64(2*len(tokens)), sum(promptTokensMetric, image))

	assert.Equal(t, float64(0), sum(predictedCachedTokensMetric, mmobs.ModalityNone),
		"a multimodal request must not record in the text-only series")
	assert.Equal(t, float64(0), sum(promptTokensMetric, mmobs.ModalityNone),
		"a multimodal request must not record in the text-only series")
}

// A prompt whose length is not a multiple of the block size still hashes its
// trailing partial block, so the block-to-token conversion has to be bounded by
// the prompt length or a full match reports more tokens than the prompt holds.
func TestPreRequestPredictionBoundedByPromptLength(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-partial-block"
	p := producerForPrediction(t, name, 4)
	endpoints, result := endpointAndResult()

	// 5 tokens at block size 4 hash to 2 blocks, the second covering 1 token.
	tokens := []uint32{1, 2, 3, 4, 5}
	runPrediction(t, p, "seed", tokens, endpoints, result)
	require.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode))

	runPrediction(t, p, "repeat", tokens, endpoints, result)
	assert.Equal(t, float64(len(tokens)), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode),
		"a full match must report the prompt's 5 tokens, not 2 blocks * 4 tokens")
}

// Each prompt is bounded on its own: clamping the aggregate would let a short
// prompt's overshoot hide under a long prompt's length.
func TestPreRequestPredictionBoundsEachPromptSeparately(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-multi-prompt"
	p := producerForPrediction(t, name, 4)
	endpoints, result := endpointAndResult()

	// 5 tokens (2 blocks, 1 partial) alongside 8 tokens (2 full blocks).
	body := &fwkrh.InferenceRequestBody{
		TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{
			{TokenIDs: []uint32{1, 2, 3, 4, 5}},
			{TokenIDs: []uint32{6, 7, 8, 9, 10, 11, 12, 13}},
		}},
	}
	runPredictionWithBody(t, p, "seed", body, endpoints, result)
	runPredictionWithBody(t, p, "repeat", body, endpoints, result)

	assert.Equal(t, float64(13), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode))
	assert.Equal(t, float64(26), metricSum(t, promptTokensMetric, name, prefixmetrics.RoleDecode))
}

// A token cap below the block size hashes nothing, so no endpoint can be
// predicted to hold any of the prompt. The prompt still has to reach the
// denominator, or the misconfiguration leaves the metric empty instead of
// reporting a zero hit rate.
func TestPreRequestPredictionCountsUnhashedPrompts(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-no-hashes"
	p, err := newDataProducer(context.Background(), name, config{
		BlockSizeTokens:        4,
		MaxPrefixTokensToMatch: 2,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}, testHandle())
	require.NoError(t, err)
	endpoints, result := endpointAndResult()

	tokens := []uint32{1, 2, 3, 4, 5}
	runPrediction(t, p, "unhashed", tokens, endpoints, result)

	assert.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode))
	assert.Equal(t, float64(len(tokens)), metricSum(t, promptTokensMetric, name, prefixmetrics.RoleDecode))
}

// A disaggregated request's cached-token count comes back from the prefiller,
// so the prediction is recorded for the prefill endpoint rather than the
// primary one.
func TestPreRequestPredictionFollowsPrefillEndpoint(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-pd"
	p := producerForPrediction(t, name, 2)
	decode := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "decode", Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes())
	prefill := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "prefill", Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{decode, prefill}
	decodeOnly := &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{decode}},
		},
	}
	disaggregated := &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode":                          {TargetEndpoints: []fwksched.Endpoint{decode}},
			experimentalDefaultPrefillProfile: {TargetEndpoints: []fwksched.Endpoint{prefill}},
		},
	}

	// Only the decode endpoint holds the prompt after this request.
	tokens := []uint32{1, 2, 3, 4}
	runPrediction(t, p, "seed", tokens, endpoints, decodeOnly)
	require.Equal(t, float64(len(tokens)), metricSum(t, promptTokensMetric, name, prefixmetrics.RoleDecode))

	// The prefill endpoint holds nothing yet, so the prediction is zero even
	// though the primary endpoint would fully match.
	runPrediction(t, p, "cold-prefill", tokens, endpoints, disaggregated)
	assert.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RolePrefill))
	assert.Equal(t, float64(len(tokens)), metricSum(t, promptTokensMetric, name, prefixmetrics.RolePrefill))

	runPrediction(t, p, "warm-prefill", tokens, endpoints, disaggregated)
	assert.Equal(t, float64(len(tokens)), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RolePrefill))
	assert.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode))
}

// The best the picker could have chosen spans every scored candidate, not just
// the one it took, so a request routed away from the cached pod reports the hit
// it passed up.
func TestPreRequestBestPredictedSpansScoredCandidates(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-best-scored"
	p := producerForPrediction(t, name, 2)
	cold, cached := namedEndpoint("cold"), namedEndpoint("cached")
	pods := []fwksched.Endpoint{cold, cached}
	tokens := []uint32{1, 2, 3, 4}

	// Seed the cached pod by routing the prompt to it once.
	runPrediction(t, p, "seed", tokens, pods, resultWith(cached, cold, cached))
	// Route the same prompt to the cold pod instead.
	runPrediction(t, p, "reroute", tokens, pods, resultWith(cold, cold, cached))

	assert.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode),
		"both requests landed on a pod holding nothing")
	assert.Equal(t, float64(len(tokens)), metricSum(t, bestPredictedMetric, name, prefixmetrics.RoleDecode),
		"the second request could have reached the cached pod")
}

// A pod that holds the prefix but is not among the request's candidates does
// not raise either maximum, since the router was never offered it.
func TestPreRequestBestIgnoresNonCandidateServers(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-best-non-candidate"
	p := producerForPrediction(t, name, 2)
	other, candidate := namedEndpoint("other-pool"), namedEndpoint("candidate")
	tokens := []uint32{1, 2, 3, 4}

	// Seed the indexer with a pod that the next request cannot reach.
	runPrediction(t, p, "seed", tokens, []fwksched.Endpoint{other}, resultWith(other, other))
	runPrediction(t, p, "scoped", tokens, []fwksched.Endpoint{candidate}, resultWith(candidate, candidate))

	assert.Equal(t, float64(0), metricSum(t, bestPredictedMetric, name, prefixmetrics.RoleDecode))
	assert.Equal(t, float64(0), metricSum(t, bestAvailableMetric, name, prefixmetrics.RoleDecode))
}

// A candidate dropped by a filter never reaches the picker, so the reuse it
// held shows up as available but not as a hit the picker could have taken.
func TestPreRequestBestAvailableSpansFilteredOutCandidates(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-best-available"
	p := producerForPrediction(t, name, 2)
	survivor, dropped := namedEndpoint("survivor"), namedEndpoint("dropped")
	pods := []fwksched.Endpoint{survivor, dropped}
	tokens := []uint32{1, 2, 3, 4}

	runPrediction(t, p, "seed", tokens, pods, resultWith(dropped, survivor, dropped))
	// The cached pod is still a candidate, but a filter kept it from the picker.
	runPrediction(t, p, "filtered", tokens, pods, resultWith(survivor, survivor))

	assert.Equal(t, float64(0), metricSum(t, bestPredictedMetric, name, prefixmetrics.RoleDecode),
		"the picker only saw the pod holding nothing")
	assert.Equal(t, float64(len(tokens)), metricSum(t, bestAvailableMetric, name, prefixmetrics.RoleDecode),
		"the filtered-out candidate still held the prefix")
}

// A profile that reports no scored candidates leaves the chosen endpoint as the
// only evidence, so both maxima fall back to it rather than dropping to zero
// and reporting a hit the router never had the chance to miss.
func TestPreRequestBestFallsBackToSelected(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-best-no-scored"
	p := producerForPrediction(t, name, 2)
	endpoints, result := endpointAndResult()
	require.Empty(t, result.ProfileResults["default"].ScoredCandidates)
	tokens := []uint32{1, 2, 3, 4}

	runPrediction(t, p, "seed", tokens, endpoints, result)
	runPrediction(t, p, "repeat", tokens, endpoints, result)

	selected := metricSum(t, predictedCachedTokensMetric, name, prefixmetrics.RoleDecode)
	assert.Equal(t, float64(len(tokens)), selected)
	assert.Equal(t, selected, metricSum(t, bestPredictedMetric, name, prefixmetrics.RoleDecode))
	assert.Equal(t, selected, metricSum(t, bestAvailableMetric, name, prefixmetrics.RoleDecode))
}

// A profile handler that rebuilds the result from its targets alone, such as
// the data-parallel one, leaves no scored candidates. The pre-filter maximum
// falls back with the picker-side one, so a routing miss toward a warmer
// candidate is not reported as reuse lost to filtering.
func TestPreRequestBestAvailableFallsBackWithoutScoredCandidates(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-best-available-no-scored"
	p := producerForPrediction(t, name, 2)
	cold, cached := namedEndpoint("cold"), namedEndpoint("cached")
	pods := []fwksched.Endpoint{cold, cached}
	tokens := []uint32{1, 2, 3, 4}

	runPrediction(t, p, "seed", tokens, pods, resultWith(cached, cold, cached))
	runPrediction(t, p, "targets-only", tokens, pods, resultWith(cold))

	assert.Equal(t, float64(0), metricSum(t, bestPredictedMetric, name, prefixmetrics.RoleDecode))
	assert.Equal(t, float64(0), metricSum(t, bestAvailableMetric, name, prefixmetrics.RoleDecode))
}

// The two maxima carry the modalities the request holds, so the reuse routing
// and filtering left behind can be split between multimodal and text-only
// traffic.
func TestPreRequestMaximaCarryModality(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-best-modality"
	p := producerForPrediction(t, name, 2)
	endpoints, result := endpointAndResult()

	body := tokenizedBody([]uint32{1, 2, 3, 4})
	body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrh.MultiModalFeature{
		{Modality: fwkrh.ModalityImage, Hash: "img"},
	}
	runPredictionWithBody(t, p, "mm", body, endpoints, result)

	image := string(fwkrh.ModalityImage)
	assert.Equal(t, image, metricModality(t, bestPredictedMetric, name))
	assert.Equal(t, image, metricModality(t, bestAvailableMetric, name))
}

func namedEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes())
}

// resultWith selects target and reports scored as the candidates that reached
// the picker. A candidate the scheduler filtered out is left out of scored.
func resultWith(target fwksched.Endpoint, scored ...fwksched.Endpoint) *fwksched.SchedulingResult {
	candidates := make([]fwksched.ScoredEndpoint, 0, len(scored))
	for _, endpoint := range scored {
		candidates = append(candidates, fwksched.ScoredEndpoint{Endpoint: endpoint})
	}
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{target}, ScoredCandidates: candidates},
		},
	}
}

func producerForPrediction(t *testing.T, name string, blockSize int) *dataProducer {
	t.Helper()
	p, err := newDataProducer(context.Background(), name, config{
		BlockSizeTokens:        blockSize,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}, testHandle())
	require.NoError(t, err)
	return p
}

func endpointAndResult() ([]fwksched.Endpoint, *fwksched.SchedulingResult) {
	endpoints := []fwksched.Endpoint{fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes())}
	return endpoints, &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: endpoints},
		},
	}
}

func runPrediction(t *testing.T, p *dataProducer, id string, tokens []uint32,
	endpoints []fwksched.Endpoint, result *fwksched.SchedulingResult,
) {
	t.Helper()
	runPredictionWithBody(t, p, id, tokenizedBody(tokens), endpoints, result)
}

func runPredictionWithBody(t *testing.T, p *dataProducer, id string, body *fwkrh.InferenceRequestBody,
	endpoints []fwksched.Endpoint, result *fwksched.SchedulingResult,
) {
	t.Helper()
	req := &fwksched.InferenceRequest{RequestID: id, TargetModel: "m", Body: body}
	require.NoError(t, p.Produce(context.Background(), req, endpoints))
	require.NoError(t, p.PreRequest(context.Background(), req, result))
	p.wg.Wait()
}

const (
	predictedCachedTokensMetric = "llm_d_epp_prefix_predicted_cached_tokens"      //nolint:gosec // G101: metric name, not a credential
	bestPredictedMetric         = "llm_d_epp_prefix_best_predicted_cached_tokens" //nolint:gosec // G101: metric name, not a credential
	bestAvailableMetric         = "llm_d_epp_prefix_best_available_cached_tokens" //nolint:gosec // G101: metric name, not a credential
	promptTokensMetric          = "llm_d_epp_prefix_prompt_tokens"                //nolint:gosec // G101: metric name, not a credential
)

// metricSum reads a shared prefix metric out of the registry it is registered
// against, since those metrics live in another package.
func metricSum(t *testing.T, metricName, pluginName, role string) float64 {
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
				return metric.GetHistogram().GetSampleSum()
			}
		}
	}
	return 0
}

// metricModality returns the modality label of the plugin's only series of a
// shared prefix metric.
func metricModality(t *testing.T, metricName, pluginName string) string {
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
