/*
Copyright 2026 The Kubernetes Authors.

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

package predictedlatency

import (
	"math"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Locality accuracy is asserted through the exposed metric families rather than
// the recorder's return values: what a dashboard consumes is the contract, and
// the whole point of the outcomes is that a dropped observation stays visible as
// a count instead of silently becoming a zero sample in a histogram.

const (
	localityDecodePod   = "qwen-decode-0-rank-0"
	localityPrefillPod  = "qwen-prefill-1-rank-0"
	localityOtherPod    = "qwen-other-2-rank-0"
	localityTargetModel = "qwen38-flash-next"
)

// setupLocalityMetrics registers the EPP metric families exactly once per test
// binary and clears every series before and after the test, so counts asserted
// below belong to this test alone.
func setupLocalityMetrics(t *testing.T) {
	t.Helper()
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)
}

// family returns the gathered metric family, or nil when nothing was recorded
// into it. An absent family and an empty family are both "no observations"; the
// distinction is deliberately not asserted, because a *Vec keeps a zero-valued
// child once touched and both cases must read as "not observed".
func family(t *testing.T, name string) []*dto.Metric {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err, "gather EPP metrics")
	for _, f := range families {
		if f.GetName() == name {
			return f.GetMetric()
		}
	}
	return nil
}

func labelsMatch(metric *dto.Metric, want map[string]string) bool {
	got := map[string]string{}
	for _, lp := range metric.GetLabel() {
		got[lp.GetName()] = lp.GetValue()
	}
	for k, v := range want {
		if got[k] != v {
			return false
		}
	}
	return true
}

// histogramObservation returns how many values were observed in the named
// histogram for the given endpoint/model, and their sum.
func histogramObservation(t *testing.T, name, endpoint, model string) (uint64, float64) {
	t.Helper()
	want := map[string]string{"model_server_endpoint": endpoint, "target_model_name": model}
	for _, metric := range family(t, name) {
		if labelsMatch(metric, want) {
			return metric.GetHistogram().GetSampleCount(), metric.GetHistogram().GetSampleSum()
		}
	}
	return 0, 0
}

// localityOutcomeCount returns observations_total for one outcome.
func localityOutcomeCount(t *testing.T, outcome, endpoint, model string) uint64 {
	t.Helper()
	want := map[string]string{
		"model_server_endpoint": endpoint,
		"target_model_name":     model,
		"outcome":               outcome,
	}
	for _, metric := range family(t, "llm_d_epp_cache_locality_observations_total") {
		if labelsMatch(metric, want) {
			return uint64(metric.GetCounter().GetValue())
		}
	}
	return 0
}

// totalObservations sums every outcome series, which is how a test asserts that
// exactly one outcome was recorded for the request.
func totalObservations(t *testing.T) uint64 {
	t.Helper()
	var total uint64
	for _, metric := range family(t, "llm_d_epp_cache_locality_observations_total") {
		total += uint64(metric.GetCounter().GetValue())
	}
	return total
}

// localityContext builds a request context as Produce plus PreRequest would have:
// a captured prediction for the scored endpoints and a selected target endpoint.
func localityContext(request *fwksched.InferenceRequest, decode, prefill *fwkdl.EndpointMetadata, scores map[string]float64, known map[string]bool) *predictedLatencyCtx {
	ctx := newPredictedLatencyContext(request)
	ctx.targetMetadata = decode
	ctx.prefillTargetMetadata = prefill
	ctx.prefixCacheScoresForEndpoints = scores
	ctx.prefixCacheScoreKnownForEndpoints = known
	return ctx
}

func metadataFor(name string) *fwkdl.EndpointMetadata {
	return createTestEndpoint(name, 0.5, 1, 0).GetMetadata()
}

func usage(cached, prompt int, withDetails bool) fwkrh.Usage {
	usage := fwkrh.Usage{PromptTokens: prompt, CompletionTokens: 3}
	if withDetails {
		usage.PromptTokenDetails = &fwkrh.PromptTokenDetails{CachedTokens: cached}
	}
	return usage
}

// TestRecordCacheLocality_ObservesSelectedEndpoint proves the comparison is made
// against the prediction captured for the endpoint that served the request, and
// that the error splits into the two non-negative magnitudes.
func TestRecordCacheLocality_ObservesSelectedEndpoint(t *testing.T) {
	setupLocalityMetrics(t)
	router := createTestRouter()
	request := createTestInferenceRequest("loc-selected", 100, 50)
	request.TargetModel = localityTargetModel
	decode := metadataFor(localityDecodePod)

	ctx := localityContext(request, decode, nil,
		map[string]float64{localityDecodePod: 0.75, localityOtherPod: 0.1},
		map[string]bool{localityDecodePod: true, localityOtherPod: true})
	response := &requestcontrol.Response{EndOfStream: true, Usage: usage(50, 100, true)}

	router.recordCacheLocality(t.Context(), request, response, ctx)

	count, sum := histogramObservation(t, "llm_d_epp_cache_locality_predicted_fraction", localityDecodePod, localityTargetModel)
	assert.Equal(t, uint64(1), count, "predicted fraction observed once for the selected endpoint")
	assert.InDelta(t, 0.75, sum, 1e-9, "predicted fraction is the selected endpoint's score, not another candidate's")

	count, sum = histogramObservation(t, "llm_d_epp_cache_locality_actual_fraction", localityDecodePod, localityTargetModel)
	assert.Equal(t, uint64(1), count)
	assert.InDelta(t, 0.5, sum, 1e-9, "actual is cached/prompt from the final usage block")

	// predicted 0.75 > actual 0.5: the whole error lands in overprediction.
	count, sum = histogramObservation(t, "llm_d_epp_cache_locality_overprediction_fraction", localityDecodePod, localityTargetModel)
	assert.Equal(t, uint64(1), count)
	assert.InDelta(t, 0.25, sum, 1e-9)
	count, sum = histogramObservation(t, "llm_d_epp_cache_locality_underprediction_fraction", localityDecodePod, localityTargetModel)
	assert.Equal(t, uint64(1), count, "the complementary magnitude is still observed, as zero")
	assert.InDelta(t, 0.0, sum, 1e-9)

	assert.Equal(t, uint64(1), localityOutcomeCount(t, "observed", localityDecodePod, localityTargetModel))
	assert.Equal(t, uint64(1), totalObservations(t), "exactly one outcome per request")

	// The unscored candidate must not gain a series of its own.
	count, _ = histogramObservation(t, "llm_d_epp_cache_locality_predicted_fraction", localityOtherPod, localityTargetModel)
	assert.Zero(t, count, "only the selected endpoint is observed")
}

// TestRecordCacheLocality_PrefersPrefillTarget pins the disaggregated case: the
// prompt is matched against the prefill endpoint, so that is the prediction the
// actual usage must be compared with — the same attribution the TTFT training
// record makes.
func TestRecordCacheLocality_PrefersPrefillTarget(t *testing.T) {
	setupLocalityMetrics(t)
	router := createTestRouter()
	request := createTestInferenceRequest("loc-prefill", 100, 50)
	request.TargetModel = localityTargetModel

	ctx := localityContext(request,
		metadataFor(localityDecodePod), metadataFor(localityPrefillPod),
		map[string]float64{localityDecodePod: 0.9, localityPrefillPod: 0.4},
		map[string]bool{localityDecodePod: true, localityPrefillPod: true})
	response := &requestcontrol.Response{EndOfStream: true, Usage: usage(40, 100, true)}

	router.recordCacheLocality(t.Context(), request, response, ctx)

	assert.Equal(t, uint64(1), localityOutcomeCount(t, "observed", localityPrefillPod, localityTargetModel),
		"locality is attributed to the prefill target")
	count, sum := histogramObservation(t, "llm_d_epp_cache_locality_predicted_fraction", localityPrefillPod, localityTargetModel)
	require.Equal(t, uint64(1), count)
	assert.InDelta(t, 0.4, sum, 1e-9, "prefill's prediction, not the decode endpoint's")
	count, _ = histogramObservation(t, "llm_d_epp_cache_locality_predicted_fraction", localityDecodePod, localityTargetModel)
	assert.Zero(t, count)
}

// TestRecordCacheLocality_MissingUsageIsNotZero is the absent-vs-zero guarantee:
// a response that never reported cached tokens must not enter the actual-fraction
// histogram as 0.0, and must not enter the predicted one either.
func TestRecordCacheLocality_MissingUsageIsNotZero(t *testing.T) {
	setupLocalityMetrics(t)
	router := createTestRouter()
	request := createTestInferenceRequest("loc-no-usage", 100, 50)
	request.TargetModel = localityTargetModel

	ctx := localityContext(request, metadataFor(localityDecodePod), nil,
		map[string]float64{localityDecodePod: 0.6}, map[string]bool{localityDecodePod: true})
	response := &requestcontrol.Response{EndOfStream: true, Usage: usage(0, 0, false)}

	router.recordCacheLocality(t.Context(), request, response, ctx)

	assert.Equal(t, uint64(1), localityOutcomeCount(t, "missing_usage", localityDecodePod, localityTargetModel))
	assert.Equal(t, uint64(1), totalObservations(t))
	for _, name := range []string{
		"llm_d_epp_cache_locality_predicted_fraction",
		"llm_d_epp_cache_locality_actual_fraction",
		"llm_d_epp_cache_locality_underprediction_fraction",
		"llm_d_epp_cache_locality_overprediction_fraction",
	} {
		count, sum := histogramObservation(t, name, localityDecodePod, localityTargetModel)
		assert.Zero(t, count, "%s must not observe a request without usage", name)
		assert.Zero(t, sum)
	}
}

// TestRecordCacheLocality_ZeroActualIsObserved is the other half of the pair
// above: a genuine 0% hit is an observation, distinguishable from the missing
// case because it lands in the histograms.
func TestRecordCacheLocality_ZeroActualIsObserved(t *testing.T) {
	setupLocalityMetrics(t)
	router := createTestRouter()
	request := createTestInferenceRequest("loc-zero", 100, 50)
	request.TargetModel = localityTargetModel

	ctx := localityContext(request, metadataFor(localityDecodePod), nil,
		map[string]float64{localityDecodePod: 0.3}, map[string]bool{localityDecodePod: true})
	response := &requestcontrol.Response{EndOfStream: true, Usage: usage(0, 100, true)}

	router.recordCacheLocality(t.Context(), request, response, ctx)

	assert.Equal(t, uint64(1), localityOutcomeCount(t, "observed", localityDecodePod, localityTargetModel))
	count, sum := histogramObservation(t, "llm_d_epp_cache_locality_actual_fraction", localityDecodePod, localityTargetModel)
	require.Equal(t, uint64(1), count, "a real zero hit rate is still an observation")
	assert.Zero(t, sum)
	count, sum = histogramObservation(t, "llm_d_epp_cache_locality_overprediction_fraction", localityDecodePod, localityTargetModel)
	require.Equal(t, uint64(1), count)
	assert.InDelta(t, 0.3, sum, 1e-9, "predicted 0.3 above actual 0 is overprediction, the direction that costs prefill work")
}

// TestRecordCacheLocality_MissingPrediction covers a request the prefix producer
// never scored — the stored score is 0.0 there too, so presence must be tracked
// separately or a missing producer reads as a predicted miss.
func TestRecordCacheLocality_MissingPrediction(t *testing.T) {
	setupLocalityMetrics(t)
	for _, tc := range []struct {
		name   string
		scores map[string]float64
		known  map[string]bool
	}{
		{
			name:   "attribute absent",
			scores: map[string]float64{localityDecodePod: 0.0},
			known:  map[string]bool{},
		},
		{
			name:   "zero denominator scored as unknown",
			scores: map[string]float64{localityDecodePod: 0.0},
			known:  map[string]bool{localityDecodePod: false},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eppmetrics.Reset()
			router := createTestRouter()
			request := createTestInferenceRequest("loc-missing-pred", 100, 50)
			request.TargetModel = localityTargetModel

			ctx := localityContext(request, metadataFor(localityDecodePod), nil, tc.scores, tc.known)
			response := &requestcontrol.Response{EndOfStream: true, Usage: usage(10, 100, true)}

			router.recordCacheLocality(t.Context(), request, response, ctx)

			assert.Equal(t, uint64(1), localityOutcomeCount(t, "missing_prediction", localityDecodePod, localityTargetModel))
			assert.Equal(t, uint64(1), totalObservations(t))
			count, _ := histogramObservation(t, "llm_d_epp_cache_locality_predicted_fraction", localityDecodePod, localityTargetModel)
			assert.Zero(t, count, "no prediction, so nothing to compare")
		})
	}
}

// TestRecordCacheLocality_NoTargetEndpoint asserts an undispatched request still
// counts exactly one outcome, without inventing an endpoint label.
func TestRecordCacheLocality_NoTargetEndpoint(t *testing.T) {
	setupLocalityMetrics(t)
	router := createTestRouter()
	request := createTestInferenceRequest("loc-no-target", 100, 50)
	request.TargetModel = localityTargetModel

	ctx := localityContext(request, nil, nil, map[string]float64{}, map[string]bool{})
	response := &requestcontrol.Response{EndOfStream: true, Usage: usage(10, 100, true)}

	router.recordCacheLocality(t.Context(), request, response, ctx)

	assert.Equal(t, uint64(1), localityOutcomeCount(t, "missing_prediction", "", localityTargetModel))
	assert.Equal(t, uint64(1), totalObservations(t))
}

// TestRecordCacheLocality_InvalidUsage covers the ratios that cannot be formed:
// a zero token denominator, cached tokens exceeding the prompt, and a prediction
// outside [0,1].
func TestRecordCacheLocality_InvalidUsage(t *testing.T) {
	setupLocalityMetrics(t)
	for _, tc := range []struct {
		name      string
		predicted float64
		known     bool
		usage     fwkrh.Usage
	}{
		{name: "no prompt tokens", predicted: 0.5, known: true, usage: usage(0, 0, true)},
		{name: "negative prompt tokens", predicted: 0.5, known: true, usage: usage(0, -4, true)},
		{name: "cached exceeds prompt", predicted: 0.5, known: true, usage: usage(101, 100, true)},
		{name: "negative cached", predicted: 0.5, known: true, usage: usage(-1, 100, true)},
		{name: "prediction above one", predicted: 1.5, known: true, usage: usage(50, 100, true)},
		{name: "prediction below zero", predicted: -0.1, known: true, usage: usage(50, 100, true)},
		{name: "prediction NaN", predicted: math.NaN(), known: true, usage: usage(50, 100, true)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eppmetrics.Reset()
			router := createTestRouter()
			request := createTestInferenceRequest("loc-invalid", 100, 50)
			request.TargetModel = localityTargetModel

			ctx := localityContext(request, metadataFor(localityDecodePod), nil,
				map[string]float64{localityDecodePod: tc.predicted},
				map[string]bool{localityDecodePod: tc.known})
			response := &requestcontrol.Response{EndOfStream: true, Usage: tc.usage}

			router.recordCacheLocality(t.Context(), request, response, ctx)

			assert.Equal(t, uint64(1), localityOutcomeCount(t, "invalid_usage", localityDecodePod, localityTargetModel))
			assert.Equal(t, uint64(1), totalObservations(t), "an unusable observation is counted, not dropped")
			for _, name := range []string{
				"llm_d_epp_cache_locality_predicted_fraction",
				"llm_d_epp_cache_locality_actual_fraction",
			} {
				count, _ := histogramObservation(t, name, localityDecodePod, localityTargetModel)
				assert.Zero(t, count, "%s must not observe invalid data", name)
			}
		})
	}
}

// TestResponseBody_RecordsLocalityOncePerRequest is the stream-once guarantee:
// intermediate chunks must not observe locality, and the terminal chunk observes
// it exactly once even if end-of-stream is delivered twice.
func TestResponseBody_RecordsLocalityOncePerRequest(t *testing.T) {
	setupLocalityMetrics(t)
	router := createTestRouter()
	router.config.StreamingMode = true
	router.latencypredictor = new(mockPredictor)

	request := createTestInferenceRequest("loc-stream", 100, 50)
	request.TargetModel = localityTargetModel
	endpoint := createTestEndpoint(localityDecodePod, 0.5, 1, 0)

	ctx := localityContext(request, endpoint.GetMetadata(), nil,
		map[string]float64{localityDecodePod: 0.8}, map[string]bool{localityDecodePod: true})
	ctx.requestReceivedTimestamp = time.Now().Add(-100 * time.Millisecond)
	ctx.schedulingResult = createTestSchedulingResult(endpoint.GetMetadata())
	router.setPredictedLatencyContextForRequest(request, ctx)

	queue := newRequestPriorityQueue()
	queue.Add(request.Headers[reqcommon.RequestIDHeaderKey], 50.0)
	router.runningRequestLists.Store(endpoint.GetMetadata().ID, queue)

	// Chunks carrying partial usage: the actual fraction is not final yet, so no
	// observation may be recorded from them.
	for _, chunk := range []*requestcontrol.Response{
		{EndOfStream: false, Usage: usage(0, 100, false)},
		{EndOfStream: false, Usage: usage(40, 100, true)},
		{EndOfStream: false, Usage: usage(60, 100, true)},
	} {
		router.ResponseBody(t.Context(), request, chunk, endpoint.GetMetadata())
	}
	assert.Zero(t, totalObservations(t), "no locality observation before end of stream")

	// The terminal chunk carries the final usage; earlier chunks may have merged
	// into it, so this is the only place a complete actual exists.
	retrieved, err := router.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	retrieved.ttft = 80
	router.setPredictedLatencyContextForRequest(request, retrieved)
	router.ResponseBody(t.Context(), request, &requestcontrol.Response{
		EndOfStream: true,
		Usage:       usage(60, 100, true),
	}, endpoint.GetMetadata())

	assert.Equal(t, uint64(1), totalObservations(t), "one outcome per request")
	assert.Equal(t, uint64(1), localityOutcomeCount(t, "observed", localityDecodePod, localityTargetModel))
	count, sum := histogramObservation(t, "llm_d_epp_cache_locality_actual_fraction", localityDecodePod, localityTargetModel)
	require.Equal(t, uint64(1), count, "the actual fraction is observed once, from the final usage")
	assert.InDelta(t, 0.6, sum, 1e-9)

	// A duplicated terminal delivery (the context is gone after the first) must
	// not double count.
	router.ResponseBody(t.Context(), request, &requestcontrol.Response{
		EndOfStream: true,
		Usage:       usage(60, 100, true),
	}, endpoint.GetMetadata())
	assert.Equal(t, uint64(1), totalObservations(t), "a repeated end of stream does not record twice")
}

// TestProduce_RecordsPrefixScorePresence ties the presence flag to what the
// endpoint attribute actually carried, including the two cases that both store a
// 0.0 score but mean different things.
func TestProduce_RecordsPrefixScorePresence(t *testing.T) {
	setupLocalityMetrics(t)
	router := createTestRouter()
	router.latencypredictor = new(mockPredictor)

	scored := createTestEndpoint(localityDecodePod, 0.5, 1, 0)
	scored.Put(router.prefixMatchDataKey, attrprefix.NewPrefixCacheMatchInfo(8, 10, 16))
	unscored := createTestEndpoint(localityOtherPod, 0.5, 1, 0)
	emptyDenominator := createTestEndpoint(localityPrefillPod, 0.5, 1, 0)
	emptyDenominator.Put(router.prefixMatchDataKey, attrprefix.NewPrefixCacheMatchInfo(0, 0, 16))

	request := createTestInferenceRequest("loc-produce", 100, 50)
	endpoints := []fwksched.Endpoint{scored, unscored, emptyDenominator}

	require.NoError(t, router.Produce(t.Context(), request, endpoints))

	retrieved, err := router.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)

	assert.True(t, retrieved.prefixCacheScoreKnownForEndpoints[localityDecodePod])
	assert.InDelta(t, 0.8, retrieved.prefixCacheScoresForEndpoints[localityDecodePod], 1e-9)

	assert.False(t, retrieved.prefixCacheScoreKnownForEndpoints[localityOtherPod],
		"an endpoint with no match attribute is unknown, not a predicted miss")
	assert.Zero(t, retrieved.prefixCacheScoresForEndpoints[localityOtherPod],
		"the score stays 0.0 so prediction features are unchanged")

	assert.False(t, retrieved.prefixCacheScoreKnownForEndpoints[localityPrefillPod],
		"a zero denominator is unknown, not a predicted miss")
	assert.Zero(t, retrieved.prefixCacheScoresForEndpoints[localityPrefillPod])
}
