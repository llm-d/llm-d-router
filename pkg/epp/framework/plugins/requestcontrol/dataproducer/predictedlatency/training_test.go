/*
Copyright 2025 The Kubernetes Authors.

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
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestBulkPredictWithMetrics(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
			"0.6": {TTFT: 0.6, TPOT: 0.04},
		},
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
		{KVCacheUsagePercent: 0.6},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			ID: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
		{
			ID: types.NamespacedName{Namespace: "default", Name: "pod2"},
		},
	}
	inputTokenLengths := []int{1, 1}
	generatedTokenCounts := []int{1, 1}
	prefixCacheScores := []float64{0.0, 0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil, nil, nil)

	assert.NoError(t, err)
	assert.Len(t, results, 2)
	assert.Equal(t, 0.5, results[0].TTFT)
	assert.Equal(t, 0.03, results[0].TPOT)
	assert.Equal(t, 0.6, results[1].TTFT)
	assert.Equal(t, 0.04, results[1].TPOT)
}

// TestBulkPredictWithMetrics_PropagatesInFlightOverrides verifies that the
// per-endpoint numRequestRunnings and prefillTokensInFlights slices are written
// onto the outgoing PredictionRequests, overriding the metrics-sourced defaults.
func TestBulkPredictWithMetrics_PropagatesInFlightOverrides(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
			"0.6": {TTFT: 0.6, TPOT: 0.04},
		},
	}

	// RunningRequestsSize is non-zero and distinct from the override values, so a
	// passing assertion proves the override replaced the metrics default.
	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5, RunningRequestsSize: 1},
		{KVCacheUsagePercent: 0.6, RunningRequestsSize: 2},
	}
	pods := []*fwkdl.EndpointMetadata{
		{ID: types.NamespacedName{Namespace: "default", Name: "pod1"}},
		{ID: types.NamespacedName{Namespace: "default", Name: "pod2"}},
	}
	inputTokenLengths := []int{1, 1}
	generatedTokenCounts := []int{1, 1}
	prefixCacheScores := []float64{0.0, 0.0}
	prefillTokensInFlights := []int64{100, 200}
	numRequestRunnings := []int{7, 13}

	_, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor,
		metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores,
		prefillTokensInFlights, numRequestRunnings, nil, nil)
	require.NoError(t, err)

	require.Len(t, mockPredictor.capturedBulkStrictRequests, 2)
	assert.Equal(t, 7, mockPredictor.capturedBulkStrictRequests[0].NumRequestRunning)
	assert.Equal(t, 13, mockPredictor.capturedBulkStrictRequests[1].NumRequestRunning)
	assert.Equal(t, int64(100), mockPredictor.capturedBulkStrictRequests[0].PrefillTokensInFlight)
	assert.Equal(t, int64(200), mockPredictor.capturedBulkStrictRequests[1].PrefillTokensInFlight)
}

// TestBulkPredictWithMetrics_PropagatesEncoderSizes verifies that the
// per-endpoint encoder-cache size slices are written onto the outgoing
// PredictionRequests.
func TestBulkPredictWithMetrics_PropagatesEncoderSizes(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
			"0.6": {TTFT: 0.6, TPOT: 0.04},
		},
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
		{KVCacheUsagePercent: 0.6},
	}
	pods := []*fwkdl.EndpointMetadata{
		{ID: types.NamespacedName{Namespace: "default", Name: "pod1"}},
		{ID: types.NamespacedName{Namespace: "default", Name: "pod2"}},
	}
	inputTokenLengths := []int{1, 1}
	generatedTokenCounts := []int{1, 1}
	prefixCacheScores := []float64{0.0, 0.0}
	encoderInputSizes := []int{3, 3}
	encoderMatchedSizes := []int{2, 0}

	_, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor,
		metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores,
		nil, nil, encoderInputSizes, encoderMatchedSizes)
	require.NoError(t, err)

	require.Len(t, mockPredictor.capturedBulkStrictRequests, 2)
	assert.Equal(t, 3, mockPredictor.capturedBulkStrictRequests[0].EncoderInputSize)
	assert.Equal(t, 2, mockPredictor.capturedBulkStrictRequests[0].EncoderMatchedSize)
	assert.Equal(t, 3, mockPredictor.capturedBulkStrictRequests[1].EncoderInputSize)
	assert.Equal(t, 0, mockPredictor.capturedBulkStrictRequests[1].EncoderMatchedSize)
}

// TestBulkPredictWithMetrics_ClampsMatchedWithoutInputSizes verifies that a
// matched-size slice without a corresponding input-size slice is clamped to
// the input size (0) instead of producing requests that fail validation.
func TestBulkPredictWithMetrics_ClampsMatchedWithoutInputSizes(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
		},
	}

	metricsStates := []*fwkdl.Metrics{{KVCacheUsagePercent: 0.5}}
	pods := []*fwkdl.EndpointMetadata{
		{ID: types.NamespacedName{Namespace: "default", Name: "pod1"}},
	}

	_, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor,
		metricsStates, "", pods, []int{1}, []int{1}, []float64{0.0},
		nil, nil, nil, []int{5})
	require.NoError(t, err)

	require.Len(t, mockPredictor.capturedBulkStrictRequests, 1)
	assert.Equal(t, 0, mockPredictor.capturedBulkStrictRequests[0].EncoderInputSize)
	assert.Equal(t, 0, mockPredictor.capturedBulkStrictRequests[0].EncoderMatchedSize)
}

func TestBuildPredictionRequestAndTrainingEntry_EncoderSizes(t *testing.T) {
	m := &fwkdl.Metrics{KVCacheUsagePercent: 0.5}
	pod := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod1"}}

	req := buildPredictionRequest("", pod, m, 10, 1, 0.0, 5, 4)
	assert.Equal(t, 5, req.EncoderInputSize)
	assert.Equal(t, 4, req.EncoderMatchedSize)

	entry := buildTrainingEntry("", pod, m, 10, 100, 0, time.Now(), 0, 0.0, 5, 4)
	assert.Equal(t, 5, entry.EncoderInputSize)
	assert.Equal(t, 4, entry.EncoderMatchedSize)
}

func TestBulkPredictWithMetrics_Error(t *testing.T) {
	mockPredictor := &mockPredictor{
		err: errors.New("prediction failed"),
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			ID: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	inputTokenLengths := []int{1}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil, nil, nil)

	assert.Error(t, err)
	assert.Nil(t, results)
}

func TestBulkPredictWithMetrics_InputMismatch(t *testing.T) {
	mockPredictor := &mockPredictor{}
	metricsStates := []*fwkdl.Metrics{{}}
	pods := []*fwkdl.EndpointMetadata{
		{
			ID: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	inputTokenLengths := []int{1, 1} // Mismatch length
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil, nil, nil)

	assert.Error(t, err)
	assert.Nil(t, results)
	assert.True(t, strings.Contains(err.Error(), "input slice lengths must match"))
}

func TestBulkPredictWithMetrics_WithPredictedLatencyCtx(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
		},
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			ID: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	inputTokenLengths := []int{1}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	plCtx := &predictedLatencyCtx{
		schedulingRequest: fwksched.InferenceRequest{
			TargetModel: "test-model",
		},
		incomingModelName: "incoming-model",
	}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", plCtx, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil, nil, nil)

	assert.NoError(t, err)
	assert.Len(t, results, 1)
	assert.Equal(t, 0.5, results[0].TTFT)
	assert.Equal(t, 0.03, results[0].TPOT)
}

func TestBulkPredictWithMetrics_ChatCompletionsInputTokenLength(t *testing.T) {
	mp := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
		},
	}

	metricsStates := []*fwkdl.Metrics{{KVCacheUsagePercent: 0.5}}
	pods := []*fwkdl.EndpointMetadata{
		{ID: types.NamespacedName{Namespace: "default", Name: "pod1"}},
	}

	inputTokenLengths := []int{2}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mp, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, []int64{0}, nil, nil, nil)

	assert.NoError(t, err)
	assert.Len(t, results, 1)
	assert.Equal(t, 0.5, results[0].TTFT)
}

func TestBulkPredictWithMetrics_NilMetricsState(t *testing.T) {
	mockPredictor := &mockPredictor{}
	metricsStates := []*fwkdl.Metrics{nil} // Nil metrics state
	pods := []*fwkdl.EndpointMetadata{
		{
			ID: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	inputTokenLengths := []int{1}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil, nil, nil)

	assert.Error(t, err)
	assert.Nil(t, results)
	assert.True(t, strings.Contains(err.Error(), "metrics state at index 0 cannot be nil"))
}

func TestPredictedLatencyMSPtr(t *testing.T) {
	assert.Nil(t, predictedLatencyMSPtr(0))
	assert.Nil(t, predictedLatencyMSPtr(-1))
	assert.Nil(t, predictedLatencyMSPtr(math.NaN()))
	assert.Nil(t, predictedLatencyMSPtr(math.Inf(1)))
	require.NotNil(t, predictedLatencyMSPtr(12.5))
	assert.Equal(t, 12.5, *predictedLatencyMSPtr(12.5))
}

func assertPredictedJSONFields(t *testing.T, entry latencypredictor.TrainingEntry, wantTTFT *float64, wantTPOT *float64) {
	t.Helper()
	raw, err := json.Marshal(entry)
	require.NoError(t, err)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(raw, &payload))

	if wantTTFT == nil {
		_, ok := payload["predicted_ttft_ms"]
		assert.False(t, ok, "predicted_ttft_ms should be omitted: %s", raw)
	} else {
		got, ok := payload["predicted_ttft_ms"].(float64)
		require.True(t, ok, "predicted_ttft_ms missing: %s", raw)
		assert.Equal(t, *wantTTFT, got)
	}
	if wantTPOT == nil {
		_, ok := payload["predicted_tpot_ms"]
		assert.False(t, ok, "predicted_tpot_ms should be omitted: %s", raw)
	} else {
		got, ok := payload["predicted_tpot_ms"].(float64)
		require.True(t, ok, "predicted_tpot_ms missing: %s", raw)
		assert.Equal(t, *wantTPOT, got)
	}
}

func TestRecordTTFTTrainingData_LookupSelectedPredictedTTFT(t *testing.T) {
	mp := &mockPredictor{}
	endpoint := createTestEndpoint("selected-pod", 0.5, 1, 1)
	request := createTestInferenceRequest("ttft-pred", 100, 50)
	plCtx := newPredictedLatencyContext(request)
	plCtx.ttft = 120
	plCtx.targetMetadata = endpoint.GetMetadata()
	plCtx.predictionsForScheduling = map[string]endpointPredictionResult{
		"selected-pod": {Endpoint: endpoint, TTFT: 80, TPOT: 10, TTFTValid: false, TPOTValid: false, IsValid: false},
		"other-pod":    {TTFT: 999, TPOT: 10, TTFTValid: true, TPOTValid: true, IsValid: true},
	}

	// Resolve selected prediction the same way PreRequest does. SLO-invalid
	// selected predictions must still train (drift needs those samples).
	processPreRequestForLatencyPrediction(context.Background(), plCtx)
	assert.Equal(t, 80.0, plCtx.predictedTTFT)

	recordTTFTTrainingData(
		context.Background(),
		mp,
		"",
		plCtx,
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5, WaitingQueueSize: 1, RunningRequestsSize: 1},
		endpoint.GetMetadata(),
		time.Now(),
		0.4,
		0,
	)

	require.Len(t, mp.capturedTrainingEntries, 1)
	entry := mp.capturedTrainingEntries[0]
	assert.Equal(t, 120.0, entry.ActualTTFT)
	assert.Equal(t, 0.0, entry.ActualTPOT)
	require.NotNil(t, entry.PredictedTTFT)
	assert.Equal(t, 80.0, *entry.PredictedTTFT)
	assert.Nil(t, entry.PredictedTPOT)
	want := 80.0
	assertPredictedJSONFields(t, entry, &want, nil)
}

func TestRecordTTFTTrainingData_OmitsPredictedWhenMissing(t *testing.T) {
	mp := &mockPredictor{}
	endpoint := createTestEndpoint("pod1", 0.5, 1, 1)
	request := createTestInferenceRequest("ttft-missing", 100, 50)
	plCtx := newPredictedLatencyContext(request)
	plCtx.ttft = 100
	plCtx.targetMetadata = endpoint.GetMetadata()
	plCtx.predictionsForScheduling = map[string]endpointPredictionResult{}

	processPreRequestForLatencyPrediction(context.Background(), plCtx)
	assert.Equal(t, 0.0, plCtx.predictedTTFT)

	recordTTFTTrainingData(
		context.Background(),
		mp,
		"",
		plCtx,
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5},
		endpoint.GetMetadata(),
		time.Now(),
		0,
		0,
	)

	require.Len(t, mp.capturedTrainingEntries, 1)
	assert.Nil(t, mp.capturedTrainingEntries[0].PredictedTTFT)
	assert.Nil(t, mp.capturedTrainingEntries[0].PredictedTPOT)
	assert.Equal(t, 100.0, mp.capturedTrainingEntries[0].ActualTTFT)
	assertPredictedJSONFields(t, mp.capturedTrainingEntries[0], nil, nil)
}

func TestRecordTTFTTrainingData_PrefillLookupSelectedPredictedTTFT(t *testing.T) {
	mp := &mockPredictor{}
	prefill := createTestEndpoint("prefill-pod", 0.4, 1, 0)
	decode := createTestEndpoint("decode-pod", 0.5, 1, 1)
	request := createTestInferenceRequest("ttft-pd", 100, 50)
	plCtx := newPredictedLatencyContext(request)
	plCtx.ttft = 150
	plCtx.targetMetadata = decode.GetMetadata()
	plCtx.prefillTargetMetadata = prefill.GetMetadata()
	plCtx.predictionsForScheduling = map[string]endpointPredictionResult{
		"prefill-pod": {Endpoint: prefill, TTFT: 95, TPOT: 0},
		"decode-pod":  {Endpoint: decode, TTFT: 40, TPOT: 12},
	}

	processPreRequestForLatencyPrediction(context.Background(), plCtx)
	assert.Equal(t, 95.0, plCtx.predictedTTFT, "TTFT must come from prefill, not decode")

	recordTTFTTrainingData(
		context.Background(),
		mp,
		"",
		plCtx,
		&fwkdl.Metrics{KVCacheUsagePercent: 0.4},
		prefill.GetMetadata(),
		time.Now(),
		0.2,
		0,
	)

	require.Len(t, mp.capturedTrainingEntries, 1)
	require.NotNil(t, mp.capturedTrainingEntries[0].PredictedTTFT)
	assert.Equal(t, 95.0, *mp.capturedTrainingEntries[0].PredictedTTFT)
	assert.Nil(t, mp.capturedTrainingEntries[0].PredictedTPOT)
	want := 95.0
	assertPredictedJSONFields(t, mp.capturedTrainingEntries[0], &want, nil)
}

func TestResponseBody_TPOTTrainingIncludesSelectedPredictedTPOT(t *testing.T) {
	router := createTestRouter()
	mp := &mockPredictor{}
	router.latencypredictor = mp

	endpoint := createTestEndpoint("decode-pod", 0.5, 1, 1)
	request := createTestInferenceRequest("tpot-pred", 100, 50)
	response := &requestcontrol.Response{EndOfStream: true}
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	plCtx := newPredictedLatencyContext(request)
	plCtx.targetMetadata = endpoint.GetMetadata()
	plCtx.schedulingResult = schedulingResult
	plCtx.schedulingRequest = *request
	plCtx.incomingModelName = testModelName
	plCtx.requestReceivedTimestamp = time.Now().Add(-200 * time.Millisecond)
	plCtx.ttft = 80
	plCtx.avgTPOT = 25
	plCtx.generatedTokenCount = 1 // keep pre-set avgTPOT; do not recompute
	plCtx.predictionsForScheduling = map[string]endpointPredictionResult{
		"decode-pod": {Endpoint: endpoint, TTFT: 85, TPOT: 22},
		"other-pod":  {TTFT: 10, TPOT: 999},
	}
	predictFirstTPOT(context.Background(), plCtx)
	assert.Equal(t, 22.0, plCtx.avgPredictedTPOT)

	plCtx.lastSeenMetrics["default"] = &fwkdl.Metrics{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}
	router.setPredictedLatencyContextForRequest(request, plCtx)

	queue := newRequestPriorityQueue()
	queue.Add(request.Headers[reqcommon.RequestIDHeaderKey], 50.0)
	router.runningRequestLists.Store(endpoint.GetMetadata().ID, queue)

	router.ResponseBody(context.Background(), request, response, endpoint.GetMetadata())

	require.NotEmpty(t, mp.capturedTrainingEntries)
	var tpotEntry *latencypredictor.TrainingEntry
	for i := range mp.capturedTrainingEntries {
		if mp.capturedTrainingEntries[i].ActualTPOT > 0 {
			tpotEntry = &mp.capturedTrainingEntries[i]
			break
		}
	}
	require.NotNil(t, tpotEntry, "expected a TPOT training entry")
	assert.Equal(t, 0.0, tpotEntry.ActualTTFT)
	require.NotNil(t, tpotEntry.PredictedTPOT)
	assert.Equal(t, 22.0, *tpotEntry.PredictedTPOT)
	assert.Nil(t, tpotEntry.PredictedTTFT)
	want := 22.0
	assertPredictedJSONFields(t, *tpotEntry, nil, &want)
}

func TestResponseBody_PDDecodeTPOTUsesSelectedDecodePrediction(t *testing.T) {
	router := createTestRouter()
	mp := &mockPredictor{}
	router.latencypredictor = mp

	prefill := createTestEndpoint("prefill-pod", 0.4, 1, 0)
	decodeSelected := createTestEndpoint("decode-selected", 0.5, 1, 1)
	decodeOther := createTestEndpoint("decode-other", 0.6, 1, 1)
	request := createTestInferenceRequest("tpot-pd", 100, 50)
	response := &requestcontrol.Response{EndOfStream: true}

	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{decodeSelected}},
			"prefill": {TargetEndpoints: []fwksched.Endpoint{prefill}},
		},
	}

	plCtx := newPredictedLatencyContext(request)
	plCtx.targetMetadata = decodeSelected.GetMetadata()
	plCtx.prefillTargetMetadata = prefill.GetMetadata()
	plCtx.schedulingResult = schedulingResult
	plCtx.schedulingRequest = *request
	plCtx.incomingModelName = testModelName
	plCtx.requestReceivedTimestamp = time.Now().Add(-200 * time.Millisecond)
	plCtx.ttft = 80
	plCtx.avgTPOT = 25
	plCtx.generatedTokenCount = 1
	plCtx.predictionsForScheduling = map[string]endpointPredictionResult{
		"prefill-pod":     {Endpoint: prefill, TTFT: 90, TPOT: 1},
		"decode-selected": {Endpoint: decodeSelected, TTFT: 40, TPOT: 18},
		"decode-other":    {Endpoint: decodeOther, TTFT: 35, TPOT: 777},
	}
	predictFirstTPOT(context.Background(), plCtx)
	assert.Equal(t, 18.0, plCtx.avgPredictedTPOT, "TPOT must come from selected decode, not prefill/other")

	plCtx.lastSeenMetrics["default"] = &fwkdl.Metrics{KVCacheUsagePercent: 0.5}
	plCtx.lastSeenMetrics["prefill"] = &fwkdl.Metrics{KVCacheUsagePercent: 0.4}
	router.setPredictedLatencyContextForRequest(request, plCtx)

	queue := newRequestPriorityQueue()
	queue.Add(request.Headers[reqcommon.RequestIDHeaderKey], 50.0)
	router.runningRequestLists.Store(decodeSelected.GetMetadata().ID, queue)

	router.ResponseBody(context.Background(), request, response, decodeSelected.GetMetadata())

	require.NotEmpty(t, mp.capturedTrainingEntries)
	var tpotEntry *latencypredictor.TrainingEntry
	for i := range mp.capturedTrainingEntries {
		if mp.capturedTrainingEntries[i].ActualTPOT > 0 {
			tpotEntry = &mp.capturedTrainingEntries[i]
			break
		}
	}
	require.NotNil(t, tpotEntry)
	require.NotNil(t, tpotEntry.PredictedTPOT)
	assert.Equal(t, 18.0, *tpotEntry.PredictedTPOT)
	assert.NotEqual(t, 1.0, *tpotEntry.PredictedTPOT)
	assert.NotEqual(t, 777.0, *tpotEntry.PredictedTPOT)
	want := 18.0
	assertPredictedJSONFields(t, *tpotEntry, nil, &want)
}

func TestResponseBody_TPOTTrainingOmitsPredictedWhenMissing(t *testing.T) {
	router := createTestRouter()
	mp := &mockPredictor{}
	router.latencypredictor = mp

	endpoint := createTestEndpoint("decode-pod", 0.5, 1, 1)
	request := createTestInferenceRequest("tpot-missing", 100, 50)
	response := &requestcontrol.Response{EndOfStream: true}
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	plCtx := newPredictedLatencyContext(request)
	plCtx.targetMetadata = endpoint.GetMetadata()
	plCtx.schedulingResult = schedulingResult
	plCtx.schedulingRequest = *request
	plCtx.incomingModelName = testModelName
	plCtx.requestReceivedTimestamp = time.Now().Add(-200 * time.Millisecond)
	plCtx.ttft = 80
	plCtx.avgTPOT = 25
	plCtx.generatedTokenCount = 1
	plCtx.avgPredictedTPOT = 0
	plCtx.lastSeenMetrics["default"] = &fwkdl.Metrics{KVCacheUsagePercent: 0.5}
	router.setPredictedLatencyContextForRequest(request, plCtx)

	queue := newRequestPriorityQueue()
	queue.Add(request.Headers[reqcommon.RequestIDHeaderKey], 50.0)
	router.runningRequestLists.Store(endpoint.GetMetadata().ID, queue)

	router.ResponseBody(context.Background(), request, response, endpoint.GetMetadata())

	require.NotEmpty(t, mp.capturedTrainingEntries)
	found := false
	for _, entry := range mp.capturedTrainingEntries {
		if entry.ActualTPOT > 0 {
			found = true
			assert.Nil(t, entry.PredictedTPOT)
			assert.Nil(t, entry.PredictedTTFT)
			assertPredictedJSONFields(t, entry, nil, nil)
		}
	}
	assert.True(t, found)
}
