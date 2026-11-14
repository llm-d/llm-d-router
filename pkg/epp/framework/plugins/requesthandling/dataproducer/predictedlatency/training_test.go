/*
Copyright 2025 The Kubernetes Authors.
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

package predictedlatency

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"

	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestBulkPredict(t *testing.T) {
	predictor := &mockPredictor{predictions: map[string]*latencypredictor.PredictionResponse{
		"0.5": {TTFT: 0.5, TPOT: 0.03}, "0.6": {TTFT: 0.6, TPOT: 0.04},
	}}
	requests := []latencypredictor.PredictionRequest{{KVCachePercentage: 0.5}, {KVCachePercentage: 0.6}}
	results, err := bulkPredict(context.Background(), "test-plugin", "test-type", nil, predictor, requests)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, 0.5, results[0].TTFT)
	assert.Equal(t, 0.04, results[1].TPOT)
	assert.Equal(t, requests, predictor.capturedBulkStrictRequests)
}

func TestBulkPredict_Error(t *testing.T) {
	predictor := &mockPredictor{err: errors.New("prediction failed")}
	results, err := bulkPredict(context.Background(), "test-plugin", "test-type", nil, predictor, []latencypredictor.PredictionRequest{{}})
	require.Error(t, err)
	assert.Nil(t, results)
}

func TestBulkPredict_Empty(t *testing.T) {
	results, err := bulkPredict(context.Background(), "test-plugin", "test-type", nil, nil, nil)
	require.NoError(t, err)
	assert.Empty(t, results)
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
		plCtx,
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5, WaitingQueueSize: 1, RunningRequestsSize: 1},
		endpoint.GetMetadata(),
		time.Now(),
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
		plCtx,
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5},
		endpoint.GetMetadata(),
		time.Now(),
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
		plCtx,
		&fwkdl.Metrics{KVCacheUsagePercent: 0.4},
		prefill.GetMetadata(),
		time.Now(),
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
	response := &fwkrh.Response{EndOfStream: true}
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	plCtx := newPredictedLatencyContext(request)
	plCtx.targetMetadata = endpoint.GetMetadata()
	plCtx.schedulingResult = schedulingResult
	plCtx.targetModel = request.TargetModel
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
	response := &fwkrh.Response{EndOfStream: true}

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
	plCtx.targetModel = request.TargetModel
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
	response := &fwkrh.Response{EndOfStream: true}
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	plCtx := newPredictedLatencyContext(request)
	plCtx.targetMetadata = endpoint.GetMetadata()
	plCtx.schedulingResult = schedulingResult
	plCtx.targetModel = request.TargetModel
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

// TestBulkPredictWithMetrics_PredictionDurationModelLabels pins which model name
// lands on which label of the prediction-duration metrics. The incoming model
// belongs on model_name and the post-rewrite target on target_model_name, as
// every other recorder in this package does it.
func TestBulkPredictWithMetrics_PredictionDurationModelLabels(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))

	mp := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
		},
	}
	plCtx := &predictedLatencyCtx{
		targetModel:       "target-model",
		incomingModelName: "incoming-model",
	}

	_, err := bulkPredict(context.Background(), "test-plugin", "test-type", plCtx, mp,
		[]latencypredictor.PredictionRequest{{KVCachePercentage: 0.5}})
	require.NoError(t, err)

	families, err := registry.Gather()
	require.NoError(t, err)

	for _, name := range []string{
		"llm_d_epp_request_ttft_prediction_duration_seconds",
		"llm_d_epp_request_tpot_prediction_duration_seconds",
	} {
		labels := labelsForFamily(t, families, name)
		assert.Equal(t, "incoming-model", labels["model_name"],
			"%s: model_name must carry the incoming model", name)
		assert.Equal(t, "target-model", labels["target_model_name"],
			"%s: target_model_name must carry the target model", name)
	}
}

// labelsForFamily returns the label set of the single sample in the named family.
func labelsForFamily(t *testing.T, families []*dto.MetricFamily, name string) map[string]string {
	t.Helper()
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		require.Len(t, family.GetMetric(), 1, "%s: expected exactly one sample", name)
		labels := map[string]string{}
		for _, pair := range family.GetMetric()[0].GetLabel() {
			labels[pair.GetName()] = pair.GetValue()
		}
		return labels
	}
	t.Fatalf("metric family %q not found", name)
	return nil
}
