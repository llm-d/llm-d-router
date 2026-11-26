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

package predictedlatency

import (
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
)

// modelLabeledVecs are the plugin's metrics that carry model_name and target_model_name.
var modelLabeledVecs = map[string]interface {
	prometheus.Collector
	DeletePartialMatch(prometheus.Labels) int
}{
	"inference_request_metric":                 llmdInferenceGauges,
	"request_predicted_ttft_seconds":           llmdRequestPredictedTTFT,
	"request_ttft_prediction_duration_seconds": llmdRequestTTFTPredictionDuration,
	"request_predicted_tpot_seconds":           llmdRequestPredictedTPOT,
	"request_tpot_prediction_duration_seconds": llmdRequestTPOTPredictionDuration,
	"request_slo_violation_total":              llmdSloViolationCounter,
}

// seriesFor returns the label sets of the series c holds for pluginName, so a test
// sees only what it recorded even when other tests write to the same vectors.
func seriesFor(t *testing.T, c prometheus.Collector, pluginName string) []map[string]string {
	t.Helper()
	registry := prometheus.NewRegistry()
	require.NoError(t, registry.Register(c))
	families, err := registry.Gather()
	require.NoError(t, err)
	var out []map[string]string
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["plugin_name"] == pluginName {
				out = append(out, labels)
			}
		}
	}
	return out
}

func deleteSeriesFor(pluginName string) {
	for _, vec := range modelLabeledVecs {
		vec.DeletePartialMatch(prometheus.Labels{"plugin_name": pluginName})
	}
}

// Both model labels come from the request body when no InferenceModelRewrite rule
// matches, so a client sending made-up model names must not get a series per name.
// Not parallel: it swaps the boundModelName hook that every recorder reads.
func TestRecordersBoundModelLabels(t *testing.T) {
	const (
		testCap    = 3
		pluginName = "bound-model-labels"
	)
	old := boundModelName
	boundModelName = metricsutil.NewBoundedLabel(testCap).BoundWithOverflow
	t.Cleanup(func() {
		boundModelName = old
		deleteSeriesFor(pluginName)
	})

	ctx := t.Context()
	for i := range 100 {
		model, target := fmt.Sprintf("model-%d", i), fmt.Sprintf("target-%d", i)
		recordRequestTTFT(ctx, pluginName, "test-type", model, target, 0.5)
		recordRequestTTFTWithSLO(ctx, pluginName, "test-type", model, target, 2, 1)
		recordRequestPredictedTTFT(ctx, pluginName, "test-type", model, target, 0.4)
		recordRequestTTFTPredictionDuration(ctx, pluginName, "test-type", model, target, 0.1)
		recordRequestTPOT(ctx, pluginName, "test-type", model, target, 0.05)
		recordRequestTPOTWithSLO(ctx, pluginName, "test-type", model, target, 3, 1)
		recordRequestPredictedTPOT(ctx, pluginName, "test-type", model, target, 0.04)
		recordRequestTPOTPredictionDuration(ctx, pluginName, "test-type", model, target, 0.2)
	}

	for name, vec := range modelLabeledVecs {
		values := map[string]bool{}
		for _, labels := range seriesFor(t, vec, pluginName) {
			values[labels["model_name"]] = true
			values[labels["target_model_name"]] = true
		}
		if vec == llmdInferenceGauges {
			// Set-value gauges skip overflowed names instead of sharing one series.
			assert.Falsef(t, values[metricsutil.OverflowValue], "%s: a set-value gauge must not write an %q series", name, metricsutil.OverflowValue)
			assert.LessOrEqualf(t, len(values), testCap, "%s: only names under the cap may get gauge series", name)
			continue
		}
		// model_name and target_model_name share one cap, as on the core request metrics.
		assert.Equalf(t, testCap+1, len(values), "%s: 200 distinct model names must fold to cap+overflow label values", name)
		assert.Truef(t, values[metricsutil.OverflowValue], "%s: names past the cap must report as %q", name, metricsutil.OverflowValue)
	}
}

// An overflowed model must not overwrite the SLO violation flag of another model:
// before, a fast request from one overflowed model reset the shared "other" gauge
// to 0 while the violation counter still said 1.
func TestSLOViolationGaugeSkipsOverflowedModels(t *testing.T) {
	const pluginName = "slo-gauge-overflow"
	old := boundModelName
	boundModelName = metricsutil.NewBoundedLabel(2).BoundWithOverflow
	t.Cleanup(func() {
		boundModelName = old
		deleteSeriesFor(pluginName)
	})

	ctx := t.Context()
	// model-a and target-a fill the cap; the next two models overflow.
	recordRequestTTFTWithSLO(ctx, pluginName, "test-type", "model-a", "target-a", 2, 1)
	recordRequestTTFTWithSLO(ctx, pluginName, "test-type", "model-b", "target-b", 2, 1)
	recordRequestTTFTWithSLO(ctx, pluginName, "test-type", "model-c", "target-c", 0.5, 1)

	for _, labels := range seriesFor(t, llmdInferenceGauges, pluginName) {
		assert.NotEqual(t, metricsutil.OverflowValue, labels["model_name"], "overflowed models must not get a gauge series")
		assert.NotEqual(t, metricsutil.OverflowValue, labels["target_model_name"], "overflowed models must not get a gauge series")
	}
	counted := 0
	for _, labels := range seriesFor(t, llmdSloViolationCounter, pluginName) {
		if labels["model_name"] == metricsutil.OverflowValue {
			counted++
		}
	}
	assert.Equal(t, 1, counted, "the violation counter still folds overflowed models into %q", metricsutil.OverflowValue)
}

// model_name is the model the client asked for, the same value the core request
// metrics put on model_name. The director passes it in the request body, so the
// request context has to take it from there.
func TestRequestMetricsLabelIncomingModel(t *testing.T) {
	const pluginName = "incoming-model-label"
	predictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{"0.5": {TTFT: 0.5, TPOT: 0.03}},
	}
	pl := NewPredictedLatency(pluginName, DefaultConfig, predictor)
	t.Cleanup(func() { deleteSeriesFor(pluginName) })

	ctx := t.Context()
	endpoint := createTestEndpoint("pod-a", 0.5, 0, 0)
	request := createTestInferenceRequest("incoming-model-label", 50, 0)
	request.Body.Model = "client-model"
	request.TargetModel = "served-model"

	require.NoError(t, pl.Produce(ctx, request, []fwksched.Endpoint{endpoint}))
	require.NoError(t, pl.PreRequest(ctx, request, createTestSchedulingResult(endpoint.GetMetadata())))
	plCtx, err := pl.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	// Backdate dispatch so the TTFT is a whole number of milliseconds above the 50 ms SLO.
	plCtx.requestReceivedTimestamp = time.Now().Add(-100 * time.Millisecond)
	pl.ResponseBody(ctx, request, &fwkrh.Response{EndOfStream: true}, endpoint.GetMetadata())

	// One recorder from each call site: Produce's prediction and ResponseBody's observation.
	require.NotEmpty(t, seriesFor(t, llmdRequestTTFTPredictionDuration, pluginName), "Produce must record the prediction duration")
	require.NotEmpty(t, seriesFor(t, llmdSloViolationCounter, pluginName), "ResponseBody must record the TTFT SLO violation")
	for name, vec := range modelLabeledVecs {
		for _, labels := range seriesFor(t, vec, pluginName) {
			if !assert.Equalf(t, "client-model", labels["model_name"], "%s: model_name must carry the model from the request body", name) ||
				!assert.Equalf(t, "served-model", labels["target_model_name"], "%s: target_model_name must carry the target model", name) {
				break
			}
		}
	}
}
