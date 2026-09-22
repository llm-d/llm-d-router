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
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry))
	require.True(t, recordRequestTTFT(t.Context(), "test-plugin", "test-type", "model", "target", 0.5))
	require.True(t, recordRequestPredictedTTFT(t.Context(), "test-plugin", "test-type", "model", "target", 0.4))
	require.True(t, recordRequestTTFTPredictionDuration(t.Context(), "test-plugin", "test-type", "model", "target", 0.1))
	require.True(t, recordRequestTTFTWithSLO(t.Context(), "test-plugin", "test-type", "model", "target", 2, 1))
	require.True(t, recordRequestTPOT(t.Context(), "test-plugin", "test-type", "model", "target", 0.05))
	require.True(t, recordRequestPredictedTPOT(t.Context(), "test-plugin", "test-type", "model", "target", 0.04))
	require.True(t, recordRequestTPOTPredictionDuration(t.Context(), "test-plugin", "test-type", "model", "target", 0.2))
	require.True(t, recordRequestTPOTWithSLO(t.Context(), "test-plugin", "test-type", "model", "target", 3, 1))

	families, err := registry.Gather()
	require.NoError(t, err)
	wantFamilies := map[string]bool{
		"llm_d_epp_inference_request_metric":                 false,
		"llm_d_epp_request_predicted_ttft_seconds":           false,
		"llm_d_epp_request_ttft_prediction_duration_seconds": false,
		"llm_d_epp_request_predicted_tpot_seconds":           false,
		"llm_d_epp_request_tpot_prediction_duration_seconds": false,
		"llm_d_epp_request_slo_violation_total":              false,
	}
	for _, family := range families {
		require.NotContains(t, family.GetName(), "inference_objective_")
		if _, found := wantFamilies[family.GetName()]; found {
			wantFamilies[family.GetName()] = true
		}
	}
	for family, found := range wantFamilies {
		require.Truef(t, found, "current metric family %q must be registered", family)
	}
}

func TestRecordRequestLatencyMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)
	ctx := t.Context()

	require.True(t, recordRequestTTFT(ctx, "test-plugin", "test-type", "model", "target", 0.5))
	require.True(t, recordRequestPredictedTTFT(ctx, "test-plugin", "test-type", "model", "target", 0.4))
	require.True(t, recordRequestTTFTPredictionDuration(ctx, "test-plugin", "test-type", "model", "target", 0.1))
	require.True(t, recordRequestTTFTWithSLO(ctx, "test-plugin", "test-type", "model", "target", 2, 1))
	require.True(t, recordRequestTPOT(ctx, "test-plugin", "test-type", "model", "target", 0.05))
	require.True(t, recordRequestPredictedTPOT(ctx, "test-plugin", "test-type", "model", "target", 0.04))
	require.True(t, recordRequestTPOTPredictionDuration(ctx, "test-plugin", "test-type", "model", "target", 0.2))
	require.True(t, recordRequestTPOTWithSLO(ctx, "test-plugin", "test-type", "model", "target", 3, 1))

	llmdPredictedTtft, err := getHistogram(llmdRequestPredictedTTFT, "test-plugin", "test-type", "model", "target")
	require.NoError(t, err)
	require.Equal(t, uint64(1), llmdPredictedTtft.GetSampleCount())
	require.Equal(t, 0.4, llmdPredictedTtft.GetSampleSum())

	llmdTtftDuration, err := getHistogram(llmdRequestTTFTPredictionDuration, "test-plugin", "test-type", "model", "target")
	require.NoError(t, err)
	require.Equal(t, uint64(1), llmdTtftDuration.GetSampleCount())
	require.Equal(t, 0.1, llmdTtftDuration.GetSampleSum())

	llmdPredictedTpot, err := getHistogram(llmdRequestPredictedTPOT, "test-plugin", "test-type", "model", "target")
	require.NoError(t, err)
	require.Equal(t, uint64(1), llmdPredictedTpot.GetSampleCount())
	require.Equal(t, 0.04, llmdPredictedTpot.GetSampleSum())

	llmdTpotDuration, err := getHistogram(llmdRequestTPOTPredictionDuration, "test-plugin", "test-type", "model", "target")
	require.NoError(t, err)
	require.Equal(t, uint64(1), llmdTpotDuration.GetSampleCount())
	require.Equal(t, 0.2, llmdTpotDuration.GetSampleSum())

	require.Equal(t, 0.5, testutil.ToFloat64(llmdInferenceGauges.WithLabelValues("test-plugin", "test-type", "model", "target", typeTTFT)))
	require.Equal(t, 0.05, testutil.ToFloat64(llmdInferenceGauges.WithLabelValues("test-plugin", "test-type", "model", "target", typeTPOT)))
	require.Equal(t, 0.4, testutil.ToFloat64(llmdInferenceGauges.WithLabelValues("test-plugin", "test-type", "model", "target", typePredictedTTFT)))
	require.Equal(t, 0.1, testutil.ToFloat64(llmdInferenceGauges.WithLabelValues("test-plugin", "test-type", "model", "target", typeTTFTPredictionDuration)))
	require.Equal(t, 0.04, testutil.ToFloat64(llmdInferenceGauges.WithLabelValues("test-plugin", "test-type", "model", "target", typePredictedTPOT)))
	require.Equal(t, 0.2, testutil.ToFloat64(llmdInferenceGauges.WithLabelValues("test-plugin", "test-type", "model", "target", typeTPOTPredictionDuration)))
	require.Equal(t, float64(1), testutil.ToFloat64(llmdSloViolationCounter.WithLabelValues("test-plugin", "test-type", "model", "target", typeTTFT)))
	require.Equal(t, float64(1), testutil.ToFloat64(llmdSloViolationCounter.WithLabelValues("test-plugin", "test-type", "model", "target", typeTPOT)))
}

func TestRecordRequestLatencyMetricsRejectNegativeValues(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)
	ctx := t.Context()

	require.False(t, recordRequestTTFT(ctx, "test-plugin", "test-type", "model", "target", -1))
	require.False(t, recordRequestPredictedTTFT(ctx, "test-plugin", "test-type", "model", "target", -1))
	require.False(t, recordRequestTTFTPredictionDuration(ctx, "test-plugin", "test-type", "model", "target", -1))
	require.False(t, recordRequestTTFTWithSLO(ctx, "test-plugin", "test-type", "model", "target", -1, 1))
	require.False(t, recordRequestTPOT(ctx, "test-plugin", "test-type", "model", "target", -1))
	require.False(t, recordRequestPredictedTPOT(ctx, "test-plugin", "test-type", "model", "target", -1))
	require.False(t, recordRequestTPOTPredictionDuration(ctx, "test-plugin", "test-type", "model", "target", -1))
	require.False(t, recordRequestTPOTWithSLO(ctx, "test-plugin", "test-type", "model", "target", -1, 1))
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
	llmdInferenceGauges.Reset()
	llmdRequestPredictedTTFT.Reset()
	llmdRequestTTFTPredictionDuration.Reset()
	llmdRequestPredictedTPOT.Reset()
	llmdRequestTPOTPredictionDuration.Reset()
	llmdSloViolationCounter.Reset()
}
