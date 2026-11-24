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
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
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

const (
	tpotRangeLoSeconds = 0.005
	tpotRangeHiSeconds = 0.05
	tpotMaxGapSeconds  = 0.005
	tpotFloatSlack     = 1e-9
)

// checkTPOTBucketSpacing validates edge ordering and that no bucket overlapping
// the 5-50 ms range is wider than 5 ms.
func checkTPOTBucketSpacing(edges []float64) error {
	for i, e := range edges {
		if math.IsNaN(e) || math.IsInf(e, 0) || e <= 0 {
			return fmt.Errorf("edge %d (%v) must be finite and positive", i, e)
		}
		if i > 0 && e <= edges[i-1] {
			return fmt.Errorf("edge %d (%v) must be greater than edge %d (%v)", i, e, i-1, edges[i-1])
		}
	}
	for i := 1; i < len(edges); i++ {
		lo, hi := edges[i-1], edges[i]
		overlaps := lo < tpotRangeHiSeconds-tpotFloatSlack && hi > tpotRangeLoSeconds+tpotFloatSlack
		if overlaps && hi-lo > tpotMaxGapSeconds+tpotFloatSlack {
			return fmt.Errorf("bucket (%v, %v] is wider than %v inside the 5-50 ms range", lo, hi, tpotMaxGapSeconds)
		}
	}
	return nil
}

func TestTPOTBucketsSpacing(t *testing.T) {
	require.NoError(t, checkTPOTBucketSpacing(tpotBuckets))
}

func TestTPOTBucketsSpacingRejectsCoarseEdges(t *testing.T) {
	coarse := []float64{0.0005, 0.00205, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1}
	require.Error(t, checkTPOTBucketSpacing(coarse))
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
	llmdRequestPredictionFailures.Reset()
}

func TestPredictedLatencyHistogramsTraceExemplars(t *testing.T) {
	t.Cleanup(resetMetrics)

	provider := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	recordingCtx, span := provider.Tracer("predicted-latency-test").Start(context.Background(), "request")
	defer span.End()

	propagatedCtx := trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    span.SpanContext().TraceID(),
		SpanID:     span.SpanContext().SpanID(),
		TraceFlags: trace.FlagsSampled,
		Remote:     true,
	}))
	unsampledCtx := trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: span.SpanContext().TraceID(),
		SpanID:  span.SpanContext().SpanID(),
	}))

	const pluginName, pluginType, model, target = "exemplar-test", "test-type", "model", "target"
	histograms := []struct {
		name      string
		histogram *prometheus.HistogramVec
		record    func(ctx context.Context) bool
	}{
		{"predicted ttft", llmdRequestPredictedTTFT, func(ctx context.Context) bool {
			return recordRequestPredictedTTFT(ctx, pluginName, pluginType, model, target, 0.4)
		}},
		{"ttft prediction duration", llmdRequestTTFTPredictionDuration, func(ctx context.Context) bool {
			return recordRequestTTFTPredictionDuration(ctx, pluginName, pluginType, model, target, 0.01)
		}},
		{"predicted tpot", llmdRequestPredictedTPOT, func(ctx context.Context) bool {
			return recordRequestPredictedTPOT(ctx, pluginName, pluginType, model, target, 0.04)
		}},
		{"tpot prediction duration", llmdRequestTPOTPredictionDuration, func(ctx context.Context) bool {
			return recordRequestTPOTPredictionDuration(ctx, pluginName, pluginType, model, target, 0.01)
		}},
	}
	contexts := []struct {
		name                string
		ctx                 context.Context
		wantTrace, wantSpan bool
	}{
		{"recording", recordingCtx, true, true},
		{"propagated", propagatedCtx, true, false},
		{"unsampled", unsampledCtx, false, false},
		{"no span", context.Background(), false, false},
	}

	for _, hist := range histograms {
		for _, ctxCase := range contexts {
			t.Run(hist.name+"/"+ctxCase.name, func(t *testing.T) {
				hist.histogram.Reset()
				require.True(t, hist.record(ctxCase.ctx))

				observed, err := getHistogram(hist.histogram, pluginName, pluginType, model, target)
				require.NoError(t, err)
				require.Equal(t, uint64(1), observed.GetSampleCount())
				var exemplar *dto.Exemplar
				for _, bucket := range observed.GetBucket() {
					if bucket.GetExemplar() != nil {
						exemplar = bucket.GetExemplar()
						break
					}
				}
				if !ctxCase.wantTrace {
					require.Nil(t, exemplar)
					return
				}
				require.NotNil(t, exemplar)
				labels := map[string]string{}
				for _, label := range exemplar.GetLabel() {
					labels[label.GetName()] = label.GetValue()
				}
				require.Equal(t, span.SpanContext().TraceID().String(), labels[metricsutil.ExemplarTraceIDLabel])
				if ctxCase.wantSpan {
					require.Equal(t, span.SpanContext().SpanID().String(), labels[metricsutil.ExemplarSpanIDLabel])
				} else {
					require.NotContains(t, labels, metricsutil.ExemplarSpanIDLabel)
				}

				registry := prometheus.NewRegistry()
				require.NoError(t, registry.Register(hist.histogram))
				req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
				req.Header.Set("Accept", "application/openmetrics-text;version=1.0.0")
				rec := httptest.NewRecorder()
				promhttp.HandlerFor(registry, promhttp.HandlerOpts{EnableOpenMetrics: true}).ServeHTTP(rec, req)
				require.Equal(t, http.StatusOK, rec.Code)
				require.Contains(t, rec.Header().Get("Content-Type"), "application/openmetrics-text")
				require.Contains(t, rec.Body.String(), `trace_id="`+span.SpanContext().TraceID().String()+`"`)
			})
		}
	}
}
