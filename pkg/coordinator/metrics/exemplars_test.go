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

package metrics

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
)

const stepDurationMetric = LLMDRouterCoordinatorSubsystem + "_step_duration_seconds"

// recordingSpan returns a context with a real SDK span, like the one the
// pipeline starts per step.
func recordingSpan(t *testing.T) (context.Context, trace.Span) {
	t.Helper()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	t.Cleanup(func() { require.NoError(t, tp.Shutdown(context.Background())) })
	return tp.Tracer("coordinator-exemplar-test").Start(context.Background(), "step")
}

// unsampledSpanContext returns a context carrying a valid but unsampled trace.
func unsampledSpanContext(t *testing.T) context.Context {
	t.Helper()
	traceID, err := trace.TraceIDFromHex("4bf92f3577b34da6a3ce929d0e0e4736")
	require.NoError(t, err)
	spanID, err := trace.SpanIDFromHex("0102030405060708")
	require.NoError(t, err)
	return trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: traceID,
		SpanID:  spanID,
	}))
}

// The step span is still recording when the duration is observed, so the
// exemplar carries the span ID as well as the trace ID and a backend can open
// the exact span that produced the observation.
func TestRecordStepDurationAttachesTraceExemplar(t *testing.T) {
	reg := metricstest.NewRegistry(t, Register, Reset)

	ctx, span := recordingSpan(t)
	defer span.End()
	RecordStepDuration(ctx, "render", 50*time.Millisecond)

	exemplars := metricstest.HistogramExemplarLabels(t, reg, stepDurationMetric, map[string]string{"step": "render"})
	require.Len(t, exemplars, 1, "a sampled step must attach exactly one exemplar")

	sc := span.SpanContext()
	require.Equal(t, map[string]string{
		metricsutil.ExemplarTraceIDLabel: sc.TraceID().String(),
		metricsutil.ExemplarSpanIDLabel:  sc.SpanID().String(),
	}, exemplars[0])
}

// An unsampled trace was never exported, so an exemplar would link to a trace
// that does not exist. The observation itself must still be recorded.
func TestRecordStepDurationSkipsExemplarForUnsampledSpan(t *testing.T) {
	reg := metricstest.NewRegistry(t, Register, Reset)

	RecordStepDuration(unsampledSpanContext(t), "render", 50*time.Millisecond)

	require.Empty(t, metricstest.HistogramExemplarLabels(t, reg, stepDurationMetric, map[string]string{"step": "render"}))
	require.Equal(t, uint64(1), metricstest.HistogramCount(t, reg, stepDurationMetric, map[string]string{"step": "render"}),
		"the duration must be observed whether or not the trace is sampled")
}

// A step recorded outside any trace still lands in the histogram.
func TestRecordStepDurationWithoutSpanStillObserves(t *testing.T) {
	reg := metricstest.NewRegistry(t, Register, Reset)

	RecordStepDuration(context.Background(), "render", 50*time.Millisecond)

	require.Empty(t, metricstest.HistogramExemplarLabels(t, reg, stepDurationMetric, map[string]string{"step": "render"}))
	require.Equal(t, uint64(1), metricstest.HistogramCount(t, reg, stepDurationMetric, map[string]string{"step": "render"}))
}
