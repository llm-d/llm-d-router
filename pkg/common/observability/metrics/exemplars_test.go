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

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// newHistogram returns a bare histogram to observe into, so these tests assert
// exemplar behavior without depending on a registered metric's labels.
func newHistogram() prometheus.Histogram {
	return prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "test_exemplar_histogram",
		Buckets: []float64{0.1, 1, 10},
	})
}

// collectExemplars returns the exemplar attached to each bucket that carries one.
func collectExemplars(t *testing.T, h prometheus.Histogram) []*dto.Exemplar {
	t.Helper()
	var m dto.Metric
	require.NoError(t, h.(prometheus.Metric).Write(&m))
	require.NotNil(t, m.Histogram)

	var found []*dto.Exemplar
	for _, bucket := range m.Histogram.Bucket {
		if bucket.Exemplar != nil {
			found = append(found, bucket.Exemplar)
		}
	}
	return found
}

func ctxWithSpan(t *testing.T, sampled bool) context.Context {
	t.Helper()
	traceID, err := trace.TraceIDFromHex("4bf92f3577b34da6a3ce929d0e0e4736")
	require.NoError(t, err)
	spanID, err := trace.SpanIDFromHex("0102030405060708")
	require.NoError(t, err)

	var flags trace.TraceFlags
	if sampled {
		flags = trace.FlagsSampled
	}
	return trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: flags,
	}))
}

// ctxWithRecordingSpan returns a context with a real SDK span, like the one a
// component starts per request.
func ctxWithRecordingSpan(t *testing.T) (context.Context, trace.Span) {
	t.Helper()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	t.Cleanup(func() { require.NoError(t, tp.Shutdown(context.Background())) })
	return tp.Tracer("exemplar-test").Start(context.Background(), "request")
}

// exemplarLabels returns an exemplar's labels as a map.
func exemplarLabels(e *dto.Exemplar) map[string]string {
	labels := map[string]string{}
	for _, l := range e.Label {
		labels[l.GetName()] = l.GetValue()
	}
	return labels
}

func TestObserveWithTraceExemplar_RecordingSpanAttachesBothIDs(t *testing.T) {
	h := newHistogram()

	ctx, span := ctxWithRecordingSpan(t)
	defer span.End()
	ObserveWithTraceExemplar(ctx, h, 0.5)

	exemplars := collectExemplars(t, h)
	require.Len(t, exemplars, 1, "a sampled span should attach exactly one exemplar")

	sc := span.SpanContext()
	labels := exemplarLabels(exemplars[0])
	require.Equal(t, map[string]string{
		ExemplarTraceIDLabel: sc.TraceID().String(),
		ExemplarSpanIDLabel:  sc.SpanID().String(),
	}, labels)

	// OpenMetrics caps an exemplar's whole label set at 128 runes; exceeding it
	// makes client_golang reject the observation at runtime.
	runes := 0
	for name, value := range labels {
		runes += len([]rune(name)) + len([]rune(value))
	}
	require.LessOrEqual(t, runes, prometheus.ExemplarMaxRunes)
	require.InDelta(t, 0.5, exemplars[0].GetValue(), 1e-9)
}

// A component with its own tracing off still sees the caller's sampled span.
// Only trace_id is attached, since the span belongs to the caller.
func TestObserveWithTraceExemplar_PropagatedSpanAttachesTraceIDOnly(t *testing.T) {
	h := newHistogram()

	ObserveWithTraceExemplar(ctxWithSpan(t, true), h, 0.5)

	exemplars := collectExemplars(t, h)
	require.Len(t, exemplars, 1, "a sampled trace still links, even without a local span")
	require.Equal(t, map[string]string{
		ExemplarTraceIDLabel: "4bf92f3577b34da6a3ce929d0e0e4736",
	}, exemplarLabels(exemplars[0]), "the span ID would name a span this component never created")
}

func TestObserveWithTraceExemplar_UnsampledSpanAttachesNothing(t *testing.T) {
	h := newHistogram()

	// The span carries a valid trace ID, but no trace was exported for it, so an
	// exemplar would link a dashboard to a trace that does not exist.
	ObserveWithTraceExemplar(ctxWithSpan(t, false), h, 0.5)

	require.Empty(t, collectExemplars(t, h), "an unsampled span must not attach an exemplar")
}

func TestObserveWithTraceExemplar_NoSpanStillObserves(t *testing.T) {
	h := newHistogram()

	ObserveWithTraceExemplar(context.Background(), h, 0.5)

	require.Empty(t, collectExemplars(t, h), "no span means no exemplar")

	var m dto.Metric
	require.NoError(t, h.(prometheus.Metric).Write(&m))
	require.Equal(t, uint64(1), m.Histogram.GetSampleCount(), "the observation itself must still be recorded")
}

// A plain Observer that cannot carry exemplars still records the observation.
func TestObserveWithTraceExemplar_NonExemplarObserverStillObserves(t *testing.T) {
	observer := &countingObserver{}

	ctx, span := ctxWithRecordingSpan(t)
	defer span.End()
	ObserveWithTraceExemplar(ctx, observer, 0.5)

	require.Equal(t, 1, observer.calls, "a non-exemplar observer must still see the observation")
	require.InDelta(t, 0.5, observer.last, 1e-9)
}

// countingObserver implements prometheus.Observer and not ExemplarObserver.
type countingObserver struct {
	calls int
	last  float64
}

func (o *countingObserver) Observe(value float64) {
	o.calls++
	o.last = value
}
