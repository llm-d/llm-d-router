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
	"go.opentelemetry.io/otel/trace"
)

// newHistogram returns a bare histogram to observe into, so these tests assert
// exemplar behavior without depending on the registered metric's labels.
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

func TestObserveWithTraceExemplar_SampledSpanAttachesTraceID(t *testing.T) {
	h := newHistogram()

	observeWithTraceExemplar(ctxWithSpan(t, true), h, 0.5)

	exemplars := collectExemplars(t, h)
	require.Len(t, exemplars, 1, "a sampled span should attach exactly one exemplar")

	labels := map[string]string{}
	for _, l := range exemplars[0].Label {
		labels[l.GetName()] = l.GetValue()
	}
	require.Equal(t, map[string]string{
		"trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
		"span_id":  "0102030405060708",
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

func TestObserveWithTraceExemplar_UnsampledSpanAttachesNothing(t *testing.T) {
	h := newHistogram()

	// The span carries a valid trace ID, but no trace was exported for it, so an
	// exemplar would link a dashboard to a trace that does not exist.
	observeWithTraceExemplar(ctxWithSpan(t, false), h, 0.5)

	require.Empty(t, collectExemplars(t, h), "an unsampled span must not attach an exemplar")
}

func TestObserveWithTraceExemplar_NoSpanStillObserves(t *testing.T) {
	h := newHistogram()

	observeWithTraceExemplar(context.Background(), h, 0.5)

	require.Empty(t, collectExemplars(t, h), "no span means no exemplar")

	var m dto.Metric
	require.NoError(t, h.(prometheus.Metric).Write(&m))
	require.Equal(t, uint64(1), m.Histogram.GetSampleCount(), "the observation itself must still be recorded")
}
