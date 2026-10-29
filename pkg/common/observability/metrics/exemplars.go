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

	"github.com/prometheus/client_golang/prometheus"
	"go.opentelemetry.io/otel/trace"
)

// Exemplar label names. Grafana links exemplars to traces by the trace_id label
// name, so don't rename it.
const (
	ExemplarTraceIDLabel = "trace_id"
	ExemplarSpanIDLabel  = "span_id"
)

// ObserveWithTraceExemplar records an observation and attaches the request's
// trace as a Prometheus exemplar, so a point on a latency graph links to its trace.
//
// Nothing is attached for unsampled spans: they have a trace ID, but no trace was
// exported, so the link would go nowhere.
//
// span_id is only added when the span is recording, meaning the observing
// component started it. With that component's tracing off, the context still
// holds the caller's span, which is not the component's own, so only trace_id is
// attached.
//
// Exemplars only reach the wire over OpenMetrics, so the component serving the
// observation must enable it on its /metrics handler.
func ObserveWithTraceExemplar(ctx context.Context, observer prometheus.Observer, value float64) {
	exemplarObserver, ok := observer.(prometheus.ExemplarObserver)
	if !ok {
		observer.Observe(value)
		return
	}

	span := trace.SpanFromContext(ctx)
	spanCtx := span.SpanContext()
	if !spanCtx.IsSampled() {
		observer.Observe(value)
		return
	}

	// trace_id + span_id is 63 runes, under OpenMetrics' 128-rune exemplar limit.
	labels := prometheus.Labels{ExemplarTraceIDLabel: spanCtx.TraceID().String()}
	if span.IsRecording() {
		labels[ExemplarSpanIDLabel] = spanCtx.SpanID().String()
	}
	exemplarObserver.ObserveWithExemplar(value, labels)
}
