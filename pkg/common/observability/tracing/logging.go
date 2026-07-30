/*
Copyright 2025 The llm-d Authors.

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

package tracing

import (
	"context"

	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// LogKeyTraceID is the structured-log field carrying the active trace ID.
	// It matches the W3C trace-context identifier used across the llm-d stack
	// so logs from every component can be queried uniformly by trace.
	LogKeyTraceID = "trace_id"

	// LogKeySpanID is the structured-log field carrying the active span ID.
	LogKeySpanID = "span_id"
)

// LoggerWithSpanContext enriches the logger stored in ctx with the trace_id and
// span_id of span, returning a context that carries the enriched logger.
//
// Call it immediately after starting a span. Downstream code that retrieves the
// logger via log.FromContext then emits log lines tagged with the active trace,
// so logs and traces can be correlated by trace_id. Because the span_id reflects
// the span just started, log lines belong to the span that produced them rather
// than to a parent span injected further up the call chain.
//
// If span has no valid span context - tracing is disabled, the span was not
// sampled, or it is a no-op span - ctx is returned unchanged.
func LoggerWithSpanContext(ctx context.Context, span trace.Span) context.Context {
	sc := span.SpanContext()
	if !sc.IsValid() {
		return ctx
	}

	logger := log.FromContext(ctx).WithValues(
		LogKeyTraceID, sc.TraceID().String(),
		LogKeySpanID, sc.SpanID().String(),
	)
	return log.IntoContext(ctx, logger)
}
