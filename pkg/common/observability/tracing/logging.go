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

// correlatedKey marks a context whose logger already carries the correlation
// fields, so nested span boundaries do not append a second pair.
type correlatedKey struct{}

// LoggerWithSpanContext enriches the logger stored in ctx with the trace_id and
// span_id of span. Call it immediately after starting a span at a request entry
// point; downstream code reading the logger via log.FromContext then emits log
// lines tagged with the active trace.
//
// Enrichment happens once per context chain. logr accumulates values rather than
// replacing them, so calling this at a nested span boundary would repeat trace_id
// and span_id on every log line, and which pair a backend keeps for duplicate keys
// is undefined. The logged span_id is therefore the entry span of the request.
//
// The marker lives on the context while the fields live on the logger, so the two
// desync if a caller installs a logger not derived from log.FromContext of an
// already-correlated context. Derive from the context logger rather than building
// one from scratch.
func LoggerWithSpanContext(ctx context.Context, span trace.Span) context.Context {
	if ctx.Value(correlatedKey{}) != nil {
		return ctx
	}

	// An invalid span context means tracing is off or the span is a no-op.
	sc := span.SpanContext()
	if !sc.IsValid() {
		return ctx
	}

	logger := log.FromContext(ctx).WithValues(
		LogKeyTraceID, sc.TraceID().String(),
		LogKeySpanID, sc.SpanID().String(),
	)
	return log.IntoContext(context.WithValue(ctx, correlatedKey{}, struct{}{}), logger)
}
