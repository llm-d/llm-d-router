/*
Copyright 2025 The Kubernetes Authors.

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

package logging

import (
	"context"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/trace"
)

// OpenTelemetry-aligned field names for correlating structured logs with traces.
// https://opentelemetry.io/docs/specs/otel/compatibility/logging_trace_context/
const (
	TraceIDKey = "trace_id"
	SpanIDKey  = "span_id"
)

// WithTrace returns a logger enriched with trace_id and span_id from the active
// span in ctx. When no valid span is present, logger is returned unchanged.
func WithTrace(ctx context.Context, logger logr.Logger) logr.Logger {
	sc := trace.SpanFromContext(ctx).SpanContext()
	if !sc.IsValid() {
		return logger
	}
	return logger.WithValues(
		TraceIDKey, sc.TraceID().String(),
		SpanIDKey, sc.SpanID().String(),
	)
}
