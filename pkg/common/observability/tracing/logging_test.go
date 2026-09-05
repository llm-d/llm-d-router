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
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// captureLogger returns a logr.Logger that appends every emitted record's
// formatted key/values to *out, so tests can assert on what was logged.
func captureLogger(out *[]string) logr.Logger {
	return funcr.New(func(_, args string) {
		*out = append(*out, args)
	}, funcr.Options{})
}

// useTracerProvider installs tp for the duration of the test and restores the
// previous global provider afterwards.
func useTracerProvider(t *testing.T, tp trace.TracerProvider) {
	t.Helper()

	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })
}

// startSampledSpan starts a real, always-sampled span so its span context is
// valid (a no-op span would report IsValid() == false).
func startSampledSpan(ctx context.Context, t *testing.T) (context.Context, trace.Span) {
	t.Helper()

	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))
	return Tracer().Start(ctx, "test")
}

func TestLoggerWithSpanContextInjectsFields(t *testing.T) {
	var logged []string
	ctx := log.IntoContext(context.Background(), captureLogger(&logged))

	ctx, span := startSampledSpan(ctx, t)
	defer span.End()

	sc := span.SpanContext()
	if !sc.IsValid() {
		t.Fatal("expected a valid span context for the sampled span")
	}

	ctx = LoggerWithSpanContext(ctx, span)
	log.FromContext(ctx).Info("hello")

	if len(logged) != 1 {
		t.Fatalf("expected 1 log record, got %d", len(logged))
	}
	rec := logged[0]
	if !strings.Contains(rec, LogKeyTraceID) || !strings.Contains(rec, sc.TraceID().String()) {
		t.Errorf("log record missing trace_id: %q", rec)
	}
	if !strings.Contains(rec, LogKeySpanID) || !strings.Contains(rec, sc.SpanID().String()) {
		t.Errorf("log record missing span_id: %q", rec)
	}
}

// A nested span boundary must not append a second pair of correlation fields:
// duplicate JSON keys are ambiguous for log backends.
func TestLoggerWithSpanContextEnrichesOncePerContext(t *testing.T) {
	var logged []string
	ctx := log.IntoContext(context.Background(), captureLogger(&logged))

	ctx, parent := startSampledSpan(ctx, t)
	defer parent.End()
	ctx = LoggerWithSpanContext(ctx, parent)

	ctx, child := Tracer().Start(ctx, "child")
	defer child.End()
	ctx = LoggerWithSpanContext(ctx, child)

	log.FromContext(ctx).Info("hello")

	if len(logged) != 1 {
		t.Fatalf("expected 1 log record, got %d", len(logged))
	}
	rec := logged[0]
	if got := strings.Count(rec, LogKeyTraceID); got != 1 {
		t.Errorf("trace_id appears %d times, want 1: %q", got, rec)
	}
	if got := strings.Count(rec, LogKeySpanID); got != 1 {
		t.Errorf("span_id appears %d times, want 1: %q", got, rec)
	}
	if !strings.Contains(rec, parent.SpanContext().SpanID().String()) {
		t.Errorf("log record must keep the outermost span_id: %q", rec)
	}
	if strings.Contains(rec, child.SpanContext().SpanID().String()) {
		t.Errorf("log record must not carry the nested span_id: %q", rec)
	}
}

// Values already on the context logger survive enrichment, and so do values
// added between the entry point and a nested boundary.
func TestLoggerWithSpanContextPreservesExistingValues(t *testing.T) {
	var logged []string
	base := captureLogger(&logged).WithValues("x-request-id", "req-1")
	ctx := log.IntoContext(context.Background(), base)

	ctx, span := startSampledSpan(ctx, t)
	defer span.End()
	ctx = LoggerWithSpanContext(ctx, span)

	ctx = log.IntoContext(ctx, log.FromContext(ctx).WithValues("model", "sim"))

	ctx, child := Tracer().Start(ctx, "child")
	defer child.End()
	ctx = LoggerWithSpanContext(ctx, child)

	log.FromContext(ctx).Info("hello")

	if len(logged) != 1 {
		t.Fatalf("expected 1 log record, got %d", len(logged))
	}
	for _, want := range []string{"req-1", "sim", span.SpanContext().TraceID().String()} {
		if !strings.Contains(logged[0], want) {
			t.Errorf("log record missing %q: %q", want, logged[0])
		}
	}
}

func TestLoggerWithSpanContextNoopForInvalidSpan(t *testing.T) {
	var logged []string
	ctx := log.IntoContext(context.Background(), captureLogger(&logged))

	// A span from the no-op tracer has an invalid span context.
	useTracerProvider(t, noop.NewTracerProvider())
	ctx, span := otel.Tracer("test").Start(ctx, "noop")
	defer span.End()

	if span.SpanContext().IsValid() {
		t.Skip("tracer unexpectedly produced a valid span context")
	}

	ctx = LoggerWithSpanContext(ctx, span)

	log.FromContext(ctx).Info("hello")
	if len(logged) != 1 {
		t.Fatalf("expected 1 log record, got %d", len(logged))
	}
	if strings.Contains(logged[0], LogKeyTraceID) || strings.Contains(logged[0], LogKeySpanID) {
		t.Errorf("expected no correlation fields for invalid span, got: %q", logged[0])
	}
}
