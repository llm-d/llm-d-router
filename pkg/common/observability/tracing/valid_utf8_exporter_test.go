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

package tracing

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// One string that is not valid UTF-8 must not cost the other spans in the
// batch: OTLP rejects the whole export request when any string field is invalid.
func TestInitTracingExportsBatchHoldingInvalidUTF8(t *testing.T) {
	clearEnv(t, otlpTransportEnv...)
	clearEnv(t, otlpProtocolEnv...)
	clearEnv(t, "OTEL_TRACES_SAMPLER_ARG")
	c, addr := startCollector(t, nil)
	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_TRACES_SAMPLER", "always_on")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://"+addr)

	origTP, origHandler, origProp := otel.GetTracerProvider(), otel.GetErrorHandler(), otel.GetTextMapPropagator()
	t.Cleanup(func() {
		otel.SetTracerProvider(origTP)
		otel.SetErrorHandler(origHandler)
		otel.SetTextMapPropagator(origProp)
	})

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	shutdown, err := InitTracing(ctx, testr.New(t), testServiceName)
	if err != nil {
		t.Fatalf("InitTracing() error = %v", err)
	}

	_, valid := Tracer().Start(ctx, "valid")
	valid.End()

	_, attr := Tracer().Start(ctx, "attribute")
	attr.SetAttributes(attribute.String("id", "tenant-\xff"))
	attr.End()

	_, event := Tracer().Start(ctx, "event")
	event.RecordError(errors.New("path /v1/\xff"))
	event.End()

	_, status := Tracer().Start(ctx, "status")
	status.SetStatus(codes.Error, "path /v1/\xff")
	status.End()

	_, name := Tracer().Start(ctx, "GET /v1/\xff")
	name.End()

	// Shutdown flushes the batcher, so the collector has the spans once it returns.
	if err := shutdown(ctx); err != nil {
		t.Fatalf("shutdown() error = %v", err)
	}

	if spans, _ := c.received(); spans != 5 {
		t.Errorf("collector received %d spans, want 5", spans)
	}
}

func TestValidUTF8ExporterReplacesInvalidStrings(t *testing.T) {
	ctx := context.Background()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))

	_, span := tp.Tracer("test").Start(ctx, "GET /v1/\xff")
	span.SetAttributes(
		attribute.String("id", "tenant-\xff\xfe-a"),
		attribute.String("model", "mod\u00e8le"),
		attribute.Int("count", 3),
		attribute.String("request", "req-\xff"),
	)
	span.AddEvent("exception", trace.WithAttributes(
		attribute.String("exception.message", "path /v1/\xff"),
		attribute.String("exception.type", "*errors.errorString"),
	))
	span.AddEvent("retry")
	span.SetStatus(codes.Error, "path /v1/\xff")
	span.End()

	_, valid := tp.Tracer("test").Start(ctx, "valid")
	valid.SetAttributes(attribute.String("id", "tenant-a"))
	valid.SetStatus(codes.Error, "upstream unavailable")
	valid.End()

	ended := recorder.Ended()
	// The spans' slices are shared, so the caller's view is copied here.
	callerAttrs := slices.Clone(ended[0].Attributes())
	callerEventAttrs := slices.Clone(ended[0].Events()[0].Attributes)
	callerSpan := ended[0]

	inner := &captureExporter{InMemoryExporter: tracetest.NewInMemoryExporter()}
	if err := (validUTF8Exporter{SpanExporter: inner}).ExportSpans(ctx, ended); err != nil {
		t.Fatalf("ExportSpans() error = %v", err)
	}
	got := inner.GetSpans()
	if len(got) != 2 {
		t.Fatalf("exported %d spans, want 2", len(got))
	}

	opts := cmp.Options{cmp.Comparer(func(a, b attribute.KeyValue) bool { return a == b })}

	wantAttrs := []attribute.KeyValue{
		attribute.String("id", "tenant-\ufffd-a"),
		attribute.String("model", "mod\u00e8le"),
		attribute.Int("count", 3),
		attribute.String("request", "req-\ufffd"),
	}
	if diff := cmp.Diff(wantAttrs, got[0].Attributes, opts); diff != "" {
		t.Errorf("attributes mismatch (-want +got):\n%s", diff)
	}
	if len(got[0].Events) != 2 {
		t.Fatalf("exported %d events, want 2", len(got[0].Events))
	}
	wantEventAttrs := []attribute.KeyValue{
		attribute.String("exception.message", "path /v1/\ufffd"),
		attribute.String("exception.type", "*errors.errorString"),
	}
	if diff := cmp.Diff(wantEventAttrs, got[0].Events[0].Attributes, opts); diff != "" {
		t.Errorf("event attributes mismatch (-want +got):\n%s", diff)
	}
	if got[0].Events[0].Name != "exception" || got[0].Events[1].Name != "retry" {
		t.Errorf("event names = %q, %q, want exception, retry", got[0].Events[0].Name, got[0].Events[1].Name)
	}
	if want := (sdktrace.Status{Code: codes.Error, Description: "path /v1/\ufffd"}); got[0].Status != want {
		t.Errorf("status = %s %q, want %s %q", got[0].Status.Code, got[0].Status.Description, want.Code, want.Description)
	}
	if got[0].Name != "GET /v1/\ufffd" {
		t.Errorf("span name = %q, want %q", got[0].Name, "GET /v1/\ufffd")
	}
	if got[0].SpanContext.SpanID() != ended[0].SpanContext().SpanID() {
		t.Errorf("span ID = %s, want %s", got[0].SpanContext.SpanID(), ended[0].SpanContext().SpanID())
	}

	// A span with nothing to replace is handed on as is, not wrapped.
	if inner.spans[1] != ended[1] {
		t.Error("valid span was replaced")
	}

	// The slices a span returns are shared with the SDK and with other span
	// processors, so the replacement must not write into them.
	if ended[0] != callerSpan {
		t.Error("caller's span slice was modified")
	}
	if diff := cmp.Diff(callerAttrs, callerSpan.Attributes(), opts); diff != "" {
		t.Errorf("caller's span attributes were modified (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(callerEventAttrs, callerSpan.Events()[0].Attributes, opts); diff != "" {
		t.Errorf("caller's event attributes were modified (-want +got):\n%s", diff)
	}
}

// captureExporter keeps the spans it is handed; the in-memory exporter stores
// only copies of them.
type captureExporter struct {
	*tracetest.InMemoryExporter

	spans []sdktrace.ReadOnlySpan
}

func (e *captureExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.spans = spans
	return e.InMemoryExporter.ExportSpans(ctx, spans)
}
