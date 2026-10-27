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

package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
)

// tracedContext returns a context carrying a sampled remote span context and
// the trace ID an injected traceparent must reference.
func tracedContext(t *testing.T) (context.Context, string) {
	t.Helper()

	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))

	traceID, err := trace.TraceIDFromHex("0123456789abcdef0123456789abcdef")
	if err != nil {
		t.Fatalf("parse trace ID: %v", err)
	}
	spanID, err := trace.SpanIDFromHex("0123456789abcdef")
	if err != nil {
		t.Fatalf("parse span ID: %v", err)
	}
	sc := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: trace.FlagsSampled,
		Remote:     true,
	})

	return trace.ContextWithSpanContext(context.Background(), sc), traceID.String()
}

// recordTraceparent stands up a gateway stub that captures the traceparent of
// the request it receives.
func recordTraceparent(t *testing.T, got *string) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*got = r.Header.Get("traceparent")
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)

	return srv
}

// Requests to the gateway carry W3C trace context so EPP and the sidecar join
// the trace the coordinator is serving.
func TestClientInjectsTraceContext(t *testing.T) {
	var gotTraceparent string
	gw := recordTraceparent(t, &gotTraceparent)
	ctx, traceID := tracedContext(t)

	resp, err := New(config.GatewayConfig{Address: gw.URL}).Post(ctx, "/v1/chat/completions", []byte(`{}`), nil)
	if err != nil {
		t.Fatalf("Post: %v", err)
	}
	_ = resp.Body.Close()

	if !strings.Contains(gotTraceparent, traceID) {
		t.Errorf("traceparent = %q, want trace ID %s", gotTraceparent, traceID)
	}
}

// The decode and passthrough reverse proxies dial through Transport(), so it
// has to inject as well.
func TestClientTransportInjectsTraceContext(t *testing.T) {
	var gotTraceparent string
	gw := recordTraceparent(t, &gotTraceparent)
	ctx, traceID := tracedContext(t)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, gw.URL, http.NoBody)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	resp, err := (&http.Client{Transport: New(config.GatewayConfig{Address: gw.URL}).Transport()}).Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	_ = resp.Body.Close()

	if !strings.Contains(gotTraceparent, traceID) {
		t.Errorf("traceparent = %q, want trace ID %s", gotTraceparent, traceID)
	}
}

// The passthrough forwards arbitrary client paths through this transport, so
// the outbound span name must not carry the path.
func TestClientSpanNameOmitsPath(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	var gotTraceparent string
	gw := recordTraceparent(t, &gotTraceparent)

	// otelhttp takes its tracer provider from the span already in the context,
	// so the parent has to be a real span rather than a bare span context: a
	// bare one is non-recording and reports the noop provider. In the
	// coordinator the pipeline span fills this role.
	ctx, parent := provider.Tracer("test").Start(context.Background(), "parent")

	const path = "/v1/files/file-abc123"
	resp, err := New(config.GatewayConfig{Address: gw.URL}).Post(ctx, path, []byte(`{}`), nil)
	if err != nil {
		t.Fatalf("Post: %v", err)
	}
	_ = resp.Body.Close()
	parent.End()

	var client sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.SpanKind() == trace.SpanKindClient {
			client = span
		}
	}
	if client == nil {
		t.Fatal("no client span recorded for the gateway call")
	}
	if got := client.Name(); got == "" || strings.Contains(got, path) {
		t.Errorf("span name = %q, want a fixed name without the path", got)
	}
}
