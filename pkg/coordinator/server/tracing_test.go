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

package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	otelsemconv "go.opentelemetry.io/otel/semconv/v1.41.0"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const (
	upstreamTraceID = "0123456789abcdef0123456789abcdef"
	upstreamSpanID  = "0123456789abcdef"
)

// setupSpanRecorder installs an in-memory recorder and the W3C propagator as
// the process defaults for the duration of the test.
func setupSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()

	recorder := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder)))
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	return recorder
}

func postTracedInference(t *testing.T, srv *Server) {
	t.Helper()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("traceparent", "00-"+upstreamTraceID+"-"+upstreamSpanID+"-01")
	srv.httpServer.Handler.ServeHTTP(httptest.NewRecorder(), req)
}

// serverSpan returns the single ended span of server kind. The pipeline and its
// steps contribute their own spans to the same trace.
func serverSpan(t *testing.T, recorder *tracetest.SpanRecorder) sdktrace.ReadOnlySpan {
	t.Helper()

	var found sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.SpanKind() != trace.SpanKindServer {
			continue
		}
		if found != nil {
			t.Fatal("expected one server span, got more than one")
		}
		found = span
	}
	if found == nil {
		t.Fatal("no server span recorded")
	}
	return found
}

// A client traceparent parents the coordinator's server span, so client,
// coordinator, and EPP spans land in one trace.
func TestServerJoinsIncomingTrace(t *testing.T) {
	recorder := setupSpanRecorder(t)

	postTracedInference(t, newTestServer(nil))

	span := serverSpan(t, recorder)
	if got := span.SpanContext().TraceID().String(); got != upstreamTraceID {
		t.Errorf("trace ID = %s, want %s", got, upstreamTraceID)
	}
	if got := span.Parent().SpanID().String(); got != upstreamSpanID {
		t.Errorf("parent span ID = %s, want %s", got, upstreamSpanID)
	}
	if got, want := span.Name(), "POST /v1/chat/completions"; got != want {
		t.Errorf("span name = %q, want %q", got, want)
	}
}

// The pipeline span nests under the server span, so one trace covers the
// request from the listener down through the steps.
func TestPipelineSpanNestsUnderServerSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)

	postTracedInference(t, newTestServer(nil))

	server := serverSpan(t, recorder)
	var pipelineSpan sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.Name() == "pipeline" {
			pipelineSpan = span
		}
	}
	if pipelineSpan == nil {
		t.Fatal("no pipeline span recorded")
	}
	if got := pipelineSpan.Parent().SpanID(); got != server.SpanContext().SpanID() {
		t.Errorf("pipeline span parent = %s, want the server span %s", got, server.SpanContext().SpanID())
	}
}

// Steps execute under the server span, which is what carries the trace into
// their outbound calls.
func TestPipelineStepRunsInRequestTrace(t *testing.T) {
	setupSpanRecorder(t)

	var stepSpan trace.SpanContext
	capture := stubStep{name: "capture", fn: func(ctx context.Context, _ *pipeline.RequestContext) error {
		stepSpan = trace.SpanContextFromContext(ctx)
		return nil
	}}
	srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{capture}), gateway.NewWithTransport(nil, stubGatewayURL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	postTracedInference(t, srv)

	if !stepSpan.IsValid() {
		t.Fatal("step context carries no span")
	}
	if got := stepSpan.TraceID().String(); got != upstreamTraceID {
		t.Errorf("step trace ID = %s, want %s", got, upstreamTraceID)
	}
}

func routeAttr(span sdktrace.ReadOnlySpan) (string, bool) {
	for _, attr := range span.Attributes() {
		if attr.Key == otelsemconv.HTTPRouteKey {
			return attr.Value.AsString(), true
		}
	}
	return "", false
}

// A route carrying an ID is named after its template, so every request to it
// shares one span name.
func TestServerSpanNamedAfterRouteTemplate(t *testing.T) {
	recorder := setupSpanRecorder(t)
	step := &auxRouteStep{stubStep: stubStep{name: "aux"}}
	srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(nil, stubGatewayURL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	srv.httpServer.Handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/v1/requests/req-123", nil))

	span := serverSpan(t, recorder)
	if got, want := span.Name(), "GET /v1/requests/{id}"; got != want {
		t.Errorf("span name = %q, want %q", got, want)
	}
	if got, _ := routeAttr(span); got != "/v1/requests/{id}" {
		t.Errorf("http.route = %q, want %q", got, "/v1/requests/{id}")
	}
}

// The passthrough serves any path the coordinator does not register, so its
// span is named after the method alone.
func TestPassthroughSpanNamedAfterMethod(t *testing.T) {
	recorder := setupSpanRecorder(t)
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer gw.Close()

	newTestServerWithGateway(nil, gw.URL).httpServer.Handler.ServeHTTP(
		httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/v1/models/some-model", nil))

	span := serverSpan(t, recorder)
	if got := span.Name(); got != http.MethodGet {
		t.Errorf("span name = %q, want %q", got, http.MethodGet)
	}
	if got, ok := routeAttr(span); ok {
		t.Errorf("http.route = %q, want unset", got)
	}
}

type panicRouteStep struct{ stubStep }

func (panicRouteStep) RegisterRoutes(r chi.Router) {
	r.Get("/v1/panic/{id}", func(http.ResponseWriter, *http.Request) { panic("boom") })
}

// A handler panic is recovered inside the router, and the span still takes
// the route name.
func TestServerSpanNamedAfterRouteOnPanic(t *testing.T) {
	recorder := setupSpanRecorder(t)
	step := panicRouteStep{stubStep{name: "panic"}}
	srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(nil, stubGatewayURL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	rec := httptest.NewRecorder()
	srv.httpServer.Handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/panic/1", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}

	if got, want := serverSpan(t, recorder).Name(), "GET /v1/panic/{id}"; got != want {
		t.Errorf("span name = %q, want %q", got, want)
	}
}

// The kubelet polls the probe routes for the life of the pod, so tracing them
// would outnumber the spans for real requests.
func TestProbeRoutesAreNotTraced(t *testing.T) {
	recorder := setupSpanRecorder(t)
	srv := newTestServer(nil)

	for _, path := range []string{pathHealthz, pathReadyz} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		srv.httpServer.Handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s returned %d, want 200", path, rec.Code)
		}
	}

	if spans := recorder.Ended(); len(spans) != 0 {
		t.Errorf("probe requests produced %d spans, want none", len(spans))
	}
}

// handleInference clears the write deadline through http.ResponseController
// and the reverse proxies flush, so the otel wrapper has to stay unwrappable.
// Both failures are silent: ErrNotSupported is tolerated, and buffered bytes
// still reach the client once the response ends.
func TestOTelHandlerPreservesStreamingInterfaces(t *testing.T) {
	var deadlineErr error
	var canFlush bool

	ts := httptest.NewServer(otelHandler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		deadlineErr = http.NewResponseController(w).SetWriteDeadline(time.Time{})
		_, canFlush = w.(http.Flusher)
	})))
	defer ts.Close()

	resp, err := ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if deadlineErr != nil {
		t.Errorf("clearing the write deadline through the otel wrapper: %v", deadlineErr)
	}
	if !canFlush {
		t.Error("otel wrapper hides http.Flusher, streamed chunks would not reach the client")
	}
}

// With span export off the provider is a no-op, and the incoming trace context
// still reaches the gateway unchanged on both the pipeline and passthrough
// paths, so client and EPP spans stay in one trace.
func TestPropagationWithExportOff(t *testing.T) {
	prevProvider, prevPropagator := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	otel.SetTracerProvider(noop.NewTracerProvider())
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	t.Cleanup(func() {
		otel.SetTracerProvider(prevProvider)
		otel.SetTextMapPropagator(prevPropagator)
	})

	incoming := "00-" + upstreamTraceID + "-" + upstreamSpanID + "-01"
	var got []string
	gwStub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = append(got, r.Header.Get("traceparent"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer gwStub.Close()

	gw := gateway.NewWithTransport(&http.Transport{}, gwStub.URL)
	callGateway := stubStep{name: "call-gateway", fn: func(ctx context.Context, _ *pipeline.RequestContext) error {
		resp, err := gw.Post(ctx, "/v1/chat/completions", []byte(`{}`), nil)
		if err != nil {
			return err
		}
		return resp.Body.Close()
	}}
	srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{callGateway}), gw)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	for _, req := range []*http.Request{
		httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`)),
		httptest.NewRequest(http.MethodGet, "/v1/models", nil),
	} {
		req.Header.Set("traceparent", incoming)
		srv.httpServer.Handler.ServeHTTP(httptest.NewRecorder(), req)
	}

	if len(got) != 2 {
		t.Fatalf("gateway received %d requests, want 2", len(got))
	}
	for i, traceparent := range got {
		if traceparent != incoming {
			t.Errorf("gateway request %d traceparent = %q, want %q", i, traceparent, incoming)
		}
	}
}
