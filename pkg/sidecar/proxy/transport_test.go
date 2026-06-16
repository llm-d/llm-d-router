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

package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

// newProxyTransport must inject W3C trace context into outbound requests so that
// EPP -> routing-proxy -> vLLM share a single trace.
func TestNewProxyTransportInjectsTraceContext(t *testing.T) {
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))

	var gotTraceparent string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotTraceparent = r.Header.Get("traceparent")
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()

	s := NewProxy(Config{})
	client := &http.Client{Transport: s.newProxyTransport("http", false)}

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
	ctx := trace.ContextWithSpanContext(context.Background(), sc)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, backend.URL, nil)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	_ = resp.Body.Close()

	if gotTraceparent == "" {
		t.Fatal("expected traceparent header to be injected into outbound request, got none")
	}
	if !strings.Contains(gotTraceparent, traceID.String()) {
		t.Fatalf("expected outbound traceparent to carry trace ID %s, got %q", traceID, gotTraceparent)
	}
}
