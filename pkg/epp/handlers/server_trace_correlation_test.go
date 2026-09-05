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

package handlers

import (
	"context"
	"io"
	"strings"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

const (
	upstreamTraceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0c9902b7-01"
	// upstreamTraceID is the trace ID of upstreamTraceparent: the trace the EPP
	// must join rather than starting a fresh one.
	upstreamTraceID = "4bf92f3577b34da6a3ce929d0e0e4736"
)

// scriptedProcessServer replays one RequestHeaders message, then reports EOF so
// Process returns cleanly.
type scriptedProcessServer struct {
	mockProcessServer
	ctx  context.Context
	req  *extProcPb.ProcessingRequest
	sent bool
}

func (m *scriptedProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if m.sent {
		return nil, io.EOF
	}
	m.sent = true
	return m.req, nil
}

func (m *scriptedProcessServer) Context() context.Context { return m.ctx }

func newRequestHeaders(headers map[string]string) *extProcPb.ProcessingRequest {
	values := make([]*configPb.HeaderValue, 0, len(headers))
	for key, value := range headers {
		values = append(values, &configPb.HeaderValue{Key: key, RawValue: []byte(value)})
	}
	return &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &configPb.HeaderMap{Headers: values},
				// EndOfStream would route to a random endpoint and need a director.
				EndOfStream: false,
			},
		},
	}
}

// runProcess drives Process over a single RequestHeaders message and returns the
// lines it logged.
func runProcess(t *testing.T, headers map[string]string) []string {
	t.Helper()

	var logged []string
	capture := funcr.New(func(prefix, args string) {
		logged = append(logged, prefix+" "+args)
	}, funcr.Options{Verbosity: 2})

	srv := &scriptedProcessServer{
		ctx: log.IntoContext(context.Background(), capture),
		req: newRequestHeaders(headers),
	}
	require.NoError(t, NewStreamingServer(nil, nil, nil, 0).Process(srv))

	return logged
}

func entryLine(t *testing.T, logged []string) string {
	t.Helper()

	for _, rec := range logged {
		if strings.Contains(rec, "EPP received request") {
			return rec
		}
	}
	t.Fatalf("entry-point log line not found in %v", logged)
	return ""
}

// Pins the entry-point wiring rather than the helper: this fails if server.go
// stops installing the enriched logger.
func TestProcessCorrelatesRequestLogsWithTrace(t *testing.T) {
	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

	entry := entryLine(t, runProcess(t, map[string]string{
		"traceparent":  upstreamTraceparent,
		"x-request-id": "req-correlation-1",
	}))

	require.Contains(t, entry, tracing.LogKeyTraceID)
	require.Contains(t, entry, tracing.LogKeySpanID)
	require.Contains(t, entry, upstreamTraceID, "must join the upstream trace, not start a fresh one")
	require.Contains(t, entry, "req-correlation-1", "correlation must not displace the request ID")
	require.Equal(t, 1, strings.Count(entry, tracing.LogKeyTraceID), "trace_id must appear once: %q", entry)
}

// With tracing off the span context is invalid, so request logs are unchanged.
func TestProcessOmitsCorrelationWhenTracingDisabled(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())

	entry := entryLine(t, runProcess(t, map[string]string{"x-request-id": "req-correlation-2"}))

	require.Contains(t, entry, "req-correlation-2")
	require.NotContains(t, entry, tracing.LogKeyTraceID)
	require.NotContains(t, entry, tracing.LogKeySpanID)
}

// useTracerProvider installs tp and the W3C propagator for the duration of the
// test, restoring the previous globals afterwards.
func useTracerProvider(t *testing.T, tp trace.TracerProvider) {
	t.Helper()

	prevTP, prevProp := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() {
		otel.SetTracerProvider(prevTP)
		otel.SetTextMapPropagator(prevProp)
	})
}
