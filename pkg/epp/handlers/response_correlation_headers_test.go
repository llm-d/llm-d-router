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

package handlers

import (
	"context"
	"strings"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/baggage"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// responseHeaders collects the generated response headers, lowercasing keys and
// counting repeats so a duplicate emission is visible.
func responseHeaders(ctx context.Context, t *testing.T, s *StreamingServer, reqCtx *RequestContext) (map[string]string, map[string]int) {
	t.Helper()

	values := make(map[string]string)
	counts := make(map[string]int)
	for _, h := range s.generateResponseHeaders(ctx, reqCtx) {
		key := strings.ToLower(h.Header.Key)
		values[key] = string(h.Header.RawValue)
		counts[key]++
	}
	return values, counts
}

func TestGenerateResponseHeaders_ReturnsRequestID(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())

	for _, tc := range []struct {
		name     string
		enabled  bool
		resolved string
		want     string
	}{
		{name: "enabled returns the resolved id", enabled: true, resolved: "req-from-client", want: "req-from-client"},
		{name: "disabled omits the header", enabled: false, resolved: "req-from-client"},
		{name: "enabled with no resolved id omits the header", enabled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{requestIDResponseHeader: tc.enabled}
			reqCtx := &RequestContext{
				resolvedRequestID: tc.resolved,
				Response:          &Response{Headers: map[string]string{}},
			}

			got, counts := responseHeaders(context.Background(), t, server, reqCtx)

			if tc.want == "" {
				assert.NotContains(t, got, reqcommon.RequestIDHeaderKey)
				return
			}
			assert.Equal(t, tc.want, got[reqcommon.RequestIDHeaderKey])
			assert.Equal(t, 1, counts[reqcommon.RequestIDHeaderKey])
		})
	}
}

// NewStreamingServer returns the request ID without an explicit opt-in, so the
// constructor default matches the documented default.
func TestNewStreamingServer_ReturnsRequestIDByDefault(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())

	server := NewStreamingServer(nil, nil, nil, 0)
	reqCtx := &RequestContext{
		resolvedRequestID: "req-default",
		Response:          &Response{Headers: map[string]string{}},
	}

	got, _ := responseHeaders(context.Background(), t, server, reqCtx)

	assert.Equal(t, "req-default", got[reqcommon.RequestIDHeaderKey])
}

// The model server may echo its own x-request-id. EPP's resolved value is the one
// the client correlates against, and the client must receive exactly one.
func TestGenerateResponseHeaders_BackendRequestIDDoesNotDuplicate(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())

	server := &StreamingServer{requestIDResponseHeader: true}
	reqCtx := &RequestContext{
		resolvedRequestID: "req-epp",
		Response: &Response{Headers: map[string]string{
			reqcommon.RequestIDHeaderKey: "req-from-model-server",
		}},
	}

	got, counts := responseHeaders(context.Background(), t, server, reqCtx)

	assert.Equal(t, 1, counts[reqcommon.RequestIDHeaderKey], "client must receive exactly one x-request-id")
	assert.Equal(t, "req-epp", got[reqcommon.RequestIDHeaderKey])
}

// With the response header disabled EPP does not own x-request-id, so a value the
// model server set passes through like any other backend header.
func TestGenerateResponseHeaders_BackendRequestIDPassesThroughWhenDisabled(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())

	server := &StreamingServer{requestIDResponseHeader: false}
	reqCtx := &RequestContext{
		resolvedRequestID: "req-epp",
		Response: &Response{Headers: map[string]string{
			reqcommon.RequestIDHeaderKey: "req-from-model-server",
		}},
	}

	got, counts := responseHeaders(context.Background(), t, server, reqCtx)

	assert.Equal(t, 1, counts[reqcommon.RequestIDHeaderKey])
	assert.Equal(t, "req-from-model-server", got[reqcommon.RequestIDHeaderKey])
}

func TestGenerateResponseHeaders_TraceparentOffByDefault(t *testing.T) {
	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

	ctx, span := tracing.Tracer("test").Start(context.Background(), "request", trace.WithSpanKind(trace.SpanKindServer))
	defer span.End()

	server := &StreamingServer{requestIDResponseHeader: true}
	reqCtx := &RequestContext{Response: &Response{Headers: map[string]string{}}}

	got, _ := responseHeaders(ctx, t, server, reqCtx)

	assert.NotContains(t, got, "traceparent", "trace IDs must not reach the client unless enabled")
}

// The returned traceparent must identify the trace an operator will open, and
// must not carry the rest of the propagator's payload: baggage can hold internal
// values that have no business reaching the client.
func TestGenerateResponseHeaders_TraceparentReturnsActiveTraceOnly(t *testing.T) {
	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))
	// useTracerProvider installs trace context alone; the composite propagator is
	// what EPP runs with, and it also injects baggage.
	tracing.InitTextMapPropagator()

	member, err := baggage.NewMember("tenant", "internal-acme")
	require.NoError(t, err)
	bag, err := baggage.New(member)
	require.NoError(t, err)

	ctx := baggage.ContextWithBaggage(context.Background(), bag)
	ctx, span := tracing.Tracer("test").Start(ctx, "request", trace.WithSpanKind(trace.SpanKindServer))
	defer span.End()

	server := &StreamingServer{traceparentResponseHeader: true}
	reqCtx := &RequestContext{Response: &Response{Headers: map[string]string{}}}

	got, counts := responseHeaders(ctx, t, server, reqCtx)

	require.Equal(t, 1, counts["traceparent"], "expected exactly one traceparent header")
	assert.Contains(t, got["traceparent"], span.SpanContext().TraceID().String(),
		"returned traceparent must carry the active EPP span trace ID")
	assert.NotContains(t, got, "baggage", "baggage must not be returned to the client")
	assert.NotContains(t, got, "tracestate", "tracestate must not be returned to the client")
}

// With tracing disabled the span context is invalid, so there is no trace to
// point the client at and the header is omitted rather than returned empty.
func TestGenerateResponseHeaders_TraceparentOmittedWhenTracingDisabled(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())

	ctx, span := tracing.Tracer("test").Start(context.Background(), "request", trace.WithSpanKind(trace.SpanKindServer))
	defer span.End()

	server := &StreamingServer{traceparentResponseHeader: true}
	reqCtx := &RequestContext{Response: &Response{Headers: map[string]string{}}}

	got, _ := responseHeaders(ctx, t, server, reqCtx)

	assert.NotContains(t, got, "traceparent")
}

// responseHeaderValues returns the x-request-id values Process set on the
// response-headers mutation of the stream.
func responseHeaderValues(t *testing.T, sent []*extProcPb.ProcessingResponse, key string) []string {
	t.Helper()

	var values []string
	for _, resp := range sent {
		headers := resp.GetResponseHeaders()
		if headers == nil {
			continue
		}
		for _, opt := range headers.GetResponse().GetHeaderMutation().GetSetHeaders() {
			if strings.EqualFold(opt.GetHeader().GetKey(), key) {
				values = append(values, string(opt.GetHeader().GetRawValue()))
			}
		}
	}
	return values
}

// The request ID is resolved in the request-headers phase, so what the response
// carries depends on whether that phase ran. A stream that starts at the
// response phase has no ID to return, and a request that supplies none gets the
// generated one rather than an empty header.
func TestProcessReturnsRequestIDOnlyWhenResolved(t *testing.T) {
	responsePhase := []*extProcPb.ProcessingRequest{
		{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
			Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: reqcommon.HeaderContentType, RawValue: []byte(reqcommon.ContentTypeJSON)},
			}},
		}}},
		{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
			Body: []byte(`{}`), EndOfStream: true,
		}}},
	}

	for _, tc := range []struct {
		name           string
		requestHeaders map[string]string
		skipRequest    bool
		want           string
	}{
		{
			name:           "client request ID is returned",
			requestHeaders: map[string]string{":path": reqcommon.PathChatCompletions, reqcommon.RequestIDHeaderKey: "req-from-client"},
			want:           "req-from-client",
		},
		{
			name:           "generated request ID is returned",
			requestHeaders: map[string]string{":path": reqcommon.PathChatCompletions},
		},
		{
			name:        "no request phase returns no request ID",
			skipRequest: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			useTracerProvider(t, noop.NewTracerProvider())

			reqs := make([]*extProcPb.ProcessingRequest, 0, 4)
			if !tc.skipRequest {
				reqs = append(reqs,
					newRequestHeaders(tc.requestHeaders),
					&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
						Body: []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`), EndOfStream: true,
					}}},
				)
			}
			reqs = append(reqs, responsePhase...)

			srv := &replayProcessServer{ctx: context.Background(), reqs: reqs}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			_ = NewStreamingServer(nil, &mockDirector{}, registry, 0).Process(srv)

			got := responseHeaderValues(t, srv.sentResponses, reqcommon.RequestIDHeaderKey)

			if tc.skipRequest {
				require.Empty(t, got, "a stream with no request phase has no request ID to return")
				return
			}
			require.Len(t, got, 1, "the client must receive exactly one x-request-id")
			if tc.want != "" {
				require.Equal(t, tc.want, got[0])
				return
			}
			require.NotEmpty(t, got[0], "a request without an x-request-id gets the generated one")
		})
	}
}
