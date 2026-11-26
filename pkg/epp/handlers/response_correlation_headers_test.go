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
	"net/http"
	"strings"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/baggage"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// responseHeaders resolves the correlation headers the way Process does, then
// collects the generated response headers, lowercasing keys and counting repeats
// so a duplicate emission is visible.
func responseHeaders(ctx context.Context, t *testing.T, s *StreamingServer, reqCtx *RequestContext) (map[string]string, map[string]int) {
	t.Helper()

	reqCtx.correlationHeaders = s.correlationResponseHeaders(ctx, reqCtx)

	values := make(map[string]string)
	counts := make(map[string]int)
	for _, h := range s.generateResponseHeaders(reqCtx) {
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

// immediateResponseHeaders returns the headers set on the ImmediateResponse of
// the single reply sent on the stream.
func immediateResponseHeaders(t *testing.T, sent []*extProcPb.ProcessingResponse) (map[string]string, map[string]int) {
	t.Helper()

	require.Len(t, sent, 1, "expected exactly one reply")
	imm := sent[0].GetImmediateResponse()
	require.NotNil(t, imm, "reply must be an ImmediateResponse")
	require.NotNil(t, imm.Headers, "ImmediateResponse must carry a header mutation")

	values := make(map[string]string)
	counts := make(map[string]int)
	for _, h := range imm.Headers.SetHeaders {
		key := strings.ToLower(h.Header.Key)
		values[key] = string(h.Header.RawValue)
		counts[key]++
	}
	return values, counts
}

// getRequestHeaders is a bodyless GET carrying a client request ID, the shape a
// Responder answers and the shape that reaches no response phase.
func getRequestHeaders(requestID string) *extProcPb.ProcessingRequest {
	headers := []*configPb.HeaderValue{
		{Key: ":method", Value: http.MethodGet},
		{Key: ":path", Value: "/v1/models"},
	}
	if requestID != "" {
		headers = append(headers, &configPb.HeaderValue{Key: reqcommon.RequestIDHeaderKey, Value: requestID})
	}
	return &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers:     &configPb.HeaderMap{Headers: headers},
				EndOfStream: true,
			},
		},
	}
}

// A request EPP rejects or answers itself is the one a user reports, so those
// replies carry the same correlation headers as a routed response. They end the
// exchange with an ImmediateResponse, which never reaches
// generateResponseHeaders.
func TestProcessImmediateResponsesCarryCorrelationHeaders(t *testing.T) {
	const requestID = "req-from-client"

	for _, tc := range []struct {
		name       string
		responder  *testResponder
		wantStatus envoyTypePb.StatusCode
	}{
		{
			name: "EPP rejects the request",
			responder: &testResponder{err: errcommon.Error{
				Code: errcommon.ServiceUnavailable,
				Msg:  "model data unavailable",
			}},
			wantStatus: envoyTypePb.StatusCode_ServiceUnavailable,
		},
		{
			name: "EPP answers the request itself",
			responder: &testResponder{response: &fwkrh.LocalResponse{
				StatusCode: http.StatusOK,
				Headers:    map[string]string{reqcommon.HeaderContentType: reqcommon.ContentTypeJSON},
				Body:       []byte(`{"object":"list"}`),
			}},
			wantStatus: envoyTypePb.StatusCode_OK,
		},
		{
			name: "a responder's own request ID does not reach the client twice",
			responder: &testResponder{response: &fwkrh.LocalResponse{
				StatusCode: http.StatusOK,
				Headers:    map[string]string{reqcommon.RequestIDHeaderKey: "req-from-responder"},
			}},
			wantStatus: envoyTypePb.StatusCode_OK,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

			server := NewStreamingServer(&responderTestDatastore{}, &mockDirector{}, nil, 0)
			server.SetResponders([]fwkrh.Responder{tc.responder})
			server.SetTraceparentResponseHeader(true)
			process := &responderProcessServer{request: getRequestHeaders(requestID)}

			require.NoError(t, server.Process(process))

			require.Equal(t, tc.wantStatus, process.sentResponses[0].GetImmediateResponse().Status.Code)
			got, counts := immediateResponseHeaders(t, process.sentResponses)

			assert.Equal(t, requestID, got[reqcommon.RequestIDHeaderKey])
			assert.Equal(t, 1, counts[reqcommon.RequestIDHeaderKey], "client must receive exactly one x-request-id")
			assert.NotEmpty(t, got[reqcommon.TraceparentHeaderKey], "reply must point at the request trace")
			assert.Equal(t, 1, counts[reqcommon.TraceparentHeaderKey])
		})
	}
}

// A request shed by flow control is answered from the state machine, which holds
// no server and no context, so it reads the headers resolved on the request.
func TestEvictedResponseCarriesCorrelationHeaders(t *testing.T) {
	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

	ctx, span := tracing.Tracer("test").Start(context.Background(), "request", trace.WithSpanKind(trace.SpanKindServer))
	defer span.End()

	server := &StreamingServer{requestIDResponseHeader: true, traceparentResponseHeader: true}
	reqCtx := &RequestContext{
		requestState:      requestEvicted,
		resolvedRequestID: "req-evicted",
	}
	reqCtx.correlationHeaders = server.correlationResponseHeaders(ctx, reqCtx)

	srv := &mockProcessServer{}
	require.NoError(t, reqCtx.updateStateAndSendIfNeeded(srv, logr.Discard(), span))

	got, counts := immediateResponseHeaders(t, srv.sentResponses)

	assert.Equal(t, "req-evicted", got[reqcommon.RequestIDHeaderKey])
	assert.Equal(t, 1, counts[reqcommon.RequestIDHeaderKey])
	assert.Contains(t, got[reqcommon.TraceparentHeaderKey], span.SpanContext().TraceID().String())
	// The eviction reason and content type the reply already carried are kept.
	assert.Equal(t, reqcommon.ContentTypeJSON, got[reqcommon.HeaderContentType])
}

// An ImmediateResponse built before the request ID is resolved, such as a stream
// that fails on its first body chunk, has nothing to correlate and must not gain
// an empty header.
func TestAppendCorrelationHeadersWithoutResolvedRequest(t *testing.T) {
	imm := &extProcPb.ImmediateResponse{}
	appendCorrelationHeaders(&RequestContext{}, imm)
	assert.Nil(t, imm.Headers, "no correlation headers means no mutation at all")

	appendCorrelationHeaders(&RequestContext{correlationHeaders: []*configPb.HeaderValueOption{{
		Header: &configPb.HeaderValue{Key: reqcommon.RequestIDHeaderKey, RawValue: []byte("req-1")},
	}}}, nil)
}

// formatTraceparent stands in for an Inject call on the W3C propagator, so it
// has to render exactly what that propagator writes, including which flag bits
// survive: masking the sampled bit alone would drop the random-trace-ID flag.
func TestFormatTraceparentMatchesW3CPropagator(t *testing.T) {
	traceID, err := trace.TraceIDFromHex("4bf92f3577b34da6a3ce929d0e0e4736")
	require.NoError(t, err)
	spanID, err := trace.SpanIDFromHex("00f067aa0b0902b7")
	require.NoError(t, err)

	for _, tc := range []struct {
		name  string
		flags trace.TraceFlags
	}{
		{name: "no flags"},
		{name: "sampled", flags: trace.FlagsSampled},
		{name: "random trace ID", flags: trace.FlagsRandom},
		{name: "sampled and random", flags: trace.FlagsSampled | trace.FlagsRandom},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
				TraceID:    traceID,
				SpanID:     spanID,
				TraceFlags: tc.flags,
			}))

			carrier := propagation.MapCarrier{}
			propagation.TraceContext{}.Inject(ctx, carrier)

			require.Equal(t, carrier.Get(reqcommon.TraceparentHeaderKey), formatTraceparent(ctx))
		})
	}
}

func TestFormatTraceparentEmptyWithoutSpanContext(t *testing.T) {
	assert.Empty(t, formatTraceparent(context.Background()))
}
