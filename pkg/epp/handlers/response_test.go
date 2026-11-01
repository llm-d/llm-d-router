/*
Copyright 2025 The Kubernetes Authors.
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
	"fmt"
	"io"
	"strconv"
	"strings"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-router/pkg/common/envoy"
	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	body = `
	{
		"id": "cmpl-573498d260f2423f9e42817bbba3743a",
		"object": "text_completion",
		"created": 1732563765,
		"model": "Qwen/Qwen3-32B",
		"choices": [
			{
				"index": 0,
				"text": " Chronicle\nThe San Francisco Chronicle has a new book review section, and it's a good one. The reviews are short, but they're well-written and well-informed. The Chronicle's book review section is a good place to start if you're looking for a good book review.\nThe Chronicle's book review section is a good place to start if you're looking for a good book review. The Chronicle's book review section",
				"logprobs": null,
				"finish_reason": "length",
				"stop_reason": null,
				"prompt_logprobs": null
			}
		],
		"usage": {
			"prompt_tokens": 11,
			"total_tokens": 111,
			"completion_tokens": 100
		}
	}
	`

	bodyWithoutUsage = `
	{
		"id": "cmpl-573498d260f2423f9e42817bbba3743a",
		"object": "text_completion",
		"created": 1732563765,
		"model": "Qwen/Qwen3-32B",
		"choices": [
			{
				"index": 0,
				"text": " Chronicle\nThe San Francisco Chronicle has a new book review section, and it's a good one. The reviews are short, but they're well-written and well-informed. The Chronicle's book review section is a good place to start if you're looking for a good book review.\nThe Chronicle's book review section is a good place to start if you're looking for a good book review. The Chronicle's book review section",
				"logprobs": null,
				"finish_reason": "length",
				"stop_reason": null,
				"prompt_logprobs": null
			}
		]
	}
	`

	bodyInvalidJSON = `
	{
		"id": "cmpl-573498d260f2423f9e42817bbba3743a",
		"object": "text_completion",
		"created": 1732563765,
		"model": "Qwen/Qwen3-32B",
		"choices": [
			{
				"invalid json"
			}
		]
	}
	`

	//#nosec G101 -- JSON response body fixture, not a credential
	bodyWithCachedTokens = `
	{
		"id": "cmpl-573498d260f2423f9e42817bbba3743a",
		"object": "text_completion",
		"created": 1732563765,
		"model": "Qwen/Qwen3-32B",
		"choices": [
			{
				"index": 0,
				"text": " Chronicle\nThe San Francisco Chronicle has a new book review section, and it's a good one. The reviews are short, but they're well-written and well-informed. The Chronicle's book review section is a good place to start if you're looking for a good book review.\nThe Chronicle's book review section is a good place to start if you're looking for a good book review. The Chronicle's book review section",
				"logprobs": null,
				"finish_reason": "length",
				"stop_reason": null,
				"prompt_logprobs": null
			}
		],
		"usage": {
			"prompt_tokens": 11,
			"total_tokens": 111,
			"completion_tokens": 100,
			"prompt_tokens_details": {
				"cached_tokens": 10
			}
		}
	}
	`

	streamingBodyWithoutUsage = `data: {"id":"cmpl-41764c93-f9d2-4f31-be08-3ba04fa25394","object":"text_completion","created":1740002445,"model":"food-review-0","choices":[],"usage":null}
	`

	streamingBodyWithUsage = `data: {"id":"cmpl-41764c93-f9d2-4f31-be08-3ba04fa25394","object":"text_completion","created":1740002445,"model":"food-review-0","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
data: [DONE]
	`
	//#nosec G101 -- JSON response body fixture, not a credential
	streamingBodyWithUsageAndCachedTokens = `data: {"id":"cmpl-41764c93-f9d2-4f31-be08-3ba04fa25394","object":"text_completion","created":1740002445,"model":"food-review-0","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":5}}}
data: [DONE]
	`
)

type mockDirector struct{ stream bool }

func (m *mockDirector) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext {
	return reqCtx
}
func (m *mockDirector) HandleResponseHeader(ctx context.Context, reqCtx *RequestContext) *RequestContext {
	return reqCtx
}
func (m *mockDirector) GetRandomEndpoint() *fwkdl.EndpointMetadata {
	return &fwkdl.EndpointMetadata{}
}
func (m *mockDirector) HandleRequest(ctx context.Context, reqCtx *RequestContext, inferenceRequestBody *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	if m.stream {
		reqCtx.SchedulingRequest = &fwksched.InferenceRequest{Body: &fwkrh.InferenceRequestBody{Stream: true}}
	}
	return reqCtx, nil
}

func TestHandleResponseBody(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name   string
		body   []byte
		reqCtx *RequestContext
		want   fwkrh.Usage
	}{
		{
			name: "success",
			body: []byte(body),
			want: fwkrh.Usage{
				PromptTokens:     11,
				TotalTokens:      111,
				CompletionTokens: 100,
			},
		},
		{
			name: "success with cached tokens",
			body: []byte(bodyWithCachedTokens),
			want: fwkrh.Usage{
				PromptTokens:     11,
				TotalTokens:      111,
				CompletionTokens: 100,
				PromptTokenDetails: &fwkrh.PromptTokenDetails{
					CachedTokens: 10,
				},
			},
		},
		{
			name: "success body without usage, the HandleResponseBody should still return non-nil error",
			body: []byte(bodyWithoutUsage),
			want: fwkrh.Usage{}, // Since the usage is not set in the responseBody, this usage should be empty.
		},
		{
			name: "success invalid joson body, the HandleResponseBody should still return non-nil error",
			body: []byte(bodyInvalidJSON),
			want: fwkrh.Usage{}, // Since the response is invalid json, the usage cannot be extrcated.
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &StreamingServer{
				parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
			}
			server.director = &mockDirector{}
			reqCtx := test.reqCtx
			if reqCtx == nil {
				reqCtx = &RequestContext{
					Response:          &Response{},
					SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
				}
			}
			server.HandleResponseBody(ctx, reqCtx, test.body, true)
			if diff := cmp.Diff(test.want, reqCtx.Usage); diff != "" {
				t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}

type terminatedProcessServer struct {
	replayProcessServer
	terminalErr     error
	failBodySend    int
	failTrailerSend bool
	bodySends       int
	evictCh         chan struct{}
	evictAfterBody  bool
}

func (srv *terminatedProcessServer) Send(resp *extProcPb.ProcessingResponse) error {
	if resp.GetResponseBody() != nil {
		srv.bodySends++
		if srv.bodySends == srv.failBodySend {
			return status.Error(codes.Unavailable, "body send failed")
		}
	}
	if resp.GetResponseTrailers() != nil && srv.failTrailerSend {
		return status.Error(codes.Unavailable, "trailer send failed")
	}
	err := srv.replayProcessServer.Send(resp)
	if srv.evictCh != nil && ((!srv.evictAfterBody && resp.GetResponseHeaders() != nil) || (srv.evictAfterBody && resp.GetResponseBody() != nil)) {
		close(srv.evictCh)
	}
	return err
}

func (srv *terminatedProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(srv.reqs) == 0 && srv.evictCh != nil {
		<-srv.Context().Done()
		return nil, srv.Context().Err()
	}
	if len(srv.reqs) == 0 && srv.terminalErr != nil {
		return nil, srv.terminalErr
	}
	return srv.replayProcessServer.Recv()
}

type responseEvictLookup struct{ ch chan struct{} }

func (lookup responseEvictLookup) Get(string) chan struct{}                 { return lookup.ch }
func (responseEvictLookup) GetReason(string) errcommon.RequestDroppedReason { return "" }
func (responseEvictLookup) Deregister(string)                               {}

func TestProcessRetainsToolCallingTelemetryOnResponseTermination(t *testing.T) {
	for _, api := range []struct{ path, json, sse string }{
		{reqcommon.PathChatCompletions, `{"choices":[{"message":{"tool_calls":[{"function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}`, `{"choices":[{"delta":{"tool_calls":[{"function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}`},
		{reqcommon.PathMessages, `{"content":[{"type":"tool_use","name":"sentinel_name","input":{"secret":"sentinel_arguments"}}]}`, `{"type":"content_block_start","content_block":{"type":"tool_use","name":"sentinel_name","input":{"secret":"sentinel_arguments"}}}`},
	} {
		toolEvent := "data: " + api.sse + "\n\n"
		toolJSON := api.json
		padding := `{"padding":"` + strings.Repeat("x", envoy.BodyByteLimit) + `",`
		largeJSON := padding + toolJSON[1:]
		largeEvent := "data: " + padding + api.sse[1:] + "\n\n"
		for _, tt := range []struct {
			name                 string
			eventStream          bool
			body                 string
			endOfStream          bool
			terminalErr          error
			wantPresent          bool
			wantErr              bool
			nonTool              bool
			skipResponse         bool
			unreadable           bool
			trailers             bool
			finalBody            string
			failBodySend         int
			failTrailerSend      bool
			wantForwardedMissing bool
			unsampled            bool
			telemetryDisabled    bool
			evicted              bool
			streamRequest        bool
		}{
			{name: "SSE EOF after tool call", eventStream: true, body: toolEvent, terminalErr: io.EOF, wantPresent: true},
			{name: "SSE cancelled after tool call", eventStream: true, body: toolEvent, terminalErr: status.Error(codes.Canceled, "cancelled"), wantPresent: true},
			{name: "SSE receive failure after tool call", eventStream: true, body: toolEvent, terminalErr: status.Error(codes.Unavailable, "upstream disconnected"), wantPresent: true, wantErr: true},
			{name: "SSE incomplete event", eventStream: true, body: `data: {"choices":[{"delta":{"tool_calls":[`, terminalErr: io.EOF},
			{name: "JSON incomplete response", body: `{"choices":[`, terminalErr: io.EOF},
			{name: "response headers only", terminalErr: io.EOF},
			{name: "completed SSE", eventStream: true, body: toolEvent, endOfStream: true, terminalErr: io.EOF, wantPresent: true},
			{name: "SSE missing final newline", eventStream: true, body: strings.TrimSuffix(toolEvent, "\n"), endOfStream: true, terminalErr: io.EOF, unreadable: true},
			{name: "SSE unterminated line at EOS", eventStream: true, body: strings.TrimSuffix(toolEvent, "\n\n"), endOfStream: true, terminalErr: io.EOF, unreadable: true},
			{name: "SSE completed through trailers", eventStream: true, body: toolEvent, trailers: true, terminalErr: io.EOF, wantPresent: true},
			{name: "SSE unfinished event through trailers", eventStream: true, body: strings.TrimSuffix(toolEvent, "\n"), trailers: true, terminalErr: io.EOF, unreadable: true},
			{name: "completed JSON", body: toolJSON, endOfStream: true, terminalErr: io.EOF, wantPresent: true},
			{name: "telemetry disabled", body: toolJSON, endOfStream: true, terminalErr: io.EOF, telemetryDisabled: true},
			{name: "malformed SSE at EOS", eventStream: true, body: "data: {invalid json}\n\n", endOfStream: true, terminalErr: io.EOF, unreadable: true},
			{name: "truncated JSON at EOS", body: `{"choices":[`, endOfStream: true, terminalErr: io.EOF, unreadable: true},
			{name: "non-tool request", body: `{"choices":[]}`, terminalErr: io.EOF, nonTool: true},
			{name: "request phase only", terminalErr: io.EOF, skipResponse: true},
			{name: "cancelled before tool call", terminalErr: status.Error(codes.Canceled, "cancelled")},
			{name: "receive failure before tool call", terminalErr: status.Error(codes.Unavailable, "disconnected"), wantErr: true},
			{name: "JSON first send failure", body: toolJSON, endOfStream: true, failBodySend: 1, wantPresent: true, wantForwardedMissing: true, wantErr: true},
			{name: "large JSON success", body: largeJSON, endOfStream: true, terminalErr: io.EOF, wantPresent: true},
			{name: "large JSON first send failure", body: largeJSON, endOfStream: true, failBodySend: 1, wantPresent: true, wantForwardedMissing: true, wantErr: true},
			{name: "large JSON final send failure", body: largeJSON, endOfStream: true, failBodySend: 2, wantPresent: true, wantForwardedMissing: true, wantErr: true},
			{name: "SSE first send failure", eventStream: true, body: toolEvent, endOfStream: true, failBodySend: 1, wantPresent: true, wantForwardedMissing: true, wantErr: true},
			{name: "SSE final EOS send failure", eventStream: true, body: toolEvent, finalBody: "data: [DONE]\n\n", endOfStream: true, failBodySend: 2, wantPresent: true, wantErr: true},
			{name: "SSE split event send failure", eventStream: true, body: strings.TrimSuffix(toolEvent, "\n"), finalBody: "\n", endOfStream: true, failBodySend: 2, wantPresent: true, wantForwardedMissing: true, wantErr: true},
			{name: "large SSE success", eventStream: true, body: largeEvent, endOfStream: true, terminalErr: io.EOF, wantPresent: true},
			{name: "large SSE final send failure", eventStream: true, body: largeEvent, endOfStream: true, failBodySend: 2, wantPresent: true, wantForwardedMissing: true, wantErr: true},
			{name: "SSE sent event before later fragment failure", eventStream: true, body: toolEvent + strings.Repeat(": ping\n", envoy.BodyByteLimit/7+1), endOfStream: true, failBodySend: 2, wantPresent: true, wantErr: true},
			{name: "JSON completed through trailers", body: toolJSON, trailers: true, terminalErr: io.EOF, wantPresent: true},
			{name: "JSON trailer send failure", body: toolJSON, trailers: true, failTrailerSend: true, wantPresent: true, wantErr: true},
			{name: "SSE trailer send failure", eventStream: true, body: toolEvent, trailers: true, failTrailerSend: true, wantPresent: true, wantErr: true},
			{name: "non-tool send failure", body: `{"choices":[]}`, endOfStream: true, failBodySend: 1, nonTool: true, wantErr: true},
			{name: "unsampled send failure", body: largeJSON, endOfStream: true, failBodySend: 2, unsampled: true, wantErr: true},
			{name: "evicted before tool call", eventStream: true, evicted: true},
			{name: "evicted after tool call", eventStream: true, body: toolEvent, evicted: true, wantPresent: true},
			{name: "JSON reply to streaming request", streamRequest: true, body: toolJSON, endOfStream: true, terminalErr: io.EOF, wantPresent: true},
			{name: "split JSON reply to streaming request", streamRequest: true, body: toolJSON[:len(toolJSON)/2], finalBody: toolJSON[len(toolJSON)/2:], endOfStream: true, terminalErr: io.EOF, wantPresent: true},
			{name: "split JSON reply to streaming request with trailers", streamRequest: true, body: toolJSON[:len(toolJSON)/2], finalBody: toolJSON[len(toolJSON)/2:], trailers: true, terminalErr: io.EOF, wantPresent: true},
			{name: "large JSON reply to streaming request", streamRequest: true, body: largeJSON, endOfStream: true, terminalErr: io.EOF, wantPresent: true},
			{name: "JSON send failure for streaming request", streamRequest: true, body: largeJSON, endOfStream: true, failBodySend: 2, wantPresent: true, wantForwardedMissing: true, wantErr: true},
		} {
			t.Run(api.path+"/"+tt.name, func(t *testing.T) {
				recorder := tracetest.NewSpanRecorder()
				sampler := sdktrace.AlwaysSample()
				if tt.unsampled {
					sampler = sdktrace.NeverSample()
				}
				provider := sdktrace.NewTracerProvider(sdktrace.WithSampler(sampler), sdktrace.WithSpanProcessor(recorder))
				useTracerProvider(t, provider)
				t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })

				contentType := "application/json"
				if tt.eventStream {
					contentType = "text/event-stream"
				}
				requestBody := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hello"}]}`
				if !tt.nonTool {
					requestBody = `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hello"}],"tools":[]}`
				}
				ctx, cancel := context.WithCancel(context.Background())
				t.Cleanup(cancel)
				srv := &terminatedProcessServer{
					replayProcessServer: replayProcessServer{
						ctx: ctx,
						reqs: []*extProcPb.ProcessingRequest{
							newRequestHeaders(map[string]string{":path": api.path}),
							{Request: &extProcPb.ProcessingRequest_RequestBody{
								RequestBody: &extProcPb.HttpBody{Body: []byte(requestBody), EndOfStream: true},
							}},
						},
					},
					terminalErr:     tt.terminalErr,
					failBodySend:    tt.failBodySend,
					failTrailerSend: tt.failTrailerSend,
				}
				if tt.evicted {
					srv.evictCh, srv.evictAfterBody = make(chan struct{}), tt.body != ""
				}
				if !tt.skipResponse {
					srv.reqs = append(srv.reqs, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: "content-type", RawValue: []byte(contentType)},
						}}},
					}})
				}
				if tt.body != "" {
					srv.reqs = append(srv.reqs, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{Body: []byte(tt.body), EndOfStream: tt.endOfStream && tt.finalBody == ""},
					}})
				}
				if tt.finalBody != "" {
					srv.reqs = append(srv.reqs, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{Body: []byte(tt.finalBody), EndOfStream: tt.endOfStream},
					}})
				}
				if tt.trailers {
					srv.reqs = append(srv.reqs, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseTrailers{
						ResponseTrailers: &extProcPb.HttpTrailers{Trailers: &configPb.HeaderMap{}},
					}})
				}
				registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser(), anthropic.NewAnthropicParser()}, logr.Discard())
				server := NewStreamingServer(nil, &mockDirector{stream: tt.streamRequest}, registry, 0)
				server.SetToolCallingTelemetryEnabled(!tt.telemetryDisabled)
				if tt.evicted {
					server.SetEvictChannelLookup(responseEvictLookup{srv.evictCh})
				}
				err := server.Process(srv)
				if tt.wantErr {
					require.Error(t, err)
				} else {
					require.NoError(t, err)
				}
				if tt.evicted {
					require.NotNil(t, srv.sentResponses[len(srv.sentResponses)-1].GetImmediateResponse())
				}
				if tt.failBodySend > 0 {
					require.Equal(t, tt.failBodySend, srv.bodySends)
				}
				var forwarded []byte
				for _, sent := range srv.sentResponses {
					if sent.GetResponseBody() != nil {
						forwarded = append(forwarded, sent.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse().GetBody()...)
					}
				}
				wantBody := tt.body + tt.finalBody
				if tt.failBodySend > 0 {
					wantBody = tt.body[:min(len(tt.body), (tt.failBodySend-1)*envoy.BodyByteLimit)]
				} else if !tt.eventStream && !tt.endOfStream && !tt.trailers {
					wantBody = ""
				}
				require.Equal(t, wantBody, string(forwarded))

				ended := recorder.Ended()
				if tt.unsampled {
					require.Empty(t, ended)
					return
				}
				require.Len(t, ended, 1)
				attributes := make(map[string]bool)
				for _, attr := range ended[0].Attributes() {
					attributes[string(attr.Key)] = attr.Value.AsBool()
					require.NotContains(t, attr.Value.String(), "sentinel_")
				}
				for key, want := range map[string]bool{
					string(semconv.LLMDToolCallingResponseUpstreamPresentKey):              tt.wantPresent,
					string(semconv.LLMDToolCallingResponseForwardedPresentKey):             tt.wantPresent && !tt.wantForwardedMissing,
					string(semconv.LLMDToolCallingResponseUpstreamDetectionIncompleteKey):  (!tt.endOfStream && !tt.trailers) || tt.unreadable,
					string(semconv.LLMDToolCallingResponseForwardedDetectionIncompleteKey): (!tt.endOfStream && !tt.trailers) || tt.unreadable || tt.failBodySend > 0 || tt.failTrailerSend,
				} {
					if tt.nonTool || tt.skipResponse || tt.telemetryDisabled {
						require.NotContains(t, attributes, key)
						continue
					}
					require.Contains(t, attributes, key)
					require.Equal(t, want, attributes[key], key)
				}
			})
		}
	}
}

func TestToolCallingResponseSkipsNonRecordingSpan(t *testing.T) {
	provider := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.NeverSample()))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	_, unsampled := provider.Tracer("test").Start(context.Background(), "request")
	t.Cleanup(func() { unsampled.End() })
	_, disabled := noop.NewTracerProvider().Tracer("test").Start(context.Background(), "request")
	t.Cleanup(func() { disabled.End() })

	for _, tt := range []struct {
		name string
		span trace.Span
	}{
		{name: "nil"},
		{name: "disabled", span: disabled},
		{name: "unsampled", span: unsampled},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.span != nil {
				require.False(t, tt.span.IsRecording())
			}
			for _, eventStream := range []bool{false, true} {
				name := "JSON"
				if eventStream {
					name = "SSE"
				}
				t.Run(name, func(t *testing.T) {
					reqCtx := &RequestContext{
						toolCallingSurface:             reqcommon.APITypeChatCompletions,
						toolCallingRequest:             true,
						toolCallingResponseEventStream: eventStream,
					}
					payload := []byte(`{"choices":[{"message":{"tool_calls":[{"id":"call_1"}]}}]}`)
					if eventStream {
						payload = []byte("data: " + string(payload) + "\n\n")
						reqCtx.observeToolCallingResponse(payload[:3], false, false, tt.span)
						reqCtx.observeToolCallingResponse(payload[:3], false, true, tt.span)
						assert.Nil(t, reqCtx.toolCallingUpstreamDetector)
						assert.Nil(t, reqCtx.toolCallingForwardedDetector)
						payload = payload[3:]
					}
					reqCtx.observeToolCallingResponse(payload, true, false, tt.span)
					reqCtx.observeToolCallingResponse(payload, true, true, tt.span)
					assert.Nil(t, reqCtx.toolCallingUpstreamDetector)
					assert.Nil(t, reqCtx.toolCallingForwardedDetector)
					assert.False(t, reqCtx.toolCallingUpstreamPresent)
					assert.False(t, reqCtx.toolCallingForwardedPresent)
					assert.False(t, reqCtx.toolCallingResponseRecorded)
				})
			}
			t.Run("interrupted", func(t *testing.T) {
				reqCtx := &RequestContext{toolCallingRequest: true}
				reqCtx.recordToolCallingResponse(tt.span)
				assert.False(t, reqCtx.toolCallingResponseRecorded)
			})
		})
	}
}

func TestObserveToolCallingResponseRecordsUpstreamAndForwardedPresence(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	_, span := provider.Tracer("test").Start(context.Background(), "request")

	reqCtx := &RequestContext{
		toolCallingSurface: reqcommon.APITypeChatCompletions,
		toolCallingRequest: true,
	}
	upstream := []byte(`{"model":"internal","choices":[{"message":{"tool_calls":[{"function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}`)
	forwarded := []byte(`{"model":"public","choices":[{"message":{"content":"no tool call"}}]}`)
	reqCtx.observeToolCallingResponse(upstream, true, false, span)
	reqCtx.observeToolCallingResponse(forwarded, true, true, span)
	reqCtx.responseComplete, reqCtx.toolCallingForwardedComplete = true, true
	reqCtx.recordToolCallingResponse(span)
	span.End()

	ended := recorder.Ended()
	require.Len(t, ended, 1)
	attributes := make(map[string]bool, len(ended[0].Attributes()))
	for _, attribute := range ended[0].Attributes() {
		attributes[string(attribute.Key)] = attribute.Value.AsBool()
		assert.NotContains(t, attribute.Value.String(), "sentinel_")
	}
	assert.True(t, attributes[string(semconv.LLMDToolCallingResponseUpstreamPresentKey)])
	assert.False(t, attributes[string(semconv.LLMDToolCallingResponseForwardedPresentKey)])
	assert.False(t, attributes[string(semconv.LLMDToolCallingResponseUpstreamDetectionIncompleteKey)])
	assert.False(t, attributes[string(semconv.LLMDToolCallingResponseForwardedDetectionIncompleteKey)])
}

func TestObserveToolCallingResponseReportsOversizedSSEDetection(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	_, span := provider.Tracer("test").Start(context.Background(), "request")

	reqCtx := &RequestContext{
		toolCallingSurface:             reqcommon.APITypeChatCompletions,
		toolCallingRequest:             true,
		toolCallingResponseEventStream: true,
	}
	oversized := make([]byte, 2<<20)
	copy(oversized, "data: ")
	for i := len("data: "); i < len(oversized); i++ {
		oversized[i] = 'x'
	}
	reqCtx.observeToolCallingResponse(oversized, true, false, span)
	reqCtx.observeToolCallingResponse(oversized, true, true, span)
	reqCtx.responseComplete, reqCtx.toolCallingForwardedComplete = true, true
	reqCtx.recordToolCallingResponse(span)
	span.End()

	ended := recorder.Ended()
	require.Len(t, ended, 1)
	attributes := make(map[string]bool, len(ended[0].Attributes()))
	for _, attribute := range ended[0].Attributes() {
		attributes[string(attribute.Key)] = attribute.Value.AsBool()
	}
	assert.False(t, attributes[string(semconv.LLMDToolCallingResponseUpstreamPresentKey)])
	assert.False(t, attributes[string(semconv.LLMDToolCallingResponseForwardedPresentKey)])
	assert.True(t, attributes[string(semconv.LLMDToolCallingResponseUpstreamDetectionIncompleteKey)])
	assert.True(t, attributes[string(semconv.LLMDToolCallingResponseForwardedDetectionIncompleteKey)])
}

func TestObserveToolCallingResponseHandlesFragmentedSSE(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	_, span := provider.Tracer("test").Start(context.Background(), "request")

	reqCtx := &RequestContext{
		toolCallingSurface:             reqcommon.APITypeMessages,
		toolCallingRequest:             true,
		toolCallingResponseEventStream: true,
	}
	first := []byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",")
	second := []byte("\"content_block\":{\"type\":\"tool_use\"}}\n\n")
	reqCtx.observeToolCallingResponse(first, false, false, span)
	reqCtx.observeToolCallingResponse(first, false, true, span)
	require.False(t, reqCtx.toolCallingUpstreamPresent)
	require.False(t, reqCtx.toolCallingForwardedPresent)
	require.False(t, reqCtx.toolCallingResponseRecorded)
	reqCtx.observeToolCallingResponse(second, true, false, span)
	reqCtx.observeToolCallingResponse(second, true, true, span)
	require.False(t, reqCtx.toolCallingResponseRecorded)
	reqCtx.responseComplete, reqCtx.toolCallingForwardedComplete = true, true
	reqCtx.recordToolCallingResponse(span)

	require.True(t, reqCtx.toolCallingResponseRecorded)
	require.True(t, reqCtx.toolCallingUpstreamPresent)
	require.True(t, reqCtx.toolCallingForwardedPresent)
	span.End()
	ended := recorder.Ended()
	require.Len(t, ended, 1)
	require.ElementsMatch(t, (toolcalling.ResponseSummary{
		ToolCallingRequested:     true,
		UpstreamToolCallPresent:  true,
		ForwardedToolCallPresent: true,
	}).SpanAttributes(), ended[0].Attributes())
}

func TestObserveToolCallingResponseSkipsNonToolRequest(t *testing.T) {
	provider := sdktrace.NewTracerProvider()
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	_, span := provider.Tracer("test").Start(context.Background(), "request")
	t.Cleanup(func() { span.End() })
	require.True(t, span.IsRecording())

	reqCtx := &RequestContext{toolCallingSurface: reqcommon.APITypeChatCompletions}
	reqCtx.observeToolCallingResponse([]byte(`{"choices":[]}`), true, false, span)
	reqCtx.observeToolCallingResponse([]byte(`{"choices":[]}`), true, true, span)

	assert.Nil(t, reqCtx.toolCallingUpstreamDetector)
	assert.Nil(t, reqCtx.toolCallingForwardedDetector)
	assert.False(t, reqCtx.toolCallingResponseRecorded)
}

func TestClearToolCallingResponseDetectors(t *testing.T) {
	upstreamDetector, err := toolcalling.NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	forwardedDetector, err := toolcalling.NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	partial := []byte(`data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"sentinel_arguments"}}]}}]}`)
	upstreamDetector.Observe(partial, false)
	forwardedDetector.Observe(partial, false)

	reqCtx := &RequestContext{
		toolCallingUpstreamDetector:  upstreamDetector,
		toolCallingForwardedDetector: forwardedDetector,
	}
	reqCtx.clearToolCallingResponseDetectors()

	assert.Nil(t, reqCtx.toolCallingUpstreamDetector)
	assert.Nil(t, reqCtx.toolCallingForwardedDetector)
	assert.False(t, upstreamDetector.Observe(nil, true), "closed upstream detector must discard the partial event")
	assert.False(t, forwardedDetector.Observe(nil, true), "closed forwarded detector must discard the partial event")
}

func TestHandleStreamedResponseBody(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	tests := []struct {
		name    string
		body    []byte
		want    fwkrh.Usage
		wantErr bool
	}{
		{
			name:    "streaming request without usage",
			body:    []byte(streamingBodyWithoutUsage),
			wantErr: false,
			// In the middle of streaming response, so request context response is not set yet.
		},
		{
			name:    "streaming request with usage",
			body:    []byte(streamingBodyWithUsage),
			wantErr: false,
			want: fwkrh.Usage{
				PromptTokens:     7,
				TotalTokens:      17,
				CompletionTokens: 10,
			},
		},
		{
			name:    "streaming request with usage and cached tokens",
			body:    []byte(streamingBodyWithUsageAndCachedTokens),
			wantErr: false,
			want: fwkrh.Usage{
				PromptTokens:     7,
				TotalTokens:      17,
				CompletionTokens: 10,
				PromptTokenDetails: &fwkrh.PromptTokenDetails{
					CachedTokens: 5,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &StreamingServer{
				parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
			}
			server.director = &mockDirector{}
			reqCtx := &RequestContext{
				Response: &Response{
					Headers: map[string]string{
						"content-type": "text/event-stream; charset=utf-8",
					},
				},
				SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
			}
			server.HandleResponseBody(ctx, reqCtx, test.body, true) // Hard coded to true since openAIParser does not endOfStream to switch logic.

			if diff := cmp.Diff(test.want, reqCtx.Usage); diff != "" {
				t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}

func TestHandleResponseBodyWithoutSchedulingRequest(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
	}
	server.director = &mockDirector{}
	timeBaseline := time.Now()
	reqCtx := &RequestContext{
		IncomingModelName:         "incoming-model",
		TargetModelName:           "target-model",
		Priority:                  3,
		RequestReceivedTimestamp:  timeBaseline,
		responseCompleteTimestamp: timeBaseline.Add(time.Second),
		Response: &Response{
			Headers: map[string]string{},
		},
	}

	require.NotPanics(t, func() {
		server.HandleResponseBody(ctx, reqCtx, []byte(body), true)
	})

	histogram := findHistogramMetric(t, "llm_d_epp_request_ntpot_seconds", map[string]string{
		"model_name":        "incoming-model",
		"target_model_name": "target-model",
		"fairness_id":       reqcommon.DefaultFairnessID,
		"priority":          "3",
	})
	require.Equal(t, uint64(1), histogram.GetSampleCount())
}

func findHistogramMetric(t *testing.T, name string, labels map[string]string) *dto.Histogram {
	t.Helper()

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			if metricHasLabels(metric, labels) {
				return metric.GetHistogram()
			}
		}
	}
	t.Fatalf("metric %q with labels %v not found", name, labels)
	return nil
}

// histogramSampleCount reports how many observations a histogram holds for the
// given labels, and zero when the series was never created.
func histogramSampleCount(t *testing.T, name string, labels map[string]string) uint64 {
	t.Helper()

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			if metricHasLabels(metric, labels) {
				return metric.GetHistogram().GetSampleCount()
			}
		}
	}
	return 0
}

func metricHasLabels(metric *dto.Metric, labels map[string]string) bool {
	got := make(map[string]string, len(metric.GetLabel()))
	for _, label := range metric.GetLabel() {
		got[label.GetName()] = label.GetValue()
	}
	for key, want := range labels {
		if got[key] != want {
			return false
		}
	}
	return true
}

func TestHandleResponseMetricsByAPISurface(t *testing.T) {
	for _, apiType := range []reqcommon.APIType{reqcommon.APITypeMessages, reqcommon.APITypeChatCompletions} {
		for _, mode := range []string{"json", "streaming", "interrupted"} {
			t.Run(apiType.String()+"/"+mode, func(t *testing.T) {
				eppmetrics.Register()
				eppmetrics.Reset()
				t.Cleanup(eppmetrics.Reset)
				streaming := mode != "json"
				complete := mode != "interrupted"
				contentType := reqcommon.ContentTypeJSON
				chunks := [][]byte{[]byte(`{"content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":2,"output_tokens":3,"cache_read_input_tokens":1}}`)}
				if apiType == reqcommon.APITypeChatCompletions {
					chunks = [][]byte{[]byte(`{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":3,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":1}}}`)}
				}
				if streaming {
					contentType = fwkrequest.MediaTypeEventStream
					if apiType == reqcommon.APITypeMessages {
						chunks = [][]byte{
							[]byte("event: message_start\ndata: " + `{"type":"message_start","message":{"usage":{"input_tokens":2,"cache_read_input_tokens":1}}}` + "\n\n"),
							[]byte("event: message_delta\ndata: " + `{"type":"message_delta","usage":{"output_tokens":3}}` + "\n\n"),
							[]byte("event: message_stop\ndata: " + `{"type":"message_stop"}` + "\n\n"),
						}
					} else {
						chunks = [][]byte{
							[]byte("data: " + `{"choices":[{"delta":{"content":"hi"}}]}` + "\n\n"),
							[]byte("data: " + `{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":1}}}` + "\n\n"),
							[]byte("data: [DONE]\n\n"),
						}
					}
				}
				server := NewStreamingServer(nil, &mockDirector{}, NewParserRegistry([]fwkrh.Parser{
					openai.NewOpenAIParser(), anthropic.NewAnthropicParser(),
				}, logr.Discard()), 0)
				reqCtx := &RequestContext{
					IncomingModelName: "model", TargetModelName: "model", apiSurface: apiType.Surface(),
					RequestReceivedTimestamp: time.Now().Add(-time.Second), modelServerStreaming: streaming,
					Request:  &Request{Headers: map[string]string{":path": apiType.Path()}},
					Response: &Response{Headers: map[string]string{reqcommon.HeaderContentType: contentType}},
				}
				ctx := logutil.NewTestLoggerIntoContext(context.Background())
				size := 0
				for i, chunk := range chunks {
					last := complete && i == len(chunks)-1
					if last {
						reqCtx.responseCompleteTimestamp = time.Now()
					}
					server.HandleResponseBody(ctx, reqCtx, chunk, last)
					size += len(chunk)
				}
				labels := map[string]string{"api_surface": apiType.String()}
				for name, sum := range map[string]float64{
					"llm_d_epp_request_input_tokens": 3, "llm_d_epp_request_output_tokens": 3,
					"llm_d_epp_request_cached_tokens": 1, "llm_d_epp_response_size_bytes": float64(size),
				} {
					if complete {
						histogram := findHistogramMetric(t, name, labels)
						require.Equal(t, uint64(1), histogram.GetSampleCount(), name)
						require.Equal(t, sum, histogram.GetSampleSum(), name)
					} else {
						require.Zero(t, histogramSampleCount(t, name, labels), name)
					}
				}
				wantCount := uint64(0)
				if complete {
					wantCount = 1
				}
				for _, name := range []string{"llm_d_epp_request_ntpot_seconds", "llm_d_epp_request_ttft_seconds"} {
					require.Equal(t, wantCount, histogramSampleCount(t, name, labels), name)
				}
				if !streaming {
					wantCount = 0
				}
				require.Equal(t, wantCount, histogramSampleCount(t, "llm_d_epp_request_streaming_tpot_seconds", labels))
				wantITL := uint64(0)
				if streaming {
					wantITL = uint64(len(chunks) - 1)
				}
				require.Equal(t, wantITL, histogramSampleCount(t, "llm_d_epp_request_streaming_itl_seconds", labels))
			})
		}
	}
}

func TestHandleResponseBodyModelStreaming_TokenAccumulation(t *testing.T) {
	t.Parallel()

	type chunkStream struct {
		body        []byte
		endOfStream bool
	}

	tests := []struct {
		name      string
		chunks    []chunkStream
		wantUsage fwkrh.Usage
	}{
		{
			name: "Standard: Usage and DONE in same chunk",
			chunks: []chunkStream{
				{body: []byte(`data: {"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}` + "\n" + `data: [DONE]`), endOfStream: true},
			},
			wantUsage: fwkrh.Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "Split: Usage in Chunk 1, DONE in Chunk 2",
			chunks: []chunkStream{
				// Chunk 1: Usage data arrives
				{body: []byte(`data: {"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}` + "\n"), endOfStream: false},
				// Chunk 2: Stream termination. Should NOT overwrite the usage from Chunk 1.
				{body: []byte(`data: [DONE]`), endOfStream: true},
			},
			wantUsage: fwkrh.Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "Fragmented: Content -> Usage -> DONE",
			chunks: []chunkStream{
				{body: []byte(`data: {"choices":[{"text":"Hello"}]}` + "\n"), endOfStream: false},
				{body: []byte(`data: {"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}` + "\n"), endOfStream: false},
				{body: []byte(`data: [DONE]`), endOfStream: true},
			},
			wantUsage: fwkrh.Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "Split mid-event: usage line cut across chunks",
			chunks: []chunkStream{
				{body: []byte(`data: {"choices":[{"text":"Hello"}]}` + "\n" + `data: {"usage":{"prompt_tok`), endOfStream: false},
				{body: []byte(`ens":5,"completion_tokens":10,"total_tokens":15}}` + "\n"), endOfStream: false},
				{body: []byte(`data: [DONE]`), endOfStream: true},
			},
			wantUsage: fwkrh.Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "Split mid-event: usage line ends at end of stream without newline",
			chunks: []chunkStream{
				{body: []byte(`data: {"usage":{"prompt_tokens":5,`), endOfStream: false},
				{body: []byte(`"completion_tokens":10,"total_tokens":15}}`), endOfStream: true},
			},
			wantUsage: fwkrh.Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "No Usage Data",
			chunks: []chunkStream{
				{body: []byte(`data: {"choices":[{"text":"Hello"}]}` + "\n"), endOfStream: false},
				{body: []byte(`data: [DONE]`), endOfStream: true},
			},
			wantUsage: fwkrh.Usage{}, // Zero values
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{
				parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
				director:       &mockDirector{},
			}
			reqCtx := &RequestContext{
				Response: &Response{
					Headers: map[string]string{
						"content-type": "text/event-stream",
					},
				},
				SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
			}

			for _, chunk := range tc.chunks {
				server.HandleResponseBody(context.Background(), reqCtx, chunk.body, chunk.endOfStream)
			}

			assert.Equal(t, tc.wantUsage, reqCtx.Usage, "Usage data should match expected accumulation")
		})
	}
}

// A large response.completed event split across chunks must still yield usage.
func TestHandleResponseBodyModelStreaming_LargeResponsesCompletedSplit(t *testing.T) {
	t.Parallel()

	stream := "event: response.output_text.delta\n" +
		`data: {"type":"response.output_text.delta","delta":"hi"}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":{"instructions":"` + strings.Repeat("x", 200_000) +
		`","usage":{"input_tokens":290000,"output_tokens":900,"total_tokens":290900}}}` + "\n\n"
	want := fwkrh.Usage{PromptTokens: 290000, CompletionTokens: 900, TotalTokens: 290900}

	for _, chunkSize := range []int{len(stream), 65536, 16384, 1000} {
		t.Run(strconv.Itoa(chunkSize), func(t *testing.T) {
			server := &StreamingServer{
				parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
				director:       &mockDirector{},
			}
			reqCtx := &RequestContext{
				Response:          &Response{Headers: map[string]string{"content-type": "text/event-stream"}},
				SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
			}
			for i := 0; i < len(stream); i += chunkSize {
				end := min(i+chunkSize, len(stream))
				server.HandleResponseBody(context.Background(), reqCtx, []byte(stream[i:end]), end == len(stream))
			}
			assert.Equal(t, want.PromptTokens, reqCtx.Usage.PromptTokens)
			assert.Equal(t, want.CompletionTokens, reqCtx.Usage.CompletionTokens)
			assert.Equal(t, 2, reqCtx.StreamedEvents)
		})
	}
}

func TestCompleteSSELines(t *testing.T) {
	t.Parallel()

	oversized := strings.Repeat("x", maxSSELineBytes)
	tests := []struct {
		name   string
		chunks []string
		want   []string // parser input per chunk
	}{
		{
			name:   "partial line carried to the next chunk",
			chunks: []string{"a\nb", "c", "d\ne", "f"},
			want:   []string{"a\n", "", "bcd\n", "ef"},
		},
		{
			name:   "final chunk returned whole with the carry",
			chunks: []string{"a\nb", "c\nd"},
			want:   []string{"a\n", "bc\nd"},
		},
		{
			name:   "oversized line dropped up to its newline",
			chunks: []string{"a\n" + oversized, "x", "tail\nb\n", "c"},
			want:   []string{"a\n", "", "b\n", "c"},
		},
		{
			name:   "oversized tail in one chunk dropped up to its newline",
			chunks: []string{"a\n" + oversized + "x", "tail\nb\n", "c"},
			want:   []string{"a\n", "b\n", "c"},
		},
		{
			name:   "oversized complete line dropped without skipping the next chunk",
			chunks: []string{"a\n" + oversized + "x\n", "b\n"},
			want:   []string{"a\n", "b\n"},
		},
		{
			name:   "short lines over the cap in total kept",
			chunks: []string{strings.Repeat(strings.Repeat("y", 1023)+"\n", 5000), "b\n"},
			want:   []string{strings.Repeat(strings.Repeat("y", 1023)+"\n", 5000), "b\n"},
		},
		{
			name:   "oversized last line at end of stream dropped",
			chunks: []string{"a\n" + oversized, "x"},
			want:   []string{"a\n", ""},
		},
		{
			name:   "oversized tail of the final chunk dropped",
			chunks: []string{"a\n", "b\n" + oversized + "x"},
			want:   []string{"a\n", "b\n"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reqCtx := &RequestContext{}
			for i, chunk := range tc.chunks {
				got := completeSSELines(context.Background(), reqCtx, []byte(chunk), i == len(tc.chunks)-1)
				assert.Equal(t, tc.want[i], string(got), "chunk %d", i)
			}
		})
	}

	assert.True(t, isEventStream("text/event-stream; charset=utf-8"))
	assert.False(t, isEventStream("application/grpc"))
}

// The Anthropic streaming format reports prompt tokens in message_start and completion
// tokens in a message_delta that reaches the EPP in a later chunk.
func TestHandleResponseBodyModelStreaming_AnthropicUsageAccumulation(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	chunks := [][]byte{
		[]byte(`event: message_start` + "\n" + `data: {"type":"message_start","message":{"usage":{"input_tokens":1000,"cache_read_input_tokens":800}}}` + "\n\n"),
		[]byte(`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}` + "\n\n"),
		[]byte(`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":200}}` + "\n\n"),
		[]byte(`event: message_stop` + "\n" + `data: {"type":"message_stop"}`),
	}

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{anthropic.NewAnthropicParser()}, logr.Discard()),
		director:       &mockDirector{},
	}
	reqCtx := &RequestContext{
		IncomingModelName: "incoming-model",
		TargetModelName:   "target-model",
		apiSurface:        reqcommon.APITypeMessages.Surface(),
		Request: &Request{
			Headers: map[string]string{
				":path": "/v1/messages",
			},
		},
		Response: &Response{
			Headers: map[string]string{
				"content-type": "text/event-stream",
			},
		},
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
	}

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	for i, chunk := range chunks {
		server.HandleResponseBody(ctx, reqCtx, chunk, i == len(chunks)-1)
	}

	// 1000 uncached plus 800 read from the cache.
	wantUsage := fwkrh.Usage{
		PromptTokens:       1800,
		CompletionTokens:   200,
		TotalTokens:        2000,
		PromptTokenDetails: &fwkrh.PromptTokenDetails{CachedTokens: 800},
	}
	assert.Equal(t, wantUsage, reqCtx.Usage, "message_delta must not discard the usage reported by message_start")

	labels := map[string]string{
		"model_name":        "incoming-model",
		"target_model_name": "target-model",
		"fairness_id":       reqcommon.DefaultFairnessID,
		"priority":          "0",
		"api_surface":       reqcommon.APITypeMessages.String(),
	}
	// Each token count belongs to one request, so accumulating usage across chunks must not
	// turn into a second observation on the chunk that completes it.
	inputTokens := findHistogramMetric(t, "llm_d_epp_request_input_tokens", labels)
	require.Equal(t, uint64(1), inputTokens.GetSampleCount())
	require.Equal(t, float64(1800), inputTokens.GetSampleSum())

	cachedTokens := findHistogramMetric(t, "llm_d_epp_request_cached_tokens", labels)
	require.Equal(t, uint64(1), cachedTokens.GetSampleCount())
	require.Equal(t, float64(800), cachedTokens.GetSampleSum())

	outputTokens := findHistogramMetric(t, "llm_d_epp_request_output_tokens", labels)
	require.Equal(t, uint64(1), outputTokens.GetSampleCount())
	require.Equal(t, float64(200), outputTokens.GetSampleSum())
}

// Both message_start and message_delta can carry the cumulative input and cache
// counts. The counts belong to one request and must reach the histograms once.
func TestHandleResponseBodyModelStreaming_AnthropicCumulativeUsage(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	chunks := [][]byte{
		[]byte(`event: message_start` + "\n" + `data: {"type":"message_start","message":{"usage":{"input_tokens":1000,"cache_read_input_tokens":800}}}` + "\n\n"),
		[]byte(`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}` + "\n\n"),
		[]byte(`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1000,"cache_read_input_tokens":800,"output_tokens":200}}` + "\n\n"),
		[]byte(`event: message_stop` + "\n" + `data: {"type":"message_stop"}`),
	}

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{anthropic.NewAnthropicParser()}, logr.Discard()),
		director:       &mockDirector{},
	}
	reqCtx := &RequestContext{
		IncomingModelName: "incoming-model",
		TargetModelName:   "target-model",
		apiSurface:        reqcommon.APITypeMessages.Surface(),
		Request: &Request{
			Headers: map[string]string{
				":path": "/v1/messages",
			},
		},
		Response: &Response{
			Headers: map[string]string{
				"content-type": "text/event-stream",
			},
		},
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
	}

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	for i, chunk := range chunks {
		server.HandleResponseBody(ctx, reqCtx, chunk, i == len(chunks)-1)
	}

	wantUsage := fwkrh.Usage{
		PromptTokens:       1800,
		CompletionTokens:   200,
		TotalTokens:        2000,
		PromptTokenDetails: &fwkrh.PromptTokenDetails{CachedTokens: 800},
	}
	assert.Equal(t, wantUsage, reqCtx.Usage)

	labels := map[string]string{
		"model_name":        "incoming-model",
		"target_model_name": "target-model",
		"fairness_id":       reqcommon.DefaultFairnessID,
		"priority":          "0",
		"api_surface":       reqcommon.APITypeMessages.String(),
	}
	inputTokens := findHistogramMetric(t, "llm_d_epp_request_input_tokens", labels)
	require.Equal(t, uint64(1), inputTokens.GetSampleCount())
	require.Equal(t, float64(1800), inputTokens.GetSampleSum())

	cachedTokens := findHistogramMetric(t, "llm_d_epp_request_cached_tokens", labels)
	require.Equal(t, uint64(1), cachedTokens.GetSampleCount())
	require.Equal(t, float64(800), cachedTokens.GetSampleSum())

	outputTokens := findHistogramMetric(t, "llm_d_epp_request_output_tokens", labels)
	require.Equal(t, uint64(1), outputTokens.GetSampleCount())
	require.Equal(t, float64(200), outputTokens.GetSampleSum())
}

// A stream that never reaches end of stream records no token counts. The usage
// parsed so far still accumulates on the request context for the response record.
func TestHandleResponseBodyModelStreaming_IncompleteStreamRecordsNoTokens(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	chunks := [][]byte{
		[]byte(`event: message_start` + "\n" + `data: {"type":"message_start","message":{"usage":{"input_tokens":1000,"cache_read_input_tokens":800}}}` + "\n\n"),
		[]byte(`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}` + "\n\n"),
	}

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{anthropic.NewAnthropicParser()}, logr.Discard()),
		director:       &mockDirector{},
	}
	reqCtx := &RequestContext{
		IncomingModelName: "incoming-model",
		TargetModelName:   "target-model",
		Request: &Request{
			Headers: map[string]string{
				":path": "/v1/messages",
			},
		},
		Response: &Response{
			Headers: map[string]string{
				"content-type": "text/event-stream",
			},
		},
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
	}

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	for _, chunk := range chunks {
		server.HandleResponseBody(ctx, reqCtx, chunk, false)
	}

	assert.Equal(t, 1800, reqCtx.Usage.PromptTokens)

	labels := map[string]string{
		"model_name":        "incoming-model",
		"target_model_name": "target-model",
		"fairness_id":       reqcommon.DefaultFairnessID,
		"priority":          "0",
	}
	assert.Zero(t, histogramSampleCount(t, "llm_d_epp_request_input_tokens", labels))
	assert.Zero(t, histogramSampleCount(t, "llm_d_epp_request_cached_tokens", labels))
	assert.Zero(t, histogramSampleCount(t, "llm_d_epp_request_output_tokens", labels))
}

func TestGenerateResponseHeaders_Sanitization(t *testing.T) {
	server := &StreamingServer{}
	reqCtx := &RequestContext{
		Response: &Response{
			Headers: map[string]string{
				"x-backend-server":              "vllm-v0.6.3",                 // should passthrough
				metadata.ObjectiveKey:           "sensitive-objective-id",      // should be stripped
				metadata.OldObjectiveKey:        "old-sensitive-objective-id",  // should be stripped
				metadata.DestinationEndpointKey: "10.2.0.5:8080",               // should be stripped
				"content-length":                "500",                         // should be stripped
				"x-llm-d-peer-topology":         "host=node12,zone=us-east1-a", // stamped by topology-stamp-handler, must reach the coordinator
			},
		},
	}

	results := server.generateResponseHeaders(reqCtx)

	gotHeaders := make(map[string]string)
	for _, h := range results {
		gotHeaders[h.Header.Key] = string(h.Header.RawValue)
	}

	assert.Contains(t, gotHeaders, "x-backend-server")
	assert.Contains(t, gotHeaders, "x-went-into-resp-headers")
	assert.NotContains(t, gotHeaders, metadata.ObjectiveKey)
	assert.NotContains(t, gotHeaders, metadata.OldObjectiveKey)
	assert.NotContains(t, gotHeaders, metadata.DestinationEndpointKey)
	assert.NotContains(t, gotHeaders, "content-length")
	assert.Equal(t, "host=node12,zone=us-east1-a", gotHeaders["x-llm-d-peer-topology"])
}

func TestGenerateResponseHeaders_FlowQueueDuration(t *testing.T) {
	server := &StreamingServer{}

	tests := []struct {
		name      string
		admitted  bool
		duration  time.Duration
		wantValue string
		wantEmit  bool
	}{
		{
			name:     "not admitted omits header",
			admitted: false,
			wantEmit: false,
		},
		{
			name:      "admitted with zero wait emits 0",
			admitted:  true,
			duration:  500 * time.Microsecond,
			wantValue: "0",
			wantEmit:  true,
		},
		{
			name:      "admitted with measurable wait emits milliseconds",
			admitted:  true,
			duration:  1500 * time.Millisecond,
			wantValue: "1500",
			wantEmit:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reqCtx := &RequestContext{
				FlowControlAdmitted:      tc.admitted,
				FlowControlQueueDuration: tc.duration,
				Response:                 &Response{Headers: map[string]string{}},
			}

			gotHeaders := make(map[string]string)
			for _, h := range server.generateResponseHeaders(reqCtx) {
				gotHeaders[h.Header.Key] = string(h.Header.RawValue)
			}

			if tc.wantEmit {
				assert.Equal(t, tc.wantValue, gotHeaders[metadata.FlowQueueDurationHeaderKey])
			} else {
				assert.NotContains(t, gotHeaders, metadata.FlowQueueDurationHeaderKey)
			}
		})
	}
}

func TestRewriteModelName(t *testing.T) {
	tests := []struct {
		name          string
		body          string
		targetModel   string
		incomingModel string
		want          string
		wantMutated   bool
	}{
		{
			name:          "non-streaming response with model rewrite",
			body:          `{"id":"cmpl-123","model":"vllm-backend-01","choices":[]}`,
			targetModel:   "vllm-backend-01",
			incomingModel: "gpt-4-proxy",
			want:          `{"id":"cmpl-123","model":"gpt-4-proxy","choices":[]}`,
			wantMutated:   true,
		},
		{
			name:          "streaming SSE chunk with model rewrite",
			body:          `data: {"id":"cmpl-123","model":"vllm-backend-01","choices":[]}` + "\n\n",
			targetModel:   "vllm-backend-01",
			incomingModel: "gpt-4-proxy",
			want:          `data: {"id":"cmpl-123","model":"gpt-4-proxy","choices":[]}` + "\n\n",
			wantMutated:   true,
		},
		{
			name:          "no rewrite when names are the same",
			body:          `{"model":"same-model"}`,
			targetModel:   "same-model",
			incomingModel: "same-model",
			want:          `{"model":"same-model"}`,
			wantMutated:   false,
		},
		{
			name:          "no rewrite when target is empty",
			body:          `{"model":"some-model"}`,
			targetModel:   "",
			incomingModel: "gpt-4-proxy",
			want:          `{"model":"some-model"}`,
			wantMutated:   false,
		},
		{
			name:          "no rewrite when incoming is empty",
			body:          `{"model":"some-model"}`,
			targetModel:   "some-model",
			incomingModel: "",
			want:          `{"model":"some-model"}`,
			wantMutated:   false,
		},
		{
			name:          "model field with space after colon",
			body:          `{"model": "vllm-backend-01"}`,
			targetModel:   "vllm-backend-01",
			incomingModel: "gpt-4-proxy",
			want:          `{"model": "gpt-4-proxy"}`,
			wantMutated:   true,
		},
		{
			name:          "body without model field is unchanged",
			body:          `{"id":"cmpl-123","choices":[]}`,
			targetModel:   "vllm-backend-01",
			incomingModel: "gpt-4-proxy",
			want:          `{"id":"cmpl-123","choices":[]}`,
			wantMutated:   false,
		},
		{
			name:          "DONE marker is not affected",
			body:          "data: [DONE]\n",
			targetModel:   "vllm-backend-01",
			incomingModel: "gpt-4-proxy",
			want:          "data: [DONE]\n",
			wantMutated:   false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, mutated := rewriteModelName([]byte(tc.body), tc.targetModel, tc.incomingModel)
			assert.Equal(t, tc.want, string(got))
			assert.Equal(t, tc.wantMutated, mutated)
		})
	}
}

func TestResponseSizeAccumulation(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name             string
		chunks           [][]byte
		wantResponseSize int
	}{
		{
			name:             "single chunk",
			chunks:           [][]byte{[]byte("hello world")},
			wantResponseSize: 11,
		},
		{
			name:             "multiple chunks",
			chunks:           [][]byte{[]byte("chunk1"), []byte("chunk2"), []byte("chunk3")},
			wantResponseSize: 18,
		},
		{
			name:             "empty chunk",
			chunks:           [][]byte{[]byte("")},
			wantResponseSize: 0,
		},
		{
			name:             "mixed chunks with empty",
			chunks:           [][]byte{[]byte("data"), []byte(""), []byte("more")},
			wantResponseSize: 8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := &StreamingServer{
				parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
				director:       &mockDirector{},
			}
			reqCtx := &RequestContext{
				Response: &Response{
					Headers: map[string]string{},
				},
				SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
			}
			for i, chunk := range tt.chunks {
				endOfStream := i == len(tt.chunks)-1
				server.HandleResponseBody(ctx, reqCtx, chunk, endOfStream)
			}
			assert.Equal(t, tt.wantResponseSize, reqCtx.responseSize)
		})
	}
}

func TestStreamedEventAccumulation(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name               string
		headers            map[string]string
		chunks             [][]byte
		wantStreamedEvents int
	}{
		{
			name:    "events accumulate across chunks",
			headers: map[string]string{"content-type": "text/event-stream"},
			chunks: [][]byte{
				[]byte(`data: {"choices":[{"text":"He"}]}` + "\n" + `data: {"choices":[{"text":"llo"}]}` + "\n"),
				[]byte(`data: {"choices":[{"text":"!"}]}` + "\n" + `data: [DONE]`),
			},
			wantStreamedEvents: 3,
		},
		{
			name:               "a truncated stream keeps the count it reached",
			headers:            map[string]string{"content-type": "text/event-stream"},
			chunks:             [][]byte{[]byte(`data: {"choices":[{"text":"He"}]}` + "\n")},
			wantStreamedEvents: 1,
		},
		{
			name:    "an event cut after the prefix at a chunk boundary counts once",
			headers: map[string]string{"content-type": "text/event-stream"},
			chunks: [][]byte{
				[]byte(`data: {"choices":[{"text":"He"}]}` + "\n" + `data: {"cho`),
				[]byte(`ices":[{"text":"llo"}]}` + "\n" + `data: [DONE]`),
			},
			wantStreamedEvents: 2,
		},
		{
			name:               "a non-streamed response counts nothing",
			headers:            map[string]string{"content-type": "application/json"},
			chunks:             [][]byte{[]byte(body)},
			wantStreamedEvents: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := &StreamingServer{
				parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
				director:       &mockDirector{},
			}
			reqCtx := &RequestContext{
				Response:          &Response{Headers: tt.headers},
				SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
			}
			for i, chunk := range tt.chunks {
				server.HandleResponseBody(ctx, reqCtx, chunk, i == len(tt.chunks)-1)
			}
			assert.Equal(t, tt.wantStreamedEvents, reqCtx.StreamedEvents)
		})
	}
}

// TestGenerateResponseBodyResponses_DynamicMetadata verifies that DynamicMetadata is attached
// to the last ProcessingResponse chunk and not to earlier chunks. This is a regression test for
// the bug where the metadata was computed by plugins but silently dropped before reaching Envoy.
func TestGenerateResponseBodyResponses_DynamicMetadata(t *testing.T) {
	meta := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"envoy.lb": structpb.NewStructValue(&structpb.Struct{
				Fields: map[string]*structpb.Value{
					"x-gateway-inference-request-cost": structpb.NewNumberValue(42),
				},
			}),
		},
	}

	tests := []struct {
		name            string
		body            []byte
		dynamicMetadata *structpb.Struct
		wantMetaOnLast  bool
	}{
		{
			name:            "metadata attached to last chunk when provided",
			body:            []byte(`{"result":"ok"}`),
			dynamicMetadata: meta,
			wantMetaOnLast:  true,
		},
		{
			name:            "no metadata when nil",
			body:            []byte(`{"result":"ok"}`),
			dynamicMetadata: nil,
			wantMetaOnLast:  false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			responses := generateResponseBodyResponses(tc.body, true, tc.dynamicMetadata)
			require.NotEmpty(t, responses, "expected at least one response")

			last := responses[len(responses)-1]
			if tc.wantMetaOnLast {
				require.NotNil(t, last.DynamicMetadata, "expected DynamicMetadata on last chunk")
				envoyLb, ok := last.DynamicMetadata.Fields["envoy.lb"]
				require.True(t, ok, "expected envoy.lb namespace in DynamicMetadata")
				cost, ok := envoyLb.GetStructValue().Fields["x-gateway-inference-request-cost"]
				require.True(t, ok)
				assert.Equal(t, float64(42), cost.GetNumberValue())
			} else {
				assert.Nil(t, last.DynamicMetadata, "expected no DynamicMetadata when none provided")
			}

			// Earlier chunks must never carry metadata.
			for i, r := range responses[:len(responses)-1] {
				assert.Nil(t, r.DynamicMetadata, "chunk %d should not have DynamicMetadata", i)
			}
		})
	}
}

// BenchmarkObserveToolCallingResponse includes span creation, response observation,
// deferred response finalization, detector cleanup, and span completion. Providers
// have no exporters; the benchmark measures local work for one complete response.
func BenchmarkObserveToolCallingResponse(b *testing.B) {
	ctx := context.Background()
	recording := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	unsampled := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.NeverSample()))
	b.Cleanup(func() {
		if err := recording.Shutdown(ctx); err != nil {
			b.Fatal(err)
		}
		if err := unsampled.Shutdown(ctx); err != nil {
			b.Fatal(err)
		}
	})

	for _, size := range []int{1024, 64 * 1024} {
		padding := strings.Repeat("x", size)
		for _, api := range []struct {
			surface reqcommon.APIType
			json    string
			sse     string
		}{
			{
				surface: reqcommon.APITypeChatCompletions,
				json:    `{"choices":[{"message":{"content":"` + padding + `","tool_calls":[{"id":"call_1","type":"function","function":{"name":"sentinel_name","arguments":"{}"}}]}}]}`,
				sse:     "data: " + `{"choices":[{"delta":{"content":"` + padding + `"}}]}` + "\n\ndata: " + `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"sentinel_name","arguments":"{}"}}]}}]}` + "\n\n",
			},
			{
				surface: reqcommon.APITypeMessages,
				json:    `{"content":[{"type":"text","text":"` + padding + `"},{"type":"tool_use","id":"call_1","name":"sentinel_name","input":{}}]}`,
				sse:     "event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + padding + `"}}` + "\n\nevent: content_block_start\ndata: " + `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_1","name":"sentinel_name","input":{}}}` + "\n\n",
			},
		} {
			for _, mode := range []struct {
				name        string
				body        []byte
				eventStream bool
			}{
				{name: "JSON", body: []byte(api.json)},
				{name: "SSE", body: []byte(api.sse), eventStream: true},
			} {
				chunks := [][]byte{mode.body}
				if mode.eventStream {
					chunks = nil
					for offset := 0; offset < len(mode.body); offset += 128 {
						chunks = append(chunks, mode.body[offset:min(offset+128, len(mode.body))])
					}
				}
				detector, err := toolcalling.NewResponseDetector(api.surface, mode.eventStream)
				if err != nil {
					b.Fatal(err)
				}
				for i, chunk := range chunks {
					if found := detector.Observe(chunk, i == len(chunks)-1); i == len(chunks)-1 && !found {
						b.Fatal("benchmark response must contain a detectable tool call")
					}
				}
				for _, tracing := range []struct {
					name     string
					provider trace.TracerProvider
				}{
					{name: "disabled", provider: noop.NewTracerProvider()},
					{name: "unsampled", provider: unsampled},
					{name: "recording", provider: recording},
				} {
					for _, toolRequest := range []bool{false, true} {
						request := "non_tool"
						if toolRequest {
							request = "tool"
						}
						b.Run(fmt.Sprintf("%s/%s/%d/%s/%s", api.surface, mode.name, size, tracing.name, request), func(b *testing.B) {
							tracer := tracing.provider.Tracer("benchmark")
							b.ReportAllocs()
							for b.Loop() {
								_, span := tracer.Start(ctx, "request")
								reqCtx := &RequestContext{
									toolCallingSurface:             api.surface,
									toolCallingRequest:             toolRequest,
									toolCallingResponseEventStream: mode.eventStream,
								}
								for i, chunk := range chunks {
									reqCtx.observeToolCallingResponse(chunk, i == len(chunks)-1, false, span)
									reqCtx.observeToolCallingResponse(chunk, i == len(chunks)-1, true, span)
								}
								reqCtx.responseComplete, reqCtx.toolCallingForwardedComplete = true, true
								reqCtx.recordToolCallingResponse(span)
								reqCtx.clearToolCallingResponseDetectors()
								span.End()
							}
						})
					}
				}
			}
		}
	}
}

// TestUsageFinalEventWins asserts the recorded usage is the final usage event
// of the stream, even when an ext_proc boundary falls inside a usage line.
func TestUsageFinalEventWins(t *testing.T) {
	hdr := map[string]string{"content-type": "text/event-stream"}
	usageLine := func(c int) string {
		return fmt.Sprintf("data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":%d,\"total_tokens\":%d}}\n", c, 5+c)
	}

	server := &StreamingServer{
		parserRegistry: NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()),
		director:       &mockDirector{},
	}
	reqCtx := &RequestContext{
		Response:          &Response{Headers: hdr},
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: reqcommon.DefaultFairnessID},
	}
	chunks := []string{
		usageLine(1) + `data: {"usage":{"prompt_tokens":5,"completion_to`,
		`kens":2,"total_tokens":7}}` + "\n" + usageLine(3),
	}
	for i, c := range chunks {
		server.HandleResponseBody(context.Background(), reqCtx, []byte(c), i == len(chunks)-1)
	}

	assert.Equal(t, 3, reqCtx.Usage.CompletionTokens, "the final usage event must win")
	assert.Equal(t, 8, reqCtx.Usage.TotalTokens)
}

// TestCapAppliesToWholeLine asserts maxSSELineBytes bounds every line,
// not only the one carried across chunks.
func TestCapAppliesToWholeLine(t *testing.T) {
	t.Run("oversized line complete in one chunk", func(t *testing.T) {
		reqCtx := &RequestContext{}
		got := completeSSELines(context.Background(), reqCtx,
			[]byte(strings.Repeat("x", maxSSELineBytes+1)+"\n"), false)
		assert.Empty(t, got, "a line over the cap must be dropped")
	})

	t.Run("oversized line closed by a later chunk", func(t *testing.T) {
		reqCtx := &RequestContext{}
		completeSSELines(context.Background(), reqCtx, []byte(strings.Repeat("x", maxSSELineBytes-10)), false)
		got := completeSSELines(context.Background(), reqCtx, []byte(strings.Repeat("y", 100)+"\n"), false)
		assert.Empty(t, got, "a line over the cap must be dropped")
	})
}

func TestRequestContextLifecycleAccessors(t *testing.T) {
	first := time.Unix(100, 0)
	complete := time.Unix(105, 0)
	reqCtx := &RequestContext{
		firstTokenTimestamp:       first,
		responseCompleteTimestamp: complete,
		modelServerStreaming:      true,
	}
	assert.Equal(t, first, reqCtx.FirstTokenTimestamp())
	assert.Equal(t, complete, reqCtx.ResponseCompleteTimestamp())
	assert.True(t, reqCtx.ModelServerStreaming())
}
