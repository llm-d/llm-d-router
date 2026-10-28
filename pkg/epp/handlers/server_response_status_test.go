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
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace/noop"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

type apiSurfaceDirector struct {
	mockDirector
	rewritePath string
}

func (d *apiSurfaceDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.IncomingModelName = "m"
	reqCtx.TargetModelName = "m"
	if d.rewritePath != "" {
		reqCtx.Request.Headers[":path"] = d.rewritePath
	}
	return reqCtx, nil
}

func TestProcessRequestMetricsByAPISurface(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())
	for _, tt := range []struct {
		name, path, body, surface, rewritePath, status string
		wantRequests, wantErrors, wantLatency          float64
		interrupted                                    bool
	}{
		{name: "plain messages", path: reqcommon.PathMessages, body: `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, surface: "messages", wantRequests: 1, wantLatency: 1},
		{name: "plain chat", path: reqcommon.PathChatCompletions, body: `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, surface: "chat_completions", wantRequests: 1, wantLatency: 1},
		{name: "rewritten messages path", path: reqcommon.PathMessages, body: `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, surface: "messages", rewritePath: reqcommon.PathChatCompletions, wantRequests: 1, wantLatency: 1},
		{name: "messages backend error", path: reqcommon.PathMessages, body: `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, surface: "messages", status: "502", wantRequests: 1, wantErrors: 1, wantLatency: 1},
		{name: "messages interrupted response", path: reqcommon.PathMessages, body: `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, surface: "messages", interrupted: true, wantRequests: 1},
		{name: "malformed messages", path: reqcommon.PathMessages, body: `{"model":`, surface: "messages", wantErrors: 1},
		{name: "unknown path", path: "/v1/embeddings", body: `{}`, surface: "unknown", wantErrors: 1},
		{name: "missing path", body: `{`, surface: "unknown", wantErrors: 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)
			reqs := []*extProcPb.ProcessingRequest{
				newRequestHeaders(map[string]string{":path": tt.path}),
				{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{Body: []byte(tt.body), EndOfStream: true}}},
			}
			if tt.wantRequests != 0 {
				responseStatus := tt.status
				if responseStatus == "" {
					responseStatus = "200"
				}
				reqs = append(reqs,
					&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{{Key: ":status", RawValue: []byte(responseStatus)}, {Key: "content-type", RawValue: []byte("application/json")}}},
					}}},
					&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body: []byte(`{"model":"m","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":2,"output_tokens":1}}`), EndOfStream: !tt.interrupted,
					}}},
				)
			}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser(), anthropic.NewAnthropicParser()}, logr.Discard())
			srv := &scriptedProcessServer{ctx: context.Background(), reqs: reqs}
			require.NoError(t, NewStreamingServer(nil, &apiSurfaceDirector{rewritePath: tt.rewritePath}, registry, 0).Process(srv))

			want := map[string]float64{
				"llm_d_epp_request_total": tt.wantRequests, "llm_d_epp_request_error_total": tt.wantErrors,
				"llm_d_epp_request_duration_seconds":             tt.wantLatency,
				"llm_d_epp_request_size_bytes":                   tt.wantRequests,
				"llm_d_epp_response_size_bytes":                  tt.wantLatency,
				"llm_d_epp_request_running":                      tt.wantRequests,
				"llm_d_epp_request_processing_duration_seconds":  1,
				"llm_d_epp_response_processing_duration_seconds": tt.wantRequests,
			}
			got := make(map[string]float64)
			families, err := ctrlmetrics.Registry.Gather()
			require.NoError(t, err)
			for _, family := range families {
				if _, ok := want[family.GetName()]; !ok {
					continue
				}
				for _, metric := range family.GetMetric() {
					require.True(t, metricHasLabels(metric, map[string]string{"api_surface": tt.surface}), "wrong API attribution for %s", family.GetName())
					switch {
					case metric.GetCounter() != nil:
						got[family.GetName()] += metric.GetCounter().GetValue()
					case metric.GetGauge() != nil:
						require.Zero(t, metric.GetGauge().GetValue(), "running requests must be released")
						got[family.GetName()]++
					default:
						got[family.GetName()] += float64(metric.GetHistogram().GetSampleCount())
					}
				}
			}
			for name, count := range want {
				require.Equal(t, count, got[name], name)
			}
		})
	}
}

func TestProcessCountsModelServerErrorFromResponseStatus(t *testing.T) {
	tests := []struct {
		name      string
		key       string
		value     string
		wantCount float64
	}{
		{name: "Envoy pseudo-header with an error status", key: ":status", value: "502", wantCount: 1},
		{name: "Envoy pseudo-header with 200", key: ":status", value: "200", wantCount: 0},
		{name: "plain status header with an error status", key: "status", value: "502", wantCount: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)

			srv := &replayProcessServer{
				ctx: logutil.NewTestLoggerIntoContext(context.Background()),
				reqs: []*extProcPb.ProcessingRequest{{
					Request: &extProcPb.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
								{Key: tt.key, RawValue: []byte(tt.value)},
							}},
						},
					},
				}},
			}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())

			require.NoError(t, NewStreamingServer(nil, &mockDirector{}, registry, 0).Process(srv))

			require.Equal(t, tt.wantCount, requestErrorCount(t, errcommon.ModelServerError))
		})
	}
}

// A request evicted after dispatch is answered with 429 and must be counted
// like any other failed request.
func TestProcessCountsEvictionAsRequestError(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)
	srv := &blockingReplayProcessServer{replayProcessServer: replayProcessServer{
		ctx: ctx,
		reqs: []*extProcPb.ProcessingRequest{
			newRequestHeaders(map[string]string{":path": "/v1/completions", "x-request-id": "req-evicted"}),
			{Request: &extProcPb.ProcessingRequest_RequestBody{
				RequestBody: &extProcPb.HttpBody{Body: []byte(`{"model":"m","prompt":"hi"}`), EndOfStream: true},
			}},
		},
	}}
	server := NewStreamingServer(nil, &mockDirector{}, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)
	server.SetEvictChannelLookup(closedEvictLookup{})

	require.NoError(t, server.Process(srv))

	last := srv.sentResponses[len(srv.sentResponses)-1]
	require.Equal(t, envoyTypePb.StatusCode_TooManyRequests, last.GetImmediateResponse().GetStatus().GetCode())
	require.Equal(t, 1.0, requestErrorCount(t, errcommon.ResourceExhausted))
}

// blockingReplayProcessServer replays its requests, then blocks like an open stream.
type blockingReplayProcessServer struct {
	replayProcessServer
}

func (m *blockingReplayProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(m.reqs) == 0 {
		<-m.ctx.Done()
		return nil, m.ctx.Err()
	}
	return m.replayProcessServer.Recv()
}

// closedEvictLookup evicts every request as soon as it is dispatched.
type closedEvictLookup struct{}

func (closedEvictLookup) Get(string) chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func (closedEvictLookup) GetReason(string) errcommon.RequestDroppedReason {
	return errcommon.RequestDroppedReasonEvicted
}

func (closedEvictLookup) Deregister(string) {}

// requestErrorCount sums llm_d_epp_request_error_total over every series with the given error code.
func requestErrorCount(t *testing.T, code string) float64 {
	t.Helper()

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	var total float64
	for _, family := range families {
		if family.GetName() != "llm_d_epp_request_error_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			if metricHasLabels(metric, map[string]string{"error_code": code}) {
				total += metric.GetCounter().GetValue()
			}
		}
	}
	return total
}
