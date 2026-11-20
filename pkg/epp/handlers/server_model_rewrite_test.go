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
	"fmt"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	modelRewriteRequestBody = `{"model":"llama","messages":[{"role":"user","content":"hi"}],"stream":true}`

	// modelRewriteTarget is the internal backend name an InferenceModelRewrite rule
	// picks and modelRewriteIncoming is the name the client asked for.
	modelRewriteTarget   = "llama-canary"
	modelRewriteIncoming = "llama"

	modelRewriteUpstreamStream = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"" + modelRewriteTarget + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"" + modelRewriteTarget + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"

	modelRewriteClientStream = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"" + modelRewriteIncoming + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"" + modelRewriteIncoming + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
)

type modelRewriteDirector struct {
	mockDirector
}

func (d *modelRewriteDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrequest.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.IncomingModelName = modelRewriteIncoming
	reqCtx.TargetModelName = modelRewriteTarget
	return reqCtx, nil
}

func modelRewriteResponseBody(body []byte, endOfStream bool) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseBody{
			ResponseBody: &extProcPb.HttpBody{Body: body, EndOfStream: endOfStream},
		},
	}
}

// modelRewriteReqs builds the ext_proc message sequence for one streamed response
// whose body is cut into two chunks at cut. cut <= 0 keeps the body in one chunk.
func modelRewriteReqs(cut int) []*extProcPb.ProcessingRequest {
	reqs := []*extProcPb.ProcessingRequest{
		newRequestHeaders(map[string]string{":path": "/v1/chat/completions", "content-type": "application/json"}),
		{Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: []byte(modelRewriteRequestBody), EndOfStream: true},
		}},
		{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: ":status", RawValue: []byte("200")},
				{Key: "content-type", RawValue: []byte("text/event-stream")},
			}}},
		}},
	}
	stream := []byte(modelRewriteUpstreamStream)
	if cut <= 0 || cut >= len(stream) {
		return append(reqs, modelRewriteResponseBody(stream, true))
	}
	return append(reqs, modelRewriteResponseBody(stream[:cut], false), modelRewriteResponseBody(stream[cut:], true))
}

// forwardedBody concatenates the streamed response bodies the EPP sent back.
func forwardedBody(resps []*extProcPb.ProcessingResponse) []byte {
	var out []byte
	for _, resp := range resps {
		body := resp.GetResponseBody()
		if body == nil {
			continue
		}
		out = append(out, body.GetResponse().GetBodyMutation().GetStreamedResponse().GetBody()...)
	}
	return out
}

func runModelRewrite(t *testing.T, cut int) []byte {
	t.Helper()

	srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: modelRewriteReqs(cut)}
	server := NewStreamingServer(nil, &modelRewriteDirector{}, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)
	require.NoError(t, server.Process(srv))
	return forwardedBody(srv.sentResponses)
}

// A streamed response keeps the client-facing model name wherever Envoy splits the
// body: the rewrite must not depend on the pattern arriving inside one chunk.
func TestStreamedResponseRewritesModelNameAcrossChunkBoundary(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	require.Equal(t, modelRewriteClientStream, string(runModelRewrite(t, 0)), "single chunk")

	var leaks []string
	for cut := 1; cut < len(modelRewriteUpstreamStream); cut++ {
		if got := string(runModelRewrite(t, cut)); got != modelRewriteClientStream {
			leaks = append(leaks, fmt.Sprintf("cut %d: %q", cut, got))
		}
	}
	require.Empty(t, leaks, "stream leaked the target model name when the body was split at these offsets")
}

// The carry holds back only bytes that could still become a model-name occurrence,
// so a body that is forwarded without a rewrite on the way still arrives intact.
func TestStreamedResponseWithoutModelRewriteForwardsEveryByte(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	stream := []byte(modelRewriteUpstreamStream)
	for cut := 1; cut < len(stream); cut++ {
		reqs := modelRewriteReqs(cut)
		srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: reqs}
		// Same names: the rewrite is a no-op, so the chunks must pass through unchanged.
		server := NewStreamingServer(nil, &mockDirector{}, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)
		require.NoError(t, server.Process(srv))
		require.Equal(t, string(stream), string(forwardedBody(srv.sentResponses)), "cut %d", cut)
	}
}

// A streamed body can end on the trailers rather than on an end-of-stream body chunk.
// The held-back prefix must still be forwarded instead of being dropped with the request.
func TestStreamedResponseFlushesModelRewriteCarryBeforeTrailers(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	// This cut leaves the first byte of the model-name occurrence held back.
	const cut = 51
	reqs := modelRewriteReqs(cut)
	reqs = reqs[:len(reqs)-1]
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &extProcPb.HttpTrailers{Trailers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: "grpc-status", RawValue: []byte("0")},
			}}},
		},
	})

	srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: reqs}
	server := NewStreamingServer(nil, &modelRewriteDirector{}, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)
	require.NoError(t, server.Process(srv))

	require.Equal(t, modelRewriteUpstreamStream[:cut], string(forwardedBody(srv.sentResponses)))
}
