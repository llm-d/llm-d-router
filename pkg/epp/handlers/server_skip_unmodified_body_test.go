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

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// mutatingDirector records the inbound body and optionally rewrites RawBody,
// marking SchedulingRequest.Body.Mutated when it does.
type mutatingDirector struct {
	mutate   bool
	outbound []byte
}

func (m *mutatingDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, body *fwkrequest.InferenceRequestBody) (*RequestContext, error) {
	if body == nil {
		body = &fwkrequest.InferenceRequestBody{}
	}
	if m.mutate {
		body.Mutated = true
		reqCtx.Request.RawBody = append([]byte(nil), m.outbound...)
	}
	reqCtx.SchedulingRequest = &fwksched.InferenceRequest{Body: body}
	reqCtx.TargetEndpoint = "10.0.0.1:8000"
	return reqCtx, nil
}

func (m *mutatingDirector) HandleResponseHeader(_ context.Context, reqCtx *RequestContext) *RequestContext {
	return reqCtx
}

func (m *mutatingDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, _ bool) *RequestContext {
	return reqCtx
}

func (m *mutatingDirector) GetRandomEndpoint() *fwkdl.EndpointMetadata {
	return &fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "8000"}
}

func (m *mutatingDirector) GetRandomDecodeEndpoint() *fwkdl.EndpointMetadata {
	return m.GetRandomEndpoint()
}

func TestProcessSkipUnmodifiedRequestBody(t *testing.T) {
	inbound := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}]}`)
	rewritten := []byte(`{"model":"rewritten","messages":[{"role":"user","content":"hello"}]}`)

	tests := []struct {
		name         string
		skip         bool
		mutate       bool
		trailers     bool
		wantMutation bool
		wantBody     []byte
	}{
		{
			name:         "default echoes an unmodified body",
			wantMutation: true,
			wantBody:     inbound,
		},
		{
			name:         "skip omits mutation when the body is unchanged",
			skip:         true,
			wantMutation: false,
		},
		{
			name:         "skip still forwards a rewritten body",
			skip:         true,
			mutate:       true,
			wantMutation: true,
			wantBody:     rewritten,
		},
		{
			name:         "skip with trailers still ends the stream on the trailers response",
			skip:         true,
			trailers:     true,
			wantMutation: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reqs := []*extProcPb.ProcessingRequest{
				newRequestHeaders(map[string]string{":path": reqcommon.PathChatCompletions}),
				{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{Body: inbound, EndOfStream: !tt.trailers}}},
			}
			if tt.trailers {
				reqs = append(reqs, &extProcPb.ProcessingRequest{
					Request: &extProcPb.ProcessingRequest_RequestTrailers{RequestTrailers: &extProcPb.HttpTrailers{}},
				})
			}
			srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: reqs}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			director := &mutatingDirector{mutate: tt.mutate, outbound: rewritten}
			server := NewStreamingServer(nil, director, registry, 0)
			server.SetSkipUnmodifiedRequestBody(tt.skip)
			require.NoError(t, server.Process(srv))

			sent := srv.sentResponses
			if tt.trailers {
				require.Len(t, sent, 3, "header, body, and trailers responses")
				require.NotNil(t, sent[2].GetRequestTrailers())
			} else {
				require.Len(t, sent, 2, "header and body responses")
			}
			require.NotNil(t, sent[0].GetRequestHeaders())
			bodyResp := sent[1].GetRequestBody().GetResponse()
			require.NotNil(t, bodyResp)
			if tt.wantMutation {
				streamed := bodyResp.GetBodyMutation().GetStreamedResponse()
				require.NotNil(t, streamed)
				require.Equal(t, tt.wantBody, streamed.GetBody())
			} else {
				require.Nil(t, bodyResp.GetBodyMutation())
			}
		})
	}
}
