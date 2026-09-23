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
	"io"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// sequenceProcessServer replays its messages in order, then reports EOF.
type sequenceProcessServer struct {
	mockProcessServer
	reqs []*extProcPb.ProcessingRequest
}

func (m *sequenceProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(m.reqs) == 0 {
		return nil, io.EOF
	}
	req := m.reqs[0]
	m.reqs = m.reqs[1:]
	return req, nil
}

// answeringDirector schedules a pod and, when answer is set, asks the server to
// answer the caller instead of forwarding. It records end-of-stream cleanups.
type answeringDirector struct {
	answer          map[string]string
	endOfStreamSeen int
	cause           fwkrc.TerminationCause
}

func (d *answeringDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.TargetPod = &fwkdl.EndpointMetadata{Address: "10.0.3.7", Port: "8000"}
	reqCtx.TargetEndpoint = "10.0.3.7:8000"
	reqCtx.AnswerHeaders = d.answer
	return reqCtx, nil
}

func (d *answeringDirector) HandleResponseHeader(_ context.Context, reqCtx *RequestContext) *RequestContext {
	return reqCtx
}

func (d *answeringDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext {
	if endOfStream {
		d.endOfStreamSeen++
		d.cause = reqCtx.TerminationCause
	}
	return reqCtx
}

func (d *answeringDirector) GetRandomEndpoint() *fwkdl.EndpointMetadata { return nil }

func TestProcessAnswersWithoutForwarding(t *testing.T) {
	director := &answeringDirector{answer: map[string]string{routing.ReservedEndpointHeader: "10.0.3.7:8000"}}
	server := NewStreamingServer(nil, director, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)

	srv := &sequenceProcessServer{reqs: []*extProcPb.ProcessingRequest{
		newRequestHeaders(map[string]string{":path": "/v1/chat/completions", "x-request-id": "req-ask"}),
		{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
			Body:        []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`),
			EndOfStream: true,
		}}},
	}}

	require.NoError(t, server.Process(srv))

	require.Len(t, srv.sentResponses, 1, "the answer is the only response; nothing is forwarded")
	ir := srv.sentResponses[0].GetImmediateResponse()
	require.NotNil(t, ir)
	assert.Equal(t, envoyTypePb.StatusCode_OK, ir.Status.Code)
	require.NotNil(t, ir.Headers)
	require.Len(t, ir.Headers.SetHeaders, 1)
	assert.Equal(t, routing.ReservedEndpointHeader, ir.Headers.SetHeaders[0].Header.Key)
	assert.Equal(t, "10.0.3.7:8000", string(ir.Headers.SetHeaders[0].Header.RawValue))
	assert.Empty(t, ir.Body)
	assert.Equal(t, 1, director.endOfStreamSeen, "stream-end cleanup releases per-request plugin state")
	assert.Equal(t, fwkrc.TerminationCauseAnswered, director.cause)
}
