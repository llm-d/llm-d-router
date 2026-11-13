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
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

func TestProcessSchedulesRequestWithTrailers(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}]}`)
	reqs := []*extProcPb.ProcessingRequest{
		newRequestHeaders(map[string]string{":path": reqcommon.PathChatCompletions}),
		{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{Body: body}}},
		{Request: &extProcPb.ProcessingRequest_RequestTrailers{RequestTrailers: &extProcPb.HttpTrailers{}}},
	}
	srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: reqs}
	registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
	require.NoError(t, NewStreamingServer(nil, &mockDirector{}, registry, 0).Process(srv))

	sent := srv.sentResponses
	require.Len(t, sent, 3, "header, body, and trailers responses")
	require.NotNil(t, sent[0].GetRequestHeaders())
	streamed := sent[1].GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.Equal(t, body, streamed.GetBody())
	// The trailers, not the body, end the request stream.
	require.False(t, streamed.GetEndOfStream())
	require.NotNil(t, sent[2].GetRequestTrailers())
}
