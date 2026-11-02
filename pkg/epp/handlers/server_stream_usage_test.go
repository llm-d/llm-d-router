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
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// usageCapturingDirector records the usage the response carried when it ends.
type usageCapturingDirector struct {
	mockDirector
	usage fwkrequest.Usage
}

func (d *usageCapturingDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext {
	if endOfStream {
		d.usage = reqCtx.Usage
	}
	return reqCtx
}

// The client receives the stream without the usage-only frame the director
// requested, while the EPP still reads the usage it carried.
func TestProcessDropsRequestedUsageFrame(t *testing.T) {
	split := len(usageStreamContent) + 20
	for _, tt := range []struct {
		name      string
		requested bool
		chunks    []string
		trailers  bool
		want      string
	}{
		{
			name:      "requested, frame split across chunks",
			requested: true,
			chunks:    []string{usageStream[:split], usageStream[split:]},
			want:      usageStreamWant,
		},
		{
			name:      "requested, stream ended by trailers mid-line",
			requested: true,
			chunks:    []string{usageStream[:split], usageStreamFrame[20:] + "data: [DONE]"},
			trailers:  true,
			want:      usageStreamContent + "data: [DONE]",
		},
		{
			name:   "not requested",
			chunks: []string{usageStream[:split], usageStream[split:]},
			want:   usageStream,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			srv := &replayProcessServer{
				ctx: context.Background(),
				reqs: []*extProcPb.ProcessingRequest{
					newRequestHeaders(map[string]string{":path": reqcommon.PathChatCompletions}),
					{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
						Body:        []byte(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`),
						EndOfStream: true,
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: reqcommon.HeaderContentType, RawValue: []byte(reqcommon.MediaTypeEventStream)},
						}},
					}}},
				},
			}
			for i, chunk := range tt.chunks {
				srv.reqs = append(srv.reqs, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
					ResponseBody: &extProcPb.HttpBody{Body: []byte(chunk), EndOfStream: !tt.trailers && i == len(tt.chunks)-1},
				}})
			}
			if tt.trailers {
				srv.reqs = append(srv.reqs, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseTrailers{
					ResponseTrailers: &extProcPb.HttpTrailers{Trailers: &configPb.HeaderMap{}},
				}})
			}
			director := &usageCapturingDirector{mockDirector: mockDirector{stream: true, usageRequested: tt.requested}}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			require.NoError(t, NewStreamingServer(nil, director, registry, 0).Process(srv))

			var forwarded []byte
			for _, sent := range srv.sentResponses {
				if body := sent.GetResponseBody(); body != nil {
					forwarded = append(forwarded, body.GetResponse().GetBodyMutation().GetStreamedResponse().GetBody()...)
				}
			}
			require.Equal(t, tt.want, string(forwarded))
			require.Equal(t, fwkrequest.Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}, director.usage)
		})
	}
}
