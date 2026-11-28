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
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllminference"
)

type dynamicMetadataDirector struct {
	mockDirector
	meta *structpb.Struct
}

func (d *dynamicMetadataDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext {
	if endOfStream {
		reqCtx.Response.DynamicMetadata = d.meta
	}
	return reqCtx
}

func TestProcessResponseTrailersDynamicMetadata(t *testing.T) {
	wantMeta, err := structpb.NewStruct(map[string]any{
		"envoy.lb": map[string]any{"tokens": float64(42)},
	})
	require.NoError(t, err)

	grpcReqBody := grpctest.Frame(t, &vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3}}},
	})
	grpcRespBody := grpctest.Frame(t, &vllmpb.GenerateResponse{
		Outputs: &vllmpb.SequenceOutput{
			Text:      "hi",
			NumTokens: 1,
			FinishInfo: &vllmpb.FinishInfo{
				NumOutputTokens: 1,
				FinishReason:    vllmpb.FinishInfo_STOP,
			},
		},
	})

	registry := NewParserRegistry([]fwkrh.Parser{
		openai.NewOpenAIParser(),
		vllminference.NewVllmInferenceParser(),
	}, logr.Discard())

	tests := []struct {
		name           string
		stream         bool
		usageRequested bool
		path           string
		reqBody        []byte
		contentType    string
		respChunk      []byte
		trailers       []*configPb.HeaderValue
		wantOnBody     bool
		wantOnTrailers bool
	}{
		{
			name:        "grpc_stream_ended_by_trailers",
			stream:      true,
			path:        vllmpb.Inference_GenerateStream_FullMethodName,
			reqBody:     grpcReqBody,
			contentType: grpcframe.MediaType,
			respChunk:   grpcRespBody,
			trailers: []*configPb.HeaderValue{
				{Key: grpcframe.HeaderStatus, RawValue: []byte(grpcframe.StatusOK)},
			},
			wantOnTrailers: true,
		},
		{
			name:           "sse_stream_ended_by_trailers",
			stream:         true,
			path:           reqcommon.PathChatCompletions,
			reqBody:        []byte(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`),
			contentType:    reqcommon.MediaTypeEventStream,
			respChunk:      []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"),
			wantOnTrailers: true,
		},
		{
			name:           "sse_stream_held_line_no_duplicate",
			stream:         true,
			usageRequested: true,
			path:           reqcommon.PathChatCompletions,
			reqBody:        []byte(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`),
			contentType:    reqcommon.MediaTypeEventStream,
			respChunk:      []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]"),
			wantOnBody:     true,
		},
		{
			name:        "non_streaming_ended_by_trailers_no_duplicate",
			stream:      false,
			path:        reqcommon.PathChatCompletions,
			reqBody:     []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`),
			contentType: "application/json",
			respChunk:   []byte(`{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`),
			wantOnBody:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := &replayProcessServer{
				ctx: t.Context(),
				reqs: []*extProcPb.ProcessingRequest{
					newRequestHeaders(map[string]string{":path": tc.path}),
					{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
						Body:        tc.reqBody,
						EndOfStream: true,
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: ":status", RawValue: []byte("200")},
							{Key: reqcommon.HeaderContentType, RawValue: []byte(tc.contentType)},
						}},
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body:        tc.respChunk,
						EndOfStream: false,
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseTrailers{ResponseTrailers: &extProcPb.HttpTrailers{
						Trailers: &configPb.HeaderMap{Headers: tc.trailers},
					}}},
				},
			}

			director := &dynamicMetadataDirector{
				mockDirector: mockDirector{stream: tc.stream, usageRequested: tc.usageRequested},
				meta:         wantMeta,
			}
			require.NoError(t, NewStreamingServer(nil, director, registry, 0).Process(srv))

			var bodyMeta, trailerMeta *structpb.Struct
			bodyMetaCount, trailerMetaCount := 0, 0
			for _, sent := range srv.sentResponses {
				if sent.GetResponseBody() != nil && sent.GetDynamicMetadata() != nil {
					bodyMeta = sent.GetDynamicMetadata()
					bodyMetaCount++
				}
				if sent.GetResponseTrailers() != nil && sent.GetDynamicMetadata() != nil {
					trailerMeta = sent.GetDynamicMetadata()
					trailerMetaCount++
				}
			}

			if tc.wantOnBody {
				require.Equal(t, 1, bodyMetaCount)
				assert.Empty(t, cmp.Diff(wantMeta, bodyMeta, protocmp.Transform()))
			} else {
				assert.Zero(t, bodyMetaCount)
			}

			if tc.wantOnTrailers {
				require.Equal(t, 1, trailerMetaCount)
				assert.Empty(t, cmp.Diff(wantMeta, trailerMeta, protocmp.Transform()))
			} else {
				assert.Zero(t, trailerMetaCount)
			}
		})
	}
}
