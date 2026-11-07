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

package sglanggrpc

import (
	"bytes"
	"context"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/sglanghttp"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

func TestSGLangGRPCParser_Claims(t *testing.T) {
	parser := NewSGLangGRPCParser()
	got := parser.Claims()
	want := fwkrh.Claims{
		Paths:     []string{"sglang.api.v1.SglangService/Generate"},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("Claims() mismatch (-want +got):\n%s", diff)
	}
}

func TestSGLangGRPCParser_ParseRequest(t *testing.T) {
	maxNewTokens := int64(64)
	streamTrue := true

	tests := []struct {
		name            string
		path            string
		req             *sglangapiv1.GenerateRequest
		rawBody         []byte
		wantGenerate    *fwkrequest.GenerateRequest
		wantMaxTokens   *int64
		wantStream      bool
		wantErrContains string
	}{
		{
			name: "input_ids with cache_salt, mm_hashes, and sampling_params",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				InputIds: &sglangapiv1.TokenIdsOrList{
					Value: &sglangapiv1.TokenIdsOrList_One{
						One: &sglangapiv1.TokenIds{Ids: []int64{10, 20, math.MaxUint32}},
					},
				},
				CacheSalt: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_One{One: "tenant-a"},
				},
				ImageData: &sglangapiv1.MediaInput{
					Value: &sglangapiv1.MediaInput_Item{
						Item: &sglangapiv1.MediaItem{Value: &sglangapiv1.MediaItem_Source{Source: "https://example.com/a.png"}},
					},
				},
				MmHashes: &sglangapiv1.StringListOrList{
					Value: &sglangapiv1.StringListOrList_One{
						One: &sglangapiv1.StringList{Items: []string{"img-hash-1"}},
					},
				},
				SamplingParams: &sglangapiv1.SamplingParamsOrList{
					Value: &sglangapiv1.SamplingParamsOrList_One{
						One: &sglangapiv1.SamplingParams{MaxNewTokens: &maxNewTokens},
					},
				},
				Stream: &streamTrue,
			},
			wantGenerate: &fwkrequest.GenerateRequest{
				TokenIDs:  []uint32{10, 20, math.MaxUint32},
				CacheSalt: "tenant-a",
				Features: &tokenization.MultiModalFeatures{
					MMHashes: map[string][]string{
						string(reqcommon.ModalityImage): {"img-hash-1"},
					},
				},
			},
			wantMaxTokens: &maxNewTokens,
			wantStream:    true,
		},
		{
			name: "text prompt with extra_key and cache_salt matches HTTP composition",
			path: "sglang.api.v1.SglangService/Generate",
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_One{One: "hello world"},
				},
				ExtraKey: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_One{One: "ns-1"},
				},
				CacheSalt: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_One{One: "salt-1"},
				},
			},
			wantGenerate: &fwkrequest.GenerateRequest{
				Text:      "hello world",
				CacheSalt: `["ns-1","salt-1"]`,
			},
		},
		{
			name: "video modality inferred when only video_data is present",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_One{One: "describe video"},
				},
				VideoData: &sglangapiv1.MediaInput{
					Value: &sglangapiv1.MediaInput_Item{
						Item: &sglangapiv1.MediaItem{Value: &sglangapiv1.MediaItem_Source{Source: "https://example.com/v.mp4"}},
					},
				},
				MmHashes: &sglangapiv1.StringListOrList{
					Value: &sglangapiv1.StringListOrList_One{
						One: &sglangapiv1.StringList{Items: []string{"vid-hash-1"}},
					},
				},
			},
			wantGenerate: &fwkrequest.GenerateRequest{
				Text: "describe video",
				Features: &tokenization.MultiModalFeatures{
					MMHashes: map[string][]string{
						string(reqcommon.ModalityVideo): {"vid-hash-1"},
					},
				},
			},
		},
		{
			name: "video modality inferred when image_data is an empty many list",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_One{One: "describe video"},
				},
				ImageData: &sglangapiv1.MediaInput{
					Value: &sglangapiv1.MediaInput_Many{Many: &sglangapiv1.MediaItems{}},
				},
				VideoData: &sglangapiv1.MediaInput{
					Value: &sglangapiv1.MediaInput_Item{
						Item: &sglangapiv1.MediaItem{Value: &sglangapiv1.MediaItem_Source{Source: "https://example.com/v.mp4"}},
					},
				},
				MmHashes: &sglangapiv1.StringListOrList{
					Value: &sglangapiv1.StringListOrList_One{
						One: &sglangapiv1.StringList{Items: []string{"vid-hash-1"}},
					},
				},
			},
			wantGenerate: &fwkrequest.GenerateRequest{
				Text: "describe video",
				Features: &tokenization.MultiModalFeatures{
					MMHashes: map[string][]string{
						string(reqcommon.ModalityVideo): {"vid-hash-1"},
					},
				},
			},
		},
		{
			name:            "rejects unsupported path",
			path:            sglangapiv1.SglangService_HealthCheck_FullMethodName,
			req:             &sglangapiv1.GenerateRequest{},
			wantErrContains: "unsupported gRPC path",
		},
		{
			name:            "rejects path prefix without exact RPC match",
			path:            "/prefix/sglang.api.v1.SglangService/Generate",
			req:             &sglangapiv1.GenerateRequest{},
			wantErrContains: "unsupported gRPC path",
		},
		{
			name:            "rejects malformed gRPC frame",
			path:            sglangapiv1.SglangService_Generate_FullMethodName,
			rawBody:         []byte{0, 0, 0},
			wantErrContains: "invalid gRPC generate request",
		},
		{
			name: "rejects bytes after the gRPC frame",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			rawBody: func() []byte {
				framed, err := grpcframe.MarshalProto(&sglangapiv1.GenerateRequest{
					Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
				})
				require.NoError(t, err)
				return append(framed, 0xde, 0xad)
			}(),
			wantErrContains: "invalid gRPC generate request",
		},
		{
			name:            "rejects missing prompt",
			path:            sglangapiv1.SglangService_Generate_FullMethodName,
			req:             &sglangapiv1.GenerateRequest{},
			wantErrContains: "input_ids or text must be provided",
		},
		{
			name: "rejects empty text",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: ""}},
			},
			wantErrContains: "text cannot be empty",
		},
		{
			name: "rejects empty input_ids",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				InputIds: &sglangapiv1.TokenIdsOrList{
					Value: &sglangapiv1.TokenIdsOrList_One{One: &sglangapiv1.TokenIds{}},
				},
			},
			wantErrContains: "input_ids cannot be empty",
		},
		{
			name: "rejects negative token id",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				InputIds: &sglangapiv1.TokenIdsOrList{
					Value: &sglangapiv1.TokenIdsOrList_One{One: &sglangapiv1.TokenIds{Ids: []int64{1, -1}}},
				},
			},
			wantErrContains: "out of uint32 range",
		},
		{
			name: "rejects token id exceeding uint32",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				InputIds: &sglangapiv1.TokenIdsOrList{
					Value: &sglangapiv1.TokenIdsOrList_One{One: &sglangapiv1.TokenIds{Ids: []int64{math.MaxUint32 + 1}}},
				},
			},
			wantErrContains: "out of uint32 range",
		},
		{
			name: "rejects batched input_ids",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				InputIds: &sglangapiv1.TokenIdsOrList{
					Value: &sglangapiv1.TokenIdsOrList_Many{
						Many: &sglangapiv1.TokenIdsList{Items: []*sglangapiv1.TokenIds{{Ids: []int64{1}}}},
					},
				},
			},
			wantErrContains: "batched input_ids is not supported",
		},
		{
			name: "rejects batched text",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_Many{
						Many: &sglangapiv1.StringList{Items: []string{"a", "b"}},
					},
				},
			},
			wantErrContains: "batched text is not supported",
		},
		{
			name: "rejects batched cache_salt",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
				CacheSalt: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_Many{
						Many: &sglangapiv1.StringList{Items: []string{"s1"}},
					},
				},
			},
			wantErrContains: "cache_salt must be a single string",
		},
		{
			name: "rejects batched extra_key",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
				ExtraKey: &sglangapiv1.StringOrList{
					Value: &sglangapiv1.StringOrList_Many{
						Many: &sglangapiv1.StringList{Items: []string{"k1"}},
					},
				},
			},
			wantErrContains: "extra_key must be a single string",
		},
		{
			name: "rejects batched image_data",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
				ImageData: &sglangapiv1.MediaInput{
					Value: &sglangapiv1.MediaInput_Nested{Nested: &sglangapiv1.MediaItemsList{}},
				},
			},
			wantErrContains: "batched image_data is not supported",
		},
		{
			name: "rejects batched video_data",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
				VideoData: &sglangapiv1.MediaInput{
					Value: &sglangapiv1.MediaInput_Nested{Nested: &sglangapiv1.MediaItemsList{}},
				},
			},
			wantErrContains: "batched video_data is not supported",
		},
		{
			name: "rejects batched audio_data",
			path: sglangapiv1.SglangService_Generate_FullMethodName,
			req: &sglangapiv1.GenerateRequest{
				Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
				AudioData: &sglangapiv1.MediaInput{
					Value: &sglangapiv1.MediaInput_Nested{Nested: &sglangapiv1.MediaItemsList{}},
				},
			},
			wantErrContains: "batched audio_data is not supported",
		},
	}

	parser := NewSGLangGRPCParser()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := tt.rawBody
			if raw == nil {
				var err error
				raw, err = grpcframe.MarshalProto(tt.req)
				require.NoError(t, err)
			}
			got, err := parser.ParseRequest(context.Background(), raw, map[string]string{":path": tt.path})
			if tt.wantErrContains != "" {
				require.ErrorContains(t, err, tt.wantErrContains)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, got)
			require.False(t, got.SkipResponseProcessing)
			require.Equal(t, tt.wantGenerate, got.Body.Generate)
			require.Equal(t, tt.wantMaxTokens, got.Body.MaxOutputTokens)
			require.Equal(t, tt.wantStream, got.Body.Stream)
			// The original bytes are forwarded, not a re-encoded message.
			wire, ok := got.Body.Payload.(fwkrequest.PayloadBytes)
			require.True(t, ok)
			require.Equal(t, raw, []byte(wire))
			require.Equal(t, raw, got.Body.RawBody)
			payload, err := grpcframe.ParsePayload(wire)
			require.NoError(t, err)
			var decoded sglangapiv1.GenerateRequest
			require.NoError(t, proto.Unmarshal(payload, &decoded))
			require.True(t, proto.Equal(tt.req, &decoded))
		})
	}
}

func TestSGLangGRPCParser_CacheSaltMatchesHTTPParser(t *testing.T) {
	httpParser := sglanghttp.NewSGLangHTTPParser()
	grpcParser := NewSGLangGRPCParser()

	httpParsed, err := httpParser.ParseRequest(
		context.Background(),
		[]byte(`{"text":"hello","extra_key":"org-a","cache_salt":"user-b"}`),
		map[string]string{":path": "/generate"},
	)
	require.NoError(t, err)

	framed, err := grpcframe.MarshalProto(&sglangapiv1.GenerateRequest{
		Text:      &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hello"}},
		ExtraKey:  &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "org-a"}},
		CacheSalt: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "user-b"}},
	})
	require.NoError(t, err)
	grpcParsed, err := grpcParser.ParseRequest(
		context.Background(),
		framed,
		map[string]string{":path": sglangapiv1.SglangService_Generate_FullMethodName},
	)
	require.NoError(t, err)
	require.Equal(t, httpParsed.Body.Generate.CacheSalt, grpcParsed.Body.Generate.CacheSalt)
}

func TestSGLangGRPCParser_RewritePriority(t *testing.T) {
	parser := NewSGLangGRPCParser()
	clientPrio := int64(99)
	framed, err := grpcframe.MarshalProto(&sglangapiv1.GenerateRequest{
		Text:     &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hello"}},
		Priority: &clientPrio,
	})
	require.NoError(t, err)

	parsed, err := parser.ParseRequest(context.Background(), framed, map[string]string{":path": sglangapiv1.SglangService_Generate_FullMethodName})
	require.NoError(t, err)

	rewritten, mutated, err := parser.RewritePriority(fwkrh.PriorityRewriteContext{}, parsed.Body.Payload.(fwkrequest.MarshalablePayload), 5)
	require.NoError(t, err)
	require.True(t, mutated)

	wireBytes, err := rewritten.Marshal()
	require.NoError(t, err)
	rawPayload, err := grpcframe.ParsePayload(wireBytes)
	require.NoError(t, err)
	var out sglangapiv1.GenerateRequest
	require.NoError(t, proto.Unmarshal(rawPayload, &out))
	require.Equal(t, int64(5), out.GetPriority())
	require.Equal(t, "hello", out.GetText().GetOne())

	// The rewrite allocates the new frame and the interface box of the
	// result, nothing else, and leaves exactly one priority field.
	rawIn := parsed.Body.Payload.(fwkrequest.PayloadBytes)
	require.Equal(t, 2.0, testing.AllocsPerRun(100, func() {
		_, _, _ = parser.RewritePriority(fwkrh.PriorityRewriteContext{}, rawIn, 5)
	}))
	prioFieldNum := (&sglangapiv1.GenerateRequest{}).ProtoReflect().Descriptor().Fields().ByName("priority").Number()
	require.Equal(t, 1, bytes.Count(rawPayload, protowire.AppendTag(nil, prioFieldNum, protowire.VarintType)))
}

func TestParseResponseTokenFrameDoesNotDecode(t *testing.T) {
	ids := make([]int64, 256)
	framed, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
		Text: " hello", OutputIds: &sglangapiv1.TokenIds{Ids: ids},
		MetaInfo: &sglangapiv1.GenerateMetaInfo{PromptTokens: 8192, CompletionTokens: 256},
	}}})
	require.NoError(t, err)
	parser := NewSGLangGRPCParser()
	// One allocation: the ParsedResponse itself.
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() {
		_, _ = parser.ParseResponse(context.Background(), framed, nil, false)
	}))
}

func TestSGLangGRPCParser_ParseResponse(t *testing.T) {
	parser := NewSGLangGRPCParser()

	t.Run("empty body on stream trailer returns nil without error", func(t *testing.T) {
		res, err := parser.ParseResponse(context.Background(), nil, nil, true)
		require.NoError(t, err)
		require.Nil(t, res)
	})

	t.Run("intermediate and terminal GenerateStreamItem frames in one chunk", func(t *testing.T) {
		chunkFrame, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
			Item: &sglangapiv1.GenerateStreamItem_Frame{
				Frame: &sglangapiv1.GenerateResponse{
					Text: "Hi",
					MetaInfo: &sglangapiv1.GenerateMetaInfo{
						PromptTokens:     12,
						CompletionTokens: 1,
						CachedTokens:     8,
					},
				},
			},
		})
		require.NoError(t, err)

		finalFrame, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
			Item: &sglangapiv1.GenerateStreamItem_Frame{
				Frame: &sglangapiv1.GenerateResponse{
					Text: "Hello",
					MetaInfo: &sglangapiv1.GenerateMetaInfo{
						PromptTokens:     12,
						CompletionTokens: 4,
						CachedTokens:     8,
						FinishReason: &sglangapiv1.FinishReason{
							Kind: &sglangapiv1.FinishReason_Length{
								Length: &sglangapiv1.FinishLength{Length: proto.Uint64(4)},
							},
						},
					},
				},
			},
		})
		require.NoError(t, err)

		res, err := parser.ParseResponse(context.Background(), append(chunkFrame, finalFrame...), map[string]string{reqcommon.HeaderContentType: grpcframe.MediaType}, false)
		require.NoError(t, err)
		require.NotNil(t, res)
		require.Equal(t, 2, res.StreamedEvents)
		require.Equal(t, &fwkrequest.Usage{
			PromptTokens:     12,
			CompletionTokens: 4,
			TotalTokens:      16,
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{
				CachedTokens: 8,
			},
		}, res.Usage)
	})

	t.Run("in-stream error frame is tolerated without usage", func(t *testing.T) {
		errFrame, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
			Item: &sglangapiv1.GenerateStreamItem_Error{
				Error: &sglangapiv1.GenerateStreamError{
					Error: &sglangapiv1.ErrorBody{Message: "aborted", Code: 499},
				},
			},
		})
		require.NoError(t, err)
		res, err := parser.ParseResponse(context.Background(), errFrame, nil, true)
		require.NoError(t, err)
		require.NotNil(t, res)
		require.Equal(t, 0, res.StreamedEvents)
		require.Nil(t, res.Usage)
	})
}
