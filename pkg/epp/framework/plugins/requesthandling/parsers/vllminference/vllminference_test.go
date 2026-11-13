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

package vllminference

import (
	"bytes"
	"context"
	"math"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/vllmwire"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

func mustMarshal(t *testing.T, m proto.Message) []byte {
	t.Helper()
	b, err := proto.Marshal(m)
	require.NoError(t, err)
	return b
}

func TestVllmInferenceParser_ClaimsAndMetadata(t *testing.T) {
	plugin, err := VllmInferenceParserPluginFactory("custom", nil, nil)
	require.NoError(t, err)
	parser, ok := plugin.(*VllmInferenceParser)
	require.True(t, ok)
	require.Equal(t, fwkplugin.TypedName{Type: VllmInferenceParserType, Name: "custom"}, parser.TypedName())

	want := fwkrh.Claims{
		Paths: []string{
			"vllm.Inference/Generate",
			"vllm.Inference/GenerateStream",
		},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C},
	}
	if diff := cmp.Diff(want, parser.Claims()); diff != "" {
		t.Fatalf("Claims() mismatch (-want +got):\n%s", diff)
	}
}

func TestVllmInferenceParser_ParseRequest(t *testing.T) {
	maxTokens64 := int64(64)
	maxTokens128 := int64(128)
	mmHashFallback := "fallback-hash"

	tests := []struct {
		name            string
		path            string
		req             *vllmpb.GenerateRequest
		rawBody         []byte
		wantModel       string
		wantGenerate    *fwkrequest.GenerateRequest
		wantMaxTokens   *int64
		wantStream      bool
		wantErrContains string
		wantErrIs       error
	}{
		{
			name: "GenerateStream with token_ids, cache_salt, preprocessed media features, and max_new_tokens",
			path: vllmpb.Inference_GenerateStream_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Model:  "meta-llama/Llama-3.1-8B",
				Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{10, 20, math.MaxUint32}}},
				Stopping: &vllmpb.StoppingCriteria{
					MaxNewTokens: 64,
				},
				Kv: &vllmpb.KVCacheParameters{
					CacheSalt: "tenant-a",
				},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     "ignored-when-identifier-present",
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{
								Kwargs:     bytes.Repeat([]byte{0x91}, 2048),
								Identifier: "img-hash-1",
								MmHash:     &mmHashFallback,
								Offset:     1,
								Length:     4,
							},
						},
					},
					{
						Modality: vllmpb.Modality_MODALITY_AUDIO,
						Uuid:     "ignored-when-mmhash-present",
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{
								MmHash: &mmHashFallback,
								Offset: 6,
								Length: 2,
							},
						},
					},
					{
						Modality: vllmpb.Modality_MODALITY_VIDEO,
						Uuid:     "video-uuid-fallback",
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{
								Offset: 9,
								Length: 3,
							},
						},
					},
				},
			},
			wantModel: "meta-llama/Llama-3.1-8B",
			wantGenerate: &fwkrequest.GenerateRequest{
				TokenIDs:  []uint32{10, 20, math.MaxUint32},
				CacheSalt: "tenant-a",
				Features: &tokenization.MultiModalFeatures{
					MMHashes: map[string][]string{
						string(reqcommon.ModalityImage): {"img-hash-1"},
						string(reqcommon.ModalityAudio): {"fallback-hash"},
						string(reqcommon.ModalityVideo): {"video-uuid-fallback"},
					},
					MMPlaceholders: map[string][]kvblock.PlaceholderRange{
						string(reqcommon.ModalityImage): {{Offset: 1, Length: 4}},
						string(reqcommon.ModalityAudio): {{Offset: 6, Length: 2}},
						string(reqcommon.ModalityVideo): {{Offset: 9, Length: 3}},
					},
				},
			},
			wantMaxTokens: &maxTokens64,
			wantStream:    true,
		},
		{
			// A raw uuid item and a preprocessed item of one modality would
			// pair the uuid with the other item's token span, so that
			// modality keeps its hashes and drops its placeholder ranges.
			// A modality with preprocessed items only keeps both.
			name: "a modality that mixes raw uuid and preprocessed items keeps hashes in order and no placeholder ranges",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "describe media"},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     "raw-first",
						Source:   &vllmpb.MediaItem_Url{Url: "https://example.com/a.png"},
					},
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{Identifier: "img-hash-2", Offset: 5, Length: 4},
						},
					},
					{
						Modality: vllmpb.Modality_MODALITY_AUDIO,
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{Identifier: "audio-hash", Offset: 12, Length: 2},
						},
					},
				},
			},
			wantGenerate: &fwkrequest.GenerateRequest{
				Text: "describe media",
				Features: &tokenization.MultiModalFeatures{
					MMHashes: map[string][]string{
						string(reqcommon.ModalityImage): {"raw-first", "img-hash-2"},
						string(reqcommon.ModalityAudio): {"audio-hash"},
					},
					MMPlaceholders: map[string][]kvblock.PlaceholderRange{
						string(reqcommon.ModalityAudio): {{Offset: 12, Length: 2}},
					},
				},
			},
		},
		{
			name: "raw media items with uuid populate MMHashes without placeholder ranges while skipping unnamed raw media",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Model:  "meta-llama/Llama-3.1-8B",
				Prompt: &vllmpb.GenerateRequest_Text{Text: "describe media"},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     "url-uuid-1",
						Source:   &vllmpb.MediaItem_Url{Url: "https://example.com/a.png"},
					},
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Source:   &vllmpb.MediaItem_RawBytes{RawBytes: bytes.Repeat([]byte("x"), 1<<12)},
					},
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Uuid:     "raw-uuid-2",
						Source:   &vllmpb.MediaItem_RawBytes{RawBytes: []byte{0x89, 0x50, 0x4e, 0x47}},
					},
					{
						Modality: vllmpb.Modality_MODALITY_AUDIO,
						Uuid:     "data-uri-uuid",
						Source:   &vllmpb.MediaItem_DataUri{DataUri: "data:audio/wav;base64,AAAA"},
					},
				},
			},
			wantModel: "meta-llama/Llama-3.1-8B",
			wantGenerate: &fwkrequest.GenerateRequest{
				Text: "describe media",
				Features: &tokenization.MultiModalFeatures{
					MMHashes: map[string][]string{
						string(reqcommon.ModalityImage): {"url-uuid-1", "raw-uuid-2"},
						string(reqcommon.ModalityAudio): {"data-uri-uuid"},
					},
				},
			},
			wantStream: false,
		},
		{
			name: "lora_name wins over model, num_sequences 1 is accepted, max_new_tokens and cache_salt are read, and raw media without uuid emits no features",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Model:    "meta-llama/Llama-3.1-8B",
				LoraName: "sql-lora",
				Prompt:   &vllmpb.GenerateRequest_Text{Text: "hello"},
				Sampling: &vllmpb.RandomSampling{NumSequences: 1},
				Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 128},
				Kv:       &vllmpb.KVCacheParameters{CacheSalt: "salt-1"},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Source:   &vllmpb.MediaItem_RawBytes{RawBytes: bytes.Repeat([]byte("x"), 1<<16)},
					},
				},
			},
			wantModel: "sql-lora",
			wantGenerate: &fwkrequest.GenerateRequest{
				Text:      "hello",
				CacheSalt: "salt-1",
			},
			wantMaxTokens: &maxTokens128,
			wantStream:    false,
		},
		{
			name: "unary Generate with text prompt and lora_name sets Model to adapter",
			path: "vllm.Inference/Generate",
			req: &vllmpb.GenerateRequest{
				Model:    "meta-llama/Llama-3.1-8B",
				LoraName: "sql-lora",
				Prompt:   &vllmpb.GenerateRequest_Text{Text: "SELECT * FROM pods"},
			},
			wantModel: "sql-lora",
			wantGenerate: &fwkrequest.GenerateRequest{
				Text: "SELECT * FROM pods",
			},
			wantStream: false,
		},
		{
			name: "path with a trailing slash is accepted",
			path: vllmpb.Inference_GenerateStream_FullMethodName + "/",
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
			},
			wantGenerate: &fwkrequest.GenerateRequest{Text: "hello"},
			wantStream:   true,
		},
		{
			name: "preprocessed features without identifier, mm_hash, or uuid emit no features",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{Offset: 3, Length: 4},
						},
					},
				},
			},
			wantGenerate: &fwkrequest.GenerateRequest{Text: "hello"},
		},
		{
			name: "Generate without model is allowed for native generate API",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
			},
			wantGenerate: &fwkrequest.GenerateRequest{
				Text: "hello",
			},
			wantStream: false,
		},
		{
			name: "rejects missing prompt",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Model: "m",
			},
			wantErrContains: "prompt (text or token_ids) must be provided",
		},
		{
			name: "rejects empty text prompt",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: ""},
			},
			wantErrContains: "text cannot be empty",
		},
		{
			name: "rejects empty token_ids",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{}},
			},
			wantErrContains: "token_ids cannot be empty",
		},
		{
			name: "rejects num_sequences > 1",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt:   &vllmpb.GenerateRequest_Text{Text: "hi"},
				Sampling: &vllmpb.RandomSampling{NumSequences: 2},
			},
			wantErrContains: "num_sequences > 1 is not supported",
		},
		{
			name: "rejects preprocessed media features without modality",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_UNSPECIFIED,
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{Identifier: "h"},
						},
					},
				},
			},
			wantErrContains: "media modality is required",
		},
		{
			name: "rejects raw media with uuid when modality is unspecified",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_UNSPECIFIED,
						Uuid:     "u-1",
						Source:   &vllmpb.MediaItem_Url{Url: "https://example.com/a.png"},
					},
				},
			},
			wantErrContains: "media modality is required",
		},
		{
			name: "rejects a placeholder range beyond int",
			path: vllmpb.Inference_Generate_FullMethodName,
			req: &vllmpb.GenerateRequest{
				Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
				Media: []*vllmpb.MediaItem{
					{
						Modality: vllmpb.Modality_MODALITY_IMAGE,
						Source: &vllmpb.MediaItem_Features{
							Features: &vllmpb.PreprocessedMediaFeatures{Identifier: "h", Offset: math.MaxUint64},
						},
					},
				},
			},
			wantErrContains: "media placeholder range exceeds int limits",
		},
		{
			name: "rejects bytes after the gRPC frame",
			path: vllmpb.Inference_Generate_FullMethodName,
			rawBody: func() []byte {
				framed, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"}})
				require.NoError(t, err)
				return append(framed, 0xde, 0xad)
			}(),
			wantErrContains: "invalid gRPC generate request",
			wantErrIs:       grpcframe.ErrMalformed,
		},
		{
			name:            "rejects unsupported gRPC path",
			path:            vllmpb.Control_GetModelInfo_FullMethodName,
			req:             &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}},
			wantErrContains: "unsupported gRPC path",
		},
		{
			name:            "rejects truncated gRPC frame",
			path:            vllmpb.Inference_Generate_FullMethodName,
			rawBody:         []byte{0, 0, 0, 0, 10, 1, 2},
			wantErrContains: "invalid gRPC generate request",
		},
		{
			name:            "rejects compressed gRPC frame",
			path:            vllmpb.Inference_Generate_FullMethodName,
			rawBody:         []byte{1, 0, 0, 0, 2, 0x12, 0x00},
			wantErrContains: "invalid gRPC generate request",
			wantErrIs:       grpcframe.ErrCompressed,
		},
		{
			name:            "rejects truncated protobuf field",
			path:            vllmpb.Inference_Generate_FullMethodName,
			rawBody:         grpctest.FramePayload(t, []byte{0x12, 0x05, 'a'}),
			wantErrContains: "unmarshaling GenerateRequest",
			wantErrIs:       vllmwire.ErrMalformed,
		},
		{
			name: "rejects wrong wire type on GenerateRequest field",
			path: vllmpb.Inference_Generate_FullMethodName,
			rawBody: func() []byte {
				var bad []byte
				bad = protowire.AppendTag(bad, protoscan.FieldNum(&vllmpb.GenerateRequest{}, "model"), protowire.VarintType)
				bad = protowire.AppendVarint(bad, 1)
				return grpctest.FramePayload(t, bad)
			}(),
			wantErrContains: "unmarshaling GenerateRequest",
			wantErrIs:       vllmwire.ErrMalformed,
		},
	}

	parser := NewVllmInferenceParser()
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := tc.rawBody
			if body == nil {
				var err error
				body, err = grpcframe.MarshalProto(tc.req)
				require.NoError(t, err)
			}
			res, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": tc.path})
			if tc.wantErrContains != "" {
				require.Error(t, err)
				require.ErrorContains(t, err, tc.wantErrContains)
				if tc.wantErrIs != nil {
					require.ErrorIs(t, err, tc.wantErrIs)
				}
				return
			}
			require.NoError(t, err)
			require.False(t, res.SkipResponseProcessing)
			require.False(t, res.JSONValidated)
			require.Equal(t, tc.wantModel, res.Body.Model)
			require.Equal(t, tc.wantStream, res.Body.Stream)
			if diff := cmp.Diff(tc.wantMaxTokens, res.Body.MaxOutputTokens); diff != "" {
				t.Fatalf("MaxOutputTokens mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantGenerate, res.Body.Generate); diff != "" {
				t.Fatalf("Generate mismatch (-want +got):\n%s", diff)
			}
			require.Equal(t, body, res.Body.RawBody)
			require.Equal(t, fwkrequest.PayloadBytes(body), res.Body.Payload)
		})
	}
}

func TestVllmInferenceParser_WireEncodingAndOneofMergeRules(t *testing.T) {
	parser := NewVllmInferenceParser()
	headers := map[string]string{":path": vllmpb.Inference_Generate_FullMethodName}

	t.Run("unpacked and merged TokenIds decode in order", func(t *testing.T) {
		idsField := protoscan.FieldNum(&vllmpb.TokenIds{}, "ids")
		tokField := protoscan.FieldNum(&vllmpb.GenerateRequest{}, "token_ids")

		var unpacked []byte
		for _, id := range []uint32{0, 131, 262, 393, 524} {
			unpacked = protowire.AppendTag(unpacked, idsField, protowire.VarintType)
			unpacked = protowire.AppendVarint(unpacked, uint64(id))
		}
		var payload []byte
		payload = protowire.AppendTag(payload, tokField, protowire.BytesType)
		payload = protowire.AppendBytes(payload, unpacked)
		part2 := mustMarshal(t, &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{777, math.MaxUint32}}},
		})
		payload = slices.Concat(payload, part2)

		res, err := parser.ParseRequest(context.Background(), grpctest.FramePayload(t, payload), headers)
		require.NoError(t, err)
		require.Equal(t, []uint32{0, 131, 262, 393, 524, 777, math.MaxUint32}, res.Body.Generate.TokenIDs)
	})

	t.Run("prompt and MediaItem.source follow protobuf oneof merge rules", func(t *testing.T) {
		textFirst := mustMarshal(t, &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_Text{Text: "first"},
		})
		tokensSecond := mustMarshal(t, &vllmpb.GenerateRequest{
			Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{9, 8}}},
		})

		// text followed by token_ids -> token_ids wins.
		resTokens, err := parser.ParseRequest(context.Background(), grpctest.FramePayload(t, slices.Concat(textFirst, tokensSecond)), headers)
		require.NoError(t, err)
		require.Equal(t, []uint32{9, 8}, resTokens.Body.Generate.TokenIDs)
		require.Empty(t, resTokens.Body.Generate.Text)

		// token_ids followed by text -> text wins and clears token_ids.
		resText, err := parser.ParseRequest(context.Background(), grpctest.FramePayload(t, slices.Concat(tokensSecond, textFirst)), headers)
		require.NoError(t, err)
		require.Equal(t, "first", resText.Body.Generate.Text)
		require.Nil(t, resText.Body.Generate.TokenIDs)

		// MediaItem.source oneof: features followed by url clears features; uuid still applies as raw media.
		featItem := mustMarshal(t, &vllmpb.MediaItem{
			Modality: vllmpb.Modality_MODALITY_IMAGE,
			Uuid:     "raw-override-uuid",
			Source: &vllmpb.MediaItem_Features{Features: &vllmpb.PreprocessedMediaFeatures{
				Identifier: "img-1",
				Offset:     2,
				Length:     4,
			}},
		})
		urlItem := mustMarshal(t, &vllmpb.MediaItem{
			Source: &vllmpb.MediaItem_Url{Url: "https://example.com/a.png"},
		})
		mediaField := protoscan.FieldNum(&vllmpb.GenerateRequest{}, "media")
		var reqPayload []byte
		reqPayload = append(reqPayload, textFirst...)
		reqPayload = protowire.AppendTag(reqPayload, mediaField, protowire.BytesType)
		reqPayload = protowire.AppendBytes(reqPayload, slices.Concat(featItem, urlItem))

		resMedia, err := parser.ParseRequest(context.Background(), grpctest.FramePayload(t, reqPayload), headers)
		require.NoError(t, err)
		require.Equal(t, &tokenization.MultiModalFeatures{
			MMHashes: map[string][]string{string(reqcommon.ModalityImage): {"raw-override-uuid"}},
		}, resMedia.Body.Generate.Features)
	})
}

func TestVllmInferenceParser_RewritePriority(t *testing.T) {
	parser := NewVllmInferenceParser()

	t.Run("no-op when client omitted priority and EPP priority is zero", func(t *testing.T) {
		in, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
			Model:  "llama",
			Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
		})
		require.NoError(t, err)

		var boxed fwkrequest.MarshalablePayload = fwkrequest.PayloadBytes(in)
		got, mutated, err := parser.RewritePriority(fwkrh.PriorityRewriteContext{}, boxed, 0)
		require.NoError(t, err)
		require.False(t, mutated)
		out := got.(fwkrequest.PayloadBytes)
		require.Equal(t, len(in), len(out))
		require.Same(t, &in[0], &out[0], "no-op must return the input backing array without copying")

		allocs := testing.AllocsPerRun(100, func() {
			res, changed, callErr := parser.RewritePriority(fwkrh.PriorityRewriteContext{}, boxed, 0)
			if callErr != nil || changed || res == nil {
				t.Fatal("unexpected RewritePriority result")
			}
		})
		require.Zero(t, allocs)
	})

	t.Run("strips client priority and negates EPP priority across range", func(t *testing.T) {
		orig, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
			Model:    "llama",
			Prompt:   &vllmpb.GenerateRequest_Text{Text: "hello"},
			Priority: 99, // client-supplied priority must be stripped
		})
		require.NoError(t, err)

		cases := []struct {
			eppPriority      int
			wantVLLMPriority int32
		}{
			{eppPriority: 4, wantVLLMPriority: -4},
			{eppPriority: 0, wantVLLMPriority: 0},
			{eppPriority: -10, wantVLLMPriority: 10},
			{eppPriority: math.MaxInt32, wantVLLMPriority: -math.MaxInt32},
			{eppPriority: math.MinInt32, wantVLLMPriority: math.MaxInt32},
		}
		for _, tc := range cases {
			got, mutated, err := parser.RewritePriority(fwkrh.PriorityRewriteContext{}, fwkrequest.PayloadBytes(orig), tc.eppPriority)
			require.NoError(t, err)
			require.True(t, mutated)
			var decoded vllmpb.GenerateRequest
			grpctest.Unmarshal(t, got.(fwkrequest.PayloadBytes), &decoded)
			require.Equal(t, tc.wantVLLMPriority, decoded.GetPriority())
			require.Equal(t, "llama", decoded.GetModel())
			require.Equal(t, "hello", decoded.GetText())
		}
	})

	t.Run("double rewrite overwrites priority and composes with RewriteModelName on LoRA request", func(t *testing.T) {
		loraOrig, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
			Model:    "meta-llama/Llama-3.1-8B",
			LoraName: "lora-alias",
			Prompt:   &vllmpb.GenerateRequest_Text{Text: "hello"},
			Priority: 77,
		})
		require.NoError(t, err)

		afterModel, err := parser.RewriteModelName(fwkrequest.PayloadBytes(loraOrig), "sql-lora-v2")
		require.NoError(t, err)
		afterFirstPrio, mutated, err := parser.RewritePriority(fwkrh.PriorityRewriteContext{}, afterModel, 2)
		require.NoError(t, err)
		require.True(t, mutated)
		afterSecondPrio, mutated, err := parser.RewritePriority(fwkrh.PriorityRewriteContext{}, afterFirstPrio, 6)
		require.NoError(t, err)
		require.True(t, mutated)

		var decoded vllmpb.GenerateRequest
		grpctest.Unmarshal(t, afterSecondPrio.(fwkrequest.PayloadBytes), &decoded)
		require.Equal(t, "meta-llama/Llama-3.1-8B", decoded.GetModel())
		require.Equal(t, "sql-lora-v2", decoded.GetLoraName())
		require.Equal(t, int32(-6), decoded.GetPriority())
		require.Equal(t, "hello", decoded.GetText())
	})

	t.Run("non-PayloadBytes input and error cases", func(t *testing.T) {
		unchanged, mutated, err := parser.RewritePriority(fwkrh.PriorityRewriteContext{}, fwkrequest.PayloadMap{"a": 1}, 4)
		require.NoError(t, err)
		require.False(t, mutated)
		require.Equal(t, fwkrequest.PayloadMap{"a": 1}, unchanged)

		_, _, err = parser.RewritePriority(fwkrh.PriorityRewriteContext{}, fwkrequest.PayloadBytes{0, 0, 0, 0, 10, 1}, 4)
		require.ErrorContains(t, err, "rewriting priority")

		_, _, err = parser.RewritePriority(fwkrh.PriorityRewriteContext{}, fwkrequest.PayloadBytes(grpctest.FramePayload(t, []byte{0x12, 0x05, 'a'})), 4)
		require.ErrorIs(t, err, vllmwire.ErrMalformed)
	})
}

func TestVllmInferenceParser_RewriteModelName(t *testing.T) {
	parser := NewVllmInferenceParser()

	t.Run("rewrites base model when lora_name is empty", func(t *testing.T) {
		orig, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
			Model:  "alias-model",
			Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
		})
		require.NoError(t, err)
		got, err := parser.RewriteModelName(fwkrequest.PayloadBytes(orig), "meta-llama/Llama-3.1-8B")
		require.NoError(t, err)
		var decoded vllmpb.GenerateRequest
		grpctest.Unmarshal(t, got.(fwkrequest.PayloadBytes), &decoded)
		require.Equal(t, "meta-llama/Llama-3.1-8B", decoded.GetModel())
		require.Empty(t, decoded.GetLoraName())
	})

	t.Run("rewrites lora_name while preserving base model when lora_name is set", func(t *testing.T) {
		orig, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
			Model:    "meta-llama/Llama-3.1-8B",
			LoraName: "lora-alias-with-more-than-thirty-two-bytes",
			Prompt:   &vllmpb.GenerateRequest_Text{Text: "hello"},
		})
		require.NoError(t, err)
		got, err := parser.RewriteModelName(fwkrequest.PayloadBytes(orig), "sql-lora-v2")
		require.NoError(t, err)
		var decoded vllmpb.GenerateRequest
		grpctest.Unmarshal(t, got.(fwkrequest.PayloadBytes), &decoded)
		require.Equal(t, "meta-llama/Llama-3.1-8B", decoded.GetModel())
		require.Equal(t, "sql-lora-v2", decoded.GetLoraName())
	})

	t.Run("clears lora_name when the target is the base model", func(t *testing.T) {
		orig, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
			Model:    "meta-llama/Llama-3.1-8B",
			LoraName: "sql-lora",
			Prompt:   &vllmpb.GenerateRequest_Text{Text: "hello"},
		})
		require.NoError(t, err)
		got, err := parser.RewriteModelName(fwkrequest.PayloadBytes(orig), "meta-llama/Llama-3.1-8B")
		require.NoError(t, err)
		var decoded vllmpb.GenerateRequest
		grpctest.Unmarshal(t, got.(fwkrequest.PayloadBytes), &decoded)
		require.Equal(t, "meta-llama/Llama-3.1-8B", decoded.GetModel())
		require.Empty(t, decoded.GetLoraName(), "an empty lora_name selects the base model")
	})

	t.Run("non-PayloadBytes input and error cases", func(t *testing.T) {
		unchanged, err := parser.RewriteModelName(fwkrequest.PayloadMap{"a": 1}, "m")
		require.NoError(t, err)
		require.Equal(t, fwkrequest.PayloadMap{"a": 1}, unchanged)

		_, err = parser.RewriteModelName(fwkrequest.PayloadBytes{0, 0, 0, 0, 10, 1}, "m")
		require.ErrorContains(t, err, "invalid gRPC generate request")

		_, err = parser.RewriteModelName(fwkrequest.PayloadBytes(grpctest.FramePayload(t, []byte{0x12, 0x05, 'a'})), "m")
		require.ErrorIs(t, err, vllmwire.ErrMalformed)
	})
}

func TestVllmInferenceParser_ParseResponse(t *testing.T) {
	parser := NewVllmInferenceParser()
	cached8 := uint32(8)
	cached0 := uint32(0)

	t.Run("empty trailer chunk returns nil without error", func(t *testing.T) {
		res, err := parser.ParseResponse(context.Background(), nil, nil, true)
		require.NoError(t, err)
		require.Nil(t, res)
	})

	t.Run("unary response extracts prompt, completion, and cached tokens", func(t *testing.T) {
		frame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
			PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 15},
			Outputs: &vllmpb.SequenceOutput{
				Text:      "hello world",
				NumTokens: 2,
				FinishInfo: &vllmpb.FinishInfo{
					NumOutputTokens: 2,
					FinishReason:    vllmpb.FinishInfo_STOP,
					NumCachedTokens: &cached8,
				},
			},
		})
		require.NoError(t, err)
		res, err := parser.ParseResponse(context.Background(), frame, nil, true)
		require.NoError(t, err)
		require.Equal(t, 1, res.StreamedEvents)
		require.Equal(t, &fwkrequest.Usage{
			PromptTokens:       15,
			CompletionTokens:   2,
			TotalTokens:        17,
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 8},
		}, res.Usage)
	})

	t.Run("finish_info without num_cached_tokens leaves PromptTokenDetails nil", func(t *testing.T) {
		frame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
			PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 15},
			Outputs: &vllmpb.SequenceOutput{
				Text:       "hi",
				NumTokens:  2,
				FinishInfo: &vllmpb.FinishInfo{NumOutputTokens: 2, FinishReason: vllmpb.FinishInfo_STOP},
			},
		})
		require.NoError(t, err)
		res, err := parser.ParseResponse(context.Background(), frame, nil, false)
		require.NoError(t, err)
		require.Equal(t, &fwkrequest.Usage{PromptTokens: 15, CompletionTokens: 2, TotalTokens: 17}, res.Usage)
	})

	t.Run("streaming start frame leaves PromptTokenDetails nil until finish_info", func(t *testing.T) {
		startFrame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
			PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 15},
		})
		require.NoError(t, err)
		resStart, err := parser.ParseResponse(context.Background(), startFrame, nil, false)
		require.NoError(t, err)
		require.Equal(t, 0, resStart.StreamedEvents)
		require.Equal(t, &fwkrequest.Usage{
			PromptTokens: 15,
			TotalTokens:  15,
		}, resStart.Usage)

		deltaFrame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{Text: "hey", NumTokens: 1},
		})
		require.NoError(t, err)
		resDelta, err := parser.ParseResponse(context.Background(), deltaFrame, nil, false)
		require.NoError(t, err)
		require.Equal(t, 1, resDelta.StreamedEvents)
		require.Nil(t, resDelta.Usage)

		endFrame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{
				Text:      "lo",
				NumTokens: 1,
				FinishInfo: &vllmpb.FinishInfo{
					NumOutputTokens: 2,
					FinishReason:    vllmpb.FinishInfo_STOP,
					NumCachedTokens: &cached0,
				},
			},
		})
		require.NoError(t, err)
		resEnd, err := parser.ParseResponse(context.Background(), endFrame, nil, true)
		require.NoError(t, err)
		require.Equal(t, 1, resEnd.StreamedEvents)
		require.Equal(t, &fwkrequest.Usage{
			CompletionTokens:   2,
			TotalTokens:        2,
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 0},
		}, resEnd.Usage)

		// Combined in one chunk accumulates start + deltas + finish.
		combined := slices.Concat(startFrame, deltaFrame, endFrame)
		resCombined, err := parser.ParseResponse(context.Background(), combined, nil, true)
		require.NoError(t, err)
		require.Equal(t, 2, resCombined.StreamedEvents)
		require.Equal(t, &fwkrequest.Usage{
			PromptTokens:       15,
			CompletionTokens:   2,
			TotalTokens:        17,
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 0},
		}, resCombined.Usage)
	})

	t.Run("malformed protobuf frame reports error while continuing across remaining frames", func(t *testing.T) {
		deltaFrame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{Text: "hey", NumTokens: 1},
		})
		require.NoError(t, err)
		badFrame := grpctest.FramePayload(t, []byte{0x00}) // invalid field number 0
		endFrame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{
				Text:      "lo",
				NumTokens: 1,
				FinishInfo: &vllmpb.FinishInfo{
					NumOutputTokens: 2,
					FinishReason:    vllmpb.FinishInfo_STOP,
					NumCachedTokens: &cached8,
				},
			},
		})
		require.NoError(t, err)

		res, err := parser.ParseResponse(context.Background(), slices.Concat(deltaFrame, badFrame, endFrame), nil, true)
		require.ErrorIs(t, err, vllmwire.ErrMalformed)
		require.NotNil(t, res)
		require.Equal(t, 2, res.StreamedEvents)
		require.Equal(t, &fwkrequest.Usage{
			CompletionTokens:   2,
			TotalTokens:        2,
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 8},
		}, res.Usage)
	})

	t.Run("truncated or compressed gRPC frame stops frame loop and preserves earlier frames", func(t *testing.T) {
		deltaFrame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
			Outputs: &vllmpb.SequenceOutput{Text: "hey", NumTokens: 1},
		})
		require.NoError(t, err)

		// Truncated inside second frame header.
		resHeaderCut, err := parser.ParseResponse(context.Background(), slices.Concat(deltaFrame, []byte{0, 0, 0}), nil, true)
		require.ErrorContains(t, err, "parsing gRPC response payload")
		require.NotNil(t, resHeaderCut)
		require.Equal(t, 1, resHeaderCut.StreamedEvents)

		// Truncated inside second frame payload.
		resPayloadCut, err := parser.ParseResponse(context.Background(), slices.Concat(deltaFrame, []byte{0, 0, 0, 0, 10, 1, 2}), nil, true)
		require.ErrorContains(t, err, "parsing gRPC response payload")
		require.NotNil(t, resPayloadCut)
		require.Equal(t, 1, resPayloadCut.StreamedEvents)

		// Compressed gRPC frame is rejected with ErrCompressed.
		resCompressed, err := parser.ParseResponse(context.Background(), []byte{1, 0, 0, 0, 2, 0x08, 0x01}, nil, true)
		require.ErrorIs(t, err, grpcframe.ErrCompressed)
		require.Nil(t, resCompressed)
	})
}

// largeRequestFrame is an 8K-token request with a 1 MiB raw image.
func largeRequestFrame(tb testing.TB) []byte {
	tb.Helper()
	ids := make([]uint32, 8192)
	for i := range ids {
		ids[i] = uint32(i + 1)
	}
	return grpctest.Frame(tb, &vllmpb.GenerateRequest{
		Model:    "meta-llama/Llama-3.1-8B",
		Prompt:   &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: ids}},
		Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 256},
		Kv:       &vllmpb.KVCacheParameters{CacheSalt: "tenant-a"},
		Media: []*vllmpb.MediaItem{
			{
				Modality: vllmpb.Modality_MODALITY_IMAGE,
				Source:   &vllmpb.MediaItem_RawBytes{RawBytes: bytes.Repeat([]byte{0xff}, 1<<20)},
			},
		},
	})
}

// ParseRequest must not copy the media bytes or the token wire bytes. The
// allocations are the decoded []uint32 token slice and the result structs.
func TestVllmInferenceParser_ParseRequestAllocations(t *testing.T) {
	framed := largeRequestFrame(t)
	parser := NewVllmInferenceParser()
	headers := map[string]string{":path": vllmpb.Inference_GenerateStream_FullMethodName}
	require.LessOrEqual(t, testing.AllocsPerRun(20, func() {
		if _, err := parser.ParseRequest(context.Background(), framed, headers); err != nil {
			t.Fatal(err)
		}
	}), 11.0)
	bytesPerOp := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := parser.ParseRequest(context.Background(), framed, headers); err != nil {
				b.Fatal(err)
			}
		}
	}).AllocedBytesPerOp()
	require.Less(t, bytesPerOp, int64(64<<10), "a copy of the 1 MiB media would show here")
}

func BenchmarkVllmInferenceParser_ParseRequest8KTokens1MBMedia(b *testing.B) {
	framed := largeRequestFrame(b)
	parser := NewVllmInferenceParser()
	headers := map[string]string{":path": vllmpb.Inference_GenerateStream_FullMethodName}
	b.ReportAllocs()
	b.SetBytes(int64(len(framed)))
	for b.Loop() {
		if _, err := parser.ParseRequest(context.Background(), framed, headers); err != nil {
			b.Fatal(err)
		}
	}
}
