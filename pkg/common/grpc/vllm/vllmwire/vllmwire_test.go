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

package vllmwire

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
)

func mustMarshal(t testing.TB, m proto.Message) []byte {
	t.Helper()
	b, err := proto.Marshal(m)
	require.NoError(t, err)
	return b
}

func mustStruct(t testing.TB, m map[string]any) *structpb.Struct {
	t.Helper()
	st, err := structpb.NewStruct(m)
	require.NoError(t, err)
	return st
}

func decodeRequestFrame(t testing.TB, framed []byte) *vllmpb.GenerateRequest {
	t.Helper()
	payload, err := grpcframe.ParsePayload(framed)
	require.NoError(t, err)
	require.Len(t, framed, grpcframe.HeaderLen+len(payload))
	var got vllmpb.GenerateRequest
	require.NoError(t, proto.Unmarshal(payload, &got))
	return &got
}

func decodeResponseFrame(t testing.TB, framed []byte) *vllmpb.GenerateResponse {
	t.Helper()
	payload, err := grpcframe.ParsePayload(framed)
	require.NoError(t, err)
	require.Len(t, framed, grpcframe.HeaderLen+len(payload))
	var got vllmpb.GenerateResponse
	require.NoError(t, proto.Unmarshal(payload, &got))
	return &got
}

func requireProtoEqual(t testing.TB, want, got proto.Message) {
	t.Helper()
	if diff := cmp.Diff(want, got, protocmp.Transform()); diff != "" {
		t.Fatalf("mismatch (-want +got):\n%s", diff)
	}
}

func concat(parts ...[]byte) []byte {
	return bytes.Join(parts, nil)
}

// sampleGenerateRequest covers every top-level field, spoofed client
// transfer params, and both raw and preprocessed media.
func sampleGenerateRequest(t testing.TB) *vllmpb.GenerateRequest {
	t.Helper()
	return &vllmpb.GenerateRequest{
		RequestId:   "req-123",
		Model:       "meta-llama/Llama-3.1-8B-Instruct",
		Prompt:      &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{128000, 9906, 1917, 128009}}},
		Temperature: proto.Float32(0.7),
		Sampling: &vllmpb.RandomSampling{
			NumSequences: 1,
			TopK:         proto.Uint32(40),
			TopP:         proto.Float32(0.95),
			MinP:         proto.Float32(0.05),
			Seed:         proto.Int64(42),
		},
		Decoding: &vllmpb.DecodingParameters{
			PresencePenalty:   0.1,
			FrequencyPenalty:  0.2,
			RepetitionPenalty: 1.05,
			AllowedTokenIds:   []uint32{1, 2, 3},
			StructuredOutput:  &vllmpb.DecodingParameters_JsonObject{JsonObject: true},
		},
		Stopping: &vllmpb.StoppingCriteria{
			MaxNewTokens:        128,
			MinNewTokens:        16,
			StopTokenIds:        []uint32{128001, 128009},
			StopStrings:         []string{"</end>"},
			IncludeStopStrings:  true,
			IgnoreEos:           true,
			ThinkingTokenBudget: proto.Int64(64),
		},
		Response: &vllmpb.ResponseOptions{
			OutputText:        proto.Bool(true),
			OutputTokenIds:    true,
			SkipSpecialTokens: proto.Bool(false),
		},
		Kv: &vllmpb.KVCacheParameters{
			BypassPrefixCache: true,
			CacheSalt:         "tenant-salt-1",
			KvTransferParams:  mustStruct(t, map[string]any{"spoofed_kv": "evil"}),
			EcTransferParams:  mustStruct(t, map[string]any{"spoofed_ec": "evil"}),
		},
		TruncatePromptTokens: 2048,
		Priority:             3,
		SessionId:            proto.String("sess-1"),
		Media: []*vllmpb.MediaItem{
			{
				Modality: vllmpb.Modality_MODALITY_IMAGE,
				Source:   &vllmpb.MediaItem_RawBytes{RawBytes: bytes.Repeat([]byte{0xab}, 256)},
				MimeType: "image/png",
				Uuid:     "img-1",
			},
			{
				Modality: vllmpb.Modality_MODALITY_IMAGE,
				Source:   &vllmpb.MediaItem_Url{Url: "https://example.com/a.png"},
			},
			{
				Modality: vllmpb.Modality_MODALITY_VIDEO,
				Source: &vllmpb.MediaItem_Features{Features: &vllmpb.PreprocessedMediaFeatures{
					Kwargs:     bytes.Repeat([]byte{0xcd}, 4096),
					Identifier: "vid-ident",
					Offset:     1,
					Length:     2,
					MmHash:     proto.String("hash-vid"),
					IsEmbed:    []bool{true, true},
				}},
				Uuid: "vid-1",
			},
		},
		LoraName:     "adapter-a",
		Watermarking: proto.Bool(true),
	}
}

func cleanSample(t testing.TB) *vllmpb.GenerateRequest {
	t.Helper()
	req := sampleGenerateRequest(t)
	req.Kv.KvTransferParams = nil
	req.Kv.EcTransferParams = nil
	return req
}

func nixlPrefillKV() map[string]any {
	return map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
		"remote_engine_id":  nil,
		"remote_block_ids":  nil,
		"remote_host":       nil,
		"remote_port":       nil,
	}
}

func TestScanGenerateRequestAccessors(t *testing.T) {
	orig := sampleGenerateRequest(t)
	payload := mustMarshal(t, orig)
	snapshot := append([]byte(nil), payload...)

	req, err := ScanGenerateRequest(payload)
	require.NoError(t, err)
	require.Equal(t, snapshot, payload, "ScanGenerateRequest must not mutate its input")

	require.Equal(t, "req-123", req.RequestID())
	require.Equal(t, orig.Model, req.Model())
	require.Equal(t, "adapter-a", req.LoraName())
	require.Equal(t, PromptTokenIDs, req.Prompt)
	require.Equal(t, PromptTokenIDs, req.Prompt)
	require.Nil(t, req.Text())
	require.Equal(t, 4, req.tokenCount())
	ids, err := req.TokenIDs()
	require.NoError(t, err)
	require.Equal(t, []uint32{128000, 9906, 1917, 128009}, ids)
	ids2, err := req.TokenIDs()
	require.NoError(t, err)
	require.Same(t, &ids[0], &ids2[0], "TokenIDs decodes once and caches")

	salt, ok := req.CacheSalt()
	require.True(t, ok)
	require.Equal(t, "tenant-salt-1", string(salt))
	require.True(t, req.bypassPrefixCache)
	maxTok, ok := req.MaxNewTokens()
	require.True(t, ok)
	require.Equal(t, uint32(128), maxTok)
	n, ok := req.NumSequences()
	require.True(t, ok)
	require.Equal(t, uint32(1), n)
	prio, ok := req.Priority()
	require.True(t, ok)
	require.Equal(t, int32(3), prio)
	require.True(t, req.HadClientTransferParams())
	kvMap, ok, err := req.ClientKVTransferParams()
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, map[string]any{"spoofed_kv": "evil"}, kvMap)
	require.True(t, req.HasMedia())
	require.Equal(t, 3, req.media.count)

	// Text prompt: Text aliases the payload, TokenIDs is nil.
	textReq, err := ScanGenerateRequest(mustMarshal(t, &vllmpb.GenerateRequest{
		Prompt:   &vllmpb.GenerateRequest_Text{Text: "hello"},
		Priority: -7,
	}))
	require.NoError(t, err)
	require.Equal(t, PromptText, textReq.Prompt)
	require.Equal(t, "hello", string(textReq.Text()))
	require.Zero(t, textReq.tokenCount())
	ids, err = textReq.TokenIDs()
	require.NoError(t, err)
	require.Nil(t, ids)
	prio, ok = textReq.Priority()
	require.True(t, ok)
	require.Equal(t, int32(-7), prio)
	_, ok = textReq.MaxNewTokens()
	require.False(t, ok)
	_, ok = textReq.NumSequences()
	require.False(t, ok)
	_, ok = textReq.CacheSalt()
	require.False(t, ok)
	_, ok, err = textReq.ClientKVTransferParams()
	require.NoError(t, err)
	require.False(t, ok)

	// Empty request.
	emptyReq, err := ScanGenerateRequest(nil)
	require.NoError(t, err)
	require.Equal(t, PromptNone, emptyReq.Prompt)
	require.Equal(t, PromptNone, emptyReq.Prompt)
	require.Empty(t, emptyReq.RequestID())
	require.Empty(t, emptyReq.Model())
	require.False(t, emptyReq.HasMedia())
}

func TestScanGenerateRequestZeroCopyAndZeroAlloc(t *testing.T) {
	cleanPayload := mustMarshal(t, cleanSample(t))
	cleanReq, err := ScanGenerateRequest(cleanPayload)
	require.NoError(t, err)
	require.False(t, cleanReq.HadClientTransferParams())
	require.Zero(t, testing.AllocsPerRun(100, func() {
		_, _ = ScanGenerateRequest(cleanPayload)
	}))

	segs, err := cleanReq.FrameSegments(LegOptions{})
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(segs), 3)
	require.Len(t, segs[0], grpcframe.HeaderLen)
	require.Equal(t, grpcframe.HeaderLen, cap(segs[0]), "head slice must not have spare capacity into tail")
	// Every segment between head and tail aliases cleanPayload.
	for _, seg := range segs[1 : len(segs)-1] {
		offset := bytes.Index(cleanPayload, seg)
		require.NotEqual(t, -1, offset)
		require.Same(t, &cleanPayload[offset], &seg[0], "span must alias the input payload")
	}
	// Text and token bytes alias as well.
	tokReq, err := ScanGenerateRequest(mustMarshal(t, &vllmpb.GenerateRequest{
		Prompt: &vllmpb.GenerateRequest_Text{Text: "alias-me"},
	}))
	require.NoError(t, err)
	require.Equal(t, "alias-me", string(tokReq.Text()))

	// The stripped frame equals the proto-level sanitized request.
	framed, err := cleanReq.Frame(LegOptions{})
	require.NoError(t, err)
	requireProtoEqual(t, cleanSample(t), decodeRequestFrame(t, framed))
}

func TestScanGenerateRequestStripsClientTransferParams(t *testing.T) {
	orig := sampleGenerateRequest(t)
	req, err := ScanGenerateRequest(mustMarshal(t, orig))
	require.NoError(t, err)
	framed, err := req.Frame(LegOptions{})
	require.NoError(t, err)
	requireProtoEqual(t, cleanSample(t), decodeRequestFrame(t, framed))

	// When kv only held transfer params, stripping leaves kv unset.
	onlySpoofed := &vllmpb.GenerateRequest{
		Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
		Kv: &vllmpb.KVCacheParameters{
			KvTransferParams: mustStruct(t, map[string]any{"do_remote_decode": true}),
		},
	}
	spoofReq, err := ScanGenerateRequest(mustMarshal(t, onlySpoofed))
	require.NoError(t, err)
	require.True(t, spoofReq.HadClientTransferParams())
	spoofFramed, err := spoofReq.Frame(LegOptions{})
	require.NoError(t, err)
	requireProtoEqual(t, &vllmpb.GenerateRequest{
		Prompt: &vllmpb.GenerateRequest_Text{Text: "hello"},
	}, decodeRequestFrame(t, spoofFramed))

	// An empty kv message is preserved as an empty message.
	emptyKV, err := ScanGenerateRequest(mustMarshal(t, &vllmpb.GenerateRequest{
		Prompt: &vllmpb.GenerateRequest_Text{Text: "x"},
		Kv:     &vllmpb.KVCacheParameters{},
	}))
	require.NoError(t, err)
	emptyFramed, err := emptyKV.Frame(LegOptions{})
	require.NoError(t, err)
	require.NotNil(t, decodeRequestFrame(t, emptyFramed).Kv)
}

func TestScanGenerateRequestMergedMessages(t *testing.T) {
	part1 := mustMarshal(t, &vllmpb.GenerateRequest{
		RequestId: "first",
		Model:     "m",
		Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2}}},
		Stopping:  &vllmpb.StoppingCriteria{MaxNewTokens: 50, MinNewTokens: 10, StopStrings: []string{"A"}},
		Kv: &vllmpb.KVCacheParameters{
			CacheSalt:        "salt-1",
			KvTransferParams: mustStruct(t, map[string]any{"a": 1, "k": "one"}),
		},
		Media: []*vllmpb.MediaItem{{Uuid: "m1"}},
	})
	part2 := mustMarshal(t, &vllmpb.GenerateRequest{
		RequestId: "second",
		Prompt:    &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{3, 4}}},
		Stopping:  &vllmpb.StoppingCriteria{MaxNewTokens: 80, StopStrings: []string{"B"}},
		Kv: &vllmpb.KVCacheParameters{
			BypassPrefixCache: true,
			CacheSalt:         "salt-2",
			KvTransferParams:  mustStruct(t, map[string]any{"k": "two"}),
		},
		Media: []*vllmpb.MediaItem{{Uuid: "m2"}},
	})
	merged := concat(part1, part2)
	var want vllmpb.GenerateRequest
	require.NoError(t, proto.Unmarshal(merged, &want))

	req, err := ScanGenerateRequest(merged)
	require.NoError(t, err)
	require.Equal(t, "second", req.RequestID())
	require.Equal(t, 4, req.tokenCount())
	ids, err := req.TokenIDs()
	require.NoError(t, err)
	require.Equal(t, []uint32{1, 2, 3, 4}, ids)
	maxTok, _ := req.MaxNewTokens()
	require.Equal(t, uint32(80), maxTok)
	salt, _ := req.CacheSalt()
	require.Equal(t, "salt-2", string(salt))
	require.True(t, req.bypassPrefixCache)
	require.Equal(t, 2, req.media.count)
	kvMap, ok, err := req.ClientKVTransferParams()
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, want.GetKv().GetKvTransferParams().AsMap(), kvMap)

	// Decode leg: everything merged as proto.Unmarshal merges it, minus transfer params.
	framed, err := req.Frame(LegOptions{})
	require.NoError(t, err)
	wantClean := proto.Clone(&want).(*vllmpb.GenerateRequest)
	wantClean.Kv.KvTransferParams = nil
	requireProtoEqual(t, wantClean, decodeRequestFrame(t, framed))

	// Prefill leg: capped stopping keeps merged stop strings.
	prefill, err := req.Frame(LegOptions{CapSingleToken: true})
	require.NoError(t, err)
	got := decodeRequestFrame(t, prefill)
	require.Equal(t, uint32(1), got.GetStopping().GetMaxNewTokens())
	require.Zero(t, got.GetStopping().GetMinNewTokens())
	require.Equal(t, []string{"A", "B"}, got.GetStopping().GetStopStrings())

	// prompt oneof: a later text clears token_ids and vice versa.
	textFirst := mustMarshal(t, &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "first"}})
	tokensSecond := mustMarshal(t, &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{9, 8}}}})
	tokReq, err := ScanGenerateRequest(concat(textFirst, tokensSecond))
	require.NoError(t, err)
	require.Equal(t, PromptTokenIDs, tokReq.Prompt)
	ids, err = tokReq.TokenIDs()
	require.NoError(t, err)
	require.Equal(t, []uint32{9, 8}, ids)
	txtReq, err := ScanGenerateRequest(concat(tokensSecond, textFirst))
	require.NoError(t, err)
	require.Equal(t, PromptText, txtReq.Prompt)
	require.Equal(t, "first", string(txtReq.Text()))
	require.Zero(t, txtReq.tokenCount())
}

func TestTokenIDsEncodings(t *testing.T) {
	ids := make([]uint32, 1000)
	for i := range ids {
		ids[i] = uint32(i * 131) //#nosec G115 -- test data
	}
	ids[999] = math.MaxUint32
	req, err := ScanGenerateRequest(mustMarshal(t, &vllmpb.GenerateRequest{
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: ids}},
	}))
	require.NoError(t, err)
	require.Equal(t, 1000, req.tokenCount())
	got, err := req.TokenIDs()
	require.NoError(t, err)
	require.Equal(t, ids, got)
	require.Equal(t, len(got), cap(got), "one exact-size allocation")

	// Unpacked encoding (one varint field per element) is valid proto3 input.
	var unpacked []byte
	for _, id := range ids[:5] {
		unpacked = protowire.AppendTag(unpacked, tokenIDsIDs, protowire.VarintType)
		unpacked = protowire.AppendVarint(unpacked, uint64(id))
	}
	payload := protowire.AppendBytes(protowire.AppendTag(nil, reqTokenIDs, protowire.BytesType), unpacked)
	req, err = ScanGenerateRequest(payload)
	require.NoError(t, err)
	require.Equal(t, 5, req.tokenCount())
	got, err = req.TokenIDs()
	require.NoError(t, err)
	require.Equal(t, []uint32{0, 131, 262, 393, 524}, got)

	// Values above uint32 are rejected, never truncated.
	over := protowire.AppendBytes(protowire.AppendTag(nil, tokenIDsIDs, protowire.BytesType), protowire.AppendVarint(nil, math.MaxUint32+1))
	payload = protowire.AppendBytes(protowire.AppendTag(nil, reqTokenIDs, protowire.BytesType), over)
	req, err = ScanGenerateRequest(payload)
	require.NoError(t, err)
	_, err = req.TokenIDs()
	require.ErrorIs(t, err, ErrMalformed)
}

func TestScanGenerateRequestMalformed(t *testing.T) {
	wrapStopping := func(inner []byte) []byte {
		return protowire.AppendBytes(protowire.AppendTag(nil, reqStopping, protowire.BytesType), inner)
	}
	wrapKV := func(inner []byte) []byte {
		return protowire.AppendBytes(protowire.AppendTag(nil, reqKV, protowire.BytesType), inner)
	}
	cases := map[string][]byte{
		"bad tag":             {0xff, 0xff, 0xff, 0xff, 0xff, 0x0f},
		"truncated tag":       {0x80},
		"truncated bytes":     {0x0a, 0x08, 'a', 'b'},
		"stopping varint":     protowire.AppendVarint(protowire.AppendTag(nil, reqStopping, protowire.VarintType), 1),
		"kv varint":           protowire.AppendVarint(protowire.AppendTag(nil, reqKV, protowire.VarintType), 1),
		"request_id varint":   protowire.AppendVarint(protowire.AppendTag(nil, reqRequestID, protowire.VarintType), 1),
		"model varint":        protowire.AppendVarint(protowire.AppendTag(nil, reqModel, protowire.VarintType), 1),
		"text varint":         protowire.AppendVarint(protowire.AppendTag(nil, reqText, protowire.VarintType), 1),
		"token_ids varint":    protowire.AppendVarint(protowire.AppendTag(nil, reqTokenIDs, protowire.VarintType), 1),
		"media varint":        protowire.AppendVarint(protowire.AppendTag(nil, reqMedia, protowire.VarintType), 1),
		"sampling varint":     protowire.AppendVarint(protowire.AppendTag(nil, reqSampling, protowire.VarintType), 1),
		"lora varint":         protowire.AppendVarint(protowire.AppendTag(nil, reqLoraName, protowire.VarintType), 1),
		"priority bytes":      protowire.AppendString(protowire.AppendTag(nil, reqPriority, protowire.BytesType), "x"),
		"truncated kv body":   append(protowire.AppendTag(nil, reqKV, protowire.BytesType), 5, 0x08),
		"max_new_tokens type": wrapStopping(protowire.AppendString(protowire.AppendTag(nil, stopMaxNewTokens, protowire.BytesType), "bad")),
		"min_new_tokens type": wrapStopping(protowire.AppendString(protowire.AppendTag(nil, stopMinNewTokens, protowire.BytesType), "bad")),
		"kv transfer varint":  wrapKV(protowire.AppendVarint(protowire.AppendTag(nil, kvTransferParams, protowire.VarintType), 1)),
		"ec transfer varint":  wrapKV(protowire.AppendVarint(protowire.AppendTag(nil, ecTransferParams, protowire.VarintType), 1)),
		"cache_salt varint":   wrapKV(protowire.AppendVarint(protowire.AppendTag(nil, kvCacheSalt, protowire.VarintType), 1)),
		"bypass bytes":        wrapKV(protowire.AppendString(protowire.AppendTag(nil, kvBypassPrefixCache, protowire.BytesType), "x")),
		"kv transfer body":    wrapKV(protowire.AppendBytes(protowire.AppendTag(nil, kvTransferParams, protowire.BytesType), []byte{0x80})),
		"token_ids body":      protowire.AppendBytes(protowire.AppendTag(nil, reqTokenIDs, protowire.BytesType), []byte{0x80}),
		"media body":          protowire.AppendBytes(protowire.AppendTag(nil, reqMedia, protowire.BytesType), []byte{0x80}),
		// A malformed features body is rejected at scan time, so the
		// MediaMetadataOnly rewrite, which re-encodes features, never
		// meets one.
		"media features body": protowire.AppendBytes(protowire.AppendTag(nil, reqMedia, protowire.BytesType),
			protowire.AppendBytes(protowire.AppendTag(nil, mediaFeatures, protowire.BytesType), []byte{0x80})),
		"num_sequences bytes": protowire.AppendBytes(protowire.AppendTag(nil, reqSampling, protowire.BytesType), protowire.AppendString(protowire.AppendTag(nil, samplingNumSequences, protowire.BytesType), "x")),
	}
	for name, input := range cases {
		_, err := ScanGenerateRequest(input)
		require.ErrorIs(t, err, ErrMalformed, name)
	}
}

func TestFrameSegmentsLegOptionsMatchProto(t *testing.T) {
	orig := sampleGenerateRequest(t)
	nixlDecodeKV := map[string]any{
		"do_remote_decode":  false,
		"do_remote_prefill": true,
		"remote_engine_id":  "eng-1",
		"remote_block_ids":  []int{10, 11, 12},
		"remote_host":       "10.0.0.8",
		"remote_port":       json.Number("5557"),
		"tp_size":           4,
		"empty_str":         "",
		"zero_int":          0,
		"false_flag":        false,
		"string_list":       []string{"10.0.1.5", "10.0.1.6"},
		"map_list":          []map[string]any{{"transfer_id": "tx-1"}},
	}
	wantDecodeKV := mustStruct(t, map[string]any{
		"do_remote_decode":  false,
		"do_remote_prefill": true,
		"remote_engine_id":  "eng-1",
		"remote_block_ids":  []any{10, 11, 12},
		"remote_host":       "10.0.0.8",
		"remote_port":       float64(5557),
		"tp_size":           4,
		"empty_str":         "",
		"zero_int":          0,
		"false_flag":        false,
		"string_list":       []any{"10.0.1.5", "10.0.1.6"},
		"map_list":          []any{map[string]any{"transfer_id": "tx-1"}},
	})
	ecParams := map[string]any{
		"ec_connector_extra": map[string]any{"keys": []string{"mm-hash-1", "mm-hash-2"}},
	}
	wantEC := mustStruct(t, map[string]any{
		"ec_connector_extra": map[string]any{"keys": []any{"mm-hash-1", "mm-hash-2"}},
	})
	metadataOnlyMedia := func(items []*vllmpb.MediaItem) []*vllmpb.MediaItem {
		out := make([]*vllmpb.MediaItem, len(items))
		for i, item := range items {
			cp := proto.Clone(item).(*vllmpb.MediaItem)
			switch src := cp.Source.(type) {
			case *vllmpb.MediaItem_Features:
				src.Features.Kwargs = nil
			default:
				cp.Source = nil
			}
			out[i] = cp
		}
		return out
	}

	tests := []struct {
		name string
		base *vllmpb.GenerateRequest
		opts LegOptions
		want func(*vllmpb.GenerateRequest)
	}{
		{
			name: "encode leg caps max_new_tokens=1, clears min_new_tokens, keeps all media",
			base: orig,
			opts: LegOptions{CapSingleToken: true, ECTransferParams: ecParams},
			want: func(r *vllmpb.GenerateRequest) {
				r.Stopping.MaxNewTokens = 1
				r.Stopping.MinNewTokens = 0
				r.Kv.KvTransferParams = nil
				r.Kv.EcTransferParams = wantEC
			},
		},
		{
			name: "prefill leg attaches kv and ec transfer params while keeping cache_salt",
			base: orig,
			opts: LegOptions{CapSingleToken: true, KVTransferParams: nixlPrefillKV(), ECTransferParams: ecParams},
			want: func(r *vllmpb.GenerateRequest) {
				r.Stopping.MaxNewTokens = 1
				r.Stopping.MinNewTokens = 0
				r.Kv.KvTransferParams = mustStruct(t, nixlPrefillKV())
				r.Kv.EcTransferParams = wantEC
			},
		},
		{
			name: "prefill after encode keeps media metadata only",
			base: orig,
			opts: LegOptions{CapSingleToken: true, Media: MediaMetadataOnly, KVTransferParams: nixlPrefillKV()},
			want: func(r *vllmpb.GenerateRequest) {
				r.Stopping.MaxNewTokens = 1
				r.Stopping.MinNewTokens = 0
				r.Kv.KvTransferParams = mustStruct(t, nixlPrefillKV())
				r.Kv.EcTransferParams = nil
				r.Media = metadataOnlyMedia(r.Media)
			},
		},
		{
			name: "decode leg preserves client stopping and drops media",
			base: orig,
			opts: LegOptions{Media: MediaNone, KVTransferParams: nixlDecodeKV},
			want: func(r *vllmpb.GenerateRequest) {
				r.Kv.KvTransferParams = wantDecodeKV
				r.Kv.EcTransferParams = nil
				r.Media = nil
			},
		},
		{
			name: "request id override replaces the client id",
			base: orig,
			opts: LegOptions{RequestID: "sidecar-uuid"},
			want: func(r *vllmpb.GenerateRequest) {
				r.RequestId = "sidecar-uuid"
				r.Kv.KvTransferParams = nil
				r.Kv.EcTransferParams = nil
			},
		},
		{
			name: "request id override on a request without one",
			base: &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}},
			opts: LegOptions{RequestID: "new-id"},
			want: func(r *vllmpb.GenerateRequest) { r.RequestId = "new-id" },
		},
		{
			name: "prefill leg synthesizes stopping when client omitted stopping and kv",
			base: &vllmpb.GenerateRequest{
				RequestId: "r-minimal",
				Prompt:    &vllmpb.GenerateRequest_Text{Text: "hi"},
			},
			opts: LegOptions{CapSingleToken: true, KVTransferParams: map[string]any{"do_remote_decode": true}},
			want: func(r *vllmpb.GenerateRequest) {
				r.Stopping = &vllmpb.StoppingCriteria{MaxNewTokens: 1}
				r.Kv = &vllmpb.KVCacheParameters{KvTransferParams: mustStruct(t, map[string]any{"do_remote_decode": true})}
			},
		},
		{
			name: "decode leg with no stopping and no kv emits neither",
			base: &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}},
			opts: LegOptions{},
			want: func(*vllmpb.GenerateRequest) {},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			scanned, err := ScanGenerateRequest(mustMarshal(t, tc.base))
			require.NoError(t, err)
			segs, err := scanned.FrameSegments(tc.opts)
			require.NoError(t, err)
			require.Len(t, segs[0], grpcframe.HeaderLen)
			require.Equal(t, grpcframe.HeaderLen, cap(segs[0]))

			want := proto.Clone(tc.base).(*vllmpb.GenerateRequest)
			tc.want(want)
			requireProtoEqual(t, want, decodeRequestFrame(t, concat(segs...)))

			framed, err := scanned.Frame(tc.opts)
			require.NoError(t, err)
			require.Len(t, framed, len(concat(segs...)))
			requireProtoEqual(t, want, decodeRequestFrame(t, framed))
		})
	}

	// Fields that the descriptor does not know pass through once in every
	// media mode.
	t.Run("unknown fields survive", func(t *testing.T) {
		payload := mustMarshal(t, orig)
		payload = protowire.AppendVarint(protowire.AppendTag(payload, 99, protowire.VarintType), 7)
		payload = protowire.AppendString(protowire.AppendTag(payload, 100, protowire.BytesType), "opaque")
		payload = protowire.AppendFixed64(protowire.AppendTag(payload, 101, protowire.Fixed64Type), 42)
		scanned, err := ScanGenerateRequest(payload)
		require.NoError(t, err)
		for _, mode := range []MediaMode{MediaAll, MediaMetadataOnly, MediaNone} {
			framed, err := scanned.Frame(LegOptions{Media: mode, CapSingleToken: true, RequestID: "x"})
			require.NoError(t, err)
			got := decodeRequestFrame(t, framed)
			unknown := got.ProtoReflect().GetUnknown()
			var nums []protowire.Number
			for len(unknown) > 0 {
				num, wt, n := protowire.ConsumeTag(unknown)
				m := protowire.ConsumeFieldValue(num, wt, unknown[n:])
				nums = append(nums, num)
				unknown = unknown[n+m:]
			}
			require.Equal(t, []protowire.Number{99, 100, 101}, nums, "mode %d", mode)
		}
	})

	t.Run("unknown media mode", func(t *testing.T) {
		scanned, err := ScanGenerateRequest(mustMarshal(t, orig))
		require.NoError(t, err)
		_, err = scanned.FrameSegments(LegOptions{Media: MediaMode(9)})
		require.Error(t, err)
	})

	t.Run("bad transfer params", func(t *testing.T) {
		scanned, err := ScanGenerateRequest(mustMarshal(t, orig))
		require.NoError(t, err)
		_, err = scanned.FrameSegments(LegOptions{KVTransferParams: map[string]any{"x": math.NaN()}})
		require.Error(t, err)
		_, err = scanned.FrameSegments(LegOptions{ECTransferParams: map[string]any{"x": struct{}{}}})
		require.Error(t, err)
	})
}

func TestFrameSegmentsMediaOverride(t *testing.T) {
	orig := sampleGenerateRequest(t)
	scanned, err := ScanGenerateRequest(mustMarshal(t, orig))
	require.NoError(t, err)
	items, err := scanned.MediaItems()
	require.NoError(t, err)
	require.Len(t, items, 3)

	require.Error(t, scanned.SetMediaBodies([][]byte{nil}), "count mismatch")
	bodies := [][]byte{
		items[0].RewrittenBody("content-0", nil),
		items[1].RewrittenBody("content-1", nil),
		items[2].Body,
	}
	require.NoError(t, scanned.SetMediaBodies(bodies))

	// MediaAll with overrides forwards the rewritten uuids.
	framed, err := scanned.Frame(LegOptions{})
	require.NoError(t, err)
	want := cleanSample(t)
	want.Media[0].Uuid = "content-0"
	want.Media[1].Uuid = "content-1"
	requireProtoEqual(t, want, decodeRequestFrame(t, framed))

	// MediaItems reflects the overrides.
	items, err = scanned.MediaItems()
	require.NoError(t, err)
	require.Equal(t, "content-0", items[0].UUID)
	require.Equal(t, "content-1", items[1].UUID)

	// MediaMetadataOnly applies on top of the overrides.
	metaFramed, err := scanned.Frame(LegOptions{Media: MediaMetadataOnly})
	require.NoError(t, err)
	got := decodeRequestFrame(t, metaFramed)
	require.Len(t, got.Media, 3)
	require.Equal(t, "content-0", got.Media[0].GetUuid())
	require.Nil(t, got.Media[0].Source)
	require.Equal(t, "image/png", got.Media[0].GetMimeType())
	require.Nil(t, got.Media[1].Source)
	require.Nil(t, got.Media[2].GetFeatures().Kwargs)
	require.Equal(t, "hash-vid", got.Media[2].GetFeatures().GetMmHash())
	require.Equal(t, uint64(1), got.Media[2].GetFeatures().GetOffset())

	// Restore client bytes.
	require.NoError(t, scanned.SetMediaBodies(nil))
	framed, err = scanned.Frame(LegOptions{})
	require.NoError(t, err)
	requireProtoEqual(t, cleanSample(t), decodeRequestFrame(t, framed))
}

func TestFrameSegmentsAllocations(t *testing.T) {
	payload := mustMarshal(t, cleanSample(t))
	req, err := ScanGenerateRequest(payload)
	require.NoError(t, err)

	require.LessOrEqual(t, testing.AllocsPerRun(100, func() {
		_, _ = req.FrameSegments(LegOptions{})
	}), 2.0, "MediaAll without KV: segments slice plus scratch buffer")

	kv := nixlPrefillKV()
	require.LessOrEqual(t, testing.AllocsPerRun(100, func() {
		_, _ = req.FrameSegments(LegOptions{CapSingleToken: true, KVTransferParams: kv})
	}), 2.0, "6-key NIXL KV map: segments slice plus scratch buffer")

	require.LessOrEqual(t, testing.AllocsPerRun(100, func() {
		_, _ = req.FrameSegments(LegOptions{Media: MediaNone, RequestID: "override"})
	}), 2.0, "MediaNone with request id override")

	require.LessOrEqual(t, testing.AllocsPerRun(100, func() {
		_, _ = req.FrameSegments(LegOptions{Media: MediaMetadataOnly, CapSingleToken: true})
	}), 2.0, "MediaMetadataOnly: the media segment is counted in the segments slice")

	// Every scratch-buffer segment is capped so a caller append cannot write
	// into the next segment.
	segs, err := req.FrameSegments(LegOptions{Media: MediaMetadataOnly, RequestID: "override", CapSingleToken: true})
	require.NoError(t, err)
	for i, seg := range segs {
		if len(seg) > 0 && bytes.Contains(payload, seg) {
			continue // aliases the payload
		}
		require.Equal(t, len(seg), cap(seg), "segment %d", i)
	}
}

func TestEncoderSubFrame(t *testing.T) {
	orig := sampleGenerateRequest(t)
	scanned, err := ScanGenerateRequest(mustMarshal(t, orig))
	require.NoError(t, err)
	items, err := scanned.MediaItems()
	require.NoError(t, err)

	ec := map[string]any{"consumer": "zmq://x", "items": []map[string]any{{"transfer_id": "t-1"}}}
	offset := uint64(1)
	frame, err := scanned.EncoderSubFrame(EncoderSubFrameOptions{
		RequestID:        "req-123-mm-2",
		TokenIDs:         []uint32{128000, 9906, 1917},
		Item:             items[2],
		UUIDOverride:     "uuid-override",
		FeaturesOffset:   &offset,
		ECTransferParams: ec,
	})
	require.NoError(t, err)

	want := proto.Clone(orig).(*vllmpb.GenerateRequest)
	want.RequestId = "req-123-mm-2"
	want.Prompt = &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{128000, 9906, 1917}}}
	item := proto.Clone(orig.Media[2]).(*vllmpb.MediaItem)
	item.Uuid = "uuid-override"
	item.GetFeatures().Offset = 1
	want.Media = []*vllmpb.MediaItem{item}
	want.Stopping.MaxNewTokens = 1
	want.Stopping.MinNewTokens = 0
	want.Kv.KvTransferParams = nil
	want.Kv.EcTransferParams = mustStruct(t, map[string]any{"consumer": "zmq://x", "items": []any{map[string]any{"transfer_id": "t-1"}}})
	requireProtoEqual(t, want, decodeRequestFrame(t, frame))

	// Minimal request: no stopping, no kv, raw item, empty token list, zero offset drops the field.
	minimal := &vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_Text{Text: "ignored"},
		Media: []*vllmpb.MediaItem{{
			Modality: vllmpb.Modality_MODALITY_IMAGE,
			Source:   &vllmpb.MediaItem_Features{Features: &vllmpb.PreprocessedMediaFeatures{Offset: 7, Length: 2}},
		}},
	}
	scanned, err = ScanGenerateRequest(mustMarshal(t, minimal))
	require.NoError(t, err)
	items, err = scanned.MediaItems()
	require.NoError(t, err)
	zero := uint64(0)
	frame, err = scanned.EncoderSubFrame(EncoderSubFrameOptions{Item: items[0], FeaturesOffset: &zero})
	require.NoError(t, err)
	got := decodeRequestFrame(t, frame)
	requireProtoEqual(t, &vllmpb.GenerateRequest{
		Model:  "m",
		Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{}},
		Media: []*vllmpb.MediaItem{{
			Modality: vllmpb.Modality_MODALITY_IMAGE,
			Source:   &vllmpb.MediaItem_Features{Features: &vllmpb.PreprocessedMediaFeatures{Length: 2}},
		}},
		Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 1},
	}, got)

	_, err = scanned.EncoderSubFrame(EncoderSubFrameOptions{Item: items[0], ECTransferParams: map[string]any{"x": math.Inf(1)}})
	require.Error(t, err)
}

func TestMediaItems(t *testing.T) {
	orig := sampleGenerateRequest(t)
	payload := mustMarshal(t, orig)
	scanned, err := ScanGenerateRequest(payload)
	require.NoError(t, err)
	items, err := scanned.MediaItems()
	require.NoError(t, err)
	require.Len(t, items, 3)

	require.Equal(t, vllmpb.Modality_MODALITY_IMAGE, items[0].Modality)
	require.Equal(t, "img-1", items[0].UUID)
	require.False(t, items[0].HasFeatures)
	require.Equal(t, mustMarshal(t, orig.Media[0]), items[0].Body)
	offset := bytes.Index(payload, items[0].Body)
	require.Same(t, &payload[offset], &items[0].Body[0], "Body aliases the payload")

	require.Empty(t, items[1].UUID)
	require.False(t, items[1].HasFeatures)

	require.Equal(t, vllmpb.Modality_MODALITY_VIDEO, items[2].Modality)
	require.Equal(t, "vid-1", items[2].UUID)
	require.True(t, items[2].HasFeatures)
	require.Equal(t, "vid-ident", string(items[2].Identifier))
	require.True(t, items[2].HasMMHash)
	require.Equal(t, "hash-vid", string(items[2].MMHash))
	require.Equal(t, uint64(1), items[2].Offset)
	require.Equal(t, uint64(2), items[2].Length)

	// No media.
	none, err := ScanGenerateRequest(mustMarshal(t, &vllmpb.GenerateRequest{}))
	require.NoError(t, err)
	items, err = none.MediaItems()
	require.NoError(t, err)
	require.Nil(t, items)

	// source oneof: features followed by url -> url wins, clearing features.
	featItem := mustMarshal(t, &vllmpb.MediaItem{
		Source: &vllmpb.MediaItem_Features{Features: &vllmpb.PreprocessedMediaFeatures{Identifier: "img-1", Offset: 2, Length: 4}},
	})
	urlItem := mustMarshal(t, &vllmpb.MediaItem{Source: &vllmpb.MediaItem_Url{Url: "https://example.com/a.png"}})
	mergedPayload := protowire.AppendBytes(protowire.AppendTag(nil, reqMedia, protowire.BytesType), concat(featItem, urlItem))
	mergedReq, err := ScanGenerateRequest(mergedPayload)
	require.NoError(t, err)
	items, err = mergedReq.MediaItems()
	require.NoError(t, err)
	require.False(t, items[0].HasFeatures)
	require.Nil(t, items[0].Identifier)
	require.Zero(t, items[0].Offset)

	// Malformed item fields surface from MediaItems.
	for name, bad := range map[string][]byte{
		"modality bytes":  protowire.AppendString(protowire.AppendTag(nil, mediaModality, protowire.BytesType), "x"),
		"uuid varint":     protowire.AppendVarint(protowire.AppendTag(nil, mediaUUID, protowire.VarintType), 1),
		"url varint":      protowire.AppendVarint(protowire.AppendTag(nil, mediaURL, protowire.VarintType), 1),
		"features varint": protowire.AppendVarint(protowire.AppendTag(nil, mediaFeatures, protowire.VarintType), 1),
		"offset bytes": protowire.AppendBytes(protowire.AppendTag(nil, mediaFeatures, protowire.BytesType),
			protowire.AppendString(protowire.AppendTag(nil, featOffset, protowire.BytesType), "x")),
		"mm_hash varint": protowire.AppendBytes(protowire.AppendTag(nil, mediaFeatures, protowire.BytesType),
			protowire.AppendVarint(protowire.AppendTag(nil, featMMHash, protowire.VarintType), 1)),
	} {
		badReq, err := ScanGenerateRequest(protowire.AppendBytes(protowire.AppendTag(nil, reqMedia, protowire.BytesType), bad))
		require.NoError(t, err, name)
		_, err = badReq.MediaItems()
		require.ErrorIs(t, err, ErrMalformed, name)
	}
}

func TestMediaItemContentDigestAndRewrittenBody(t *testing.T) {
	a := MediaItem{Body: mustMarshal(t, &vllmpb.MediaItem{Source: &vllmpb.MediaItem_RawBytes{RawBytes: []byte("img")}, Uuid: "a"})}
	b := MediaItem{Body: mustMarshal(t, &vllmpb.MediaItem{Source: &vllmpb.MediaItem_RawBytes{RawBytes: []byte("img")}, Uuid: "b"})}
	c := MediaItem{Body: mustMarshal(t, &vllmpb.MediaItem{Source: &vllmpb.MediaItem_RawBytes{RawBytes: []byte("other")}})}
	require.Equal(t, a.ContentDigest(), b.ContentDigest(), "uuid is excluded from the digest")
	require.NotEqual(t, a.ContentDigest(), c.ContentDigest())
	require.Len(t, a.ContentDigest(), 64)

	// No rewrite returns the same slice.
	require.Same(t, &a.Body[0], &a.RewrittenBody("", nil)[0])

	// uuid override replaces every uuid field with exactly one.
	dup := MediaItem{Body: concat(a.Body, b.Body)}
	var got vllmpb.MediaItem
	require.NoError(t, proto.Unmarshal(dup.RewrittenBody("z", nil), &got))
	require.Equal(t, "z", got.GetUuid())
	require.Equal(t, "img", string(got.GetRawBytes()))
	uuidFields := 0
	require.NoError(t, protoscan.ForEachField(dup.RewrittenBody("z", nil), func(num protowire.Number, _ protowire.Type, _, _ []byte) error {
		if num == mediaUUID {
			uuidFields++
		}
		return nil
	}))
	require.Equal(t, 1, uuidFields)

	// Features offset rewrite drops a zero offset and keeps the other fields.
	feat := MediaItem{Body: mustMarshal(t, &vllmpb.MediaItem{
		Source: &vllmpb.MediaItem_Features{Features: &vllmpb.PreprocessedMediaFeatures{
			Kwargs: []byte{1, 2}, Identifier: "id", Offset: 5, Length: 3, MmHash: proto.String("h"),
		}},
		Uuid: "u",
	})}
	for _, off := range []uint64{0, 1, 300} {
		o := off
		require.NoError(t, proto.Unmarshal(feat.RewrittenBody("", &o), &got))
		require.Equal(t, off, got.GetFeatures().GetOffset())
		require.Equal(t, uint64(3), got.GetFeatures().GetLength())
		require.Equal(t, "id", got.GetFeatures().GetIdentifier())
		require.Equal(t, "u", got.GetUuid())
		require.NoError(t, proto.Unmarshal(feat.RewrittenBody("v", &o), &got))
		require.Equal(t, "v", got.GetUuid())
		require.Equal(t, off, got.GetFeatures().GetOffset())
	}
}

func TestRewritePriority(t *testing.T) {
	withPriority, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
		Model: "old-model", LoraName: "old-lora", Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}, Priority: 99,
	})
	require.NoError(t, err)
	withoutPriority, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
		Model: "old-model", Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"},
	})
	require.NoError(t, err)

	for _, prio := range []int32{-10, -1, 0, 1, 42, math.MinInt32, math.MaxInt32} {
		for _, base := range [][]byte{withPriority, withoutPriority} {
			framed, err := RewritePriority(base, prio)
			require.NoError(t, err)
			got := decodeRequestFrame(t, framed)
			require.Equal(t, prio, got.GetPriority())
			require.Equal(t, "old-model", got.GetModel())
			require.Equal(t, "hi", got.GetText())
		}
	}

	// No client priority and priority 0: same frame, no allocation.
	same, err := RewritePriority(withoutPriority, 0)
	require.NoError(t, err)
	require.Same(t, &withoutPriority[0], &same[0])
	require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = RewritePriority(withoutPriority, 0) }))
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = RewritePriority(withPriority, -4) }))

	_, err = RewritePriority([]byte{0, 0, 0, 0, 3, 0x12, 0x05, 'a'}, -1)
	require.ErrorIs(t, err, ErrMalformed)
	_, err = RewritePriority([]byte{0, 0}, -1)
	require.Error(t, err)
}

// kvFetchHints returns a one-action KVHintsEnvelope naming source.
func kvFetchHints(t testing.TB, source string) *vllmpb.KVHintsEnvelope {
	t.Helper()
	return &vllmpb.KVHintsEnvelope{
		ProtocolVersion: "1",
		MessageId:       "msg-" + source,
		Actions: []*vllmpb.KVHintAction{{
			ActionId:   "fetch-0",
			ActionType: "kv.fetch",
			Payload:    mustStruct(t, map[string]any{"source": source}),
		}},
	}
}

func TestScanGenerateRequestKVHints(t *testing.T) {
	plain := &vllmpb.GenerateRequest{Model: "m", Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}}
	req, err := ScanGenerateRequest(mustMarshal(t, plain))
	require.NoError(t, err)
	require.False(t, req.HasKVHints())

	hinted := proto.Clone(plain).(*vllmpb.GenerateRequest)
	hinted.KvHints = kvFetchHints(t, "203.0.113.7:7000")
	req, err = ScanGenerateRequest(mustMarshal(t, hinted))
	require.NoError(t, err)
	require.True(t, req.HasKVHints())

	// The scan only records the field; frames built from it keep kv_hints.
	framed, err := req.Frame(LegOptions{})
	require.NoError(t, err)
	requireProtoEqual(t, hinted, decodeRequestFrame(t, framed))
}

func TestRewriteKVHints(t *testing.T) {
	base := &vllmpb.GenerateRequest{
		Model:    "m",
		Prompt:   &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3}}},
		Kv:       &vllmpb.KVCacheParameters{CacheSalt: "salt"},
		Priority: 2,
	}
	withoutHints, err := grpcframe.MarshalProto(base)
	require.NoError(t, err)
	client := proto.Clone(base).(*vllmpb.GenerateRequest)
	client.KvHints = kvFetchHints(t, "203.0.113.7:7000")
	withHints, err := grpcframe.MarshalProto(client)
	require.NoError(t, err)

	eppHints := kvFetchHints(t, "10.0.0.9:5600")
	envelope := mustMarshal(t, eppHints)
	want := proto.Clone(base).(*vllmpb.GenerateRequest)
	want.KvHints = eppHints

	for _, frame := range [][]byte{withHints, withoutHints} {
		framed, err := RewriteKVHints(frame, envelope)
		require.NoError(t, err)
		requireProtoEqual(t, want, decodeRequestFrame(t, framed))
	}

	// An empty envelope drops the client's hints.
	framed, err := RewriteKVHints(withHints, nil)
	require.NoError(t, err)
	requireProtoEqual(t, base, decodeRequestFrame(t, framed))

	// The engine merges repeated kv_hints fields, so every one is dropped.
	merged, err := grpcframe.FramePayload(concat(
		withHints[grpcframe.HeaderLen:],
		mustMarshal(t, &vllmpb.GenerateRequest{KvHints: kvFetchHints(t, "198.51.100.1:7000")}),
	))
	require.NoError(t, err)
	framed, err = RewriteKVHints(merged, envelope)
	require.NoError(t, err)
	requireProtoEqual(t, want, decodeRequestFrame(t, framed))

	// No hints and nothing to write: the same frame, no allocation.
	same, err := RewriteKVHints(withoutHints, nil)
	require.NoError(t, err)
	require.Same(t, &withoutHints[0], &same[0])
	require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = RewriteKVHints(withoutHints, nil) }))
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = RewriteKVHints(withHints, envelope) }))

	_, err = RewriteKVHints([]byte{0, 0, 0, 0, 3, 0x12, 0x05, 'a'}, envelope)
	require.ErrorIs(t, err, ErrMalformed)
	_, err = RewriteKVHints([]byte{0, 0}, nil)
	require.Error(t, err)
}

func TestRewriteModelAndLoraName(t *testing.T) {
	base, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
		Model: "old-model", LoraName: "old-lora", Prompt: &vllmpb.GenerateRequest_Text{Text: "hi"}, Priority: 99,
	})
	require.NoError(t, err)

	framed, err := RewriteModel(base, "new-model")
	require.NoError(t, err)
	got := decodeRequestFrame(t, framed)
	require.Equal(t, "new-model", got.GetModel())
	require.Equal(t, "old-lora", got.GetLoraName())
	require.Equal(t, int32(99), got.GetPriority())
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = RewriteModel(base, "new-model") }))

	framed, err = RewriteLoraName(base, "new-lora")
	require.NoError(t, err)
	got = decodeRequestFrame(t, framed)
	require.Equal(t, "old-model", got.GetModel())
	require.Equal(t, "new-lora", got.GetLoraName())

	// Empty value drops the field. Duplicate fields collapse to one.
	dup := concat(base, base[grpcframe.HeaderLen:])
	dup, err = grpcframe.FramePayload(dup[grpcframe.HeaderLen:])
	require.NoError(t, err)
	framed, err = RewriteLoraName(dup, "")
	require.NoError(t, err)
	got = decodeRequestFrame(t, framed)
	require.Empty(t, got.GetLoraName())
	require.Equal(t, "old-model", got.GetModel())

	_, err = RewriteModel([]byte{0, 0, 0, 0, 3, 0x12, 0x05, 'a'}, "m")
	require.ErrorIs(t, err, ErrMalformed)
	_, err = RewriteModel(nil, "m")
	require.Error(t, err)
}

func TestScanGenerateResponse(t *testing.T) {
	kvStruct := mustStruct(t, map[string]any{
		"do_remote_prefill": true,
		"remote_engine_id":  "prefill-0",
		"remote_port":       float64(5557),
	})
	ecStruct := mustStruct(t, map[string]any{"mm_hash": "abc123"})

	tests := []struct {
		name string
		resp *vllmpb.GenerateResponse
		want ResponseInfo
	}{
		{
			name: "first stream chunk with prompt info and token delta",
			resp: &vllmpb.GenerateResponse{
				PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 512, TokenIds: []uint32{1, 2, 3}},
				Outputs:    &vllmpb.SequenceOutput{Index: 0, Text: "Hey", NumTokens: 1, TokenIds: []uint32{9906}},
			},
			want: ResponseInfo{HasPromptInfo: true, NumPromptTokens: 512, HasOutputs: true},
		},
		{
			name: "intermediate token chunk",
			resp: &vllmpb.GenerateResponse{Outputs: &vllmpb.SequenceOutput{Index: 1, Text: "lo", NumTokens: 2}},
			want: ResponseInfo{HasOutputs: true},
		},
		{
			name: "terminal prefill response with cached=0 and transfer structs",
			resp: &vllmpb.GenerateResponse{
				PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 64},
				Outputs: &vllmpb.SequenceOutput{
					NumTokens: 1,
					FinishInfo: &vllmpb.FinishInfo{
						NumOutputTokens:  1,
						FinishReason:     vllmpb.FinishInfo_LENGTH,
						NumCachedTokens:  proto.Uint32(0),
						KvTransferParams: kvStruct,
						EcTransferParams: ecStruct,
					},
				},
			},
			want: ResponseInfo{
				HasPromptInfo: true, NumPromptTokens: 64, HasOutputs: true,
				HasFinishInfo: true, NumOutputTokens: 1,
				HasCachedTokens: true, NumCachedTokens: 0, hasKVTransferParams: true, hasECTransferParams: true,
			},
		},
		{
			name: "terminal without cached tokens",
			resp: &vllmpb.GenerateResponse{Outputs: &vllmpb.SequenceOutput{
				FinishInfo: &vllmpb.FinishInfo{NumOutputTokens: 7, FinishReason: vllmpb.FinishInfo_STOP},
			}},
			want: ResponseInfo{HasOutputs: true, HasFinishInfo: true, NumOutputTokens: 7},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			payload := mustMarshal(t, tc.resp)
			got, err := ScanGenerateResponse(payload)
			require.NoError(t, err)
			got.kvTransferRaw, got.ecTransferRaw = nil, nil
			got.outputsCount, got.finishInfoCount = 0, 0
			require.Equal(t, tc.want, got)
			require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = ScanGenerateResponse(payload) }))

			got, err = ScanGenerateResponse(payload)
			require.NoError(t, err)
			kv, err := got.KVTransferParams()
			require.NoError(t, err)
			ec, err := got.ECTransferParams()
			require.NoError(t, err)
			if got.hasKVTransferParams {
				require.Equal(t, kvStruct.AsMap(), kv)
				require.Equal(t, ecStruct.AsMap(), ec)
			} else {
				require.Nil(t, kv)
				require.Nil(t, ec)
			}
		})
	}

	// Repeated finish_info.kv_transfer_params bodies merge.
	part1 := mustMarshal(t, &vllmpb.GenerateResponse{Outputs: &vllmpb.SequenceOutput{FinishInfo: &vllmpb.FinishInfo{
		KvTransferParams: mustStruct(t, map[string]any{"a": 1, "k": "one"}),
		EcTransferParams: mustStruct(t, map[string]any{"e": 1}),
	}}})
	part2 := mustMarshal(t, &vllmpb.GenerateResponse{Outputs: &vllmpb.SequenceOutput{FinishInfo: &vllmpb.FinishInfo{
		KvTransferParams: mustStruct(t, map[string]any{"k": "two"}),
		EcTransferParams: mustStruct(t, map[string]any{"f": 2}),
	}}})
	merged := concat(part1, part2)
	var wantResp vllmpb.GenerateResponse
	require.NoError(t, proto.Unmarshal(merged, &wantResp))
	info, err := ScanGenerateResponse(merged)
	require.NoError(t, err)
	kv, err := info.KVTransferParams()
	require.NoError(t, err)
	require.Equal(t, wantResp.GetOutputs().GetFinishInfo().GetKvTransferParams().AsMap(), kv)
	ec, err := info.ECTransferParams()
	require.NoError(t, err)
	require.Equal(t, wantResp.GetOutputs().GetFinishInfo().GetEcTransferParams().AsMap(), ec)

	// Malformed inputs.
	payload := mustMarshal(t, tests[2].resp)
	_, err = ScanGenerateResponse(payload[:len(payload)-1])
	require.ErrorIs(t, err, ErrMalformed)
	wrapOutputs := func(inner []byte) []byte {
		return protowire.AppendBytes(protowire.AppendTag(nil, respOutputs, protowire.BytesType), inner)
	}
	wrapFinish := func(inner []byte) []byte {
		return wrapOutputs(protowire.AppendBytes(protowire.AppendTag(nil, seqFinishInfo, protowire.BytesType), inner))
	}
	for name, bad := range map[string][]byte{
		"prompt_info varint": protowire.AppendVarint(protowire.AppendTag(nil, respPromptInfo, protowire.VarintType), 1),
		"outputs varint":     protowire.AppendVarint(protowire.AppendTag(nil, respOutputs, protowire.VarintType), 1),
		"num_prompt bytes":   protowire.AppendBytes(protowire.AppendTag(nil, respPromptInfo, protowire.BytesType), protowire.AppendString(protowire.AppendTag(nil, promptNumTokens, protowire.BytesType), "x")),
		"finish varint":      wrapOutputs(protowire.AppendVarint(protowire.AppendTag(nil, seqFinishInfo, protowire.VarintType), 1)),
		"num_output bytes":   wrapFinish(protowire.AppendString(protowire.AppendTag(nil, finishNumOutputTokens, protowire.BytesType), "x")),
		"cached bytes":       wrapFinish(protowire.AppendString(protowire.AppendTag(nil, finishNumCachedTokens, protowire.BytesType), "x")),
		"kv varint":          wrapFinish(protowire.AppendVarint(protowire.AppendTag(nil, finishKVTransferParams, protowire.VarintType), 1)),
		"ec varint":          wrapFinish(protowire.AppendVarint(protowire.AppendTag(nil, finishECTransferParams, protowire.VarintType), 1)),
	} {
		_, err := ScanGenerateResponse(bad)
		require.ErrorIs(t, err, ErrMalformed, name)
	}
	// A malformed Struct body is reported by the decoder, not the scanner.
	info, err = ScanGenerateResponse(wrapFinish(protowire.AppendBytes(protowire.AppendTag(nil, finishKVTransferParams, protowire.BytesType), []byte{0x80})))
	require.NoError(t, err)
	_, err = info.KVTransferParams()
	require.ErrorIs(t, err, ErrMalformed)
}

func TestRewriteTerminalCachedTokens(t *testing.T) {
	// 1. Intermediate token frame: returned untouched (same slice, 0 allocs),
	// even with non-canonical varints.
	interPayload := mustMarshal(t, &vllmpb.GenerateResponse{
		Outputs: &vllmpb.SequenceOutput{Text: "token", NumTokens: 1, TokenIds: []uint32{42}},
	})
	interFrame, err := grpcframe.FramePayload(interPayload)
	require.NoError(t, err)
	gotInter, err := RewriteTerminalCachedTokens(interFrame, 96)
	require.NoError(t, err)
	require.Same(t, &interFrame[0], &gotInter[0])
	require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = RewriteTerminalCachedTokens(interFrame, 96) }))

	// 2. Terminal frame already matching cachedTokens: same slice, 0 allocs.
	matchingFrame, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
		Outputs: &vllmpb.SequenceOutput{FinishInfo: &vllmpb.FinishInfo{
			NumOutputTokens: 10,
			FinishReason:    vllmpb.FinishInfo_STOP,
			StopReason:      &vllmpb.FinishInfo_EosTokenId{EosTokenId: 128009},
			NumCachedTokens: proto.Uint32(48),
		}},
	})
	require.NoError(t, err)
	gotMatching, err := RewriteTerminalCachedTokens(matchingFrame, 48)
	require.NoError(t, err)
	require.Same(t, &matchingFrame[0], &gotMatching[0])
	require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = RewriteTerminalCachedTokens(matchingFrame, 48) }))

	// 3. Terminal frames that need substitution: from nil, from a value, to 0, to a multi-byte varint.
	richTerminal := &vllmpb.GenerateResponse{
		PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 256, TokenIds: []uint32{1, 2, 3}},
		Outputs: &vllmpb.SequenceOutput{
			Index: 0, Text: "world!", NumTokens: 2, TokenIds: []uint32{1917, 0},
			Logprobs: []float32{-0.1, -0.2}, Ranks: []uint32{1, 1},
			FinishInfo: &vllmpb.FinishInfo{
				NumOutputTokens:  32,
				FinishReason:     vllmpb.FinishInfo_STOP,
				StopReason:       &vllmpb.FinishInfo_StopString{StopString: "!"},
				NumCachedTokens:  proto.Uint32(4),
				KvTransferParams: mustStruct(t, map[string]any{"remote_engine_id": "e1"}),
			},
			SamplingMask: []*vllmpb.TokenIds{{Ids: []uint32{7, 8}}},
		},
	}
	for _, cached := range []uint32{0, 64, 100000} {
		in := proto.Clone(richTerminal).(*vllmpb.GenerateResponse)
		if cached == 64 {
			in.Outputs.FinishInfo.NumCachedTokens = nil
		}
		frame, err := grpcframe.MarshalProto(in)
		require.NoError(t, err)
		rewritten, err := RewriteTerminalCachedTokens(frame, cached)
		require.NoError(t, err)
		require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = RewriteTerminalCachedTokens(frame, cached) }))
		want := proto.Clone(in).(*vllmpb.GenerateResponse)
		want.Outputs.FinishInfo.NumCachedTokens = proto.Uint32(cached)
		requireProtoEqual(t, want, decodeResponseFrame(t, rewritten))
	}

	// 4. Fallback path: concatenated duplicate outputs messages on the wire.
	part1 := mustMarshal(t, &vllmpb.GenerateResponse{Outputs: &vllmpb.SequenceOutput{Text: "a", NumTokens: 1}})
	part2 := mustMarshal(t, &vllmpb.GenerateResponse{Outputs: &vllmpb.SequenceOutput{FinishInfo: &vllmpb.FinishInfo{
		NumOutputTokens: 1, FinishReason: vllmpb.FinishInfo_LENGTH, NumCachedTokens: proto.Uint32(2),
	}}})
	dupFrame, err := grpcframe.FramePayload(concat(part1, part2))
	require.NoError(t, err)
	dupRewritten, err := RewriteTerminalCachedTokens(dupFrame, 80)
	require.NoError(t, err)
	gotDup := decodeResponseFrame(t, dupRewritten)
	require.Equal(t, uint32(80), gotDup.GetOutputs().GetFinishInfo().GetNumCachedTokens())
	require.Equal(t, "a", gotDup.GetOutputs().GetText())

	// 5. Wire inputs with a repeated num_cached_tokens in one finish_info, and
	// with a repeated finish_info in one outputs, rewrite like a full decode.
	finishBody := mustMarshal(t, richTerminal.GetOutputs().GetFinishInfo())
	dupCached := protowire.AppendVarint(protowire.AppendTag(append([]byte{}, finishBody...), finishNumCachedTokens, protowire.VarintType), 9)
	seqDupCached := protowire.AppendBytes(protowire.AppendTag(nil, seqFinishInfo, protowire.BytesType), dupCached)
	seqDupFinish := protowire.AppendBytes(protowire.AppendTag(nil, seqFinishInfo, protowire.BytesType), finishBody)
	seqDupFinish = protowire.AppendBytes(protowire.AppendTag(seqDupFinish, seqFinishInfo, protowire.BytesType), finishBody)
	for name, seqBody := range map[string][]byte{"repeated num_cached_tokens": seqDupCached, "repeated finish_info": seqDupFinish} {
		payload := protowire.AppendBytes(protowire.AppendTag(nil, respOutputs, protowire.BytesType), seqBody)
		frame, err := grpcframe.FramePayload(payload)
		require.NoError(t, err)
		rewritten, err := RewriteTerminalCachedTokens(frame, 80)
		require.NoError(t, err, name)
		var want vllmpb.GenerateResponse
		require.NoError(t, proto.Unmarshal(payload, &want))
		want.Outputs.FinishInfo.NumCachedTokens = proto.Uint32(80)
		requireProtoEqual(t, &want, decodeResponseFrame(t, rewritten))
	}

	// 6. Errors: bad frame header, malformed payload.
	bad := []byte{0, 0, 0, 0, 3, 0x12, 0x05, 'a'}
	got, err := RewriteTerminalCachedTokens(bad, 1)
	require.ErrorIs(t, err, ErrMalformed)
	require.Nil(t, got)
	got, err = RewriteTerminalCachedTokens([]byte{0, 0}, 1)
	require.Error(t, err)
	require.Nil(t, got)
}

// Benchmarks.

func benchRequest(b *testing.B, tokens int) []byte {
	b.Helper()
	ids := make([]uint32, tokens)
	for i := range ids {
		ids[i] = uint32(i % 32000) //#nosec G115 -- bounded
	}
	req := cleanSample(b)
	req.Prompt = &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: ids}}
	return mustMarshal(b, req)
}

func BenchmarkScanGenerateRequest(b *testing.B) {
	payload := benchRequest(b, 8192)
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ScanGenerateRequest(payload); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFrameSegments(b *testing.B) {
	payload := benchRequest(b, 8192)
	req, err := ScanGenerateRequest(payload)
	if err != nil {
		b.Fatal(err)
	}
	nixlDecodeKV := map[string]any{
		"do_remote_decode":  false,
		"do_remote_prefill": true,
		"remote_engine_id":  "eng-1",
		"remote_block_ids":  []int{10, 11, 12},
		"remote_host":       "10.0.0.8",
		"remote_port":       json.Number("5557"),
	}
	cases := []struct {
		name string
		opts LegOptions
	}{
		{"decode/MediaNone/NoKV", LegOptions{Media: MediaNone}},
		{"encode/MediaAll/EC", LegOptions{CapSingleToken: true, ECTransferParams: map[string]any{"ec": map[string]any{"keys": []string{"a", "b"}}}}},
		{"prefill/MediaMetadataOnly/NIXL", LegOptions{CapSingleToken: true, Media: MediaMetadataOnly, KVTransferParams: nixlPrefillKV()}},
		{"decode/MediaNone/NIXL", LegOptions{Media: MediaNone, KVTransferParams: nixlDecodeKV}},
		{"passthrough/MediaAll/RequestID", LegOptions{RequestID: "sidecar-uuid"}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := req.FrameSegments(tc.opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkEncoderSubFrame(b *testing.B) {
	req, err := ScanGenerateRequest(benchRequest(b, 1024))
	if err != nil {
		b.Fatal(err)
	}
	items, err := req.MediaItems()
	if err != nil {
		b.Fatal(err)
	}
	opts := EncoderSubFrameOptions{
		RequestID:        "r-mm-0",
		TokenIDs:         []uint32{1, 2, 3, 4},
		Item:             items[2],
		ECTransferParams: map[string]any{"consumer": "zmq://x"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := req.EncoderSubFrame(opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkScanGenerateResponse(b *testing.B) {
	payload := mustMarshal(b, &vllmpb.GenerateResponse{
		Outputs: &vllmpb.SequenceOutput{Text: "tok", NumTokens: 1, TokenIds: []uint32{42}, Logprobs: []float32{-0.5}},
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ScanGenerateResponse(payload); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRewriteTerminalCachedTokens(b *testing.B) {
	inter, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
		Outputs: &vllmpb.SequenceOutput{Text: "tok", NumTokens: 1, TokenIds: []uint32{42}},
	})
	if err != nil {
		b.Fatal(err)
	}
	terminal, err := grpcframe.MarshalProto(&vllmpb.GenerateResponse{
		Outputs: &vllmpb.SequenceOutput{Text: "!", NumTokens: 1, FinishInfo: &vllmpb.FinishInfo{
			NumOutputTokens: 32, FinishReason: vllmpb.FinishInfo_STOP, NumCachedTokens: proto.Uint32(4),
		}},
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Run("intermediate", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := RewriteTerminalCachedTokens(inter, 64); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("terminal", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := RewriteTerminalCachedTokens(terminal, 64); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkRewritePriority(b *testing.B) {
	frame, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
		Model: "m", Prompt: &vllmpb.GenerateRequest_Text{Text: "hello world"}, Priority: 3,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := RewritePriority(frame, -2); err != nil {
			b.Fatal(err)
		}
	}
}
