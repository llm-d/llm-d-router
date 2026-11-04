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

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
)

func mustMarshal(t *testing.T, m proto.Message) []byte {
	t.Helper()
	b, err := proto.Marshal(m)
	require.NoError(t, err)
	return b
}

func TestScanTokenIDsMatchesProto(t *testing.T) {
	ids := make([]int64, 1000)
	for i := range ids {
		ids[i] = int64(i * 131)
	}
	ids[999] = math.MaxUint32
	packed := mustMarshal(t, &sglangapiv1.TokenIds{Ids: ids})

	got, err := scanTokenIDs(packed)
	require.NoError(t, err)
	require.Len(t, got, len(ids))
	for i, id := range ids {
		require.Equal(t, uint32(id), got[i])
	}
	require.Equal(t, len(got), cap(got), "one exact-size allocation")
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = scanTokenIDs(packed) }))

	// Unpacked encoding (one varint field per element) is valid proto3 input.
	var unpacked []byte
	for _, id := range ids[:5] {
		unpacked = protowire.AppendTag(unpacked, tokenIDsIDs, protowire.VarintType)
		unpacked = protowire.AppendVarint(unpacked, uint64(id))
	}
	got, err = scanTokenIDs(unpacked)
	require.NoError(t, err)
	require.Equal(t, []uint32{0, 131, 262, 393, 524}, got)

	// Out-of-range values report their position, as the typed path did.
	_, err = scanTokenIDs(mustMarshal(t, &sglangapiv1.TokenIds{Ids: []int64{1, -1}}))
	require.ErrorContains(t, err, "input_ids[1] out of uint32 range: -1")
	_, err = scanTokenIDs(mustMarshal(t, &sglangapiv1.TokenIds{Ids: []int64{math.MaxUint32 + 1}}))
	require.ErrorContains(t, err, "input_ids[0] out of uint32 range")

	_, err = scanTokenIDs([]byte{0x0a, 0x05, 0x80})
	require.ErrorIs(t, err, errMalformed)
}

func TestScanGenerateRequestAliasesPayload(t *testing.T) {
	req := &sglangapiv1.GenerateRequest{
		Text:      &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hello"}},
		CacheSalt: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "salt"}},
		ImageData: &sglangapiv1.MediaInput{Value: &sglangapiv1.MediaInput_Item{Item: &sglangapiv1.MediaItem{
			Value: &sglangapiv1.MediaItem_Source{Source: string(bytes.Repeat([]byte("x"), 1<<16))},
		}}},
		Stream: proto.Bool(true),
	}
	payload := mustMarshal(t, req)
	scan, err := scanGenerateRequest(payload)
	require.NoError(t, err)
	require.True(t, scan.text.present)
	require.Equal(t, "hello", string(scan.text.one))
	require.Equal(t, "salt", string(scan.cacheSalt.one))
	require.True(t, scan.hasImage)
	require.False(t, scan.hasVideo)
	require.True(t, scan.stream)
	require.False(t, scan.inputIDs.present)

	// The media bytes are not copied: the scan allocates nothing.
	require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = scanGenerateRequest(payload) }))

	// An empty text.one is present but empty, as the typed GetValue() != nil was.
	payload = mustMarshal(t, &sglangapiv1.GenerateRequest{Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: ""}}})
	scan, err = scanGenerateRequest(payload)
	require.NoError(t, err)
	require.True(t, scan.text.present)
	require.Empty(t, scan.text.one)

	_, err = scanGenerateRequest([]byte{0x12, 0x05, 'a'})
	require.ErrorIs(t, err, errMalformed)
}

func TestScanStreamItemMatchesProto(t *testing.T) {
	stop := &sglangapiv1.FinishReason{Kind: &sglangapiv1.FinishReason_Stop{Stop: &sglangapiv1.FinishStop{}}}
	items := []*sglangapiv1.GenerateStreamItem{
		{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
			Text: "hi", OutputIds: &sglangapiv1.TokenIds{Ids: []int64{1, 2, 3}},
			MetaInfo: &sglangapiv1.GenerateMetaInfo{PromptTokens: 12, CompletionTokens: 3, CachedTokens: 8},
		}}},
		{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
			MetaInfo: &sglangapiv1.GenerateMetaInfo{PromptTokens: 12, CompletionTokens: 4, CachedTokens: 8, FinishReason: stop},
		}}},
		{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{Text: "no meta"}}},
		{Item: &sglangapiv1.GenerateStreamItem_Error{Error: &sglangapiv1.GenerateStreamError{Error: &sglangapiv1.ErrorBody{Message: "boom"}}}},
	}
	for _, item := range items {
		payload := mustMarshal(t, item)
		got, err := scanStreamItem(payload)
		require.NoError(t, err)

		var decoded sglangapiv1.GenerateStreamItem
		require.NoError(t, proto.Unmarshal(payload, &decoded))
		frame := decoded.GetFrame()
		require.Equal(t, frame != nil, got.hasFrame)
		require.Equal(t, frame.GetMetaInfo().GetFinishReason() != nil, got.finished)
		require.Equal(t, frame.GetMetaInfo().GetPromptTokens(), got.promptTokens)
		require.Equal(t, frame.GetMetaInfo().GetCompletionTokens(), got.completionTokens)
		require.Equal(t, frame.GetMetaInfo().GetCachedTokens(), got.cachedTokens)
		require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = scanStreamItem(payload) }))
	}

	// frame then error in one payload: the oneof switch discards the frame.
	both := append(mustMarshal(t, items[1]), mustMarshal(t, items[3])...)
	got, err := scanStreamItem(both)
	require.NoError(t, err)
	require.Equal(t, streamItemScan{}, got)

	terminal := mustMarshal(t, items[1])
	_, err = scanStreamItem(terminal[:len(terminal)-1])
	require.ErrorIs(t, err, errMalformed)
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

func TestFramePriority(t *testing.T) {
	clientPrio := int64(99)
	req := &sglangapiv1.GenerateRequest{
		Text:     &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hello"}},
		Priority: &clientPrio,
		Stream:   proto.Bool(true),
	}
	payload := mustMarshal(t, req)
	// Two client priority fields: both must go.
	payload = protowire.AppendVarint(protowire.AppendTag(payload, reqPriority, protowire.VarintType), 7)

	for _, prio := range []int64{5, 0, -3, math.MaxInt64} {
		framed, err := framePriority(payload, prio)
		require.NoError(t, err)
		body, err := grpcframe.ParsePayload(framed)
		require.NoError(t, err)
		require.Len(t, framed, grpcframe.HeaderLen+len(body))
		var out sglangapiv1.GenerateRequest
		require.NoError(t, proto.Unmarshal(body, &out))
		want := proto.Clone(req).(*sglangapiv1.GenerateRequest)
		want.Priority = proto.Int64(prio)
		require.True(t, proto.Equal(want, &out), "priority %d", prio)
		require.Equal(t, 1, bytes.Count(body, protowire.AppendTag(nil, reqPriority, protowire.VarintType)))
		// One allocation, sized before the client fields are dropped.
		require.Equal(t, grpcframe.HeaderLen+len(payload)+protowire.SizeTag(reqPriority)+protowire.SizeVarint(uint64(prio)), cap(framed))
	}
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = framePriority(payload, 5) }))

	_, err := framePriority([]byte{0xff}, 1)
	require.ErrorIs(t, err, errMalformed)
}
