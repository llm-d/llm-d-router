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

package sglangwire

import (
	"bytes"
	"math"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
)

func optStr(v string) *sglangapiv1.OptionalStringOrList {
	return &sglangapiv1.OptionalStringOrList{Value: &sglangapiv1.OptionalStringOrList_One{One: &sglangapiv1.OptionalString{Value: proto.String(v)}}}
}

func optInt(v int64) *sglangapiv1.OptionalInt64OrList {
	return &sglangapiv1.OptionalInt64OrList{Value: &sglangapiv1.OptionalInt64OrList_One{One: &sglangapiv1.OptionalInt64{Value: proto.Int64(v)}}}
}

func testRequest() *sglangapiv1.GenerateRequest {
	return &sglangapiv1.GenerateRequest{
		Rid:      &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "rid-1"}},
		InputIds: &sglangapiv1.TokenIdsOrList{Value: &sglangapiv1.TokenIdsOrList_One{One: &sglangapiv1.TokenIds{Ids: []int64{1, 2, 300000}}}},
		Stream:   proto.Bool(true),
		// Client-supplied bootstrap_host, bootstrap_port, and bootstrap_room
		// must never reach a worker: the coordinator sets them. Other
		// bootstrap_* fields, such as bootstrap_pair_key, pass through.
		BootstrapHost:    optStr("spoofed"),
		BootstrapPort:    optInt(1),
		BootstrapRoom:    optInt(2),
		BootstrapPairKey: optStr("pair"),
		Priority:         proto.Int64(5),
		CustomLabels:     map[string]string{"k": "v"},
	}
}

// legBody assembles one leg request as the gateway client sends it.
func legBody(t *testing.T, payload []byte, b *Bootstrap) []byte {
	t.Helper()
	head, tail, err := LegEnvelope(len(payload), b)
	require.NoError(t, err)
	return bytes.Join([][]byte{head, payload, tail}, nil)
}

func decodeFrame(t *testing.T, framed []byte) *sglangapiv1.GenerateRequest {
	t.Helper()
	payload, err := grpcframe.ParsePayload(framed)
	require.NoError(t, err)
	require.Len(t, framed, grpcframe.HeaderLen+len(payload), "frame length prefix")
	var got sglangapiv1.GenerateRequest
	require.NoError(t, proto.Unmarshal(payload, &got))
	return &got
}

func mustMarshal(t *testing.T, m proto.Message) []byte {
	t.Helper()
	b, err := proto.Marshal(m)
	require.NoError(t, err)
	return b
}

func TestScanGenerateRequestStripsClientBootstrapFields(t *testing.T) {
	payload := mustMarshal(t, testRequest())
	req, err := ScanGenerateRequest(payload)
	require.NoError(t, err)
	require.True(t, req.Stream)

	want := testRequest()
	want.BootstrapHost, want.BootstrapPort, want.BootstrapRoom = nil, nil, nil
	var got sglangapiv1.GenerateRequest
	require.NoError(t, proto.Unmarshal(req.Payload, &got))
	if diff := cmp.Diff(want, &got, protocmp.Transform()); diff != "" {
		t.Fatalf("mismatch (-want +got):\n%s", diff)
	}
	require.LessOrEqual(t, cap(req.Payload), len(payload), "stripped copy is one allocation of at most the input size")
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = ScanGenerateRequest(payload) }))
	require.Equal(t, payload, mustMarshal(t, testRequest()), "input not modified")

	// Without client bootstrap fields (the normal case), Payload aliases the
	// input and the scan does not allocate.
	clean := testRequest()
	clean.BootstrapHost, clean.BootstrapPort, clean.BootstrapRoom = nil, nil, nil
	cleanPayload := mustMarshal(t, clean)
	req, err = ScanGenerateRequest(cleanPayload)
	require.NoError(t, err)
	require.Same(t, &cleanPayload[0], &req.Payload[0])
	require.Len(t, req.Payload, len(cleanPayload))
	require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = ScanGenerateRequest(cleanPayload) }))

	for _, stream := range []*bool{nil, proto.Bool(false), proto.Bool(true)} {
		req, err := ScanGenerateRequest(mustMarshal(t, &sglangapiv1.GenerateRequest{Stream: stream, Priority: proto.Int64(1)}))
		require.NoError(t, err)
		require.Equal(t, stream != nil && *stream, req.Stream)
	}

	_, err = ScanGenerateRequest([]byte{0xff, 0xff, 0xff})
	require.ErrorIs(t, err, ErrMalformed)
	// A length prefix past the end of the payload.
	_, err = ScanGenerateRequest([]byte{0x12, 0x05, 'a'})
	require.ErrorIs(t, err, ErrMalformed)
}

func TestLegEnvelopeMatchesProto(t *testing.T) {
	req, err := ScanGenerateRequest(mustMarshal(t, testRequest()))
	require.NoError(t, err)

	tests := []struct {
		name string
		b    *Bootstrap
		want func(*sglangapiv1.GenerateRequest)
	}{
		{
			name: "no bootstrap (decode-only)",
			want: func(r *sglangapiv1.GenerateRequest) {
				r.BootstrapHost, r.BootstrapPort, r.BootstrapRoom = nil, nil, nil
			},
		},
		{
			name: "port and room (prefill)",
			b:    &Bootstrap{Port: 8998, Room: -123456789012345},
			want: func(r *sglangapiv1.GenerateRequest) {
				r.BootstrapHost, r.BootstrapPort, r.BootstrapRoom = nil, optInt(8998), optInt(-123456789012345)
			},
		},
		{
			name: "host, port, and room (decode)",
			b:    &Bootstrap{Host: "10.2.3.4", Port: 8998, Room: 0},
			want: func(r *sglangapiv1.GenerateRequest) {
				r.BootstrapHost, r.BootstrapPort, r.BootstrapRoom = optStr("10.2.3.4"), optInt(8998), optInt(0)
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			want := testRequest()
			tc.want(want)
			if diff := cmp.Diff(want, decodeFrame(t, legBody(t, req.Payload, tc.b)), protocmp.Transform()); diff != "" {
				t.Fatalf("mismatch (-want +got):\n%s", diff)
			}
		})
	}

	// head and tail share one exact-size allocation, and head has no spare
	// capacity so an append cannot overwrite tail.
	head, tail, err := LegEnvelope(10, &Bootstrap{Host: "h", Port: 1, Room: 2})
	require.NoError(t, err)
	require.Len(t, head, grpcframe.HeaderLen)
	require.Equal(t, grpcframe.HeaderLen, cap(head))
	require.Equal(t, len(tail), cap(tail))
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() {
		_, _, _ = LegEnvelope(10, &Bootstrap{Host: "h", Port: 1, Room: 2})
	}))

	_, _, err = LegEnvelope(-1, nil)
	require.Error(t, err)
}

func TestScanTokenIDsMatchesProto(t *testing.T) {
	ids := make([]int64, 1000)
	for i := range ids {
		ids[i] = int64(i * 131)
	}
	ids[999] = math.MaxUint32
	packed := mustMarshal(t, &sglangapiv1.TokenIds{Ids: ids})

	got, err := ScanTokenIDs(packed)
	require.NoError(t, err)
	require.Len(t, got, len(ids))
	for i, id := range ids {
		require.Equal(t, uint32(id), got[i])
	}
	require.Equal(t, len(got), cap(got), "one exact-size allocation")
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = ScanTokenIDs(packed) }))

	// Unpacked encoding (one varint field per element) is valid proto3 input.
	var unpacked []byte
	for _, id := range ids[:5] {
		unpacked = protowire.AppendTag(unpacked, tokenIDsIDs, protowire.VarintType)
		unpacked = protowire.AppendVarint(unpacked, uint64(id))
	}
	got, err = ScanTokenIDs(unpacked)
	require.NoError(t, err)
	require.Equal(t, []uint32{0, 131, 262, 393, 524}, got)

	_, err = ScanTokenIDs(mustMarshal(t, &sglangapiv1.TokenIds{Ids: []int64{1, -1}}))
	require.ErrorContains(t, err, "input_ids: value[1] out of uint32 range: -1")
	require.ErrorIs(t, err, ErrMalformed)
	_, err = ScanTokenIDs(mustMarshal(t, &sglangapiv1.TokenIds{Ids: []int64{math.MaxUint32 + 1}}))
	require.ErrorContains(t, err, "input_ids: value[0] out of uint32 range")

	_, err = ScanTokenIDs([]byte{0x0a, 0x05, 0x80})
	require.ErrorIs(t, err, ErrMalformed)
}

func TestScanRequestFieldsAliasesPayload(t *testing.T) {
	req := &sglangapiv1.GenerateRequest{
		Text:      &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hello"}},
		CacheSalt: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "salt"}},
		ImageData: &sglangapiv1.MediaInput{Value: &sglangapiv1.MediaInput_Item{Item: &sglangapiv1.MediaItem{
			Value: &sglangapiv1.MediaItem_Source{Source: string(bytes.Repeat([]byte("x"), 1<<16))},
		}}},
		Stream: proto.Bool(true),
	}
	payload := mustMarshal(t, req)
	scan, err := ScanRequestFields(payload)
	require.NoError(t, err)
	require.True(t, scan.Text.Present)
	require.Equal(t, "hello", string(scan.Text.One))
	require.Equal(t, "salt", string(scan.CacheSalt.One))
	require.Equal(t, MediaScan{Present: true}, scan.ImageData)
	require.Equal(t, MediaScan{}, scan.VideoData)
	require.True(t, scan.Stream)
	require.False(t, scan.InputIDs.Present)

	// The media bytes are not copied: the scan allocates nothing.
	require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = ScanRequestFields(payload) }))

	// An empty text.one is present but empty, matching GetValue() != nil.
	payload = mustMarshal(t, &sglangapiv1.GenerateRequest{Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: ""}}})
	scan, err = ScanRequestFields(payload)
	require.NoError(t, err)
	require.True(t, scan.Text.Present)
	require.Empty(t, scan.Text.One)

	_, err = ScanRequestFields([]byte{0x12, 0x05, 'a'})
	require.ErrorIs(t, err, ErrMalformed)
}

func protoMediaScan(m *sglangapiv1.MediaInput) MediaScan {
	if m == nil {
		return MediaScan{}
	}
	switch v := m.GetValue().(type) {
	case *sglangapiv1.MediaInput_Item:
		return MediaScan{Present: v.Item != nil}
	case *sglangapiv1.MediaInput_Many:
		return MediaScan{Present: len(v.Many.GetItems()) > 0}
	case *sglangapiv1.MediaInput_Nested:
		return MediaScan{Nested: v.Nested != nil}
	default:
		return MediaScan{}
	}
}

func TestScanRequestFieldsMediaInput(t *testing.T) {
	itemMedia := &sglangapiv1.MediaInput{
		Value: &sglangapiv1.MediaInput_Item{
			Item: &sglangapiv1.MediaItem{Value: &sglangapiv1.MediaItem_Source{Source: "https://example.com/a.png"}},
		},
	}
	emptyManyMedia := &sglangapiv1.MediaInput{
		Value: &sglangapiv1.MediaInput_Many{Many: &sglangapiv1.MediaItems{}},
	}
	nonEmptyManyMedia := &sglangapiv1.MediaInput{
		Value: &sglangapiv1.MediaInput_Many{
			Many: &sglangapiv1.MediaItems{Items: []*sglangapiv1.OptionalMediaItem{{}}},
		},
	}
	nestedMedia := &sglangapiv1.MediaInput{
		Value: &sglangapiv1.MediaInput_Nested{Nested: &sglangapiv1.MediaItemsList{}},
	}

	reqWithImage := func(img *sglangapiv1.MediaInput) []byte {
		return mustMarshal(t, &sglangapiv1.GenerateRequest{ImageData: img})
	}

	tests := []struct {
		name    string
		payload []byte
		want    MediaScan
	}{
		{name: "unset", payload: reqWithImage(nil), want: MediaScan{}},
		{name: "empty MediaInput wrapper", payload: reqWithImage(&sglangapiv1.MediaInput{}), want: MediaScan{}},
		{name: "single item", payload: reqWithImage(itemMedia), want: MediaScan{Present: true}},
		{name: "empty many list (JSON [])", payload: reqWithImage(emptyManyMedia), want: MediaScan{}},
		{name: "non-empty many list", payload: reqWithImage(nonEmptyManyMedia), want: MediaScan{Present: true}},
		{name: "nested batch list", payload: reqWithImage(nestedMedia), want: MediaScan{Nested: true}},
		{
			name:    "item then empty many switches oneof and clears Present",
			payload: slices.Concat(reqWithImage(itemMedia), reqWithImage(emptyManyMedia)),
			want:    MediaScan{},
		},
		{
			name:    "non-empty many then empty many merges without clearing Present",
			payload: slices.Concat(reqWithImage(nonEmptyManyMedia), reqWithImage(emptyManyMedia)),
			want:    MediaScan{Present: true},
		},
		{
			name:    "nested then item switches oneof and clears Nested",
			payload: slices.Concat(reqWithImage(nestedMedia), reqWithImage(itemMedia)),
			want:    MediaScan{Present: true},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ScanRequestFields(tc.payload)
			require.NoError(t, err)
			require.Equal(t, tc.want, got.ImageData)

			var decoded sglangapiv1.GenerateRequest
			require.NoError(t, proto.Unmarshal(tc.payload, &decoded))
			require.Equal(t, protoMediaScan(decoded.GetImageData()), got.ImageData)
			require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = ScanRequestFields(tc.payload) }))
		})
	}
}

func TestScanRequestFieldsSubmessageMerge(t *testing.T) {
	t.Run("repeated sampling_params merges max_new_tokens", func(t *testing.T) {
		p1 := mustMarshal(t, &sglangapiv1.GenerateRequest{
			SamplingParams: &sglangapiv1.SamplingParamsOrList{
				Value: &sglangapiv1.SamplingParamsOrList_One{One: &sglangapiv1.SamplingParams{MaxNewTokens: proto.Int64(256)}},
			},
		})
		p2 := mustMarshal(t, &sglangapiv1.GenerateRequest{
			SamplingParams: &sglangapiv1.SamplingParamsOrList{
				Value: &sglangapiv1.SamplingParamsOrList_One{One: &sglangapiv1.SamplingParams{Temperature: proto.Float64(0.2)}},
			},
		})
		merged := slices.Concat(p1, p2)

		fields, err := ScanRequestFields(merged)
		require.NoError(t, err)
		require.True(t, fields.SamplingParams.Present)
		require.False(t, fields.SamplingParams.Many)
		gotMax, ok, err := ScanMaxNewTokens(fields.SamplingParams.One)
		require.NoError(t, err)
		require.True(t, ok)

		var decoded sglangapiv1.GenerateRequest
		require.NoError(t, proto.Unmarshal(merged, &decoded))
		require.Equal(t, decoded.GetSamplingParams().GetOne().GetMaxNewTokens(), gotMax)
		require.Equal(t, int64(256), gotMax)
	})

	t.Run("repeated input_ids and mm_hashes merge across one fields and reset on oneof switch", func(t *testing.T) {
		first := mustMarshal(t, &sglangapiv1.GenerateRequest{
			InputIds: &sglangapiv1.TokenIdsOrList{
				Value: &sglangapiv1.TokenIdsOrList_One{One: &sglangapiv1.TokenIds{Ids: []int64{10, 20}}},
			},
			MmHashes: &sglangapiv1.StringListOrList{
				Value: &sglangapiv1.StringListOrList_One{One: &sglangapiv1.StringList{Items: []string{"h1"}}},
			},
		})
		second := mustMarshal(t, &sglangapiv1.GenerateRequest{
			InputIds: &sglangapiv1.TokenIdsOrList{
				Value: &sglangapiv1.TokenIdsOrList_One{One: &sglangapiv1.TokenIds{Ids: []int64{30, 40}}},
			},
			MmHashes: &sglangapiv1.StringListOrList{
				Value: &sglangapiv1.StringListOrList_One{One: &sglangapiv1.StringList{Items: []string{"h2"}}},
			},
		})
		merged := slices.Concat(first, second)

		fields, err := ScanRequestFields(merged)
		require.NoError(t, err)
		ids, err := ScanTokenIDs(fields.InputIDs.One)
		require.NoError(t, err)
		hashes, err := ScanStringList(fields.MMHashes.One)
		require.NoError(t, err)

		var decoded sglangapiv1.GenerateRequest
		require.NoError(t, proto.Unmarshal(merged, &decoded))
		require.Equal(t, []int64{10, 20, 30, 40}, decoded.GetInputIds().GetOne().GetIds())
		require.Equal(t, []uint32{10, 20, 30, 40}, ids)
		require.Equal(t, decoded.GetMmHashes().GetOne().GetItems(), hashes)

		// Switching one -> many -> one discards the earlier one.
		many := mustMarshal(t, &sglangapiv1.GenerateRequest{
			InputIds: &sglangapiv1.TokenIdsOrList{
				Value: &sglangapiv1.TokenIdsOrList_Many{Many: &sglangapiv1.TokenIdsList{}},
			},
		})
		switched := bytes.Join([][]byte{first, many, second}, nil)
		fields, err = ScanRequestFields(switched)
		require.NoError(t, err)
		ids, err = ScanTokenIDs(fields.InputIDs.One)
		require.NoError(t, err)
		decoded.Reset()
		require.NoError(t, proto.Unmarshal(switched, &decoded))
		require.Equal(t, []int64{30, 40}, decoded.GetInputIds().GetOne().GetIds())
		require.Equal(t, []uint32{30, 40}, ids)
	})

	t.Run("empty OrList wrapper after populated field preserves earlier value without allocating", func(t *testing.T) {
		populated := mustMarshal(t, &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "keep-me"}},
		})
		empty := protowire.AppendBytes(protowire.AppendTag(nil, reqText, protowire.BytesType), nil)
		merged := slices.Concat(populated, empty)

		fields, err := ScanRequestFields(merged)
		require.NoError(t, err)
		require.True(t, fields.Text.Present)
		require.Equal(t, "keep-me", string(fields.Text.One))
		require.Zero(t, testing.AllocsPerRun(100, func() { _, _ = ScanRequestFields(merged) }))
	})
}

func TestRewritePriority(t *testing.T) {
	clientPrio := int64(99)
	req := &sglangapiv1.GenerateRequest{
		Text:     &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hello"}},
		Priority: &clientPrio,
		Stream:   proto.Bool(true),
	}
	payload := mustMarshal(t, req)
	// Two client priority fields: both must go.
	payload = protowire.AppendVarint(protowire.AppendTag(payload, reqPriority, protowire.VarintType), 7)
	frame, err := grpcframe.FramePayload(payload)
	require.NoError(t, err)

	for _, prio := range []int64{5, 0, -3, math.MaxInt64} {
		framed, err := RewritePriority(frame, prio)
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
		// One exact-size allocation.
		require.Equal(t, len(framed), cap(framed))
	}
	require.Equal(t, 1.0, testing.AllocsPerRun(100, func() { _, _ = RewritePriority(frame, 5) }))

	// An optional field keeps an explicit zero, so a request without priority
	// still gets one.
	noPrio, err := grpcframe.FramePayload(mustMarshal(t, &sglangapiv1.GenerateRequest{Stream: proto.Bool(true)}))
	require.NoError(t, err)
	framed, err := RewritePriority(noPrio, 0)
	require.NoError(t, err)
	var out sglangapiv1.GenerateRequest
	require.NoError(t, proto.Unmarshal(framed[grpcframe.HeaderLen:], &out))
	require.NotNil(t, out.Priority)
	require.Zero(t, out.GetPriority())

	malformed, err := grpcframe.FramePayload([]byte{0xff})
	require.NoError(t, err)
	_, err = RewritePriority(malformed, 1)
	require.ErrorIs(t, err, ErrMalformed)
	_, err = RewritePriority([]byte{0, 0}, 1)
	require.Error(t, err)
}

func TestScanStreamItem(t *testing.T) {
	stop := &sglangapiv1.FinishReason{Kind: &sglangapiv1.FinishReason_Stop{Stop: &sglangapiv1.FinishStop{}}}
	tests := []struct {
		name string
		item *sglangapiv1.GenerateStreamItem
		want StreamItem
	}{
		{
			name: "token frame",
			item: &sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
				Text:      "hi",
				MetaInfo:  &sglangapiv1.GenerateMetaInfo{PromptTokens: 9, CompletionTokens: 4, CachedTokens: 3},
				OutputIds: &sglangapiv1.TokenIds{Ids: []int64{42}},
			}}},
			want: StreamItem{HasFrame: true, HasMetaInfo: true, PromptTokens: 9, CompletionTokens: 4, CachedTokens: 3},
		},
		{
			name: "terminal frame with index",
			item: &sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
				Index:    proto.Uint32(2),
				MetaInfo: &sglangapiv1.GenerateMetaInfo{FinishReason: stop, PromptTokens: 64, CompletionTokens: 12, CachedTokens: 48},
			}}},
			want: StreamItem{HasFrame: true, Index: 2, HasMetaInfo: true, Finished: true, PromptTokens: 64, CompletionTokens: 12, CachedTokens: 48},
		},
		{
			name: "empty finish_reason is still set",
			item: &sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
				MetaInfo: &sglangapiv1.GenerateMetaInfo{FinishReason: &sglangapiv1.FinishReason{}},
			}}},
			want: StreamItem{HasFrame: true, HasMetaInfo: true, Finished: true},
		},
		{
			name: "frame without meta_info",
			item: &sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{Text: "x"}}},
			want: StreamItem{HasFrame: true},
		},
		{
			name: "empty item",
			item: &sglangapiv1.GenerateStreamItem{},
			want: StreamItem{},
		},
		{
			name: "error",
			item: &sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Error{Error: &sglangapiv1.GenerateStreamError{
				Error: &sglangapiv1.ErrorBody{Message: "boom", Code: 500},
			}}},
			want: StreamItem{IsError: true},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			payload, err := proto.Marshal(tc.item)
			require.NoError(t, err)
			got, err := ScanStreamItem(payload)
			require.NoError(t, err)
			require.Equal(t, tc.want, got)

			// Cross-check against the generated code.
			var decoded sglangapiv1.GenerateStreamItem
			require.NoError(t, proto.Unmarshal(payload, &decoded))
			frame := decoded.GetFrame()
			require.Equal(t, frame != nil, got.HasFrame)
			require.Equal(t, decoded.GetError() != nil, got.IsError)
			require.Equal(t, frame.GetIndex(), got.Index)
			require.Equal(t, frame.GetMetaInfo() != nil, got.HasMetaInfo)
			require.Equal(t, frame.GetMetaInfo().GetFinishReason() != nil, got.Finished)
			require.Equal(t, frame.GetMetaInfo().GetPromptTokens(), got.PromptTokens)
			require.Equal(t, frame.GetMetaInfo().GetCompletionTokens(), got.CompletionTokens)
			require.Equal(t, frame.GetMetaInfo().GetCachedTokens(), got.CachedTokens)

			allocs := testing.AllocsPerRun(100, func() { _, _ = ScanStreamItem(payload) })
			require.Zero(t, allocs)
		})
	}

	// A truncated nested message is malformed.
	payload, err := proto.Marshal(tests[1].item)
	require.NoError(t, err)
	_, err = ScanStreamItem(payload[:len(payload)-1])
	require.ErrorIs(t, err, ErrMalformed)
}

// Concatenated messages merge on the wire. frame and error share the item
// oneof, so a switch between them discards the earlier member's fields.
func TestScanStreamItemOneofSwitchDiscardsEarlierFields(t *testing.T) {
	marshal := func(item *sglangapiv1.GenerateStreamItem) []byte {
		b, err := proto.Marshal(item)
		require.NoError(t, err)
		return b
	}
	terminal := marshal(&sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
		Index: proto.Uint32(3),
		MetaInfo: &sglangapiv1.GenerateMetaInfo{
			FinishReason: &sglangapiv1.FinishReason{Kind: &sglangapiv1.FinishReason_Stop{Stop: &sglangapiv1.FinishStop{}}},
			PromptTokens: 64,
			CachedTokens: 48,
		},
	}}})
	errItem := marshal(&sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Error{Error: &sglangapiv1.GenerateStreamError{
		Error: &sglangapiv1.ErrorBody{Message: "boom"},
	}}})
	noMeta := marshal(&sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{Text: "x"}}})
	complMeta := marshal(&sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
		MetaInfo: &sglangapiv1.GenerateMetaInfo{CompletionTokens: 9},
	}}})

	tests := []struct {
		name    string
		payload []byte
		want    StreamItem
	}{
		{name: "frame, error, frame", payload: bytes.Join([][]byte{terminal, errItem, noMeta}, nil), want: StreamItem{HasFrame: true}},
		{name: "frame, error", payload: bytes.Join([][]byte{terminal, errItem}, nil), want: StreamItem{IsError: true}},
		{
			name:    "frame, frame merges",
			payload: bytes.Join([][]byte{terminal, complMeta}, nil),
			want:    StreamItem{HasFrame: true, Index: 3, HasMetaInfo: true, Finished: true, PromptTokens: 64, CompletionTokens: 9, CachedTokens: 48},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ScanStreamItem(tc.payload)
			require.NoError(t, err)
			require.Equal(t, tc.want, got)

			var decoded sglangapiv1.GenerateStreamItem
			require.NoError(t, proto.Unmarshal(tc.payload, &decoded))
			frame := decoded.GetFrame()
			require.Equal(t, frame != nil, got.HasFrame)
			require.Equal(t, decoded.GetError() != nil, got.IsError)
			require.Equal(t, frame.GetIndex(), got.Index)
			require.Equal(t, frame.GetMetaInfo() != nil, got.HasMetaInfo)
			require.Equal(t, frame.GetMetaInfo().GetFinishReason() != nil, got.Finished)
			require.Equal(t, frame.GetMetaInfo().GetPromptTokens(), got.PromptTokens)
			require.Equal(t, frame.GetMetaInfo().GetCompletionTokens(), got.CompletionTokens)
			require.Equal(t, frame.GetMetaInfo().GetCachedTokens(), got.CachedTokens)
		})
	}
}
