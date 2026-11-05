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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
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
				MetaInfo:  &sglangapiv1.GenerateMetaInfo{PromptTokens: 9, CachedTokens: 3},
				OutputIds: &sglangapiv1.TokenIds{Ids: []int64{42}},
			}}},
			want: StreamItem{HasMetaInfo: true, CachedTokens: 3},
		},
		{
			name: "terminal frame with index",
			item: &sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
				Index:    proto.Uint32(2),
				MetaInfo: &sglangapiv1.GenerateMetaInfo{FinishReason: stop, CachedTokens: 48},
			}}},
			want: StreamItem{Index: 2, HasMetaInfo: true, Finished: true, CachedTokens: 48},
		},
		{
			name: "empty finish_reason is still set",
			item: &sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{
				MetaInfo: &sglangapiv1.GenerateMetaInfo{FinishReason: &sglangapiv1.FinishReason{}},
			}}},
			want: StreamItem{HasMetaInfo: true, Finished: true},
		},
		{
			name: "frame without meta_info",
			item: &sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{Text: "x"}}},
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
			require.Equal(t, decoded.GetError() != nil, got.IsError)
			require.Equal(t, frame.GetIndex(), got.Index)
			require.Equal(t, frame.GetMetaInfo() != nil, got.HasMetaInfo)
			require.Equal(t, frame.GetMetaInfo().GetFinishReason() != nil, got.Finished)
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
			CachedTokens: 48,
		},
	}}})
	errItem := marshal(&sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Error{Error: &sglangapiv1.GenerateStreamError{
		Error: &sglangapiv1.ErrorBody{Message: "boom"},
	}}})
	noMeta := marshal(&sglangapiv1.GenerateStreamItem{Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{Text: "x"}}})

	tests := []struct {
		name    string
		payload []byte
		want    StreamItem
	}{
		{name: "frame, error, frame", payload: bytes.Join([][]byte{terminal, errItem, noMeta}, nil), want: StreamItem{}},
		{name: "frame, error", payload: bytes.Join([][]byte{terminal, errItem}, nil), want: StreamItem{IsError: true}},
		{name: "frame, frame merges", payload: bytes.Join([][]byte{terminal, noMeta}, nil), want: StreamItem{Index: 3, HasMetaInfo: true, Finished: true, CachedTokens: 48}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ScanStreamItem(tc.payload)
			require.NoError(t, err)
			require.Equal(t, tc.want, got)

			var decoded sglangapiv1.GenerateStreamItem
			require.NoError(t, proto.Unmarshal(tc.payload, &decoded))
			frame := decoded.GetFrame()
			require.Equal(t, decoded.GetError() != nil, got.IsError)
			require.Equal(t, frame.GetIndex(), got.Index)
			require.Equal(t, frame.GetMetaInfo() != nil, got.HasMetaInfo)
			require.Equal(t, frame.GetMetaInfo().GetFinishReason() != nil, got.Finished)
			require.Equal(t, frame.GetMetaInfo().GetCachedTokens(), got.CachedTokens)
		})
	}
}
