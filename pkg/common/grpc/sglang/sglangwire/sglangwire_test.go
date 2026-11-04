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
		// Client-supplied bootstrap fields must never reach a worker.
		BootstrapHost:    optStr("spoofed"),
		BootstrapPort:    optInt(1),
		BootstrapRoom:    optInt(2),
		BootstrapPairKey: optStr("pair"),
		Priority:         proto.Int64(5),
		CustomLabels:     map[string]string{"k": "v"},
	}
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

func TestFrameGenerateRequestMatchesProto(t *testing.T) {
	payload, err := proto.Marshal(testRequest())
	require.NoError(t, err)

	tests := []struct {
		name string
		b    *Bootstrap
		want func(*sglangapiv1.GenerateRequest)
	}{
		{
			name: "strip only (decode-only)",
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
			framed, err := FrameGenerateRequest(payload, tc.b)
			require.NoError(t, err)
			want := testRequest()
			tc.want(want)
			if diff := cmp.Diff(want, decodeFrame(t, framed), protocmp.Transform()); diff != "" {
				t.Fatalf("mismatch (-want +got):\n%s", diff)
			}
		})
	}

	// Without client bootstrap fields (the normal case), the frame is
	// allocated at its exact size.
	clean := testRequest()
	clean.BootstrapHost, clean.BootstrapPort, clean.BootstrapRoom = nil, nil, nil
	cleanPayload, err := proto.Marshal(clean)
	require.NoError(t, err)
	framed, err := FrameGenerateRequest(cleanPayload, &Bootstrap{Host: "h", Port: 1, Room: 2})
	require.NoError(t, err)
	require.Equal(t, len(framed), cap(framed))

	_, err = FrameGenerateRequest([]byte{0xff}, nil)
	require.ErrorIs(t, err, ErrMalformed)

	allocs := testing.AllocsPerRun(100, func() {
		_, _ = FrameGenerateRequest(payload, &Bootstrap{Host: "10.2.3.4", Port: 8998, Room: 7})
	})
	require.Equal(t, 1.0, allocs)
}

func TestScanGenerateRequest(t *testing.T) {
	for _, stream := range []*bool{nil, proto.Bool(false), proto.Bool(true)} {
		payload, err := proto.Marshal(&sglangapiv1.GenerateRequest{Stream: stream, Priority: proto.Int64(1)})
		require.NoError(t, err)
		got, err := ScanGenerateRequest(payload)
		require.NoError(t, err)
		require.Equal(t, stream != nil && *stream, got)
	}
	_, err := ScanGenerateRequest([]byte{0xff, 0xff, 0xff})
	require.ErrorIs(t, err, ErrMalformed)
	// A length prefix past the end of the payload.
	_, err = ScanGenerateRequest([]byte{0x12, 0x05, 'a'})
	require.ErrorIs(t, err, ErrMalformed)
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
