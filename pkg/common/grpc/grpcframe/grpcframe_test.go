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

package grpcframe

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"

	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestParsePayload(t *testing.T) {
	tests := []struct {
		name    string
		data    []byte
		want    []byte
		wantErr string
	}{
		{
			name:    "too short header",
			data:    []byte{0, 0, 0, 0},
			want:    nil,
			wantErr: "invalid gRPC frame: expected at least 5 bytes for header, got 4",
		},
		{
			name:    "compressed payload not supported",
			data:    []byte{1, 0, 0, 0, 0},
			want:    nil,
			wantErr: "compressed gRPC payload is not supported",
		},
		{
			name: "incomplete payload",
			data: func() []byte {
				b := make([]byte, 10)
				binary.BigEndian.PutUint32(b[1:5], 10)
				copy(b[5:], []byte("12345"))
				return b
			}(),
			want:    nil,
			wantErr: "incomplete gRPC payload: header indicates 10 bytes, but only 5 bytes are available",
		},
		{
			name: "success exact size",
			data: func() []byte {
				b, err := FramePayload([]byte("hello"))
				require.NoError(t, err)
				return b
			}(),
			want: []byte("hello"),
		},
		{
			name:    "length header overflows uint32",
			data:    []byte{0, 0xFF, 0xFF, 0xFF, 0xFF, 0xAA, 0xBB, 0xCC, 0xDD},
			want:    nil,
			wantErr: "incomplete gRPC payload: header indicates 4294967295 bytes, but only 4 bytes are available",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParsePayload(tt.data)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParsePayload() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestParsePayloads(t *testing.T) {
	frame := func(payload string) []byte {
		b, err := FramePayload([]byte(payload))
		require.NoError(t, err)
		return b
	}
	tests := []struct {
		name    string
		data    []byte
		want    [][]byte
		wantErr string
	}{
		{
			name: "no frames",
			data: nil,
		},
		{
			name: "one frame",
			data: frame("hello"),
			want: [][]byte{[]byte("hello")},
		},
		{
			name: "two frames",
			data: append(frame("hello"), frame("world")...),
			want: [][]byte{[]byte("hello"), []byte("world")},
		},
		{
			name:    "incomplete trailing frame keeps the frames before it",
			data:    append(frame("hello"), frame("world")[:8]...),
			want:    [][]byte{[]byte("hello")},
			wantErr: "incomplete gRPC payload: header indicates 5 bytes, but only 3 bytes are available",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParsePayloads(tt.data)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParsePayloads() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMarshalProtoRoundTrip(t *testing.T) {
	_, err := MarshalProto(nil)
	require.EqualError(t, err, "nil protobuf message")

	priority := int64(7)
	req := &sglangapiv1.GenerateRequest{
		Text:     &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hello sglang"}},
		Priority: &priority,
	}
	framed, err := MarshalProto(req)
	require.NoError(t, err)

	payload, err := ParsePayload(framed)
	require.NoError(t, err)

	var decoded sglangapiv1.GenerateRequest
	require.NoError(t, proto.Unmarshal(payload, &decoded))
	require.True(t, proto.Equal(req, &decoded))
}

func TestIsContentType(t *testing.T) {
	tests := []struct {
		contentType string
		want        bool
	}{
		{contentType: "application/grpc", want: true},
		{contentType: "application/grpc+proto", want: true},
		{contentType: "application/grpc+json", want: true},
		{contentType: "APPLICATION/GRPC; charset=utf-8", want: true},
		{contentType: "application/grpc-web", want: false},
		{contentType: "application/grpc-web+proto", want: false},
		{contentType: "text/event-stream", want: false},
		{contentType: "application/json", want: false},
		{contentType: "", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.contentType, func(t *testing.T) {
			require.Equal(t, tt.want, IsContentType(tt.contentType))
		})
	}
}

func TestStatusAndHasErrorStatus(t *testing.T) {
	require.False(t, HasErrorStatus(nil))

	okHeader := make(http.Header)
	okHeader.Set(HeaderStatus, StatusOK)
	code, ok := Status(okHeader)
	require.True(t, ok)
	require.Equal(t, StatusOK, code)
	require.False(t, HasErrorStatus(okHeader))

	trailerErr := make(http.Header)
	trailerErr.Add(http.TrailerPrefix+HeaderStatus, "14")
	code, ok = Status(trailerErr)
	require.True(t, ok)
	require.Equal(t, "14", code)
	require.True(t, HasErrorStatus(trailerErr))

	reverseProxyTrailerErr := make(http.Header)
	reverseProxyTrailerErr.Add(http.TrailerPrefix+http.CanonicalHeaderKey(HeaderStatus), "3")
	code, ok = Status(reverseProxyTrailerErr)
	require.True(t, ok)
	require.Equal(t, "3", code)
	require.True(t, HasErrorStatus(reverseProxyTrailerErr))
}

func TestCopyTrailersAndWriteError(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteError(rec, codes.InvalidArgument, "bad request: 50% done\nretry")
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, MediaType, rec.Header().Get(reqcommon.HeaderContentType))
	require.Equal(t, "3", rec.Header().Get(HeaderStatus))
	require.Equal(t, "bad request: 50%25 done%0Aretry", rec.Header().Get(HeaderMessage))

	dst := make(http.Header)
	CopyTrailers(dst, rec.Header())
	require.Equal(t, "3", dst.Get(http.TrailerPrefix+HeaderStatus))
	require.Equal(t, "bad request: 50%25 done%0Aretry", dst.Get(http.TrailerPrefix+HeaderMessage))

	fromReverseProxy := make(http.Header)
	fromReverseProxy.Add(http.TrailerPrefix+http.CanonicalHeaderKey(HeaderStatus), "14")
	dstFromProxy := make(http.Header)
	CopyTrailers(dstFromProxy, fromReverseProxy)
	require.Equal(t, "14", dstFromProxy.Get(http.TrailerPrefix+HeaderStatus))
}

func TestReadFrame(t *testing.T) {
	f1, err := FramePayload([]byte("first"))
	require.NoError(t, err)
	f2, err := FramePayload([]byte("second"))
	require.NoError(t, err)

	r := bytes.NewReader(append(append([]byte{}, f1...), f2...))
	got1, err := ReadFrame(r, 64)
	require.NoError(t, err)
	require.Equal(t, []byte("first"), got1)

	got2, err := ReadFrame(r, 64)
	require.NoError(t, err)
	require.Equal(t, []byte("second"), got2)

	_, err = ReadFrame(r, 64)
	require.ErrorIs(t, err, io.EOF)

	_, err = ReadFrame(bytes.NewReader(f2), 3)
	require.ErrorContains(t, err, "exceeds limit 3")

	_, err = ReadFrame(bytes.NewReader([]byte{1, 0, 0, 0, 0}), 64)
	require.ErrorIs(t, err, ErrCompressed)

	_, err = ReadFrame(bytes.NewReader(f1[:3]), 64)
	require.ErrorContains(t, err, "read gRPC frame header")
}

func TestEncodeStatusMessage(t *testing.T) {
	require.Equal(t, "plain ascii", EncodeStatusMessage("plain ascii"))
	require.Equal(t, "50%25\x20bad%0Aline2%0D%00%C3%A9", EncodeStatusMessage("50% bad\nline2\r\x00é"))
}

func TestMarshalProtoAllocatesOnce(t *testing.T) {
	req := &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hello sglang"}},
	}
	allocs := testing.AllocsPerRun(100, func() {
		if _, err := MarshalProto(req); err != nil {
			t.Fatal(err)
		}
	})
	require.Equal(t, 1.0, allocs)
}

func TestReader(t *testing.T) {
	f1, err := FramePayload([]byte("first"))
	require.NoError(t, err)
	big := bytes.Repeat([]byte("x"), 8192) // larger than the initial buffer
	f2, err := FramePayload(big)
	require.NoError(t, err)

	fr := NewReader(bytes.NewReader(append(append([]byte{}, f1...), f2...)), 1<<20)
	got, err := fr.Next()
	require.NoError(t, err)
	require.Equal(t, f1, got, "Next returns the frame with its header")
	got, err = fr.Next()
	require.NoError(t, err)
	require.Equal(t, f2, got)
	_, err = fr.Next()
	require.ErrorIs(t, err, io.EOF)

	_, err = NewReader(bytes.NewReader(f1), 3).Next()
	require.ErrorContains(t, err, "exceeds limit 3")
	_, err = NewReader(bytes.NewReader([]byte{1, 0, 0, 0, 0}), 64).Next()
	require.ErrorIs(t, err, ErrCompressed)
	_, err = NewReader(bytes.NewReader(f1[:3]), 64).Next()
	require.ErrorContains(t, err, "read gRPC frame header")
	_, err = NewReader(bytes.NewReader(f1[:7]), 64).Next()
	require.ErrorContains(t, err, "read gRPC frame payload")

	// Steady state reuses the buffer: no allocation per frame.
	stream := bytes.Repeat(f1, 100)
	r := bytes.NewReader(stream)
	fr = NewReader(r, 64)
	allocs := testing.AllocsPerRun(10, func() {
		r.Reset(stream)
		for {
			if _, err := fr.Next(); err != nil {
				break
			}
		}
	})
	require.Zero(t, allocs)
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestReadMessage(t *testing.T) {
	frame, err := FramePayload([]byte("payload"))
	require.NoError(t, err)
	const limit = 64

	got, err := ReadMessage(bytes.NewReader(frame), limit)
	require.NoError(t, err)
	require.Equal(t, frame, got)
	require.Equal(t, len(frame), cap(got), "allocated at the exact size")

	tooLargeDeclared, err := FramePayload(bytes.Repeat([]byte("x"), limit))
	require.NoError(t, err)
	tests := []struct {
		name string
		body []byte
		want error
	}{
		{name: "empty", body: nil, want: ErrMalformed},
		{name: "short header", body: []byte{0, 1}, want: ErrMalformed},
		{name: "truncated payload", body: frame[:len(frame)-1], want: ErrMalformed},
		{name: "trailing data", body: append(append([]byte{}, frame...), 0), want: ErrMalformed},
		{name: "compressed", body: []byte{1, 0, 0, 0, 0}, want: ErrCompressed},
		{name: "declared over limit, short body", body: []byte{0, 0, 0, 1, 0}, want: ErrTooLarge},
		{name: "declared over limit", body: tooLargeDeclared, want: ErrTooLarge},
		{name: "compression flag on a large body", body: bytes.Repeat([]byte("x"), limit+1), want: ErrCompressed},
		{name: "trailing data over limit", body: append(append([]byte{}, frame...), make([]byte, limit)...), want: ErrMalformed},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ReadMessage(bytes.NewReader(tc.body), limit)
			require.ErrorIs(t, err, tc.want)
		})
	}

	_, err = ReadMessage(errReader{}, limit)
	require.ErrorIs(t, err, io.ErrClosedPipe)

	// An oversized or compressed header is rejected before any payload read.
	for _, header := range [][]byte{{0, 0, 0, 1, 0}, {1, 0, 0, 0, 1}} {
		_, err = ReadMessage(io.MultiReader(bytes.NewReader(header), errReader{}), limit)
		require.NotErrorIs(t, err, io.ErrClosedPipe, "header %v", header)
	}

	// The largest limit does not overflow.
	got, err = ReadMessage(bytes.NewReader(frame), math.MaxInt64)
	require.NoError(t, err)
	require.Equal(t, frame, got)
}
