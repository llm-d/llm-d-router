/*
Copyright 2026 The Kubernetes Authors.

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

package grpcutil

import (
	"encoding/binary"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestParseGrpcPayload(t *testing.T) {
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
				b[0] = 0                               // not compressed
				binary.BigEndian.PutUint32(b[1:5], 10) // indicates 10 bytes
				copy(b[5:], []byte("12345"))
				return b
			}(),
			want:    nil,
			wantErr: "incomplete gRPC payload: header indicates 10 bytes, but only 5 bytes are available",
		},
		{
			name: "success exact size",
			data: func() []byte {
				b := make([]byte, 10)
				b[0] = 0 // not compressed
				binary.BigEndian.PutUint32(b[1:5], 5)
				copy(b[5:], []byte("hello"))
				return b
			}(),
			want:    []byte("hello"),
			wantErr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseGrpcPayload(tt.data)

			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("got error nil, want %q", tt.wantErr)
				}
				if gotErr := err.Error(); gotErr != tt.wantErr {
					t.Errorf("got error %q, want %q", gotErr, tt.wantErr)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			want := tt.want
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("ParseGrpcPayload() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
