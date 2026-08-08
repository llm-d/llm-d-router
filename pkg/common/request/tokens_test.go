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

package request

import (
	"maps"
	"testing"
)

func TestPrimeSingleTokenRequest(t *testing.T) {
	tests := []struct {
		name     string
		original map[string]any
		want     map[string]any
	}{
		{
			name:     "neither field present",
			original: map[string]any{"model": "m"},
			want:     map[string]any{"model": "m", FieldMaxTokens: 1, FieldStream: false},
		},
		{
			name:     "max_tokens only",
			original: map[string]any{"model": "m", FieldMaxTokens: 50},
			want:     map[string]any{"model": "m", FieldMaxTokens: 1, FieldStream: false},
		},
		{
			name:     "max_completion_tokens only",
			original: map[string]any{"model": "m", FieldMaxCompletionTokens: 100},
			want:     map[string]any{"model": "m", FieldMaxTokens: 1, FieldMaxCompletionTokens: 1, FieldStream: false},
		},
		{
			name:     "both fields present",
			original: map[string]any{"model": "m", FieldMaxTokens: 50, FieldMaxCompletionTokens: 100},
			want:     map[string]any{"model": "m", FieldMaxTokens: 1, FieldMaxCompletionTokens: 1, FieldStream: false},
		},
		{
			name:     "stream_options is stripped and stream is forced false",
			original: map[string]any{"model": "m", FieldStream: true, FieldStreamOptions: map[string]any{"include_usage": true}},
			want:     map[string]any{"model": "m", FieldMaxTokens: 1, FieldStream: false},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := maps.Clone(tt.original)
			PrimeSingleTokenRequest(target, tt.original)

			if len(target) != len(tt.want) {
				t.Fatalf("got %v, want %v", target, tt.want)
			}
			for k, want := range tt.want {
				if got := target[k]; got != want {
					t.Errorf("target[%q] = %v, want %v", k, got, want)
				}
			}
		})
	}
}

func TestPrimeSingleTokenRequest_TargetIsOriginal(t *testing.T) {
	// target and original may be the same map (in-place mutation).
	req := map[string]any{"model": "m", FieldMaxCompletionTokens: 100}

	PrimeSingleTokenRequest(req, req)

	if req[FieldMaxTokens] != 1 {
		t.Errorf("max_tokens = %v, want 1", req[FieldMaxTokens])
	}
	if req[FieldMaxCompletionTokens] != 1 {
		t.Errorf("max_completion_tokens = %v, want 1", req[FieldMaxCompletionTokens])
	}
	if req[FieldStream] != false {
		t.Errorf("stream = %v, want false", req[FieldStream])
	}
}

func TestPrimeSingleTokenRequest_TargetDistinctFromOriginal(t *testing.T) {
	// target starts empty; original decides which max-tokens field is set,
	// mirroring buildEncoderRequest-style callers that build a fresh map.
	original := map[string]any{"model": "m", FieldMaxTokens: 50}
	target := map[string]any{}

	PrimeSingleTokenRequest(target, original)

	if _, ok := original[FieldMaxCompletionTokens]; ok {
		t.Fatalf("original should not be mutated, got %v", original)
	}
	if target[FieldMaxTokens] != 1 {
		t.Errorf("max_tokens = %v, want 1", target[FieldMaxTokens])
	}
	if _, ok := target[FieldMaxCompletionTokens]; ok {
		t.Errorf("max_completion_tokens should be absent, got %v", target[FieldMaxCompletionTokens])
	}
}
