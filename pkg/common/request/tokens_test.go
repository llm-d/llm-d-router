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
		name  string
		input map[string]any
		want  map[string]any
	}{
		{
			name:  "neither field present",
			input: map[string]any{"model": "m"},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1, FieldMaxCompletionTokens: 1, FieldStream: false},
		},
		{
			name:  "max_tokens only",
			input: map[string]any{"model": "m", FieldMaxTokens: 50},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1, FieldMaxCompletionTokens: 1, FieldStream: false},
		},
		{
			name:  "max_completion_tokens only",
			input: map[string]any{"model": "m", FieldMaxCompletionTokens: 100},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1, FieldMaxCompletionTokens: 1, FieldStream: false},
		},
		{
			name:  "both fields present",
			input: map[string]any{"model": "m", FieldMaxTokens: 50, FieldMaxCompletionTokens: 100},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1, FieldMaxCompletionTokens: 1, FieldStream: false},
		},
		{
			name:  "stream_options is stripped and stream is forced false",
			input: map[string]any{"model": "m", FieldStream: true, FieldStreamOptions: map[string]any{"include_usage": true}},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1, FieldMaxCompletionTokens: 1, FieldStream: false},
		},
		{
			name:  "min_tokens is stripped",
			input: map[string]any{"model": "m", FieldMinTokens: 5},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1, FieldMaxCompletionTokens: 1, FieldStream: false},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := maps.Clone(tt.input)
			PrimeSingleTokenRequest(target)

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

func TestCapMaxTokensField(t *testing.T) {
	tests := []struct {
		name  string
		input map[string]any
		want  map[string]any
	}{
		{
			name:  "neither field present",
			input: map[string]any{"model": "m"},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1},
		},
		{
			name:  "max_tokens is overwritten",
			input: map[string]any{"model": "m", FieldMaxTokens: 50},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1},
		},
		{
			name:  "min_tokens is stripped",
			input: map[string]any{"model": "m", FieldMinTokens: 5},
			want:  map[string]any{"model": "m", FieldMaxTokens: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := maps.Clone(tt.input)
			CapMaxTokensField(target)

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
