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

package steps

import (
	"reflect"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
)

func TestCapSingleTokenOutput(t *testing.T) {
	tests := []struct {
		name   string
		format gateway.RequestFormat
		body   map[string]any
		want   map[string]any
	}{
		{
			name:   "chat completions caps output fields and forces non-streaming",
			format: gateway.FormatChatCompletions,
			body: map[string]any{
				"model":                 "m",
				"max_tokens":            100,
				"min_tokens":            5,
				"max_completion_tokens": 100,
				"stream":                true,
				"stream_options":        map[string]any{"include_usage": true},
			},
			want: map[string]any{
				"model":                 "m",
				"max_tokens":            1,
				"max_completion_tokens": 1,
				"stream":                false,
			},
		},
		{
			name:   "max_completion_tokens is not added when the client omitted it",
			format: gateway.FormatChatCompletions,
			body:   map[string]any{"model": "m"},
			want: map[string]any{
				"model":      "m",
				"max_tokens": 1,
				"stream":     false,
			},
		},
		{
			name:   "completions caps max_tokens, strips min_tokens, forces non-streaming",
			format: gateway.FormatCompletions,
			body:   map[string]any{"model": "m", "max_tokens": 100, "min_tokens": 5},
			want:   map[string]any{"model": "m", "max_tokens": 1, "stream": false},
		},
		{
			name:   "streaming is forced false and stream_options stripped",
			format: gateway.FormatCompletions,
			body:   map[string]any{"stream": true, "stream_options": map[string]any{"include_usage": true}},
			want:   map[string]any{"stream": false, "max_tokens": 1},
		},
		{
			name:   "generate caps max_tokens and strips min_tokens inside sampling_params",
			format: gateway.FormatGenerate,
			body: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 100, "min_tokens": 5},
			},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			name:   "generate synthesizes sampling_params when absent",
			format: gateway.FormatGenerate,
			body:   map[string]any{"model": "m"},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			name:   "generate preserves other sampling_params entries",
			format: gateway.FormatGenerate,
			body: map[string]any{
				"sampling_params": map[string]any{
					"extra_args": map[string]any{"kv_transfer_params": "x"},
				},
			},
			want: map[string]any{
				"sampling_params": map[string]any{
					"max_tokens": 1,
					"extra_args": map[string]any{"kv_transfer_params": "x"},
				},
				"stream": false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			capSingleTokenOutput(tt.body, tt.format)
			if !reflect.DeepEqual(tt.body, tt.want) {
				t.Fatalf("got %v, want %v", tt.body, tt.want)
			}
		})
	}
}
