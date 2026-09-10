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

package vllm

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const testHash = "abc123"

// testKwargs is a base64 tensor blob standing in for a real (non-cache-hit)
// kwargs_data entry.
const testKwargs = "dGVuc29y"

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
			name:   "max_completion_tokens is added even when the client omitted it",
			format: gateway.FormatChatCompletions,
			body:   map[string]any{"model": "m"},
			want: map[string]any{
				"model":                 "m",
				"max_tokens":            1,
				"max_completion_tokens": 1,
				"stream":                false,
			},
		},
		{
			name:   "completions caps max_tokens, strips min_tokens, forces non-streaming",
			format: gateway.FormatCompletions,
			body:   map[string]any{"model": "m", "max_tokens": 100, "min_tokens": 5},
			want:   map[string]any{"model": "m", "max_tokens": 1, "max_completion_tokens": 1, "stream": false},
		},
		{
			name:   "streaming is forced false and stream_options stripped",
			format: gateway.FormatCompletions,
			body:   map[string]any{"stream": true, "stream_options": map[string]any{"include_usage": true}},
			want:   map[string]any{"stream": false, "max_tokens": 1, "max_completion_tokens": 1},
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

// mmImageKwargs extracts features["kwargs_data"].image as a []any, marshaling
// through JSON so the test sees exactly what the encoder/decoder receives on the
// wire (where the cache-hit sentinel must be null, never "").
func mmImageKwargs(t *testing.T, features map[string]any) []any {
	t.Helper()
	raw, err := json.Marshal(features["kwargs_data"])
	if err != nil {
		t.Fatalf("marshal kwargs_data: %v", err)
	}
	var decoded map[string][]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal kwargs_data: %v", err)
	}
	return decoded[modalityImage]
}

func TestBuildMMFeatures_CacheHitSentinelSerializesAsNull(t *testing.T) {
	// The empty-string KwargsData is the "resolve from cache" sentinel. On the
	// wire it must be JSON null, not "": vLLM decodes "" as an inline tensor and
	// fails with "Input data was truncated", while null means a cache-hit item.
	entry := func(kwargs string) pipeline.MultimodalEntry {
		return pipeline.MultimodalEntry{Hash: testHash, KwargsData: kwargs}
	}

	t.Run("all cache-hit -> all null", func(t *testing.T) {
		features := buildMMFeatures([]pipeline.MultimodalEntry{entry(""), entry("")}, true)
		items := mmImageKwargs(t, features)
		if len(items) != 2 {
			t.Fatalf("expected 2 kwargs_data entries, got %d: %v", len(items), items)
		}
		for i, it := range items {
			if it != nil {
				t.Errorf("kwargs_data[%d] = %#v, want null", i, it)
			}
		}
		// Regression guard: the raw JSON must contain null, not "".
		raw, _ := json.Marshal(features["kwargs_data"])
		if strings.Contains(string(raw), `""`) {
			t.Errorf("kwargs_data emitted empty string instead of null: %s", raw)
		}
	})

	t.Run("mixed batch keeps inline, nulls cache hits", func(t *testing.T) {
		features := buildMMFeatures([]pipeline.MultimodalEntry{entry(testKwargs), entry("")}, true)
		items := mmImageKwargs(t, features)
		if len(items) != 2 || items[0] != testKwargs || items[1] != nil {
			t.Fatalf("expected [\"dGVuc29y\", null], got %#v", items)
		}
	})

	t.Run("includeKwargs=false omits the field", func(t *testing.T) {
		features := buildMMFeatures([]pipeline.MultimodalEntry{entry("")}, false)
		if _, ok := features["kwargs_data"]; ok {
			t.Errorf("expected kwargs_data absent when includeKwargs is false")
		}
	})
}
