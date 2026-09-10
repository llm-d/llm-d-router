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
	"reflect"
	"testing"
)

// The sidecar connectors copy the client body one level deep, cap the copy for
// the prefill leg, and marshal the original for the decode leg. Capping the
// generate API writes inside sampling_params, which a one-level copy shares, so
// CapSingleToken must replace that map rather than write through it.
func TestCapSingleToken_LeavesTheCallersNestedMapIntact(t *testing.T) {
	client := map[string]any{
		"token_ids":         []any{1, 2, 3},
		FieldSamplingParams: map[string]any{FieldMaxTokens: 200, FieldMinTokens: 10},
	}

	prefill := maps.Clone(client)
	CapSingleToken(prefill, APITypeGenerate)

	decodeLimits := client[FieldSamplingParams].(map[string]any)
	if got := decodeLimits[FieldMaxTokens]; got != 200 {
		t.Errorf("decode leg max_tokens = %v, want the client's 200", got)
	}
	if got, ok := decodeLimits[FieldMinTokens]; !ok || got != 10 {
		t.Errorf("decode leg min_tokens = %v (present %v), want the client's 10", got, ok)
	}

	prefillLimits := prefill[FieldSamplingParams].(map[string]any)
	if got := prefillLimits[FieldMaxTokens]; got != 1 {
		t.Errorf("prefill leg max_tokens = %v, want 1", got)
	}
	if _, ok := prefillLimits[FieldMinTokens]; ok {
		t.Error("prefill leg kept min_tokens")
	}
}

func TestCapSingleToken(t *testing.T) {
	tests := []struct {
		name    string
		apiType APIType
		body    map[string]any
		want    map[string]any
	}{
		{
			name:    "chat completions caps output fields and forces non-streaming",
			apiType: APITypeChatCompletions,
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
			name:    "max_completion_tokens is added even when the client omitted it",
			apiType: APITypeChatCompletions,
			body:    map[string]any{"model": "m"},
			want: map[string]any{
				"model":                 "m",
				"max_tokens":            1,
				"max_completion_tokens": 1,
				"stream":                false,
			},
		},
		{
			name:    "completions caps max_tokens, strips min_tokens, forces non-streaming",
			apiType: APITypeCompletions,
			body:    map[string]any{"model": "m", "max_tokens": 100, "min_tokens": 5},
			want:    map[string]any{"model": "m", "max_tokens": 1, "stream": false},
		},
		{
			name:    "streaming is forced false and stream_options stripped",
			apiType: APITypeCompletions,
			body:    map[string]any{"stream": true, "stream_options": map[string]any{"include_usage": true}},
			want:    map[string]any{"stream": false, "max_tokens": 1},
		},
		{
			name:    "generate caps max_tokens and strips min_tokens inside sampling_params",
			apiType: APITypeGenerate,
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
			name:    "generate synthesizes sampling_params when absent",
			apiType: APITypeGenerate,
			body:    map[string]any{"model": "m"},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			// The sidecar caps the leg straight off the client body, with no
			// equivalent of the coordinator's validateSamplingParams ahead of
			// it, so a malformed sampling_params arrives here. The leg still
			// has to carry a cap, so the field is replaced.
			name:    "generate replaces a non-object sampling_params",
			apiType: APITypeGenerate,
			body:    map[string]any{"model": "m", "sampling_params": "not-an-object"},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			name:    "generate replaces a null sampling_params",
			apiType: APITypeGenerate,
			body:    map[string]any{"model": "m", "sampling_params": nil},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			name:    "generate leaves the top-level fields alone",
			apiType: APITypeGenerate,
			body: map[string]any{
				"max_tokens":            100,
				"max_completion_tokens": 100,
				"sampling_params":       map[string]any{"max_tokens": 100},
			},
			want: map[string]any{
				"max_tokens":            100,
				"max_completion_tokens": 100,
				"sampling_params":       map[string]any{"max_tokens": 1},
				"stream":                false,
			},
		},
		{
			name:    "responses caps max_output_tokens",
			apiType: APITypeResponses,
			body:    map[string]any{"model": "m", "max_output_tokens": 800},
			want:    map[string]any{"model": "m", "max_output_tokens": 1, "stream": false},
		},
		{
			// max_tokens and max_completion_tokens are not Responses fields, so
			// tokenLimitFields does not name them and they are left as sent.
			// min_tokens is a floor and is stripped for every API.
			name:    "responses leaves fields the API does not use",
			apiType: APITypeResponses,
			body:    map[string]any{"model": "m", "max_tokens": 100, "min_tokens": 5, "max_output_tokens": 800},
			want:    map[string]any{"model": "m", "max_tokens": 100, "max_output_tokens": 1, "stream": false},
		},
		{
			name:    "generate preserves other sampling_params entries",
			apiType: APITypeGenerate,
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
			CapSingleToken(tt.body, tt.apiType)
			if !reflect.DeepEqual(tt.body, tt.want) {
				t.Fatalf("got %v, want %v", tt.body, tt.want)
			}
		})
	}
}
