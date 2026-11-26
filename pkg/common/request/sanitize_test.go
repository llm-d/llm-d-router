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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestIsTokenArrayCompletionsPromptDecodedValues(t *testing.T) {
	cases := []struct {
		name   string
		prompt any
		want   bool
	}{
		{name: "string prompt", prompt: "hello", want: false},
		{name: "string array", prompt: []any{"a", "b"}, want: false},
		{name: "empty array", prompt: []any{}, want: false},
		{name: "nil", prompt: nil, want: false},
		{name: "single token array float64", prompt: []any{float64(1), float64(2)}, want: true},
		{name: "single token array json.Number", prompt: []any{json.Number("1"), json.Number("2")}, want: true},
		{name: "batch token arrays", prompt: []any{[]any{json.Number("1")}, []any{json.Number("2"), json.Number("3")}}, want: true},
		{name: "batch with empty inner array", prompt: []any{[]any{json.Number("1")}, []any{}}, want: false},
		{name: "mixed int and string", prompt: []any{json.Number("1"), "a"}, want: false},
		{name: "negative int", prompt: []any{json.Number("-1")}, want: false},
		{name: "non-integer float", prompt: []any{float64(1.5)}, want: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, isTokenArrayCompletionsPrompt(tc.prompt))
		})
	}
}

func TestIsExternalTextRoute(t *testing.T) {
	require.True(t, IsExternalTextRoute(APITypeChatCompletions, nil))
	require.True(t, IsExternalTextRoute(APITypeMessages, nil))
	require.True(t, IsExternalTextRoute(APITypeResponses, nil))
	require.True(t, IsExternalTextRoute(APITypeCompletions, "hello"))
	require.True(t, IsExternalTextRoute(APITypeCompletions, []any{"a", "b"}))
	require.False(t, IsExternalTextRoute(APITypeCompletions, []any{json.Number("1"), json.Number("2")}))
	require.False(t, IsExternalTextRoute(APITypeVLLMGenerate, nil))
	require.False(t, IsExternalTextRoute(APITypeSGLangGenerate, nil))
	require.False(t, IsExternalTextRoute(APITypeAudioSpeech, nil))
}

func TestFindAndStripExternalTextFields(t *testing.T) {
	clean := map[string]any{"model": "m", "messages": []any{}}
	require.Nil(t, FindExternalTextFields(clean))
	require.Nil(t, StripExternalTextFields(clean))
	require.Zero(t, testing.AllocsPerRun(100, func() {
		_ = FindExternalTextFields(clean)
		_ = StripExternalTextFields(clean)
	}))

	dirty := map[string]any{
		"model":                     "m",
		"prompt_token_ids":          []any{1, 2},
		"prompt_render_fingerprint": "fp1",
		"render_fingerprint":        "fp2",
		"kv_transfer_params":        map[string]any{"prompt_token_ids": []any{3}},
		"input_ids":                 []any{4, 5},
	}
	wantFields := []string{
		FieldPromptTokenIDs,
		FieldPromptRenderFingerprint,
		FieldRenderFingerprint,
		FieldKVTransferParams,
		FieldInputIDs,
	}
	require.Equal(t, wantFields, FindExternalTextFields(dirty))
	require.Len(t, dirty, 6, "FindExternalTextFields must not mutate the map")

	stripped := StripExternalTextFields(dirty)
	require.Equal(t, wantFields, stripped)
	require.Equal(t, map[string]any{"model": "m"}, dirty)
}

func TestNestedKVTransferParams(t *testing.T) {
	const kv = `{"do_remote_prefill":true}`
	tests := []struct {
		name string
		// body builds a new request body: Strip edits decoded objects in place.
		body      func() map[string]any
		wantNames []string
		wantBody  string
	}{
		{
			name: "decoded_vllm_xargs",
			body: func() map[string]any {
				return map[string]any{"model": "m", "vllm_xargs": map[string]any{"kv_transfer_params": map[string]any{"do_remote_prefill": true}, "k": "v"}}
			},
			wantNames: []string{"vllm_xargs.kv_transfer_params"},
			wantBody:  `{"model":"m","vllm_xargs":{"k":"v"}}`,
		},
		{
			name: "raw_vllm_xargs",
			body: func() map[string]any {
				return map[string]any{"model": "m", "vllm_xargs": json.RawMessage(`{"kv_transfer_params":` + kv + `,"k":"v"}`)}
			},
			wantNames: []string{"vllm_xargs.kv_transfer_params"},
			wantBody:  `{"model":"m","vllm_xargs":{"k":"v"}}`,
		},
		{
			name: "decoded_sampling_params_extra_args",
			body: func() map[string]any {
				return map[string]any{"token_ids": []any{1}, "sampling_params": map[string]any{
					"max_tokens": 4,
					"extra_args": map[string]any{"kv_transfer_params": map[string]any{"do_remote_decode": true}, "k": "v"},
				}}
			},
			wantNames: []string{"sampling_params.extra_args.kv_transfer_params"},
			wantBody:  `{"token_ids":[1],"sampling_params":{"max_tokens":4,"extra_args":{"k":"v"}}}`,
		},
		{
			name: "raw_sampling_params_extra_args",
			body: func() map[string]any {
				return map[string]any{"token_ids": []any{1}, "sampling_params": json.RawMessage(
					`{"max_tokens":4,"extra_args":{"kv_transfer_params":` + kv + `,"k":"v"}}`)}
			},
			wantNames: []string{"sampling_params.extra_args.kv_transfer_params"},
			wantBody:  `{"token_ids":[1],"sampling_params":{"max_tokens":4,"extra_args":{"k":"v"}}}`,
		},
		{
			name: "decoded_sampling_params_with_raw_vllm_xargs",
			body: func() map[string]any {
				return map[string]any{"sampling_params": map[string]any{
					"max_tokens": 4,
					"vllm_xargs": json.RawMessage(`{"kv_transfer_params":` + kv + `}`),
				}}
			},
			wantNames: []string{"sampling_params.vllm_xargs.kv_transfer_params"},
			wantBody:  `{"sampling_params":{"max_tokens":4,"vllm_xargs":{}}}`,
		},
		{
			name: "every_copy_and_the_top_level",
			body: func() map[string]any {
				return map[string]any{
					"kv_transfer_params": map[string]any{"do_remote_prefill": true},
					"vllm_xargs":         json.RawMessage(`{"kv_transfer_params":` + kv + `}`),
					"sampling_params": map[string]any{
						"extra_args": map[string]any{"kv_transfer_params": 1},
						"vllm_xargs": map[string]any{"kv_transfer_params": 2},
					},
				}
			},
			wantNames: []string{
				"kv_transfer_params",
				"vllm_xargs.kv_transfer_params",
				"sampling_params.extra_args.kv_transfer_params",
				"sampling_params.vllm_xargs.kv_transfer_params",
			},
			wantBody: `{"vllm_xargs":{},"sampling_params":{"extra_args":{},"vllm_xargs":{}}}`,
		},
		{
			name: "raw_vllm_xargs_with_an_escaped_key",
			body: func() map[string]any {
				return map[string]any{"model": "m", "vllm_xargs": json.RawMessage(`{"kv_transfer_\u0070arams":` + kv + `,"k":"v"}`)}
			},
			wantNames: []string{"vllm_xargs.kv_transfer_params"},
			wantBody:  `{"model":"m","vllm_xargs":{"k":"v"}}`,
		},
		{
			name: "raw_parent_with_an_escape_and_without_the_field",
			body: func() map[string]any {
				return map[string]any{"sampling_params": json.RawMessage(`{"stop":["\n\n"],"extra_args":{"k":"v"}}`)}
			},
			wantBody: `{"sampling_params":{"stop":["\n\n"],"extra_args":{"k":"v"}}}`,
		},
		{
			name: "parents_without_the_field",
			body: func() map[string]any {
				return map[string]any{
					"vllm_xargs":      json.RawMessage(`{"k":"v"}`),
					"sampling_params": map[string]any{"extra_args": map[string]any{"k": "v"}, "vllm_xargs": json.RawMessage(`{"k":"v"}`)},
				}
			},
			wantBody: `{"vllm_xargs":{"k":"v"},"sampling_params":{"extra_args":{"k":"v"},"vllm_xargs":{"k":"v"}}}`,
		},
		{
			name: "parents_that_are_not_objects",
			body: func() map[string]any {
				return map[string]any{
					"vllm_xargs":      json.RawMessage(`["kv_transfer_params"]`),
					"sampling_params": map[string]any{"extra_args": "kv_transfer_params", "vllm_xargs": json.RawMessage(`null`)},
				}
			},
			wantBody: `{"vllm_xargs":["kv_transfer_params"],"sampling_params":{"extra_args":"kv_transfer_params","vllm_xargs":null}}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			unchanged, err := json.Marshal(tc.body())
			require.NoError(t, err)

			found := tc.body()
			require.Equal(t, tc.wantNames, FindKVTransferParams(found))
			afterFind, err := json.Marshal(found)
			require.NoError(t, err)
			require.JSONEq(t, string(unchanged), string(afterFind), "FindKVTransferParams must not change the body")

			stripped := tc.body()
			require.Equal(t, tc.wantNames, StripKVTransferParams(stripped))
			afterStrip, err := json.Marshal(stripped)
			require.NoError(t, err)
			require.JSONEq(t, tc.wantBody, string(afterStrip))
			require.Nil(t, FindKVTransferParams(stripped))
		})
	}
}

func TestExternalTextFieldsIncludeNestedKVTransferParams(t *testing.T) {
	newBody := func() map[string]any {
		return map[string]any{
			"model":            "m",
			"prompt_token_ids": []any{1, 2},
			"vllm_xargs":       json.RawMessage(`{"kv_transfer_params":{"do_remote_prefill":true},"k":"v"}`),
		}
	}
	want := []string{FieldPromptTokenIDs, "vllm_xargs.kv_transfer_params"}

	require.Equal(t, want, FindExternalTextFields(newBody()))

	body := newBody()
	require.Equal(t, want, StripExternalTextFields(body))
	encoded, err := json.Marshal(body)
	require.NoError(t, err)
	require.JSONEq(t, `{"model":"m","vllm_xargs":{"k":"v"}}`, string(encoded))
}

// The EPP keeps nested objects raw, and most requests carry sampling_params or
// vllm_xargs without kv_transfer_params. The check of such a request must not
// decode those objects.
func TestFindKVTransferParamsDoesNotDecodeCleanRawObjects(t *testing.T) {
	clean := map[string]any{
		"token_ids":       json.RawMessage(`[1,2,3]`),
		"vllm_xargs":      json.RawMessage(`{"k":"v"}`),
		"sampling_params": json.RawMessage(`{"max_tokens":4,"extra_args":{"k":"v"},"vllm_xargs":{"k":"v"}}`),
	}

	require.Nil(t, FindKVTransferParams(clean))
	require.Zero(t, testing.AllocsPerRun(100, func() {
		_ = FindKVTransferParams(clean)
		_ = FindExternalTextFields(clean)
	}))
}
