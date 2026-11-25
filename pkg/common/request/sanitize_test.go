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
