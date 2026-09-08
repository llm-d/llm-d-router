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
	"reflect"
	"testing"
)

func TestAPIType_String(t *testing.T) {
	cases := map[APIType]string{
		APITypeChatCompletions: "chat_completions",
		APITypeCompletions:     "completions",
		APITypeResponses:       "responses",
		APITypeGenerate:        "generate",
		APIType(7):             "APIType(7)",
	}
	for apiType, want := range cases {
		if got := apiType.String(); got != want {
			t.Errorf("APIType(%d).String() = %q, want %q", int(apiType), got, want)
		}
	}
}

func TestAPIType_Path(t *testing.T) {
	cases := map[APIType]string{
		APITypeChatCompletions: PathChatCompletions,
		APITypeCompletions:     PathCompletions,
		APITypeResponses:       PathResponses,
		APITypeGenerate:        PathGenerate,
		APIType(7):             PathGenerate,
	}
	for apiType, want := range cases {
		if got := apiType.Path(); got != want {
			t.Errorf("APIType(%d).Path() = %q, want %q", int(apiType), got, want)
		}
	}
}

func TestDetectAPIType(t *testing.T) {
	tests := []struct {
		name string
		path string
		want APIType
	}{
		{name: "chat completions", path: PathChatCompletions, want: APITypeChatCompletions},
		{name: "completions", path: PathCompletions, want: APITypeCompletions},
		{name: "responses", path: PathResponses, want: APITypeResponses},
		{name: "messages shares chat completions fields", path: PathMessages, want: APITypeChatCompletions},
		{name: "generate", path: PathGenerate, want: APITypeGenerate},
		{name: "chat completions wins over the completions substring", path: "/prefix" + PathChatCompletions, want: APITypeChatCompletions},
		{name: "prefixed completions", path: "/prefix" + PathCompletions, want: APITypeCompletions},
		{name: "unknown path falls back to generate", path: "/v1/embeddings", want: APITypeGenerate},
		{name: "empty path falls back to generate", path: "", want: APITypeGenerate},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectAPIType(tt.path); got != tt.want {
				t.Errorf("DetectAPIType(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestAPIType_TokenLimitFields(t *testing.T) {
	cases := map[APIType][]string{
		APITypeChatCompletions: {FieldMaxTokens, FieldMaxCompletionTokens, FieldMinTokens},
		APITypeCompletions:     {FieldMaxTokens, FieldMaxCompletionTokens, FieldMinTokens},
		APITypeResponses:       {FieldMaxOutputTokens},
		APITypeGenerate:        {FieldMaxTokens, FieldMinTokens},
	}
	for apiType, want := range cases {
		if got := apiType.TokenLimitFields(); !reflect.DeepEqual(got, want) {
			t.Errorf("APIType(%d).TokenLimitFields() = %v, want %v", int(apiType), got, want)
		}
	}
}

func TestAPIType_TokenLimitMap(t *testing.T) {
	t.Run("non-generate returns the body itself", func(t *testing.T) {
		body := map[string]any{"model": "m"}
		got, created := APITypeChatCompletions.TokenLimitMap(body)
		if created {
			t.Error("created = true, want false")
		}
		if !reflect.DeepEqual(got, body) {
			t.Errorf("got %v, want the body %v", got, body)
		}
		if _, ok := body[FieldSamplingParams]; ok {
			t.Error("sampling_params was added to a non-generate body")
		}
	})

	t.Run("generate returns an existing sampling_params", func(t *testing.T) {
		sp := map[string]any{FieldMaxTokens: 100}
		body := map[string]any{FieldSamplingParams: sp}
		got, created := APITypeGenerate.TokenLimitMap(body)
		if created {
			t.Error("created = true, want false")
		}
		if !reflect.DeepEqual(got, sp) {
			t.Errorf("got %v, want %v", got, sp)
		}
	})

	t.Run("generate synthesizes an absent sampling_params", func(t *testing.T) {
		body := map[string]any{"model": "m"}
		got, created := APITypeGenerate.TokenLimitMap(body)
		if !created {
			t.Error("created = false, want true")
		}
		if len(got) != 0 {
			t.Errorf("got %v, want an empty map", got)
		}
		got[FieldMaxTokens] = 1
		if sp, ok := body[FieldSamplingParams].(map[string]any); !ok || sp[FieldMaxTokens] != 1 {
			t.Errorf("body sampling_params = %v, want the returned map", body[FieldSamplingParams])
		}
	})

	t.Run("generate replaces a non-map sampling_params", func(t *testing.T) {
		body := map[string]any{FieldSamplingParams: "not-a-map"}
		got, created := APITypeGenerate.TokenLimitMap(body)
		if !created {
			t.Error("created = false, want true")
		}
		if len(got) != 0 {
			t.Errorf("got %v, want an empty map", got)
		}
	})
}
