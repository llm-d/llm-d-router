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

func TestAPIType_StringAndPath(t *testing.T) {
	cases := map[APIType]struct{ name, path string }{
		APITypeChatCompletions: {"chat_completions", PathChatCompletions},
		APITypeCompletions:     {"completions", PathCompletions},
		APITypeResponses:       {"responses", PathResponses},
		APITypeGenerate:        {"generate", PathGenerate},
		APIType(7):             {"APIType(7)", PathGenerate},
	}
	for apiType, want := range cases {
		if got := apiType.String(); got != want.name {
			t.Errorf("APIType(%d).String() = %q, want %q", int(apiType), got, want.name)
		}
		if got := apiType.Path(); got != want.path {
			t.Errorf("APIType(%d).Path() = %q, want %q", int(apiType), got, want.path)
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
		{name: "prefixed chat completions", path: "/prefix" + PathChatCompletions, want: APITypeChatCompletions},
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

func TestAPIType_tokenLimitFields(t *testing.T) {
	cases := map[APIType][]string{
		APITypeChatCompletions: {FieldMaxTokens, FieldMaxCompletionTokens},
		APITypeCompletions:     {FieldMaxTokens},
		APITypeResponses:       {FieldMaxOutputTokens},
		APITypeGenerate:        {FieldMaxTokens},
		APIType(7):             {FieldMaxTokens},
	}
	for apiType, want := range cases {
		if got := apiType.tokenLimitFields(); !reflect.DeepEqual(got, want) {
			t.Errorf("APIType(%d).tokenLimitFields() = %v, want %v", int(apiType), got, want)
		}
	}
}

func TestAPIType_tokenLimitMap(t *testing.T) {
	t.Run("non-generate returns the body itself", func(t *testing.T) {
		body := map[string]any{"model": "m"}
		got := APITypeChatCompletions.tokenLimitMap(body)
		if !reflect.DeepEqual(got, body) {
			t.Errorf("got %v, want the body %v", got, body)
		}
		if _, ok := body[FieldSamplingParams]; ok {
			t.Error("sampling_params was added to a non-generate body")
		}
	})

	t.Run("an unrecognized API is treated as generate", func(t *testing.T) {
		body := map[string]any{"model": "m"}

		got := APIType(7).tokenLimitMap(body)

		got[FieldMaxTokens] = 1
		if sp, ok := body[FieldSamplingParams].(map[string]any); !ok || sp[FieldMaxTokens] != 1 {
			t.Errorf("body sampling_params = %v, want the returned map", body[FieldSamplingParams])
		}
	})

	// A generate body may share sampling_params with the body it was cloned from,
	// so the caller gets a copy the body owns and the original stays intact.
	t.Run("generate copies an existing sampling_params", func(t *testing.T) {
		sp := map[string]any{FieldMaxTokens: 100}
		body := map[string]any{FieldSamplingParams: sp}

		got := APITypeGenerate.tokenLimitMap(body)

		if !reflect.DeepEqual(got, sp) {
			t.Errorf("got %v, want the entries of %v", got, sp)
		}
		got[FieldMaxTokens] = 1
		if sp[FieldMaxTokens] != 100 {
			t.Errorf("caller's sampling_params was written through: %v", sp)
		}
		if body[FieldSamplingParams].(map[string]any)[FieldMaxTokens] != 1 {
			t.Errorf("body sampling_params = %v, want the returned map", body[FieldSamplingParams])
		}
	})

	for name, value := range map[string]any{"absent": nil, "not a map": "not-a-map"} {
		t.Run("generate replaces a sampling_params that is "+name, func(t *testing.T) {
			body := map[string]any{"model": "m"}
			if value != nil {
				body[FieldSamplingParams] = value
			}

			got := APITypeGenerate.tokenLimitMap(body)

			if len(got) != 0 {
				t.Errorf("got %v, want an empty map", got)
			}
			got[FieldMaxTokens] = 1
			if sp, ok := body[FieldSamplingParams].(map[string]any); !ok || sp[FieldMaxTokens] != 1 {
				t.Errorf("body sampling_params = %v, want the returned map", body[FieldSamplingParams])
			}
		})
	}
}
