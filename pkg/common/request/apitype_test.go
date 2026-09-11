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
		APITypeMessages:        {"messages", PathMessages},
		APIType(7):             {"APIType(7)", PathChatCompletions},
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
		{name: "messages", path: PathMessages, want: APITypeMessages},
		{name: "generate", path: PathGenerate, want: APITypeGenerate},
		{name: "prefixed chat completions", path: "/prefix" + PathChatCompletions, want: APITypeChatCompletions},
		{name: "prefixed completions", path: "/prefix" + PathCompletions, want: APITypeCompletions},
		{name: "prefixed messages", path: "/prefix" + PathMessages, want: APITypeMessages},
		{name: "prefixed generate", path: "/prefix" + PathGenerate, want: APITypeGenerate},
		{name: "unknown path falls back to chat completions", path: "/v1/embeddings", want: APITypeChatCompletions},
		{name: "empty path falls back to chat completions", path: "", want: APITypeChatCompletions},
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
		APITypeMessages:        {FieldMaxTokens},
		APIType(7):             {FieldMaxTokens, FieldMaxCompletionTokens},
	}
	for apiType, want := range cases {
		if got := apiType.tokenLimitFields(); !reflect.DeepEqual(got, want) {
			t.Errorf("APIType(%d).tokenLimitFields() = %v, want %v", int(apiType), got, want)
		}
	}
}
