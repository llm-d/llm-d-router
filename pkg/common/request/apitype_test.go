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
		APITypeVLLMGenerate:    {"vllm_generate", PathVLLMGenerate},
		APITypeSGLangGenerate:  {"sglang_generate", PathSGLangGenerate},
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
		{name: "generate", path: PathVLLMGenerate, want: APITypeVLLMGenerate},
		{name: "sglang generate", path: PathSGLangGenerate, want: APITypeSGLangGenerate},
		{name: "prefixed chat completions", path: "/prefix" + PathChatCompletions, want: APITypeChatCompletions},
		{name: "prefixed completions", path: "/prefix" + PathCompletions, want: APITypeCompletions},
		{name: "prefixed responses", path: "/prefix" + PathResponses, want: APITypeResponses},
		{name: "prefixed messages", path: "/prefix" + PathMessages, want: APITypeMessages},
		{name: "prefixed generate", path: "/prefix" + PathVLLMGenerate, want: APITypeVLLMGenerate},
		{name: "prefixed sglang generate", path: "/prefix" + PathSGLangGenerate, want: APITypeSGLangGenerate},
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

func TestDetectAPISurface(t *testing.T) {
	tests := []struct {
		name, path, want string
	}{
		{"chat completions", PathChatCompletions, "chat_completions"},
		{"completions", PathCompletions, "completions"},
		{"responses", PathResponses, "responses"},
		{"responses trailing slash and query", PathResponses + "/?stream=true", "responses"},
		{"prefixed responses", "/provider/model" + PathResponses, "responses"},
		{"responses item read", PathResponses + "/resp_123", "unknown"},
		{"responses item read trailing slash and query", PathResponses + "/resp_123/?include=output", "unknown"},
		{"prefixed responses item read", "/provider/model" + PathResponses + "/resp_123", "unknown"},
		{"messages", PathMessages, "messages"},
		{"vllm generate", PathVLLMGenerate, "vllm_generate"},
		{"sglang generate", PathSGLangGenerate, "sglang_generate"},
		{"provider prefix", "/provider/model" + PathMessages, "messages"},
		{"unversioned provider route", "/provider/messages", "messages"},
		{"trailing slash and query", PathMessages + "/?beta=true", "messages"},
		{"unknown", "/v1/embeddings", "unknown"},
		{"missing", "", "unknown"},
		{"token counting", PathMessages + "/count_tokens", "unknown"},
		{"render", PathMessages + "/render", "unknown"},
		{"extra path segment", PathChatCompletions + "/extra", "unknown"},
		{"substring", "/v1/messages-extra", "unknown"},
		{"missing segment boundary", "/provider/notmessages", "unknown"},
		{"query is not path", "/health?target=" + PathMessages, "unknown"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectAPISurface(tt.path).String(); got != tt.want {
				t.Errorf("DetectAPISurface(%q) = %q, want %q", tt.path, got, tt.want)
			}
		})
	}
}

func TestAPISurfaceString(t *testing.T) {
	var zero APISurface
	for _, surface := range []APISurface{zero, APISurfaceUnknown, APISurface(-1), APISurface(1000)} {
		if got := surface.String(); got != "unknown" {
			t.Errorf("APISurface(%d).String() = %q, want unknown", surface, got)
		}
	}
	for _, apiType := range []APIType{
		APITypeChatCompletions, APITypeCompletions, APITypeResponses,
		APITypeVLLMGenerate, APITypeSGLangGenerate, APITypeMessages,
	} {
		if got := apiType.Surface().String(); got != apiType.String() {
			t.Errorf("%v surface = %q, want %q", apiType, got, apiType.String())
		}
	}
	for _, apiType := range []APIType{APIType(-1), APIType(1000)} {
		if got := apiType.Surface(); got != APISurfaceUnknown {
			t.Errorf("%v surface = %v, want unknown", apiType, got)
		}
	}
}

func TestAPIType_tokenLimitFields(t *testing.T) {
	cases := map[APIType][]string{
		APITypeChatCompletions: {FieldMaxTokens, FieldMaxCompletionTokens},
		APITypeCompletions:     {FieldMaxTokens},
		APITypeResponses:       {FieldMaxOutputTokens},
		APITypeVLLMGenerate:    {FieldMaxTokens},
		APITypeMessages:        {FieldMaxTokens},
		APIType(7):             {FieldMaxTokens, FieldMaxCompletionTokens},
	}
	for apiType, want := range cases {
		if got := apiType.tokenLimitFields(); !reflect.DeepEqual(got, want) {
			t.Errorf("APIType(%d).tokenLimitFields() = %v, want %v", int(apiType), got, want)
		}
	}
}
