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
	"net/http"
	"net/http/httptest"
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
		APITypeUnknown:         {"unknown", PathChatCompletions},
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
		{name: "trailing slash", path: PathResponses + "/", want: APITypeResponses},
		{name: "repeated slash", path: "//v1/chat/completions", want: APITypeChatCompletions},
		{name: "prefixed chat completions", path: "/prefix" + PathChatCompletions, want: APITypeChatCompletions},
		{name: "prefixed completions", path: "/prefix" + PathCompletions, want: APITypeCompletions},
		{name: "prefixed responses", path: "/prefix" + PathResponses, want: APITypeResponses},
		{name: "prefixed messages", path: "/prefix" + PathMessages, want: APITypeMessages},
		{name: "prefixed generate", path: "/prefix" + PathVLLMGenerate, want: APITypeVLLMGenerate},
		{name: "prefixed sglang generate", path: "/prefix" + PathSGLangGenerate, want: APITypeSGLangGenerate},
		{name: "segment ending in generate", path: "/regenerate", want: APITypeUnknown},
		{name: "sub-path ending in generate", path: PathResponses + "/generate", want: APITypeSGLangGenerate},
		{name: "chat completions render", path: "/v1/chat/completions/render", want: APITypeUnknown},
		{name: "messages count_tokens", path: "/v1/messages/count_tokens", want: APITypeUnknown},
		{name: "messages batches", path: "/v1/messages/batches", want: APITypeUnknown},
		{name: "responses input_tokens", path: "/v1/responses/input_tokens", want: APITypeUnknown},
		{name: "responses compact", path: "/v1/responses/compact", want: APITypeUnknown},
		{name: "conversations", path: "/v1/conversations", want: APITypeUnknown},
		{name: "embeddings", path: "/v1/embeddings", want: APITypeUnknown},
		{name: "empty path", path: "", want: APITypeUnknown},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectAPIType(tt.path); got != tt.want {
				t.Errorf("DetectAPIType(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestCanonicalAPIPath(t *testing.T) {
	tests := []struct {
		path   string
		want   string
		wantOK bool
	}{
		{path: PathChatCompletions, want: PathChatCompletions, wantOK: true},
		{path: PathCompletions, want: PathCompletions, wantOK: true},
		{path: PathResponses, want: PathResponses, wantOK: true},
		{path: PathMessages, want: PathMessages, wantOK: true},
		{path: PathVLLMGenerate, want: PathVLLMGenerate, wantOK: true},
		{path: PathSGLangGenerate, want: PathSGLangGenerate, wantOK: true},
		{path: PathChatCompletions + "/", want: PathChatCompletions, wantOK: true},
		{path: PathCompletions + "/", want: PathCompletions, wantOK: true},
		{path: PathResponses + "/", want: PathResponses, wantOK: true},
		{path: PathMessages + "/", want: PathMessages, wantOK: true},
		{path: PathVLLMGenerate + "/", want: PathVLLMGenerate, wantOK: true},
		{path: PathSGLangGenerate + "/", want: PathSGLangGenerate, wantOK: true},
		{path: "//v1/responses", want: PathResponses, wantOK: true},
		{path: "/v1//chat/completions", want: PathChatCompletions, wantOK: true},
		{path: "/v1/./responses", want: PathResponses, wantOK: true},
		{path: "/v1/files/../responses", want: PathResponses, wantOK: true},
		{path: "/v1/chat/completions/render", want: "/v1/chat/completions/render"},
		{path: "/v1/messages/count_tokens", want: "/v1/messages/count_tokens"},
		{path: "/v1/responses/input_tokens", want: "/v1/responses/input_tokens"},
		{path: "/v1/responses/resp_123", want: "/v1/responses/resp_123"},
		{path: "/v1/conversations", want: "/v1/conversations"},
		{path: "/v1/embeddings/", want: "/v1/embeddings"},
		{path: "/prefix" + PathChatCompletions, want: "/prefix" + PathChatCompletions},
		{path: "/V1/chat/completions", want: "/V1/chat/completions"},
		{path: "/", want: "/"},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got, ok := CanonicalAPIPath(tt.path)
			if got != tt.want || ok != tt.wantOK {
				t.Errorf("CanonicalAPIPath(%q) = (%q, %v), want (%q, %v)", tt.path, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}

func TestCanonicalizeAPIPath(t *testing.T) {
	tests := []struct {
		name        string
		target      string
		wantPath    string
		wantRawPath string
	}{
		{name: "trailing slash on an API path", target: PathResponses + "/", wantPath: PathResponses},
		{name: "repeated slash on an API path", target: "//v1/chat/completions?x=1", wantPath: PathChatCompletions},
		{name: "escaped slash on an API path", target: "/v1/messages%2F", wantPath: PathMessages},
		{name: "canonical API path", target: PathCompletions, wantPath: PathCompletions},
		{name: "sub-resource of an API path", target: "/v1/responses/compact/", wantPath: "/v1/responses/compact/"},
		{name: "unrelated path", target: "/v1/embeddings/", wantPath: "/v1/embeddings/"},
		{name: "unrelated escaped path", target: "/v1/files/a%2Fb", wantPath: "/v1/files/a/b", wantRawPath: "/v1/files/a%2Fb"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got *http.Request
			h := CanonicalizeAPIPath(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) { got = r }))
			req := httptest.NewRequest(http.MethodPost, tt.target, nil)
			origPath, origRawPath := req.URL.Path, req.URL.RawPath

			h.ServeHTTP(httptest.NewRecorder(), req)

			if got.URL.Path != tt.wantPath || got.URL.RawPath != tt.wantRawPath {
				t.Errorf("next saw (Path %q, RawPath %q), want (%q, %q)", got.URL.Path, got.URL.RawPath, tt.wantPath, tt.wantRawPath)
			}
			if req.URL.Path != origPath || req.URL.RawPath != origRawPath {
				t.Errorf("caller's request was modified to (Path %q, RawPath %q)", req.URL.Path, req.URL.RawPath)
			}
			if got.URL.RawQuery != req.URL.RawQuery {
				t.Errorf("query = %q, want %q", got.URL.RawQuery, req.URL.RawQuery)
			}
		})
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
