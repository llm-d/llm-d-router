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

package sglanghttp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/utils/ptr"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

var benchmarkSGLangParseResult *fwkrh.ParseResult

func makeSGLangTokenArrayBody(tokenCount int) []byte {
	tokens := strings.Repeat("12345,", tokenCount-1) + "12345"
	return []byte(`{"input_ids":[` + tokens + `],"sampling_params":{"max_new_tokens":1}}`)
}

func BenchmarkSGLangHTTPParser_ParseRequest(b *testing.B) {
	parser := NewSGLangHTTPParser()
	headers := map[string]string{":path": "/generate"}
	for _, tc := range []struct {
		name  string
		count int
	}{
		{"4K", 4 * 1024},
		{"32K", 32 * 1024},
		{"256K", 256 * 1024},
		{"1M", 1_000_000},
	} {
		body := makeSGLangTokenArrayBody(tc.count)
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(body)))
			for b.Loop() {
				result, err := parser.ParseRequest(context.Background(), body, headers)
				if err != nil {
					b.Fatal(err)
				}
				benchmarkSGLangParseResult = result
			}
		})
	}
}

func BenchmarkSGLangHTTPParser_ParseRequestFallback1M(b *testing.B) {
	parser := NewSGLangHTTPParser()
	headers := map[string]string{":path": "/generate"}
	tokens := strings.Repeat("12345,", 1_000_000-1) + "12345.0"
	body := []byte(`{"input_ids":[` + tokens + `],"sampling_params":{"max_new_tokens":1}}`)
	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	for b.Loop() {
		_, _ = parser.ParseRequest(context.Background(), body, headers)
	}
}

func TestNewSGLangHTTPParser(t *testing.T) {
	parser := NewSGLangHTTPParser()
	want := fwkplugin.TypedName{Type: SGLangHTTPParserType, Name: SGLangHTTPParserType}
	if parser.TypedName() != want {
		t.Errorf("TypedName() = %v, want %v", parser.TypedName(), want)
	}
}

func TestSGLangHTTPParser_ParseRequest(t *testing.T) {
	parser := NewSGLangHTTPParser()

	tests := []struct {
		name    string
		headers map[string]string
		body    map[string]any
		want    *fwkrh.InferenceRequestBody
		wantErr bool
	}{
		{
			name:    "basic input_ids",
			headers: map[string]string{":path": "/generate"},
			body:    map[string]any{"input_ids": []any{1, 2, 3}},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{TokenIDs: []uint32{1, 2, 3}},
			},
		},
		{
			name:    "extra_key mapped to CacheSalt",
			headers: map[string]string{":path": "/generate"},
			body:    map[string]any{"input_ids": []any{10, 20}, "extra_key": "salt-abc"},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{TokenIDs: []uint32{10, 20}, CacheSalt: "salt-abc"},
			},
		},
		{
			name:    "sampling_params max_new_tokens",
			headers: map[string]string{":path": "/generate"},
			body: map[string]any{
				"input_ids":       []any{1, 2, 3},
				"sampling_params": map[string]any{"max_new_tokens": 256},
			},
			want: &fwkrh.InferenceRequestBody{
				Generate:        &fwkrh.GenerateRequest{TokenIDs: []uint32{1, 2, 3}},
				MaxOutputTokens: ptr.To(int64(256)),
			},
		},
		{
			name:    "stream flag",
			headers: map[string]string{":path": "/generate"},
			body:    map[string]any{"input_ids": []any{1, 2, 3}, "stream": true},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{TokenIDs: []uint32{1, 2, 3}},
				Stream:   true,
			},
		},
		{
			name:    "unrelated fields are left for SGLang to validate",
			headers: map[string]string{":path": "/generate"},
			body: map[string]any{
				"text":      "ignored by the parser",
				"input_ids": []int{7, 8},
			},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{TokenIDs: []uint32{7, 8}},
			},
		},
		{
			name:    "per-prompt extra_key list is rejected",
			headers: map[string]string{":path": "/generate"},
			body: map[string]any{
				"input_ids": []int{1, 2},
				"extra_key": []string{"tenant-a", "tenant-b"},
			},
			wantErr: true,
		},
		{
			name:    "x-original-path header",
			headers: map[string]string{"x-original-path": "/generate"},
			body:    map[string]any{"input_ids": []any{5, 6, 7}},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{TokenIDs: []uint32{5, 6, 7}},
			},
		},
		{
			name:    "missing input_ids",
			headers: map[string]string{":path": "/generate"},
			body:    map[string]any{"sampling_params": map[string]any{"temperature": 0.8}},
			wantErr: true,
		},
		{
			name:    "empty input_ids",
			headers: map[string]string{":path": "/generate"},
			body:    map[string]any{"input_ids": []any{}},
			wantErr: true,
		},
		{
			name:    "unsupported path",
			headers: map[string]string{":path": "/v1/completions"},
			body:    map[string]any{"input_ids": []any{1, 2, 3}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, err := json.Marshal(tt.body)
			if err != nil {
				t.Fatalf("marshal body: %v", err)
			}
			got, err := parser.ParseRequest(context.Background(), bodyBytes, tt.headers)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			tt.want.Payload = fwkrh.RawPayload(bodyBytes)
			if got.SkipResponseProcessing {
				t.Errorf("SkipResponseProcessing = true, want false")
			}
			if diff := cmp.Diff(tt.want, got.Body); diff != "" {
				t.Errorf("Body mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSGLangHTTPParser_ParseRequest_ErrorPaths(t *testing.T) {
	parser := NewSGLangHTTPParser()
	headers := map[string]string{":path": "/generate"}

	tests := []struct {
		name        string
		body        string
		errContains string
	}{
		{
			name:        "negative token id",
			body:        `{"input_ids":[1,2,-1]}`,
			errContains: "input_ids must be an array of uint32 integers",
		},
		{
			name:        "fractional token id",
			body:        `{"input_ids":[1,2.5]}`,
			errContains: "input_ids must be an array of uint32 integers",
		},
		{
			name:        "overflowing token id",
			body:        `{"input_ids":[4294967296]}`,
			errContains: "input_ids must be an array of uint32 integers",
		},
		{
			name:        "empty input_ids",
			body:        `{"input_ids":[]}`,
			errContains: "input_ids cannot be empty",
		},
		{
			name:        "input embeds without input ids",
			body:        `{"input_embeds":[[0.1,0.2]]}`,
			errContains: "input_ids must be provided",
		},
		{
			name:        "text without input ids",
			body:        `{"text":"hello"}`,
			errContains: "input_ids must be provided",
		},
		{
			name:        "batched input_ids are rejected",
			body:        `{"input_ids":[[1,2],[3,4]]}`,
			errContains: "input_ids must be an array of uint32 integers",
		},
		{
			name:        "per-prompt cache salt unsupported",
			body:        `{"input_ids":[1,2],"extra_key":["a","b"]}`,
			errContains: "extra_key must be a string",
		},
		{
			name:        "unsupported path",
			body:        `{"input_ids":[1]}`,
			errContains: "unsupported path",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hdrs := headers
			if tt.name == "unsupported path" {
				hdrs = map[string]string{":path": "/v1/completions"}
			}
			_, err := parser.ParseRequest(context.Background(), []byte(tt.body), hdrs)
			if err == nil {
				t.Fatalf("ParseRequest() error = nil, want error containing %q", tt.errContains)
			}
			if !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("ParseRequest() error = %q, want substring %q", err.Error(), tt.errContains)
			}
		})
	}
}

func TestSGLangHTTPParser_ParseResponse(t *testing.T) {
	parser := NewSGLangHTTPParser()

	tests := []struct {
		name    string
		body    string
		headers map[string]string
		want    *fwkrh.Usage
		wantErr bool
	}{
		{
			name: "non-streaming response",
			body: `{"text":"hello","meta_info":{"prompt_tokens":10,"completion_tokens":5}}`,
			want: &fwkrh.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
		{
			name: "non-streaming with additional meta_info fields",
			body: `{"text":"hi","output_ids":[1,2],"meta_info":{"id":"req-1","prompt_tokens":8,"completion_tokens":3,"cached_tokens":2}}`,
			want: &fwkrh.Usage{
				PromptTokens:     8,
				CompletionTokens: 3,
				TotalTokens:      11,
				PromptTokenDetails: &fwkrh.PromptTokenDetails{
					CachedTokens: 2,
				},
			},
		},
		{
			name:    "streaming — only final chunk (finish_reason set) is used",
			headers: map[string]string{"Content-Type": "text/event-stream"},
			body: "data: {\"text\":\"he\",\"meta_info\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n" +
				"data: {\"text\":\"hello\",\"meta_info\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"finish_reason\":{\"type\":\"stop\"}}}\n\n" +
				"data: [DONE]\n\n",
			want: &fwkrh.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
		{
			name:    "streaming — intermediate chunk without finish_reason returns nil",
			headers: map[string]string{"Content-Type": "text/event-stream"},
			body:    "data: {\"text\":\"he\",\"meta_info\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
			want:    nil,
		},
		{
			name:    "streaming detected from content type",
			headers: map[string]string{"Content-Type": "text/event-stream; charset=utf-8"},
			body: "event: generation\n" +
				"data: {\"text\":\"hello\",\"meta_info\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"finish_reason\":{\"type\":\"stop\"}}}\n\n",
			want: &fwkrh.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
		{
			name:    "streaming — [DONE] only",
			headers: map[string]string{"Content-Type": "text/event-stream"},
			body:    "data: [DONE]\n\n",
			want:    nil,
		},
		{
			name: "empty body",
			body: "",
			want: nil,
		},
		{
			name: "explicit zero usage is preserved",
			body: `{"text":"","meta_info":{"prompt_tokens":0,"completion_tokens":0}}`,
			want: &fwkrh.Usage{},
		},
		{
			name:    "Invalid JSON returns error",
			body:    `{"meta_info":`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseResponse(context.Background(), []byte(tt.body), tt.headers, true)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseResponse() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			var gotUsage *fwkrh.Usage
			if got != nil {
				gotUsage = got.Usage
			}
			if diff := cmp.Diff(tt.want, gotUsage); diff != "" {
				t.Errorf("Usage mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSGLangHTTPParser_RejectsPathWithoutSegmentBoundary(t *testing.T) {
	parser := NewSGLangHTTPParser()
	for _, path := range []string{"/regenerate", "/health_generate", "/vertex_generate", "/inference/v1/generate", "/prefix/generate"} {
		t.Run(path, func(t *testing.T) {
			_, err := parser.ParseRequest(
				context.Background(),
				[]byte(`{"input_ids":[1]}`),
				map[string]string{":path": path},
			)
			if err == nil {
				t.Fatal("ParseRequest() error = nil, want unsupported path")
			}
		})
	}
}

func TestSGLangHTTPParser_Claims(t *testing.T) {
	parser := NewSGLangHTTPParser()
	got := parser.Claims()
	want := fwkrh.Claims{
		Paths:     []string{generatePathSuffix},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Claims() mismatch (-want +got):\n%s", diff)
	}
}
