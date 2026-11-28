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

/*
Copyright 2025 The llm-d Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package openai

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

// Regression test for #981: extractUsage must not panic when an upstream
// response uses non-conforming types for usage fields. Before the fix,
// each vector below crashed the goroutine processing the response via an
// unchecked type assertion.
func TestExtractUsage_MalformedFields_NoPanic(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "usage_field_is_string",
			body: `{"object":"chat.completion","usage":"oops"}`,
		},
		{
			name: "prompt_tokens_details_is_number",
			body: `{"object":"chat.completion","usage":{"prompt_tokens_details":0,"prompt_tokens":7}}`,
		},
		{
			name: "prompt_tokens_is_string",
			body: `{"object":"chat.completion","usage":{"prompt_tokens":"1024"}}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("extractUsage panicked on %q: %v", tc.body, r)
				}
			}()
			_, _ = extractUsage([]byte(tc.body))
		})
	}
}

// Some non-conforming providers emit numeric usage fields as JSON strings;
// toInt coerces these rather than panicking.
func TestExtractUsage_StringCoercion(t *testing.T) {
	body := `{"object":"chat.completion","usage":{"prompt_tokens":"1024","completion_tokens":"7","total_tokens":"1031"}}`
	u, err := extractUsage([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u == nil {
		t.Fatal("usage is nil")
	}
	if u.PromptTokens != 1024 {
		t.Errorf("PromptTokens = %d, want 1024", u.PromptTokens)
	}
	if u.CompletionTokens != 7 {
		t.Errorf("CompletionTokens = %d, want 7", u.CompletionTokens)
	}
	if u.TotalTokens != 1031 {
		t.Errorf("TotalTokens = %d, want 1031", u.TotalTokens)
	}
}

// The SSE spec allows "data:" with no space after the colon; a stream chunk
// in that form reads the same as the spaced wire form.
func TestExtractUsageStreaming_NoSpaceDataLine(t *testing.T) {
	body := "data:{\"id\":\"1\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"total_tokens\":9}}\n\n"
	u := extractUsageStreaming([]byte(body))
	if u == nil {
		t.Fatal("usage is nil")
	}
	if u.PromptTokens != 7 {
		t.Errorf("PromptTokens = %d, want 7", u.PromptTokens)
	}
	if u.TotalTokens != 9 {
		t.Errorf("TotalTokens = %d, want 9", u.TotalTokens)
	}
}

func TestExtractUsageStreaming_UsageObjectGuard(t *testing.T) {
	tests := []struct {
		name  string
		chunk string
		want  *request.Usage
	}{
		{
			name:  "vllm_usage_null",
			chunk: "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n",
		},
		{
			name:  "python_spaced_usage_null",
			chunk: "data: {\"id\": \"chatcmpl-1\", \"choices\": [{\"delta\": {\"content\": \"hi\"}}], \"usage\": null}\n\n",
		},
		{
			name:  "whitespace_usage_null",
			chunk: "data: {\"id\":\"chatcmpl-1\",\"usage\" :\t null}\n\n",
		},
		{
			name:  "responses_created_null_usage",
			chunk: "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"usage\":null}}\n\n",
		},
		{
			name:  "responses_in_progress_null_usage",
			chunk: "event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"resp_1\",\"usage\":null}}\n\n",
		},
		{
			name:  "escaped_json_in_content",
			chunk: "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"\\\"usage\\\":{\\\"prompt_tokens\\\":1}\"}}]}\n\n",
		},
		{
			name:  "speech_audio_done_null_usage",
			chunk: "event: speech.audio.done\ndata: {\"type\":\"speech.audio.done\",\"usage\":null}\n\n",
		},
		{
			name:  "speech_audio_done_null_with_nested_object",
			chunk: "event: speech.audio.done\ndata: {\"type\":\"speech.audio.done\",\"usage\":null,\"meta\":{\"usage\":{\"prompt_tokens\":1}}}\n\n",
		},
		{
			name:  "spaced_usage_object",
			chunk: "data: {\"id\": \"chatcmpl-1\", \"choices\": [], \"usage\" : { \"prompt_tokens\": 7, \"completion_tokens\": 10, \"total_tokens\": 17 }}\n\n",
			want:  &request.Usage{PromptTokens: 7, CompletionTokens: 10, TotalTokens: 17},
		},
		{
			name:  "multi_key_null_then_real_usage",
			chunk: "data: {\"meta\":{\"usage\":null},\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":10,\"total_tokens\":17}}\n\n",
			want:  &request.Usage{PromptTokens: 7, CompletionTokens: 10, TotalTokens: 17},
		},
		{
			name:  "top_level_null_then_responses_completed",
			chunk: "event: response.completed\ndata: {\"usage\":null,\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":31,\"output_tokens\":3,\"total_tokens\":34}}}\n\n",
			want:  &request.Usage{PromptTokens: 31, CompletionTokens: 3, TotalTokens: 34},
		},
		{
			name: "multi_line_null_then_real_usage",
			chunk: "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n" +
				"data: {\"id\":\"chatcmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":10,\"total_tokens\":17}}\n\n",
			want: &request.Usage{PromptTokens: 7, CompletionTokens: 10, TotalTokens: 17},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := extractUsageStreaming([]byte(tc.chunk))
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("extractUsageStreaming() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtractRawUsage(t *testing.T) {
	tests := []struct {
		name string
		raw  json.RawMessage
		want *request.Usage
	}{
		{name: "nil_raw", raw: nil},
		{name: "empty_raw", raw: json.RawMessage{}},
		{name: "null_raw", raw: json.RawMessage("null")},
		{name: "empty_object", raw: json.RawMessage("{}"), want: &request.Usage{}},
		{name: "string_value", raw: json.RawMessage(`"oops"`)},
		{name: "number_value", raw: json.RawMessage("123")},
		{name: "array_value", raw: json.RawMessage("[]")},
		{name: "invalid_json", raw: json.RawMessage("{bad")},
		{
			name: "string_coerced_counts",
			raw:  json.RawMessage(`{"input_tokens":"1024","output_tokens":"7","total_tokens":"1031"}`),
			want: &request.Usage{PromptTokens: 1024, CompletionTokens: 7, TotalTokens: 1031},
		},
		{
			name: "malformed_input_tokens_details",
			raw:  json.RawMessage(`{"input_tokens_details":0,"input_tokens":7}`),
			want: &request.Usage{PromptTokens: 7},
		},
		{
			name: "cached_tokens",
			raw:  json.RawMessage(`{"input_tokens":31,"output_tokens":3,"total_tokens":34,"input_tokens_details":{"cached_tokens":16}}`),
			want: &request.Usage{
				PromptTokens:       31,
				CompletionTokens:   3,
				TotalTokens:        34,
				PromptTokenDetails: &request.PromptTokenDetails{CachedTokens: 16},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := extractRawUsage(tc.raw)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("extractRawUsage() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtractUsageStreaming_NullUsageZeroAllocs(t *testing.T) {
	cases := []struct {
		name  string
		chunk []byte
	}{
		{
			name:  "vllm_usage_null",
			chunk: []byte("data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}],\"usage\":null}\n\n"),
		},
		{
			name:  "python_spaced_usage_null",
			chunk: []byte("data: {\"id\": \"chatcmpl-1\", \"choices\": [{\"delta\": {\"content\": \"hello\"}}], \"usage\": null}\n\n"),
		},
		{
			name:  "responses_early_null_usage",
			chunk: []byte("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"usage\":null}}\n\n"),
		},
		{
			name:  "plain_content_usage",
			chunk: []byte("data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"usage\"}}]}\n\n"),
		},
		{
			name:  "escaped_json_in_content",
			chunk: []byte("data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"\\\"usage\\\":{\\\"prompt_tokens\\\":1}\"}}]}\n\n"),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			allocs := testing.AllocsPerRun(100, func() {
				if u := extractUsageStreaming(tc.chunk); u != nil {
					t.Fatalf("unexpected usage: %+v", u)
				}
			})
			if allocs != 0 {
				t.Errorf("extractUsageStreaming allocated %.1f times per call, want 0", allocs)
			}
		})
	}

	t.Run("parse_stream_response_matches_without_usage", func(t *testing.T) {
		parser := NewOpenAIParser()
		withoutUsage := []byte(`data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hello"}}]}`)
		withNullUsage := []byte(`data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hello"}}],"usage":null}`)

		baseAllocs := testing.AllocsPerRun(100, func() {
			if _, err := parser.parseStreamResponse(withoutUsage); err != nil {
				t.Fatal(err)
			}
		})
		nullAllocs := testing.AllocsPerRun(100, func() {
			if _, err := parser.parseStreamResponse(withNullUsage); err != nil {
				t.Fatal(err)
			}
		})
		if nullAllocs != baseAllocs {
			t.Errorf("parseStreamResponse on usage:null allocated %.1f times, want %.1f (same as without_usage)", nullAllocs, baseAllocs)
		}
	})
}

func TestExtractRawUsage_DirectDecodeAllocs(t *testing.T) {
	raw := json.RawMessage(`{"input_tokens":512,"output_tokens":128,"total_tokens":640,"input_tokens_details":{"cached_tokens":256}}`)

	// Direct decode stays under 25 allocs; a json.Marshal round-trip exceeds 28 allocs.
	const maxDirectDecodeAllocs = 25
	allocs := testing.AllocsPerRun(100, func() {
		if u := extractRawUsage(raw); u == nil || u.PromptTokens != 512 {
			t.Fatalf("unexpected usage: %+v", u)
		}
	})
	if allocs > maxDirectDecodeAllocs {
		t.Errorf("extractRawUsage allocated %.1f times, ceiling %d", allocs, maxDirectDecodeAllocs)
	}
}
