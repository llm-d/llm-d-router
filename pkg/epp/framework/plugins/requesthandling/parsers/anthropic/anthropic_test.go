/*
Copyright 2025 The Kubernetes Authors.

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

package anthropic

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func TestNewAnthropicParser(t *testing.T) {
	parser := NewAnthropicParser()

	expectedName := fwkplugin.TypedName{
		Type: AnthropicParserType,
		Name: AnthropicParserType,
	}

	if diff := cmp.Diff(expectedName, parser.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}
}

func TestAnthropicParser_ParseRequest(t *testing.T) {
	parser := NewAnthropicParser()

	tests := []struct {
		name    string
		headers map[string]string
		body    map[string]any
		want    *fwkrh.InferenceRequestBody
		wantErr bool
	}{
		{
			name:    "simple text messages",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"messages": []any{
					map[string]any{"role": "user", "content": "Hello, Claude"},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hello, Claude"}},
					},
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"messages": []any{
						map[string]any{"role": "user", "content": "Hello, Claude"},
					},
				},
			},
		},
		{
			name:    "structured content blocks",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{"type": "text", "text": "Describe this image"},
						},
					},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{
							Structured: []fwkrh.AnthropicContentBlock{
								{Type: "text", Text: "Describe this image"},
							},
						}},
					},
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"messages": []any{
						map[string]any{
							"role": "user",
							"content": []any{
								map[string]any{"type": "text", "text": "Describe this image"},
							},
						},
					},
				},
			},
		},
		{
			name:    "system prompt as string",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"system":     "You are a helpful assistant.",
				"messages": []any{
					map[string]any{"role": "user", "content": "Hello"},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					System: fwkrh.AnthropicContent{Raw: "You are a helpful assistant."},
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hello"}},
					},
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"system":     "You are a helpful assistant.",
					"messages": []any{
						map[string]any{"role": "user", "content": "Hello"},
					},
				},
			},
		},
		{
			name:    "system prompt as content blocks",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"system": []any{
					map[string]any{"type": "text", "text": "You are a helpful assistant."},
				},
				"messages": []any{
					map[string]any{"role": "user", "content": "Hello"},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					System: fwkrh.AnthropicContent{
						Structured: []fwkrh.AnthropicContentBlock{
							{Type: "text", Text: "You are a helpful assistant."},
						},
					},
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hello"}},
					},
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"system": []any{
						map[string]any{"type": "text", "text": "You are a helpful assistant."},
					},
					"messages": []any{
						map[string]any{"role": "user", "content": "Hello"},
					},
				},
			},
		},
		{
			name:    "message with image content",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "image",
								"source": map[string]any{
									"type":       "base64",
									"media_type": "image/png",
									"data":       "iVBORw0KGgo=",
								},
							},
							map[string]any{"type": "text", "text": "What is in this image?"},
						},
					},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{
							Structured: []fwkrh.AnthropicContentBlock{
								{
									Type: "image",
									Source: &fwkrh.AnthropicImageSource{
										Type:      "base64",
										MediaType: "image/png",
										Data:      "iVBORw0KGgo=",
									},
								},
								{Type: "text", Text: "What is in this image?"},
							},
						}},
					},
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"messages": []any{
						map[string]any{
							"role": "user",
							"content": []any{
								map[string]any{
									"type": "image",
									"source": map[string]any{
										"type":       "base64",
										"media_type": "image/png",
										"data":       "iVBORw0KGgo=",
									},
								},
								map[string]any{"type": "text", "text": "What is in this image?"},
							},
						},
					},
				},
			},
		},
		{
			name:    "request with tools",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"tools": []any{
					map[string]any{
						"name":        "get_weather",
						"description": "Get the weather",
					},
				},
				"messages": []any{
					map[string]any{"role": "user", "content": "What's the weather?"},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					Tools: []fwkrh.AnthropicTool{
						{
							Name:        "get_weather",
							Description: "Get the weather",
						},
					},
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{Raw: "What's the weather?"}},
					},
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"tools": []any{
						map[string]any{
							"name":        "get_weather",
							"description": "Get the weather",
						},
					},
					"messages": []any{
						map[string]any{"role": "user", "content": "What's the weather?"},
					},
				},
			},
		},
		{
			name:    "request with stream flag",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"stream":     true,
				"messages": []any{
					map[string]any{"role": "user", "content": "Hello"},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hello"}},
					},
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"stream":     true,
					"messages": []any{
						map[string]any{"role": "user", "content": "Hello"},
					},
				},
				Stream: true,
			},
		},
		{
			name:    "request with cache_salt",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"cache_salt": "test-salt-123",
				"messages": []any{
					map[string]any{"role": "user", "content": "Hello"},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hello"}},
					},
					CacheSalt: "test-salt-123",
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"cache_salt": "test-salt-123",
					"messages": []any{
						map[string]any{"role": "user", "content": "Hello"},
					},
				},
			},
		},
		{
			name:    "path from x-original-path header",
			headers: map[string]string{"x-original-path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"messages": []any{
					map[string]any{"role": "user", "content": "Hello"},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				MaxOutputTokens: ptr.To(int64(1024)),
				Messages: &fwkrh.MessagesRequest{
					Messages: []fwkrh.AnthropicMessage{
						{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hello"}},
					},
				},
				Payload: fwkrh.PayloadMap{
					"model":      "claude-sonnet-4-6",
					"max_tokens": json.Number("1024"),
					"messages": []any{
						map[string]any{"role": "user", "content": "Hello"},
					},
				},
			},
		},
		{
			name:    "empty messages array",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"messages":   []any{},
			},
			wantErr: true,
		},
		{
			name:    "missing messages field",
			headers: map[string]string{":path": "/v1/messages"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
			},
			wantErr: true,
		},
		{
			name:    "nil body",
			headers: map[string]string{":path": "/v1/messages"},
			body:    nil,
			wantErr: true,
		},
		{
			name:    "unsupported path",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"messages": []any{
					map[string]any{"role": "user", "content": "Hello"},
				},
			},
			wantErr: true,
		},
		{
			name:    "no path header",
			headers: map[string]string{},
			body: map[string]any{
				"model":      "claude-sonnet-4-6",
				"max_tokens": float64(1024),
				"messages": []any{
					map[string]any{"role": "user", "content": "Hello"},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, err := json.Marshal(tt.body)
			if err != nil {
				t.Fatalf("Invalid tt.body %v: cannot convert to bytes", tt.body)
			}
			got, err := parser.ParseRequest(context.Background(), bodyBytes, tt.headers)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
			}

			if tt.wantErr {
				return
			}

			if got.SkipResponseProcessing != false {
				t.Errorf("ParseRequest() got.SkipResponseProcessing = %v, want false", got.SkipResponseProcessing)
			}

			// Model is extracted from the request body's "model" field.
			tt.want.Model, _ = tt.body["model"].(string)

			if diff := cmp.Diff(tt.want, got.Body); diff != "" {
				t.Errorf("ParseRequest() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAnthropicParser_ParseResponse(t *testing.T) {
	parser := NewAnthropicParser()

	tests := []struct {
		name    string
		body    []byte
		want    *fwkrh.ParsedResponse
		wantErr bool
	}{
		{
			name: "standard usage",
			body: []byte(`{
				"id": "msg_123",
				"type": "message",
				"role": "assistant",
				"content": [{"type": "text", "text": "Hello"}],
				"usage": {
					"input_tokens": 25,
					"output_tokens": 15
				}
			}`),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     25,
					CompletionTokens: 15,
					TotalTokens:      40,
				},
			},
		},
		{
			name: "usage with cache tokens",
			body: []byte(`{
				"id": "msg_123",
				"type": "message",
				"usage": {
					"input_tokens": 100,
					"output_tokens": 50,
					"cache_read_input_tokens": 80
				}
			}`),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     100,
					CompletionTokens: 50,
					TotalTokens:      150,
					PromptTokenDetails: &fwkrh.PromptTokenDetails{
						CachedTokens: 80,
					},
				},
			},
		},
		{
			name: "missing usage field",
			body: []byte(`{"id": "msg_123", "type": "message"}`),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:    "invalid JSON",
			body:    []byte(`{malformed`),
			wantErr: true,
		},
		{
			name: "empty body",
			body: []byte{},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseResponse(context.Background(), tt.body, map[string]string{}, false)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseResponse() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAnthropicParser_ParseResponse_Streaming(t *testing.T) {
	parser := NewAnthropicParser()

	tests := []struct {
		name  string
		chunk []byte
		want  *fwkrh.ParsedResponse
	}{
		{
			name:  "message_start with input tokens",
			chunk: []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"usage\":{\"input_tokens\":25}}}"),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens: 25,
					TotalTokens:  25,
				},
			},
		},
		{
			name:  "message_delta with output tokens",
			chunk: []byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":15}}"),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					CompletionTokens: 15,
					TotalTokens:      15,
				},
			},
		},
		{
			name: "full stream with both events",
			chunk: []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25}}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n" +
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15}}"),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     25,
					CompletionTokens: 15,
					TotalTokens:      40,
				},
			},
		},
		{
			name: "stream with cache tokens",
			chunk: []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":80}}}\n\n" +
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":50}}"),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     100,
					CompletionTokens: 50,
					TotalTokens:      150,
					PromptTokenDetails: &fwkrh.PromptTokenDetails{
						CachedTokens: 80,
					},
				},
			},
		},
		{
			name:  "content delta without usage",
			chunk: []byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}"),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:  "content delta with usage text but no usage object",
			chunk: []byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"usage\"}}"),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:  "message_stop without usage",
			chunk: []byte("event: message_stop\ndata: {\"type\":\"message_stop\"}"),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseResponse(context.Background(), tt.chunk, map[string]string{contentType: eventStreamType}, true)
			if err != nil {
				t.Fatalf("ParseResponse() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func BenchmarkAnthropicParser_ParseResponse_Streaming(b *testing.B) {
	parser := NewAnthropicParser()
	tests := []struct {
		name  string
		chunk []byte
	}{
		{
			name:  "without_usage",
			chunk: []byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}"),
		},
		{
			name:  "with_usage",
			chunk: []byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15}}"),
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := parser.parseStreamResponse(tt.chunk); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestAnthropicParser_Claims(t *testing.T) {
	parser := NewAnthropicParser()
	got := parser.Claims()
	want := fwkrh.Claims{
		Paths:     []string{messagesAPI, countTokensAPI},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Claims() mismatch (-want +got):\n%s", diff)
	}
}

func TestAnthropicParser_ParseRequest_CountTokens(t *testing.T) {
	parser := NewAnthropicParser()

	tests := []struct {
		name    string
		headers map[string]string
		body    []byte
	}{
		{
			name:    "valid count_tokens body forwarded as raw payload",
			headers: map[string]string{":path": "/v1/messages/count_tokens"},
			body: []byte(`{"model":"test-model",` +
				`"system":"You are a helpful assistant.",` +
				`"messages":[{"role":"user","content":"Hello"}]}`),
		},
		{
			name:    "empty body still forwarded",
			headers: map[string]string{":path": "/v1/messages/count_tokens"},
			body:    []byte{},
		},
		{
			name:    "non-JSON body still forwarded",
			headers: map[string]string{":path": "/v1/messages/count_tokens"},
			body:    []byte("not-json"),
		},
		{
			name:    "path read from x-original-path header",
			headers: map[string]string{"x-original-path": "/v1/messages/count_tokens"},
			body:    []byte(`{}`),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseRequest(context.Background(), tt.body, tt.headers)
			if err != nil {
				t.Fatalf("ParseRequest() error = %v", err)
			}
			if !got.SkipResponseProcessing {
				t.Errorf("ParseRequest() SkipResponseProcessing = false, want true")
			}
			want := &fwkrh.InferenceRequestBody{Payload: fwkrh.RawPayload(tt.body)}
			if diff := cmp.Diff(want, got.Body); diff != "" {
				t.Errorf("ParseRequest() body mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAnthropicParser_ParseRequest_MaxOutputTokens(t *testing.T) {
	parser := NewAnthropicParser()
	headers := map[string]string{":path": "/v1/messages"}
	msgs := []any{map[string]any{"role": "user", "content": "Hello"}}

	tests := []struct {
		name string
		body map[string]any
		want *int64
	}{
		{
			name: "max_tokens present",
			body: map[string]any{"model": "claude-sonnet-4-6", "max_tokens": float64(1024), "messages": msgs},
			want: ptr.To(int64(1024)),
		},
		{
			name: "max_tokens absent",
			body: map[string]any{"model": "claude-sonnet-4-6", "messages": msgs},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, err := json.Marshal(tt.body)
			if err != nil {
				t.Fatalf("marshal body: %v", err)
			}
			got, err := parser.ParseRequest(context.Background(), bodyBytes, headers)
			if err != nil {
				t.Fatalf("ParseRequest() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got.Body.MaxOutputTokens); diff != "" {
				t.Errorf("MaxOutputTokens mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestParseRequest_ToolBlocks verifies tool_use, tool_result, and thinking
// blocks survive parsing with their raw JSON (input order preserved) intact.
func TestParseRequest_ToolBlocks(t *testing.T) {
	parser := NewAnthropicParser()
	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 1024,
		"tools": [{
			"name": "get_weather",
			"description": "Get the weather",
			"input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
		}],
		"messages": [
			{"role": "user", "content": "Weather in Zurich?"},
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "need the tool"},
				{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Zurich"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_01", "content": [
					{"type": "text", "text": "Sunny"}
				]}
			]}
		]
	}`

	got, err := parser.ParseRequest(context.Background(), []byte(body), map[string]string{":path": "/v1/messages"})
	require.NoError(t, err)
	require.NotNil(t, got.Body.Messages)

	msgs := got.Body.Messages.Messages
	require.Len(t, msgs, 3)

	assistant := msgs[1].Content.Structured
	require.Len(t, assistant, 2)
	assert.Equal(t, "thinking", assistant[0].Type)
	assert.Equal(t, "need the tool", assistant[0].Thinking)
	assert.Equal(t, "tool_use", assistant[1].Type)
	assert.Equal(t, "toolu_01", assistant[1].ID)
	assert.Equal(t, "get_weather", assistant[1].Name)
	assert.JSONEq(t, `{"city": "Zurich"}`, string(assistant[1].Input))
	assert.Equal(t, `{"city": "Zurich"}`, string(assistant[1].Input), "input must keep wire bytes")

	toolResult := msgs[2].Content.Structured
	require.Len(t, toolResult, 1)
	assert.Equal(t, "tool_result", toolResult[0].Type)
	assert.Equal(t, "toolu_01", toolResult[0].ToolUseID)
	assert.Equal(t, "Sunny", toolResult[0].Content.Structured[0].Text)

	tools := got.Body.Messages.Tools
	require.Len(t, tools, 1)
	assert.Equal(t, "get_weather", tools[0].Name)
	assert.Equal(t, `{"type": "object", "properties": {"city": {"type": "string"}}}`, string(tools[0].InputSchema),
		"input_schema must keep wire bytes for order-faithful re-serialization")
}
