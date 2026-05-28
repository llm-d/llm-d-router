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

package requesthandling

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLLMRequestBody_PromptText(t *testing.T) {
	tests := []struct {
		name     string
		body     *InferenceRequestBody
		expected string
	}{
		{
			name: "completions request returns prompt directly",
			body: &InferenceRequestBody{
				Completions: &CompletionsRequest{
					Prompt: Prompt{Raw: "What is the meaning of life?"},
				},
			},
			expected: "What is the meaning of life?",
		},
		{
			name: "completions request with array of strings prompt",
			body: &InferenceRequestBody{
				Completions: &CompletionsRequest{
					Prompt: Prompt{Strings: []string{"Why is", "the sky blue?"}},
				},
			},
			expected: "Why is the sky blue?",
		},
		{
			name: "chat completions with single raw message",
			body: &InferenceRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					Messages: []Message{
						{Role: "user", Content: Content{Raw: "Hello, how are you?"}},
					},
				},
			},
			expected: "Hello, how are you? ",
		},
		{
			name: "chat completions with multiple messages",
			body: &InferenceRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					Messages: []Message{
						{Role: "system", Content: Content{Raw: "You are a helpful assistant."}},
						{Role: "user", Content: Content{Raw: "Tell me a joke."}},
					},
				},
			},
			expected: "You are a helpful assistant. Tell me a joke. ",
		},
		{
			name: "chat completions with structured content blocks",
			body: &InferenceRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					Messages: []Message{
						{
							Role: "user",
							Content: Content{
								Structured: []ContentBlock{
									{Type: "text", Text: "Describe this image:"},
									{Type: "image_url", ImageURL: ImageBlock{URL: "http://example.com/img.png"}},
								},
							},
						},
					},
				},
			},
			expected: "Describe this image:  ",
		},
		{
			name: "responses request with string input",
			body: &InferenceRequestBody{
				Responses: &ResponsesRequest{
					Input: "Some response input",
				},
			},
			expected: "Some response input",
		},
		{
			name: "responses request with non-string input",
			body: &InferenceRequestBody{
				Responses: &ResponsesRequest{
					Input: map[string]any{"key": "value"},
				},
			},
			expected: `{"key":"value"}`,
		},
		{
			name: "conversations request",
			body: &InferenceRequestBody{
				Conversations: &ConversationsRequest{
					Items: []ConversationItem{
						{Type: "message", Role: "user", Content: "Hello"},
					},
				},
			},
			expected: `[{"type":"message","role":"user","content":"Hello"}]`,
		},
		{
			name:     "empty body returns empty string",
			body:     &InferenceRequestBody{},
			expected: "",
		},
		{
			name: "chat completions with no messages",
			body: &InferenceRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					Messages: []Message{},
				},
			},
			expected: "",
		},
		{
			name: "embeddings request with string input",
			body: &InferenceRequestBody{
				Embeddings: &EmbeddingsRequest{
					Input: EmbeddingsInput{Raw: "hello world"},
				},
			},
			expected: "hello world",
		},
		{
			name: "embeddings request with array of strings input",
			body: &InferenceRequestBody{
				Embeddings: &EmbeddingsRequest{
					Input: EmbeddingsInput{Strings: []string{"hello", "world"}},
				},
			},
			expected: "hello world",
		},
		{
			name: "unified prompt with single text block",
			body: &InferenceRequestBody{
				Prompt: &UnifiedPrompt{
					Items: []PromptItem{
						{
							Role: "user",
							Blocks: []PromptBlock{
								{Type: BlockTypeText, Text: "hello"},
							},
						},
					},
				},
			},
			expected: "hello",
		},
		{
			name: "unified prompt with multiple text blocks and empty role",
			body: &InferenceRequestBody{
				Prompt: &UnifiedPrompt{
					Items: []PromptItem{
						{
							Role: "system",
							Blocks: []PromptBlock{
								{Type: BlockTypeText, Text: "instructions"},
							},
						},
						{
							Role: "",
							Blocks: []PromptBlock{
								{Type: BlockTypeText, Text: "user input"},
								{Type: BlockTypeImage, AssetURI: "http://image"}, // ignored in text
							},
						},
					},
				},
			},
			expected: "instructions user input",
		},
		{
			name: "unified prompt with documents",
			body: &InferenceRequestBody{
				Prompt: &UnifiedPrompt{
					Items: []PromptItem{
						{
							Role: "user",
							Blocks: []PromptBlock{
								{Type: BlockTypeText, Text: "query"},
							},
						},
					},
					Documents: []any{
						"doc1 text",
						map[string]any{"content": "doc2 text"},
					},
				},
			},
			expected: `query doc1 text {"content":"doc2 text"}`,
		},
		{
			name: "unified prompt prioritized over legacy completions",
			body: &InferenceRequestBody{
				Prompt: &UnifiedPrompt{
					Items: []PromptItem{
						{
							Role: "user",
							Blocks: []PromptBlock{
								{Type: BlockTypeText, Text: "unified"},
							},
						},
					},
				},
				Completions: &CompletionsRequest{
					Prompt: Prompt{Raw: "legacy"},
				},
			},
			expected: "unified",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.body.PromptText()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestPrompt_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    Prompt
		wantErr bool
	}{
		{
			name:  "string prompt",
			input: `"hello world"`,
			want:  Prompt{Raw: "hello world"},
		},
		{
			name:  "array of strings prompt",
			input: `["hello","world"]`,
			want:  Prompt{Strings: []string{"hello", "world"}},
		},
		{
			name:  "single-element array prompt",
			input: `["hello world"]`,
			want:  Prompt{Strings: []string{"hello world"}},
		},
		{
			name:  "array of integers prompt",
			input: `[1,2,3]`,
			want:  Prompt{TokenIDs: []uint32{1, 2, 3}},
		},
		{
			name:    "array of floats prompt is rejected",
			input:   `[1.5,2.7]`,
			wantErr: true,
		},

		{
			name:    "array of arrays of integers prompt is rejected for now",
			input:   `[[1,2],[3,4]]`,
			wantErr: true,
		},

		{
			name:    "integer prompt is rejected",
			input:   `123`,
			wantErr: true,
		},
		{
			name:    "object prompt is rejected",
			input:   `{"key":"value"}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var p Prompt
			err := p.UnmarshalJSON([]byte(tt.input))
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, p)
			}
		})
	}
}

func TestEmbeddingsInput_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    EmbeddingsInput
		wantErr bool
	}{
		{
			name:  "string input",
			input: `"hello world"`,
			want:  EmbeddingsInput{Raw: "hello world"},
		},
		{
			name:  "array of strings input",
			input: `["hello","world"]`,
			want:  EmbeddingsInput{Strings: []string{"hello", "world"}},
		},
		{
			name:  "array of integers input",
			input: `[1,2,3]`,
			want:  EmbeddingsInput{TokenIDs: []uint32{1, 2, 3}},
		},
		{
			name:    "array of floats input is rejected",
			input:   `[1.5,2.7]`,
			wantErr: true,
		},

		{
			name:    "array of arrays of integers input is rejected for now",
			input:   `[[1,2],[3,4]]`,
			wantErr: true,
		},

		{
			name:    "integer input is rejected",
			input:   `123`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var e EmbeddingsInput
			err := e.UnmarshalJSON([]byte(tt.input))
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, e)
			}
		})
	}
}

func TestInferenceRequestBody_InputTokenCountHint(t *testing.T) {
	tests := []struct {
		name     string
		body     *InferenceRequestBody
		wantHint int
	}{
		{
			name: "completions with token IDs returns count",
			body: &InferenceRequestBody{
				Completions: &CompletionsRequest{
					Prompt: Prompt{TokenIDs: []uint32{1, 2, 3}},
				},
			},
			wantHint: 3,
		},
		{
			name: "completions with text returns -1",
			body: &InferenceRequestBody{
				Completions: &CompletionsRequest{
					Prompt: Prompt{Raw: "hello"},
				},
			},
			wantHint: -1,
		},
		{
			name: "embeddings with token IDs returns count",
			body: &InferenceRequestBody{
				Embeddings: &EmbeddingsRequest{
					Input: EmbeddingsInput{TokenIDs: []uint32{1, 2, 3, 4}},
				},
			},
			wantHint: 4,
		},
		{
			name: "embeddings with text returns -1",
			body: &InferenceRequestBody{
				Embeddings: &EmbeddingsRequest{
					Input: EmbeddingsInput{Raw: "hello"},
				},
			},
			wantHint: -1,
		},
		{
			name:     "empty body returns -1",
			body:     &InferenceRequestBody{},
			wantHint: -1,
		},
		{
			name: "unified tokens returns count",
			body: &InferenceRequestBody{
				Tokens: &TokenizedInput{
					TokenIDs: []uint32{10, 20, 30},
				},
			},
			wantHint: 3,
		},
		{
			name: "unified prompt with nil tokens returns -1",
			body: &InferenceRequestBody{
				Prompt: &UnifiedPrompt{
					Items: []PromptItem{
						{Blocks: []PromptBlock{{Type: BlockTypeText, Text: "hello"}}},
					},
				},
			},
			wantHint: -1,
		},
		{
			name: "unified tokens prioritized over legacy completions",
			body: &InferenceRequestBody{
				Tokens: &TokenizedInput{
					TokenIDs: []uint32{10, 20, 30},
				},
				Completions: &CompletionsRequest{
					Prompt: Prompt{TokenIDs: []uint32{1, 2}},
				},
			},
			wantHint: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantHint, tt.body.InputTokenCountHint())
		})
	}
}

func TestPrompt_PlainText(t *testing.T) {
	tests := []struct {
		name string
		p    Prompt
		want string
	}{
		{name: "raw string", p: Prompt{Raw: "hello"}, want: "hello"},
		{name: "strings joined", p: Prompt{Strings: []string{"a", "b", "c"}}, want: "a b c"},
		{name: "single string in array", p: Prompt{Strings: []string{"hello"}}, want: "hello"},
		{name: "zero value", p: Prompt{}, want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.p.PlainText())
		})
	}
}

func TestPrompt_IsEmpty(t *testing.T) {
	assert.True(t, Prompt{}.IsEmpty())
	assert.True(t, Prompt{Strings: []string{}}.IsEmpty())
	assert.False(t, Prompt{Raw: "x"}.IsEmpty())
	assert.False(t, Prompt{Strings: []string{"x"}}.IsEmpty())
}

func TestPrompt_MarshalJSON(t *testing.T) {
	raw, _ := Prompt{Raw: "hello"}.MarshalJSON()
	assert.Equal(t, `"hello"`, string(raw))

	arr, _ := Prompt{Strings: []string{"a", "b"}}.MarshalJSON()
	assert.Equal(t, `["a","b"]`, string(arr))

	empty, _ := Prompt{}.MarshalJSON()
	assert.Equal(t, `""`, string(empty))
}

func TestGenerateRequest_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		want        []uint32
		wantErr     bool
		errContains string
	}{
		{
			name:  "valid token ids",
			input: `{"token_ids":[1,2,3]}`,
			want:  []uint32{1, 2, 3},
		},
		{
			name:  "max uint32 boundary accepted",
			input: `{"token_ids":[4294967295]}`,
			want:  []uint32{4294967295},
		},
		{
			name:        "negative token id rejected",
			input:       `{"token_ids":[1,2,-1]}`,
			wantErr:     true,
			errContains: "token_ids[2]: invalid value",
		},
		{
			name:        "non-integer token id rejected",
			input:       `{"token_ids":[1,2.5,3]}`,
			wantErr:     true,
			errContains: "token_ids[1]: invalid value",
		},
		{
			name:        "value above MaxUint32 rejected",
			input:       `{"token_ids":[4294967296]}`,
			wantErr:     true,
			errContains: "token_ids[0]: invalid value",
		},
		{
			name:        "NaN token id rejected",
			input:       `{"token_ids":[1,NaN]}`,
			wantErr:     true,
			errContains: "invalid character",
		},
		{
			name:        "malformed json rejected",
			input:       `{"token_ids":[`,
			wantErr:     true,
			errContains: "unexpected end of JSON",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var g GenerateRequest
			err := g.UnmarshalJSON([]byte(tt.input))
			if tt.wantErr {
				require.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, g.TokenIDs)
		})
	}
}

func TestInferenceRequestBody_CacheSalt(t *testing.T) {
	tests := []struct {
		name     string
		body     *InferenceRequestBody
		expected string
	}{
		{
			name: "extracted cache salt returned directly",
			body: &InferenceRequestBody{
				ExtractedCacheSalt: "salt-123",
			},
			expected: "salt-123",
		},
		{
			name: "fallback to conversations cache salt",
			body: &InferenceRequestBody{
				Conversations: &ConversationsRequest{
					CacheSalt: "conv-salt",
				},
			},
			expected: "conv-salt",
		},
		{
			name: "fallback to responses cache salt",
			body: &InferenceRequestBody{
				Responses: &ResponsesRequest{
					CacheSalt: "resp-salt",
				},
			},
			expected: "resp-salt",
		},
		{
			name: "fallback to chat completions cache salt",
			body: &InferenceRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					CacheSalt: "chat-salt",
				},
			},
			expected: "chat-salt",
		},
		{
			name: "prioritizes extracted cache salt over legacy",
			body: &InferenceRequestBody{
				ExtractedCacheSalt: "extracted",
				Conversations: &ConversationsRequest{
					CacheSalt: "conv",
				},
				Responses: &ResponsesRequest{
					CacheSalt: "resp",
				},
				ChatCompletions: &ChatCompletionsRequest{
					CacheSalt: "chat",
				},
			},
			expected: "extracted",
		},
		{
			name: "prioritizes conversations over responses in fallback",
			body: &InferenceRequestBody{
				Conversations: &ConversationsRequest{
					CacheSalt: "conv",
				},
				Responses: &ResponsesRequest{
					CacheSalt: "resp",
				},
			},
			expected: "conv",
		},
		{
			name:     "empty body returns empty string",
			body:     &InferenceRequestBody{},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.body.CacheSalt())
		})
	}
}
