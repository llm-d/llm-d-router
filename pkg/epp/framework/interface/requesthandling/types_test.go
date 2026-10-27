/*
Copyright 2025 The Kubernetes Authors.
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

package requesthandling

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

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
			want:  Prompt{TokenIDs: [][]uint32{{1, 2, 3}}},
		},
		{
			name:    "array of floats prompt is rejected",
			input:   `[1.5,2.7]`,
			wantErr: true,
		},
		{
			name:  "whole decimal token IDs",
			input: `[1.0]`,
			want:  Prompt{TokenIDs: [][]uint32{{1}}},
		},
		{
			name:  "array of arrays of integers prompt",
			input: `[[1,2],[3,4]]`,
			want:  Prompt{TokenIDs: [][]uint32{{1, 2}, {3, 4}}},
		},
		{
			name:  "single sub-array of integers prompt",
			input: `[[10,20,30]]`,
			want:  Prompt{TokenIDs: [][]uint32{{10, 20, 30}}},
		},
		{
			name:    "empty sub-array in nested prompt",
			input:   `[[1,2],[]]`,
			wantErr: true,
		},
		{
			name:    "mixed types in nested array prompt",
			input:   `[[1,2],"hello"]`,
			wantErr: true,
		},
		{
			name:    "float in sub-array prompt",
			input:   `[[1,2.5]]`,
			wantErr: true,
		},
		{
			name:    "non-numeric in sub-array prompt",
			input:   `[[1,"a"]]`,
			wantErr: true,
		},
		{
			name:    "triple nesting prompt",
			input:   `[[[1,2]]]`,
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
				assert.Equal(t, Prompt{}, p)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, p)
			}
		})
	}
}

func TestContent_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    Content
		wantErr bool
	}{
		{
			name:  "string content",
			input: `"hello world"`,
			want:  Content{Raw: "hello world"},
		},
		{
			name:  "string content with escapes",
			input: `"line one\n\"quoted\"\u0041"`,
			want:  Content{Raw: "line one\n\"quoted\"A"},
		},
		{
			name:  "empty string content",
			input: `""`,
			want:  Content{},
		},
		{
			name:  "null content",
			input: `null`,
			want:  Content{},
		},
		{
			name:  "content blocks",
			input: `[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]`,
			want: Content{Structured: []ContentBlock{
				{Type: "text", Text: "describe"},
				{Type: "image_url", ImageURL: ImageBlock{URL: "data:image/png;base64,AAAA"}},
			}},
		},
		{
			name:  "empty content blocks",
			input: `[]`,
			want:  Content{Structured: []ContentBlock{}},
		},
		{
			name:    "content block with mistyped field is rejected",
			input:   `[{"type":1}]`,
			wantErr: true,
		},
		{
			name:    "array of strings is rejected",
			input:   `["hello"]`,
			wantErr: true,
		},
		{
			name:    "object content is rejected",
			input:   `{"type":"text"}`,
			wantErr: true,
		},
		{
			name:    "number content is rejected",
			input:   `123`,
			wantErr: true,
		},
		{
			name:    "boolean content is rejected",
			input:   `true`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var c Content
			err := c.UnmarshalJSON([]byte(tt.input))
			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, Content{}, c)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, c)
			}
		})
	}
}

func TestAnthropicContent_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    AnthropicContent
		wantErr string
	}{
		{
			name:  "string content",
			input: `"hello world"`,
			want:  AnthropicContent{Raw: "hello world"},
		},
		{
			name:  "text block",
			input: `[{"type":"text","text":"hi"}]`,
			want:  AnthropicContent{Structured: []AnthropicContentBlock{{Type: "text", Text: "hi"}}},
		},
		{
			name:  "image block with object source",
			input: `[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]`,
			want: AnthropicContent{Structured: []AnthropicContentBlock{{
				Type:   "image",
				Source: json.RawMessage(`{"type":"url","url":"https://example.com/a.png"}`),
			}}},
		},
		{
			name:  "document block with object source",
			input: `[{"type":"document","title":"t","source":{"type":"text","media_type":"text/plain","data":"x"}}]`,
			want: AnthropicContent{Structured: []AnthropicContentBlock{{
				Type:   "document",
				Source: json.RawMessage(`{"type":"text","media_type":"text/plain","data":"x"}`),
			}}},
		},
		{
			name: "search_result block with string source",
			input: `[{"type":"search_result","source":"https://example.com/notes","title":"Notes",` +
				`"content":[{"type":"text","text":"code word"}]},{"type":"text","text":"reply"}]`,
			want: AnthropicContent{Structured: []AnthropicContentBlock{
				{
					Type:    "search_result",
					Source:  json.RawMessage(`"https://example.com/notes"`),
					Content: json.RawMessage(`[{"type":"text","text":"code word"}]`),
				},
				{Type: "text", Text: "reply"},
			}},
		},
		{
			name: "search_result block inside tool_result",
			input: `[{"type":"tool_result","tool_use_id":"toolu_01","content":[` +
				`{"type":"search_result","source":"https://example.com/report","title":"Report",` +
				`"content":[{"type":"text","text":"sunny"}]}]}]`,
			want: AnthropicContent{Structured: []AnthropicContentBlock{{
				Type:      "tool_result",
				ToolUseID: "toolu_01",
				Content: json.RawMessage(`[{"type":"search_result","source":"https://example.com/report","title":"Report",` +
					`"content":[{"type":"text","text":"sunny"}]}]`),
			}}},
		},
		{
			name: "web_search_tool_result with error object content",
			input: `[{"type":"web_search_tool_result","tool_use_id":"srvtoolu_01",` +
				`"content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}}]`,
			want: AnthropicContent{Structured: []AnthropicContentBlock{{
				Type:      "web_search_tool_result",
				ToolUseID: "srvtoolu_01",
				Content:   json.RawMessage(`{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}`),
			}}},
		},
		{
			name: "bash_code_execution_tool_result with object content",
			input: `[{"type":"bash_code_execution_tool_result","tool_use_id":"srvtoolu_02",` +
				`"content":{"type":"bash_code_execution_result","stdout":"1\n","stderr":"","return_code":0,"content":[]}}]`,
			want: AnthropicContent{Structured: []AnthropicContentBlock{{
				Type:      "bash_code_execution_tool_result",
				ToolUseID: "srvtoolu_02",
				Content:   json.RawMessage(`{"type":"bash_code_execution_result","stdout":"1\n","stderr":"","return_code":0,"content":[]}`),
			}}},
		},
		{
			name:    "number content",
			input:   `42`,
			wantErr: "anthropic content: must be a string or an array of content blocks",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var c AnthropicContent
			err := json.Unmarshal([]byte(tt.input), &c)
			if tt.wantErr != "" {
				assert.EqualError(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, c)
		})
	}
}

// A malformed block inside an array must surface its own decode error, once.
// A tool_result keeps its content raw, so a malformed block below it decodes
// and is reported as empty content by ToolResultContent instead.
func TestAnthropicContent_UnmarshalJSONMalformedBlock(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{name: "top level", input: `[{"type":"text","text":5}]`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var c AnthropicContent
			err := json.Unmarshal([]byte(tt.input), &c)

			var typeErr *json.UnmarshalTypeError
			require.ErrorAs(t, err, &typeErr)
			assert.Equal(t, "number", typeErr.Value)
			assert.NotContains(t, err.Error(), "must be a string or an array")
		})
	}

	t.Run("inside tool_result", func(t *testing.T) {
		var c AnthropicContent
		require.NoError(t, json.Unmarshal([]byte(`[{"type":"tool_result","tool_use_id":"toolu_01","content":[{"type":"text","text":5}]}]`), &c))
		require.Len(t, c.Structured, 1)
		assert.Equal(t, `[{"type":"text","text":5}]`, string(c.Structured[0].Content))
		assert.Equal(t, AnthropicContent{}, c.Structured[0].ToolResultContent())
	})
}

func TestAnthropicContent_UnmarshalJSONPreservesReceiverBehavior(t *testing.T) {
	stale := AnthropicContent{
		Raw: "stale",
		Structured: []AnthropicContentBlock{{
			Type:   "image",
			Name:   "stale",
			Source: json.RawMessage(`{"type":"url","url":"https://example.com/a.png"}`),
		}},
	}
	clone := func() AnthropicContent {
		c := stale
		c.Structured = append([]AnthropicContentBlock(nil), stale.Structured...)
		return c
	}

	c := clone()
	require.NoError(t, json.Unmarshal([]byte(`[{"type":"text","text":"a"}]`), &c))
	assert.Equal(t, AnthropicContent{Raw: "stale", Structured: []AnthropicContentBlock{{Type: "text", Text: "a"}}}, c,
		"decoded blocks must not inherit fields from the receiver's old blocks")

	c = clone()
	require.NoError(t, json.Unmarshal([]byte(`"hello"`), &c))
	assert.Equal(t, AnthropicContent{Raw: "hello", Structured: stale.Structured}, c)

	c = clone()
	require.Error(t, json.Unmarshal([]byte(`[{"type":"image"},{"type":"text","text":5}]`), &c))
	assert.Equal(t, stale, c, "a failed decode must leave the receiver unchanged")

	c = clone()
	require.Error(t, json.Unmarshal([]byte(`42`), &c))
	assert.Equal(t, stale, c, "a failed decode must leave the receiver unchanged")
}

func TestAnthropicContentBlock_ImageSource(t *testing.T) {
	tests := []struct {
		name  string
		block AnthropicContentBlock
		want  *AnthropicImageSource
	}{
		{
			name:  "image with base64 source",
			block: AnthropicContentBlock{Type: "image", Source: json.RawMessage(`{"type":"base64","media_type":"image/png","data":"AAAA"}`)},
			want:  &AnthropicImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"},
		},
		{
			name:  "image with url source",
			block: AnthropicContentBlock{Type: "image", Source: json.RawMessage(`{"type":"url","url":"https://example.com/a.png"}`)},
			want:  &AnthropicImageSource{Type: "url", URL: "https://example.com/a.png"},
		},
		{
			name:  "image without source",
			block: AnthropicContentBlock{Type: "image"},
		},
		{
			name:  "image with null source",
			block: AnthropicContentBlock{Type: "image", Source: json.RawMessage(`null`)},
		},
		{
			name:  "image with string source",
			block: AnthropicContentBlock{Type: "image", Source: json.RawMessage(`"https://example.com/a.png"`)},
		},
		{
			name:  "image with malformed source object",
			block: AnthropicContentBlock{Type: "image", Source: json.RawMessage(`{"type":5}`)},
		},
		{
			name:  "search_result is not an image",
			block: AnthropicContentBlock{Type: "search_result", Source: json.RawMessage(`"https://example.com/notes"`)},
		},
		{
			name:  "document is not an image",
			block: AnthropicContentBlock{Type: "document", Source: json.RawMessage(`{"type":"url","url":"https://example.com/a.pdf"}`)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.block.ImageSource())
		})
	}
}

func TestAnthropicContentBlock_ToolResultContent(t *testing.T) {
	tests := []struct {
		name  string
		block AnthropicContentBlock
		want  AnthropicContent
	}{
		{
			name:  "tool_result with string content",
			block: AnthropicContentBlock{Type: "tool_result", Content: json.RawMessage(`"Sunny, 22C"`)},
			want:  AnthropicContent{Raw: "Sunny, 22C"},
		},
		{
			name:  "tool_result with block array content",
			block: AnthropicContentBlock{Type: "tool_result", Content: json.RawMessage(`[{"type":"text","text":"sunny"}]`)},
			want:  AnthropicContent{Structured: []AnthropicContentBlock{{Type: "text", Text: "sunny"}}},
		},
		{
			name:  "tool_result without content",
			block: AnthropicContentBlock{Type: "tool_result"},
		},
		{
			name:  "tool_result with null content",
			block: AnthropicContentBlock{Type: "tool_result", Content: json.RawMessage(`null`)},
		},
		{
			name:  "tool_result with object content",
			block: AnthropicContentBlock{Type: "tool_result", Content: json.RawMessage(`{"type":"text","text":"sunny"}`)},
		},
		{
			name:  "web_search_tool_result is not a tool_result",
			block: AnthropicContentBlock{Type: "web_search_tool_result", Content: json.RawMessage(`[{"type":"web_search_result","url":"https://example.com"}]`)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.block.ToolResultContent())
		})
	}
}

func TestPrompt_UnmarshalJSONPreservesReceiverBehavior(t *testing.T) {
	stale := Prompt{
		Raw:      "stale",
		Strings:  []string{"stale"},
		TokenIDs: [][]uint32{{99}},
	}

	p := stale
	require.NoError(t, p.UnmarshalJSON([]byte(`[1,2,3]`)))
	assert.Equal(t, Prompt{Raw: "stale", TokenIDs: [][]uint32{{1, 2, 3}}}, p)

	p = stale
	require.NoError(t, p.UnmarshalJSON([]byte(`[1.0]`)))
	assert.Equal(t, Prompt{Raw: "stale", TokenIDs: [][]uint32{{1}}}, p)

	p = stale
	require.NoError(t, p.UnmarshalJSON([]byte(`"hello"`)))
	assert.Equal(t, Prompt{Raw: "hello", Strings: []string{"stale"}, TokenIDs: [][]uint32{{99}}}, p)
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
			want:  EmbeddingsInput{TokenIDs: [][]uint32{{1, 2, 3}}},
		},
		{
			name:    "array of floats input is rejected",
			input:   `[1.5,2.7]`,
			wantErr: true,
		},
		{
			name:  "exponent token IDs",
			input: `[1e0]`,
			want:  EmbeddingsInput{TokenIDs: [][]uint32{{1}}},
		},
		{
			name:  "array of arrays of integers input",
			input: `[[1,2],[3,4]]`,
			want:  EmbeddingsInput{TokenIDs: [][]uint32{{1, 2}, {3, 4}}},
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

func TestEmbeddingsInput_UnmarshalJSONPreservesReceiverBehavior(t *testing.T) {
	stale := EmbeddingsInput{
		Raw:      "stale",
		Strings:  []string{"stale"},
		TokenIDs: [][]uint32{{99}},
	}

	e := stale
	require.NoError(t, e.UnmarshalJSON([]byte(`[1,2,3]`)))
	assert.Equal(t, EmbeddingsInput{Raw: "stale", TokenIDs: [][]uint32{{1, 2, 3}}}, e)

	e = stale
	require.NoError(t, e.UnmarshalJSON([]byte(`[1e0]`)))
	assert.Equal(t, EmbeddingsInput{Raw: "stale", TokenIDs: [][]uint32{{1}}}, e)

	e = stale
	require.NoError(t, e.UnmarshalJSON([]byte(`"hello"`)))
	assert.Equal(t, EmbeddingsInput{Raw: "hello", Strings: []string{"stale"}, TokenIDs: [][]uint32{{99}}}, e)
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
	assert.False(t, Prompt{TokenIDs: [][]uint32{{1, 2}}}.IsEmpty())
}

func TestPrompt_MarshalJSON(t *testing.T) {
	raw, _ := Prompt{Raw: "hello"}.MarshalJSON()
	assert.Equal(t, `"hello"`, string(raw))

	arr, _ := Prompt{Strings: []string{"a", "b"}}.MarshalJSON()
	assert.Equal(t, `["a","b"]`, string(arr))

	nested, _ := Prompt{TokenIDs: [][]uint32{{1, 2}, {3, 4}}}.MarshalJSON()
	assert.Equal(t, `[[1,2],[3,4]]`, string(nested))

	empty, _ := Prompt{}.MarshalJSON()
	assert.Equal(t, `""`, string(empty))
}

func TestTextToSpeechRequest_String(t *testing.T) {
	assert.Equal(t, "{InputLength: 5}", (&TextToSpeechRequest{Input: "hello"}).String())
	assert.Equal(t, nilStr, (*TextToSpeechRequest)(nil).String())
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
			name:  "valid token ids with whitespace",
			input: "{\n\t\"token_ids\": [ 1,\r\n2, 3 ]\n}",
			want:  []uint32{1, 2, 3},
		},
		{
			name:  "max uint32 boundary accepted",
			input: `{"token_ids":[4294967295]}`,
			want:  []uint32{4294967295},
		},
		{
			name:  "whole decimal accepted",
			input: `{"token_ids":[1.0]}`,
			want:  []uint32{1},
		},
		{
			name:  "whole exponent accepted",
			input: `{"token_ids":[1e0]}`,
			want:  []uint32{1},
		},
		{
			name:  "negative zero accepted",
			input: `{"token_ids":[-0]}`,
			want:  []uint32{0},
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
		{
			name:        "invalid cache salt preserves error field",
			input:       `{"token_ids":[1,2,3],"cache_salt":1}`,
			wantErr:     true,
			errContains: "Go struct field .cache_salt",
		},
		{
			name:        "invalid placeholder preserves error field",
			input:       `{"token_ids":[1,2,3],"features":{"mm_placeholders":{"image":[{"offset":"x","length":1}]}}}`,
			wantErr:     true,
			errContains: "Go struct field .features.mm_placeholders.image.0.offset",
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

func TestGenerateRequest_UnmarshalJSONPreservesReceiverBehavior(t *testing.T) {
	g := GenerateRequest{
		TokenIDs:  []uint32{99},
		CacheSalt: "stale",
		Features: &tokenization.MultiModalFeatures{
			MMHashes: map[string][]string{"image": {"stale"}},
		},
	}

	err := g.UnmarshalJSON([]byte(`{"token_ids":[1,2.5,3],"cache_salt":"updated"}`))
	require.EqualError(t, err, "token_ids[1]: invalid value 2.5")
	assert.Equal(t, []uint32{1, 0, 0}, g.TokenIDs)
	assert.Equal(t, "updated", g.CacheSalt)
	assert.Equal(t, []string{"stale"}, g.Features.MMHashes["image"])

	require.NoError(t, g.UnmarshalJSON([]byte(`{}`)))
	assert.NotNil(t, g.TokenIDs)
	assert.Empty(t, g.TokenIDs)
	assert.Empty(t, g.CacheSalt)
	assert.Equal(t, []string{"stale"}, g.Features.MMHashes["image"])
}

func TestGenerateRequest_UnmarshalJSONCanonicalPreservesAbsentFeatures(t *testing.T) {
	g := GenerateRequest{
		TokenIDs:  []uint32{99},
		CacheSalt: "stale",
		Features: &tokenization.MultiModalFeatures{
			MMHashes: map[string][]string{"image": {"stale"}},
		},
	}

	require.NoError(t, g.UnmarshalJSON([]byte(`{"token_ids":[1,2,3],"cache_salt":"updated"}`)))
	assert.Equal(t, []uint32{1, 2, 3}, g.TokenIDs)
	assert.Equal(t, "updated", g.CacheSalt)
	assert.Equal(t, []string{"stale"}, g.Features.MMHashes["image"])
}

func TestMaxOutputTokensFromPayload(t *testing.T) {
	tests := []struct {
		name string
		m    PayloadMap
		keys []string
		want *int64
	}{
		{name: "absent", m: PayloadMap{"other": float64(1)}, keys: []string{"max_tokens"}, want: nil},
		{name: "float64 value", m: PayloadMap{"max_tokens": float64(64)}, keys: []string{"max_tokens"}, want: ptr.To(int64(64))},
		{name: "json.Number value", m: PayloadMap{"max_tokens": json.Number("128")}, keys: []string{"max_tokens"}, want: ptr.To(int64(128))},
		{name: "explicit zero binds", m: PayloadMap{"max_tokens": float64(0)}, keys: []string{"max_tokens"}, want: ptr.To(int64(0))},
		{name: "negative ignored", m: PayloadMap{"max_tokens": float64(-1)}, keys: []string{"max_tokens"}, want: nil},
		{name: "non-integral ignored", m: PayloadMap{"max_tokens": float64(1.5)}, keys: []string{"max_tokens"}, want: nil},
		{name: "wrong type ignored", m: PayloadMap{"max_tokens": "64"}, keys: []string{"max_tokens"}, want: nil},
		{
			name: "precedence: first present key wins",
			m:    PayloadMap{"max_completion_tokens": float64(100), "max_tokens": float64(50)},
			keys: []string{"max_completion_tokens", "max_tokens"},
			want: ptr.To(int64(100)),
		},
		{
			name: "precedence: fall back to absent second key",
			m:    PayloadMap{"max_tokens": float64(50)},
			keys: []string{"max_completion_tokens", "max_tokens"},
			want: ptr.To(int64(50)),
		},
		{
			name: "fall back when primary is negative",
			m:    PayloadMap{"max_completion_tokens": float64(-1), "max_tokens": float64(50)},
			keys: []string{"max_completion_tokens", "max_tokens"},
			want: ptr.To(int64(50)),
		},
		{
			name: "fall back when primary is wrong type",
			m:    PayloadMap{"max_completion_tokens": "bad", "max_tokens": float64(50)},
			keys: []string{"max_completion_tokens", "max_tokens"},
			want: ptr.To(int64(50)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, MaxOutputTokensFromPayload(tt.m, tt.keys...))
		})
	}
}
