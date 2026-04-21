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

package tokenizer

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

type mockTokenizer struct {
	renderFunc     func(prompt string) ([]uint32, []tokenizerTypes.Offset, error)
	renderChatFunc func(req *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error)
}

func (m *mockTokenizer) Render(_ context.Context, prompt string) ([]uint32, []tokenizerTypes.Offset, error) {
	return m.renderFunc(prompt)
}

func (m *mockTokenizer) RenderChat(_ context.Context, req *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error) {
	return m.renderChatFunc(req)
}

func newTestPlugin(tok tokenizer) *Plugin {
	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: "test"},
		tokenizer: tok,
	}
}

func TestPluginFactory_Validation(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := plugin.NewEppHandle(ctx, nil)

	tests := []struct {
		name       string
		params     string
		expectErr  bool
		errContain string
	}{
		{
			name:       "missing modelName",
			params:     `{}`,
			expectErr:  true,
			errContain: "'modelName' must be specified",
		},
		{
			name:       "nil parameters",
			params:     "",
			expectErr:  true,
			errContain: "'modelName' must be specified",
		},
		{
			name:       "invalid JSON",
			params:     `{invalid}`,
			expectErr:  true,
			errContain: "failed to parse",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rawParams json.RawMessage
			if tt.params != "" {
				rawParams = json.RawMessage(tt.params)
			}

			p, err := PluginFactory("test-tokenizer", rawParams, handle)
			if tt.expectErr {
				require.Error(t, err)
				assert.Nil(t, p)
				assert.Contains(t, err.Error(), tt.errContain)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, p)
			}
		})
	}
}

func TestProduce_PopulatesTokenizedPrompt(t *testing.T) {
	mm := &tokenization.MultiModalFeatures{
		MMHashes: map[string][]string{"image": {"hash-a", "hash-b"}},
		MMPlaceholders: map[string][]kvblock.PlaceholderRange{
			"image": {{Offset: 3, Length: 5}, {Offset: 20, Length: 7}},
		},
	}
	tok := &mockTokenizer{
		renderChatFunc: func(_ *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return []uint32{1, 2, 3, 4}, mm, nil
		},
	}
	p := newTestPlugin(tok)

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hi"}}},
			},
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedPrompt)
	assert.Equal(t, []uint32{1, 2, 3, 4}, req.Body.TokenizedPrompt.TokenIDs)
	require.Len(t, req.Body.TokenizedPrompt.MultiModalFeatures, 2)

	assert.Equal(t, 3, req.Body.TokenizedPrompt.MultiModalFeatures[0].Offset)
	assert.Equal(t, "hash-a", req.Body.TokenizedPrompt.MultiModalFeatures[0].Hash)
	assert.Equal(t, 20, req.Body.TokenizedPrompt.MultiModalFeatures[1].Offset)
	assert.Equal(t, "hash-b", req.Body.TokenizedPrompt.MultiModalFeatures[1].Hash)
	assert.Equal(t, fwkrh.ModalityImage, req.Body.TokenizedPrompt.MultiModalFeatures[0].Modality)
}

func TestProduce_SkipsWhenAlreadyPopulated(t *testing.T) {
	existing := &fwkrh.TokenizedPrompt{TokenIDs: []uint32{42}}
	p := newTestPlugin(&mockTokenizer{})
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{TokenizedPrompt: existing},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	assert.Same(t, existing, req.Body.TokenizedPrompt)
}

func TestProduce_NilBody(t *testing.T) {
	p := newTestPlugin(&mockTokenizer{})
	req := &scheduling.InferenceRequest{}
	err := p.Produce(context.Background(), req, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "request body is nil")
}

func TestProduce_TokenizerError(t *testing.T) {
	tok := &mockTokenizer{
		renderChatFunc: func(_ *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return nil, nil, assert.AnError
		},
	}
	p := newTestPlugin(tok)
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hi"}}},
			},
		},
	}
	err := p.Produce(context.Background(), req, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tokenization failed")
	assert.Nil(t, req.Body.TokenizedPrompt)
}

func TestProduce_UnsupportedBodyType(t *testing.T) {
	p := newTestPlugin(&mockTokenizer{})
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{}, // no Completions or ChatCompletions
	}
	err := p.Produce(context.Background(), req, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported request body type")
	assert.Nil(t, req.Body.TokenizedPrompt)
}

func TestConvertMMFeaturesRoundTrip(t *testing.T) {
	src := &tokenization.MultiModalFeatures{
		MMHashes: map[string][]string{"image": {"h1", "h2"}},
		MMPlaceholders: map[string][]kvblock.PlaceholderRange{
			"image": {{Offset: 1, Length: 2}, {Offset: 10, Length: 3}},
		},
	}
	upstream := convertMMFeaturesToUpstream(src)
	require.Len(t, upstream, 2)

	hashes, ranges := ConvertMMFeaturesFromUpstream(upstream)
	assert.Equal(t, []string{"h1", "h2"}, hashes["image"])
	assert.Equal(t,
		[]kvblock.PlaceholderRange{{Offset: 1, Length: 2}, {Offset: 10, Length: 3}},
		ranges["image"],
	)
}

func TestConvertMMFeaturesNil(t *testing.T) {
	assert.Nil(t, convertMMFeaturesToUpstream(nil))
	assert.Nil(t, convertMMFeaturesToUpstream(&tokenization.MultiModalFeatures{}))
	h, r := ConvertMMFeaturesFromUpstream(nil)
	assert.Nil(t, h)
	assert.Nil(t, r)
}

func TestChatCompletionsToRenderChatRequest(t *testing.T) {
	chat := &fwkrh.ChatCompletionsRequest{
		Messages: []fwkrh.Message{
			{Role: "system", Content: fwkrh.Content{Raw: "You are a helpful assistant."}},
			{Role: "user", Content: fwkrh.Content{Raw: "Hello!"}},
		},
		ChatTemplate:              "template",
		AddGenerationPrompt:       true,
		ContinueFinalMessage:      false,
		ReturnAssistantTokensMask: true,
	}

	result := ChatCompletionsToRenderChatRequest(chat)

	require.Len(t, result.Conversation, 2)
	assert.Equal(t, "system", result.Conversation[0].Role)
	assert.Equal(t, tokenizerTypes.Content{Raw: "You are a helpful assistant."}, result.Conversation[0].Content)
	assert.Equal(t, "user", result.Conversation[1].Role)
	assert.Equal(t, tokenizerTypes.Content{Raw: "Hello!"}, result.Conversation[1].Content)
	assert.Equal(t, "template", result.ChatTemplate)
	assert.True(t, result.AddGenerationPrompt)
	assert.False(t, result.ContinueFinalMessage)
	assert.True(t, result.ReturnAssistantTokensMask)
}

func TestProduce_StringArrayPrompt(t *testing.T) {
	tok := &mockTokenizer{
		renderFunc: func(prompt string) ([]uint32, []tokenizerTypes.Offset, error) {
			switch prompt {
			case "hello":
				return []uint32{10, 20, 30}, nil, nil
			case "world":
				return []uint32{40, 50}, nil, nil
			default:
				return nil, nil, nil
			}
		},
	}
	p := newTestPlugin(tok)

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Strings: []string{"hello", "world"}},
			},
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedPrompt)
	assert.Equal(t, []uint32{10, 20, 30, 40, 50}, req.Body.TokenizedPrompt.TokenIDs)
	require.Len(t, req.Body.TokenizedPrompt.PerPromptTokens, 2)
	assert.Equal(t, []uint32{10, 20, 30}, req.Body.TokenizedPrompt.PerPromptTokens[0])
	assert.Equal(t, []uint32{40, 50}, req.Body.TokenizedPrompt.PerPromptTokens[1])
}

func TestProduce_SinglePromptNoPerPromptTokens(t *testing.T) {
	tok := &mockTokenizer{
		renderFunc: func(_ string) ([]uint32, []tokenizerTypes.Offset, error) {
			return []uint32{10, 20, 30}, nil, nil
		},
	}
	p := newTestPlugin(tok)

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "hello"},
			},
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedPrompt)
	assert.Equal(t, []uint32{10, 20, 30}, req.Body.TokenizedPrompt.TokenIDs)
	assert.Nil(t, req.Body.TokenizedPrompt.PerPromptTokens, "single prompt should not set PerPromptTokens")
}

func TestChatCompletionsToRenderChatRequest_MultimodalContent(t *testing.T) {
	tests := []struct {
		name     string
		messages []fwkrh.Message
		wantConv []tokenizerTypes.Conversation
	}{
		{
			name: "single image with text",
			messages: []fwkrh.Message{
				{Role: "user", Content: fwkrh.Content{
					Structured: []fwkrh.ContentBlock{
						{Type: "text", Text: "Describe this image"},
						{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "data:image/png;base64,abc123"}},
					},
				}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "user", Content: tokenizerTypes.Content{
					Structured: []tokenizerTypes.ContentBlock{
						{Type: "text", Text: "Describe this image"},
						{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "data:image/png;base64,abc123"}},
					},
				}},
			},
		},
		{
			name: "system text message plus multimodal user message",
			messages: []fwkrh.Message{
				{Role: "system", Content: fwkrh.Content{Raw: "You are a visual analyst."}},
				{Role: "user", Content: fwkrh.Content{
					Structured: []fwkrh.ContentBlock{
						{Type: "text", Text: "Compare these two images"},
						{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "data:image/png;base64,img1"}},
						{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "data:image/png;base64,img2"}},
					},
				}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "system", Content: tokenizerTypes.Content{Raw: "You are a visual analyst."}},
				{Role: "user", Content: tokenizerTypes.Content{
					Structured: []tokenizerTypes.ContentBlock{
						{Type: "text", Text: "Compare these two images"},
						{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "data:image/png;base64,img1"}},
						{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "data:image/png;base64,img2"}},
					},
				}},
			},
		},
		{
			name: "multi-turn with image in history",
			messages: []fwkrh.Message{
				{Role: "user", Content: fwkrh.Content{
					Structured: []fwkrh.ContentBlock{
						{Type: "text", Text: "What is in this image?"},
						{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/img.jpg"}},
					},
				}},
				{Role: "assistant", Content: fwkrh.Content{Raw: "I see a dog."}},
				{Role: "user", Content: fwkrh.Content{Raw: "What breed is it?"}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "user", Content: tokenizerTypes.Content{
					Structured: []tokenizerTypes.ContentBlock{
						{Type: "text", Text: "What is in this image?"},
						{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "https://example.com/img.jpg"}},
					},
				}},
				{Role: "assistant", Content: tokenizerTypes.Content{Raw: "I see a dog."}},
				{Role: "user", Content: tokenizerTypes.Content{Raw: "What breed is it?"}},
			},
		},
		{
			name: "text-only messages produce no Structured field",
			messages: []fwkrh.Message{
				{Role: "user", Content: fwkrh.Content{Raw: "Hello!"}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "user", Content: tokenizerTypes.Content{Raw: "Hello!"}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chat := &fwkrh.ChatCompletionsRequest{Messages: tt.messages}
			result := ChatCompletionsToRenderChatRequest(chat)
			require.Len(t, result.Conversation, len(tt.wantConv))
			for i, want := range tt.wantConv {
				got := result.Conversation[i]
				assert.Equal(t, want.Role, got.Role)
				assert.Equal(t, want.Content.Raw, got.Content.Raw)
				assert.Equal(t, want.Content.Structured, got.Content.Structured,
					"message %d: Structured content mismatch", i)
			}
		})
	}
}
