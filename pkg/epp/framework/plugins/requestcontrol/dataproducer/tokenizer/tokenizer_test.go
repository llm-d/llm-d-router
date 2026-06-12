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
	"errors"
	"testing"
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

type mockTokenizer struct {
	renderFunc     func(payload fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error)
	renderChatFunc func(payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error)
}

func (m *mockTokenizer) Render(_ context.Context, payload fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
	return m.renderFunc(payload)
}

func (m *mockTokenizer) RenderChat(_ context.Context, payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	return m.renderChatFunc(payload)
}

func newTestPlugin(tok tokenizer) *Plugin {
	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: "test"},
		backend:   renderBackend{tk: tok},
	}
}

func TestProduceTimeout(t *testing.T) {
	ctx := context.Background()

	// vLLM backend surfaces its configured render timeout (default mmTimeout).
	vp, err := NewPlugin(ctx, "tok", &tokenizerPluginConfig{ModelName: "m", VLLM: &vllmConfig{}})
	require.NoError(t, err)
	assert.Equal(t, defaultHTTPRenderMMTimeout, vp.ProduceTimeout())

	// The override value is the plugin's own configurable timeout.
	vp2, err := NewPlugin(ctx, "tok", &tokenizerPluginConfig{ModelName: "m", VLLM: &vllmConfig{MMTimeout: "45s"}})
	require.NoError(t, err)
	assert.Equal(t, 45*time.Second, vp2.ProduceTimeout())

	// Estimate backend declares none, so the director keeps its default.
	ep, err := NewPlugin(ctx, "tok", &tokenizerPluginConfig{Estimate: &estimateConfig{}})
	require.NoError(t, err)
	assert.Zero(t, ep.ProduceTimeout())

	// A render backend whose tokenizer manages no timeout (e.g. UDS) keeps the default.
	assert.Zero(t, newTestPlugin(&mockTokenizer{}).ProduceTimeout())
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
			name:      "empty object selects estimate",
			params:    `{}`,
			expectErr: false,
		},
		{
			name:      "nil parameters select estimate",
			params:    "",
			expectErr: false,
		},
		{
			name:       "render backend requires modelName",
			params:     `{"vllm":{}}`,
			expectErr:  true,
			errContain: "'modelName' must be specified",
		},
		{
			name:      "estimate image static mode parses",
			params:    `{"estimate":{"image":{"mode":"static","static":{"staticToken":8}}}}`,
			expectErr: false,
		},
		{
			name:       "invalid estimate image mode",
			params:     `{"estimate":{"image":{"mode":"bogus"}}}`,
			expectErr:  true,
			errContain: "estimate.image.mode must be",
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

			p, err := PluginFactory("test-tokenizer", plugin.StrictDecoder(rawParams), handle)
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
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return []uint32{1, 2, 3, 4}, mm, nil
		},
	}
	p := newTestPlugin(tok)

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hi"}}},
			},
			Payload: fwkrh.PayloadMap{},
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedPrompt)
	assert.Equal(t, []uint32{1, 2, 3, 4}, req.Body.TokenizedPrompt.PerPromptTokens[0])
	require.Len(t, req.Body.TokenizedPrompt.MultiModalFeatures, 1)
	require.Len(t, req.Body.TokenizedPrompt.MultiModalFeatures[0], 2)

	assert.Equal(t, 3, req.Body.TokenizedPrompt.MultiModalFeatures[0][0].Offset)
	assert.Equal(t, "hash-a", req.Body.TokenizedPrompt.MultiModalFeatures[0][0].Hash)
	assert.Equal(t, 20, req.Body.TokenizedPrompt.MultiModalFeatures[0][1].Offset)
	assert.Equal(t, "hash-b", req.Body.TokenizedPrompt.MultiModalFeatures[0][1].Hash)
	assert.Equal(t, fwkrh.ModalityImage, req.Body.TokenizedPrompt.MultiModalFeatures[0][0].Modality)
}

func TestProduce_SkipsWhenAlreadyPopulated(t *testing.T) {
	existing := &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{{42}}}
	p := newTestPlugin(&mockTokenizer{})
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{TokenizedPrompt: existing},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	assert.Same(t, existing, req.Body.TokenizedPrompt)
}

func TestProduce_SetsCacheSaltOnSkipPath(t *testing.T) {
	tok := &mockTokenizer{
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Fatal("backend must not run on the skip path")
			return nil, nil, nil
		},
	}
	existing := &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{{1, 2, 3}}}
	p := newTestPlugin(tok)
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{CacheSalt: "tenant-x"},
			TokenizedPrompt: existing,
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	assert.Same(t, existing, req.Body.TokenizedPrompt)
	assert.Equal(t, "tenant-x", req.Body.TokenizedPrompt.CacheSalt)
	assert.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedPrompt.PerPromptTokens[0])
}

func TestRenderBackend_CompletionsTokenIDsPassthrough(t *testing.T) {
	tok := &mockTokenizer{
		renderFunc: func(fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			t.Fatal("render must not run when token IDs are provided")
			return nil, nil, nil
		},
	}
	tp, err := renderBackend{tk: tok}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: []uint32{5, 6, 7}}},
	})
	require.NoError(t, err)
	assert.Equal(t, []uint32{5, 6, 7}, tp.PerPromptTokens[0])
}

func TestRenderBackend_CompletionsArrayPassesArrayPayload(t *testing.T) {
	tok := &mockTokenizer{
		renderFunc: func(payload fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			pm, ok := payload.AsMap()
			require.True(t, ok)
			arr, ok := pm["prompt"].([]string)
			require.True(t, ok, "multi-string prompt must be passed as []string")
			assert.Equal(t, []string{"alpha", "beta"}, arr)
			return [][]uint32{{1, 2}, {3}}, nil, nil
		},
	}
	tp, err := renderBackend{tk: tok}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Strings: []string{"alpha", "beta"}}},
	})
	require.NoError(t, err)
	assert.Equal(t, [][]uint32{{1, 2}, {3}}, tp.PerPromptTokens)
}

func TestRenderBackend_CompletionsSingleArrayUsesPlainText(t *testing.T) {
	var got string
	tok := &mockTokenizer{
		renderFunc: func(payload fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			pm, ok := payload.AsMap()
			require.True(t, ok)
			got, _ = pm["prompt"].(string)
			return [][]uint32{{1}}, nil, nil
		},
	}
	tp, err := renderBackend{tk: tok}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Strings: []string{"alpha beta"}}},
	})
	require.NoError(t, err)
	assert.Equal(t, "alpha beta", got)
	assert.Equal(t, [][]uint32{{1}}, tp.PerPromptTokens)
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
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return nil, nil, assert.AnError
		},
	}
	p := newTestPlugin(tok)
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hi"}}},
			},
			Payload: fwkrh.PayloadMap{},
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
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{},
		},
	}
	err := p.Produce(context.Background(), req, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported request body type")
	assert.Nil(t, req.Body.TokenizedPrompt)
}

func TestProduce_GenerateUsesPreTokenizedIDs(t *testing.T) {
	// Generate requests carry pre-tokenized IDs — the tokenizer must NOT be called.
	tok := &mockTokenizer{
		renderFunc: func(_ fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			t.Error("tokenizer.Render must not be called for generate requests")
			return nil, nil, nil
		},
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Error("tokenizer.RenderChat must not be called for generate requests")
			return nil, nil, nil
		},
	}
	p := newTestPlugin(tok)

	tokenIDs := []uint32{1, 2, 3, 4, 5}
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Generate: &fwkrh.GenerateRequest{
				TokenIDs: tokenIDs,
			},
		},
	}

	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedPrompt)
	assert.Equal(t, tokenIDs, req.Body.TokenizedPrompt.PerPromptTokens[0])
	assert.Nil(t, req.Body.TokenizedPrompt.MultiModalFeatures)
}

func TestProduce_GenerateFlattensFeatures(t *testing.T) {
	// Generate requests with multimodal features must populate TokenizedPrompt.MultiModalFeatures
	// in offset-sorted prompt order, so downstream prefix-cache scoring picks up image hashes.
	tok := &mockTokenizer{
		renderFunc: func(_ fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			t.Error("tokenizer.Render must not be called for generate requests")
			return nil, nil, nil
		},
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Error("tokenizer.RenderChat must not be called for generate requests")
			return nil, nil, nil
		},
	}
	p := newTestPlugin(tok)

	tokenIDs := []uint32{151644, 872, 198, 3838, 374, 279, 6722}
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Generate: &fwkrh.GenerateRequest{
				TokenIDs: tokenIDs,
				Features: &tokenization.MultiModalFeatures{
					MMHashes: map[string][]string{
						"image": {"abc123hash", "def456hash"},
					},
					MMPlaceholders: map[string][]kvblock.PlaceholderRange{
						"image": {
							{Offset: 1, Length: 3},
							{Offset: 4, Length: 3},
						},
					},
				},
			},
		},
	}

	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedPrompt)
	assert.Equal(t, tokenIDs, req.Body.TokenizedPrompt.PerPromptTokens[0])
	assert.Equal(t,
		[][]fwkrh.MultiModalFeature{{
			{Modality: fwkrh.ModalityImage, Hash: "abc123hash", Offset: 1, Length: 3},
			{Modality: fwkrh.ModalityImage, Hash: "def456hash", Offset: 4, Length: 3},
		}},
		req.Body.TokenizedPrompt.MultiModalFeatures,
	)
}

func TestConvertMMFeaturesRoundTrip(t *testing.T) {
	src := &tokenization.MultiModalFeatures{
		MMHashes: map[string][]string{"image": {"h1", "h2"}},
		MMPlaceholders: map[string][]kvblock.PlaceholderRange{
			"image": {{Offset: 1, Length: 2}, {Offset: 10, Length: 3}},
		},
	}
	upstream := convertMMFeaturesToUpstream(src)
	require.Len(t, upstream, 1)
	require.Len(t, upstream[0], 2)

	hashes, ranges := ConvertMMFeaturesFromUpstream(upstream[0])
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
	toolCalls := []any{
		map[string]any{
			"id":   "chatcmpl-tool-1",
			"type": "function",
			"function": map[string]any{
				"name":      "bash",
				"arguments": `{"command":"ls -la"}`,
			},
		},
	}
	chat := &fwkrh.ChatCompletionsRequest{
		Messages: []fwkrh.Message{
			{Role: "system", Content: fwkrh.Content{Raw: "You are a helpful assistant."}},
			{
				Role:      "assistant",
				Content:   fwkrh.Content{Raw: "Reflection."},
				ToolCalls: toolCalls,
			},
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
	assert.Equal(t, "assistant", result.Conversation[1].Role)
	assert.Equal(t, tokenizerTypes.Content{Raw: "Reflection."}, result.Conversation[1].Content)
	assert.Equal(t, "template", result.ChatTemplate)
	assert.True(t, result.AddGenerationPrompt)
	assert.False(t, result.ContinueFinalMessage)
	assert.True(t, result.ReturnAssistantTokensMask)

	assert.Equal(t, toolCalls, result.Conversation[1].ToolCalls)
}

func TestProduce_StringArrayPrompt(t *testing.T) {
	tok := &mockTokenizer{
		renderFunc: func(payload fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			pm, _ := payload.AsMap()
			_, ok := pm["prompt"].([]string)
			require.True(t, ok, "multi-string prompt must be passed as []string")
			return [][]uint32{{10, 20, 30}, {40, 50}}, nil, nil
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
	require.Len(t, req.Body.TokenizedPrompt.PerPromptTokens, 2)
	assert.Equal(t, []uint32{10, 20, 30}, req.Body.TokenizedPrompt.PerPromptTokens[0])
	assert.Equal(t, []uint32{40, 50}, req.Body.TokenizedPrompt.PerPromptTokens[1])
}

func TestProduce_StringArrayPromptRenderError(t *testing.T) {
	tok := &mockTokenizer{
		renderFunc: func(fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			return nil, nil, errors.New("render failed")
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
	err := p.Produce(context.Background(), req, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tokenization failed")
	assert.Nil(t, req.Body.TokenizedPrompt)
}

func TestProduce_StringArrayPromptDoesNotPublishEmptyTokenResult(t *testing.T) {
	tok := &mockTokenizer{
		renderFunc: func(_ fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			return nil, nil, nil
		},
	}
	p := newTestPlugin(tok)

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Strings: []string{"", ""}},
			},
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	assert.Nil(t, req.Body.TokenizedPrompt)
}

func TestProduce_SinglePromptSetsPerPromptTokens(t *testing.T) {
	tok := &mockTokenizer{
		renderFunc: func(_ fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			return [][]uint32{{10, 20, 30}}, nil, nil
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
	assert.Equal(t, [][]uint32{{10, 20, 30}}, req.Body.TokenizedPrompt.PerPromptTokens)
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
