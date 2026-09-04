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
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
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

func (m *mockTokenizer) RenderMessages(ctx context.Context, payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	return m.RenderChat(ctx, payload)
}

func newTestPlugin(tok tokenizer) *Plugin {
	return &Plugin{
		typedName:   plugin.TypedName{Type: PluginType, Name: "test"},
		backend:     renderBackend{tk: tok},
		backendName: backendVLLM,
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

	// A render backend whose tokenizer manages no timeout keeps the default.
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

func TestProduce_PopulatesTokenizedRequest(t *testing.T) {
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
	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []uint32{1, 2, 3, 4}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
	require.Len(t, req.Body.TokenizedRequest.Prompts, 1)
	require.Len(t, req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures, 2)

	assert.Equal(t, 3, req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures[0].Offset)
	assert.Equal(t, "hash-a", req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures[0].Hash)
	assert.Equal(t, 20, req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures[1].Offset)
	assert.Equal(t, "hash-b", req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures[1].Hash)
	assert.Equal(t, fwkrh.ModalityImage, req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures[0].Modality)
}

func TestProduce_SkipsWhenAlreadyPopulated(t *testing.T) {
	existing := &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{42}}}}
	p := newTestPlugin(&mockTokenizer{})
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{TokenizedRequest: existing},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	assert.Same(t, existing, req.Body.TokenizedRequest)
}

func TestProduce_SetsCacheSaltOnSkipPath(t *testing.T) {
	tok := &mockTokenizer{
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Fatal("backend must not run on the skip path")
			return nil, nil, nil
		},
	}
	existing := &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}}}
	p := newTestPlugin(tok)
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions:  &fwkrh.ChatCompletionsRequest{CacheSalt: "tenant-x"},
			TokenizedRequest: existing,
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	assert.Same(t, existing, req.Body.TokenizedRequest)
	assert.Equal(t, "tenant-x", req.Body.TokenizedRequest.CacheSalt)
	assert.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
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
	assert.Nil(t, req.Body.TokenizedRequest)
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
	assert.Nil(t, req.Body.TokenizedRequest)
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
	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, tokenIDs, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
	assert.Nil(t, req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures)
}

func TestProduce_GenerateFlattensFeatures(t *testing.T) {
	// Generate requests with multimodal features must populate PromptTokens.MultiModalFeatures
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
	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, tokenIDs, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
	assert.Equal(t,
		[]fwkrh.MultiModalFeature{
			{Modality: fwkrh.ModalityImage, Hash: "abc123hash", Offset: 1, Length: 3},
			{Modality: fwkrh.ModalityImage, Hash: "def456hash", Offset: 4, Length: 3},
		},
		req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures,
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
