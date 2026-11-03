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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

// failingTokenizer fails the test when the backend renders, for the skip paths.
func failingTokenizer(t *testing.T) *mockTokenizer {
	t.Helper()
	return &mockTokenizer{
		renderFunc: func(fwkrequest.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			t.Error("backend must not render a request that carries prompt token IDs")
			return nil, nil, nil
		},
		renderChatFunc: func(fwkrequest.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Error("backend must not render a request that carries prompt token IDs")
			return nil, nil, nil
		},
	}
}

func chatBodyWithPayload(payload fwkrequest.RequestPayload) *fwkrequest.InferenceRequestBody {
	return &fwkrequest.InferenceRequestBody{
		ChatCompletions: &fwkrequest.ChatCompletionsRequest{CacheSalt: "tenant-x"},
		Payload:         payload,
	}
}

func TestProduce_UsesClientPromptTokenIDs(t *testing.T) {
	tests := map[string]struct {
		params any
		want   []uint32
	}{
		"raw json envelope": {
			params: json.RawMessage(`{"prompt_token_ids":[5,6,7]}`),
			want:   []uint32{5, 6, 7},
		},
		"decoded map with json numbers": {
			params: map[string]any{"prompt_token_ids": []any{json.Number("5"), json.Number("6")}},
			want:   []uint32{5, 6},
		},
		"decoded map with float numbers": {
			params: map[string]any{"prompt_token_ids": []any{float64(9), float64(8)}},
			want:   []uint32{9, 8},
		},
		"ids written by a mutation": {
			params: map[string]any{"prompt_token_ids": []uint32{4, 2}},
			want:   []uint32{4, 2},
		},
		"raw json outside the canonical form": {
			params: json.RawMessage(`{"prompt_token_ids":[1.0, 2e0]}`),
			want:   []uint32{1, 2},
		},
		"other kv params preserved alongside": {
			params: json.RawMessage(`{"remote_engine_id":"e","prompt_token_ids":[1]}`),
			want:   []uint32{1},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			p := newTestPlugin(failingTokenizer(t))
			body := chatBodyWithPayload(fwkrequest.PayloadMap{reqcommon.FieldKVTransferParams: tc.params})
			req := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, p.Produce(context.Background(), req, nil))

			require.NotNil(t, req.Body.TokenizedRequest)
			require.Len(t, req.Body.TokenizedRequest.Prompts, 1)
			assert.Equal(t, tc.want, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
			assert.Nil(t, req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures)
			assert.Equal(t, "tenant-x", req.Body.TokenizedRequest.CacheSalt)
			assert.False(t, body.Mutated)
		})
	}
}

func TestProduce_ClientPromptTokenIDsOnEveryChatShapedAPI(t *testing.T) {
	for name, body := range map[string]*fwkrequest.InferenceRequestBody{
		"messages":  {Messages: &fwkrequest.MessagesRequest{}},
		"responses": {Responses: &fwkrequest.ResponsesRequest{}},
	} {
		t.Run(name, func(t *testing.T) {
			p := newTestPlugin(failingTokenizer(t))
			body.Payload = fwkrequest.PayloadMap{
				reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[3,4]}`),
			}
			req := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, p.Produce(context.Background(), req, nil))

			require.NotNil(t, req.Body.TokenizedRequest)
			assert.Equal(t, []uint32{3, 4}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
		})
	}
}

// TestProduce_KVPromptTokenIDsIgnoredOffChatPath checks that an API outside
// vLLM's chat path keeps its own tokenization: the field is ignored there, and
// a generate request's features would be lost.
func TestProduce_KVPromptTokenIDsIgnoredOffChatPath(t *testing.T) {
	kvParams := json.RawMessage(`{"prompt_token_ids":[3,4]}`)
	features := &tokenization.MultiModalFeatures{
		MMHashes:       map[string][]string{"image": {"h0"}},
		MMPlaceholders: map[string][]kvblock.PlaceholderRange{"image": {{Offset: 0, Length: 2}}},
	}
	body := &fwkrequest.InferenceRequestBody{
		Generate: &fwkrequest.GenerateRequest{TokenIDs: []uint32{1, 2}, Features: features},
		Payload:  fwkrequest.PayloadMap{reqcommon.FieldKVTransferParams: kvParams},
	}
	req := &scheduling.InferenceRequest{Body: body}

	require.NoError(t, newTestPlugin(failingTokenizer(t)).Produce(context.Background(), req, nil))

	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []uint32{1, 2}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
	assert.Equal(t, convertMMFeaturesToUpstream(features), req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures)
}

// TestProduce_KVPromptTokenIDsRenderedWithMedia checks that a chat-shaped request
// with media still renders: vLLM ignores the ids when content is not text, and
// the render supplies the multimodal features.
func TestProduce_KVPromptTokenIDsRenderedWithMedia(t *testing.T) {
	backendTokens := []uint32{42}
	render := &mockTokenizer{
		renderChatFunc: func(fwkrequest.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return backendTokens, nil, nil
		},
	}
	image := fwkrequest.ContentBlock{Type: reqcommon.PartTypeImageURL, ImageURL: fwkrequest.ImageBlock{URL: "http://img"}}
	untypedImage := fwkrequest.ContentBlock{ImageURL: fwkrequest.ImageBlock{URL: "http://img"}}
	tests := map[string]*fwkrequest.InferenceRequestBody{
		"chat image part": {ChatCompletions: &fwkrequest.ChatCompletionsRequest{
			Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Structured: []fwkrequest.ContentBlock{image}}}},
		}},
		"chat part without a type": {ChatCompletions: &fwkrequest.ChatCompletionsRequest{
			Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Structured: []fwkrequest.ContentBlock{untypedImage}}}},
		}},
		"messages image block": {Messages: &fwkrequest.MessagesRequest{
			Messages: []fwkrequest.AnthropicMessage{{Role: "user", Content: fwkrequest.AnthropicContent{
				Structured: []fwkrequest.AnthropicContentBlock{{Type: reqcommon.BlockTypeImage}},
			}}},
		}},
		"messages image in tool result": {Messages: &fwkrequest.MessagesRequest{
			Messages: []fwkrequest.AnthropicMessage{{Role: "user", Content: fwkrequest.AnthropicContent{
				Structured: []fwkrequest.AnthropicContentBlock{{
					Type:    reqcommon.BlockTypeToolResult,
					Content: json.RawMessage(`[{"type":"image","source":{"type":"url","url":"http://img"}}]`),
				}},
			}}},
		}},
		"responses input image": {Responses: &fwkrequest.ResponsesRequest{
			Input: []any{map[string]any{"role": "user", "content": []any{
				map[string]any{"type": reqcommon.PartTypeInputImage, "image_url": "http://img"},
			}}},
		}},
		"responses image in function call output": {Responses: &fwkrequest.ResponsesRequest{
			Input: []any{map[string]any{"type": "function_call_output", "output": []any{
				map[string]any{"type": reqcommon.PartTypeInputImage, "image_url": "http://img"},
			}}},
		}},
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			body.Payload = fwkrequest.PayloadMap{
				reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[3,4]}`),
			}
			req := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, newTestPlugin(render).Produce(context.Background(), req, nil))

			require.NotNil(t, req.Body.TokenizedRequest)
			assert.Equal(t, backendTokens, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
		})
	}
}

// TestProduce_KVPromptTokenIDsWithTextOnlyParts checks that text-only structured
// content still uses the supplied ids.
func TestProduce_KVPromptTokenIDsWithTextOnlyParts(t *testing.T) {
	text := []fwkrequest.ContentBlock{{Type: reqcommon.PartTypeText, Text: "hi"}}
	tests := map[string]*fwkrequest.InferenceRequestBody{
		"chat text part": {ChatCompletions: &fwkrequest.ChatCompletionsRequest{
			Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Structured: text}}},
		}},
		"messages text block": {Messages: &fwkrequest.MessagesRequest{
			Messages: []fwkrequest.AnthropicMessage{{Role: "user", Content: fwkrequest.AnthropicContent{
				Structured: []fwkrequest.AnthropicContentBlock{{Type: reqcommon.PartTypeText, Text: "hi"}},
			}}},
		}},
		"responses input text": {Responses: &fwkrequest.ResponsesRequest{
			Input: []any{map[string]any{"role": "user", "content": []any{
				map[string]any{"type": reqcommon.PartTypeInputText, "text": "hi"},
			}}},
		}},
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			body.Payload = fwkrequest.PayloadMap{
				reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[3,4]}`),
			}
			req := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, newTestPlugin(failingTokenizer(t)).Produce(context.Background(), req, nil))

			require.NotNil(t, req.Body.TokenizedRequest)
			assert.Equal(t, []uint32{3, 4}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
		})
	}
}

func TestProduce_ClientPromptTokenIDsWithEstimateBackend(t *testing.T) {
	p, err := NewPlugin(context.Background(), "tok", &tokenizerPluginConfig{Estimate: &estimateConfig{}})
	require.NoError(t, err)
	req := &scheduling.InferenceRequest{Body: chatBodyWithPayload(fwkrequest.PayloadMap{
		reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[5,6,7]}`),
	})}

	require.NoError(t, p.Produce(context.Background(), req, nil))

	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []uint32{5, 6, 7}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
}

func TestProduce_CompletionsTokenPromptPassthrough(t *testing.T) {
	tests := map[string]struct {
		prompt fwkrequest.Prompt
		want   [][]uint32
	}{
		"single array": {
			prompt: fwkrequest.Prompt{TokenIDs: [][]uint32{{1, 2, 3}}},
			want:   [][]uint32{{1, 2, 3}},
		},
		"nested arrays": {
			prompt: fwkrequest.Prompt{TokenIDs: [][]uint32{{1, 2}, {3}}},
			want:   [][]uint32{{1, 2}, {3}},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			p := newTestPlugin(failingTokenizer(t))
			body := &fwkrequest.InferenceRequestBody{
				Completions: &fwkrequest.CompletionsRequest{Prompt: tc.prompt, CacheSalt: "s"},
				Payload:     fwkrequest.PayloadMap{"prompt": []any{1, 2}},
			}
			req := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, p.Produce(context.Background(), req, nil))

			require.NotNil(t, req.Body.TokenizedRequest)
			assert.Equal(t, tc.want[0], req.Body.TokenizedRequest.Prompts[0].TokenIDs)
			require.Len(t, req.Body.TokenizedRequest.Prompts, len(tc.want))
			assert.Equal(t, "s", req.Body.TokenizedRequest.CacheSalt)
		})
	}
}

func TestProduce_ClientPromptTokenIDsFallsBackToBackend(t *testing.T) {
	backendTokens := []uint32{42}
	render := &mockTokenizer{
		renderFunc: func(fwkrequest.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			return [][]uint32{backendTokens}, nil, nil
		},
		renderChatFunc: func(fwkrequest.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return backendTokens, nil, nil
		},
	}
	tests := map[string]*fwkrequest.InferenceRequestBody{
		"no kv transfer params": chatBodyWithPayload(fwkrequest.PayloadMap{"model": "m"}),
		"kv transfer params null": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: nil,
		}),
		"kv transfer params not an object": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: "x",
		}),
		"kv transfer params invalid json": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[1`),
		}),
		"prompt token ids absent": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: json.RawMessage(`{"remote_engine_id":"e"}`),
		}),
		"prompt token ids empty": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[]}`),
		}),
		"prompt token ids not a list": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": "x"},
		}),
		"prompt token ids mixed types": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []any{json.Number("1"), "x"}},
		}),
		"prompt token ids negative": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []any{json.Number("-1")}},
		}),
		"prompt token ids fractional": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []any{json.Number("1.5")}},
		}),
		"prompt token ids overflow": chatBodyWithPayload(fwkrequest.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []any{json.Number("4294967296")}},
		}),
		"raw payload": chatBodyWithPayload(fwkrequest.RawPayload(`{}`)),
		"completions text prompt": {
			Completions: &fwkrequest.CompletionsRequest{Prompt: fwkrequest.Prompt{Raw: "hi"}},
			Payload:     fwkrequest.PayloadMap{"prompt": "hi"},
		},
		"completions text prompt with kv ids": {
			Completions: &fwkrequest.CompletionsRequest{Prompt: fwkrequest.Prompt{Raw: "hi"}},
			Payload: fwkrequest.PayloadMap{
				"prompt":                        "hi",
				reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[7]}`),
			},
		},
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			p := newTestPlugin(render)
			req := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, p.Produce(context.Background(), req, nil))

			require.NotNil(t, req.Body.TokenizedRequest)
			assert.Equal(t, backendTokens, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
		})
	}
}
