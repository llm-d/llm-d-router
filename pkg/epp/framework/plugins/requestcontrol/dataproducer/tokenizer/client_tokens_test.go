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
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

// failingTokenizer fails the test when the backend renders, for the skip paths.
func failingTokenizer(t *testing.T) *mockTokenizer {
	t.Helper()
	return &mockTokenizer{
		renderFunc: func(fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			t.Error("backend must not render a request that carries prompt token IDs")
			return nil, nil, nil
		},
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Error("backend must not render a request that carries prompt token IDs")
			return nil, nil, nil
		},
	}
}

func chatBody(payload fwkrh.RequestPayload) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{CacheSalt: "tenant-x"},
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
		"other kv params preserved alongside": {
			params: json.RawMessage(`{"remote_engine_id":"e","prompt_token_ids":[1]}`),
			want:   []uint32{1},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			p := newTestPlugin(failingTokenizer(t))
			body := chatBody(fwkrh.PayloadMap{reqcommon.FieldKVTransferParams: tc.params})
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
	for name, body := range map[string]*fwkrh.InferenceRequestBody{
		"messages":  {Messages: &fwkrh.MessagesRequest{}},
		"responses": {Responses: &fwkrh.ResponsesRequest{}},
		"generate":  {Generate: &fwkrh.GenerateRequest{}},
	} {
		t.Run(name, func(t *testing.T) {
			p := newTestPlugin(failingTokenizer(t))
			body.Payload = fwkrh.PayloadMap{
				reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[3,4]}`),
			}
			req := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, p.Produce(context.Background(), req, nil))

			require.NotNil(t, req.Body.TokenizedRequest)
			assert.Equal(t, []uint32{3, 4}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
		})
	}
}

func TestProduce_ClientPromptTokenIDsWithEstimateBackend(t *testing.T) {
	p, err := NewPlugin(context.Background(), "tok", &tokenizerPluginConfig{Estimate: &estimateConfig{}})
	require.NoError(t, err)
	req := &scheduling.InferenceRequest{Body: chatBody(fwkrh.PayloadMap{
		reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[5,6,7]}`),
	})}

	require.NoError(t, p.Produce(context.Background(), req, nil))

	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []uint32{5, 6, 7}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
}

func TestProduce_CompletionsTokenPromptPassthrough(t *testing.T) {
	tests := map[string]struct {
		prompt fwkrh.Prompt
		want   [][]uint32
	}{
		"single array": {
			prompt: fwkrh.Prompt{TokenIDs: [][]uint32{{1, 2, 3}}},
			want:   [][]uint32{{1, 2, 3}},
		},
		"nested arrays": {
			prompt: fwkrh.Prompt{TokenIDs: [][]uint32{{1, 2}, {3}}},
			want:   [][]uint32{{1, 2}, {3}},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			p := newTestPlugin(failingTokenizer(t))
			body := &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{Prompt: tc.prompt, CacheSalt: "s"},
				Payload:     fwkrh.PayloadMap{"prompt": []any{1, 2}},
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
		renderFunc: func(fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
			return [][]uint32{backendTokens}, nil, nil
		},
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return backendTokens, nil, nil
		},
	}
	tests := map[string]*fwkrh.InferenceRequestBody{
		"no kv transfer params": chatBody(fwkrh.PayloadMap{"model": "m"}),
		"kv transfer params null": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: nil,
		}),
		"kv transfer params not an object": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: "x",
		}),
		"kv transfer params invalid json": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[1`),
		}),
		"prompt token ids absent": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: json.RawMessage(`{"remote_engine_id":"e"}`),
		}),
		"prompt token ids empty": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: json.RawMessage(`{"prompt_token_ids":[]}`),
		}),
		"prompt token ids not a list": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": "x"},
		}),
		"prompt token ids mixed types": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []any{json.Number("1"), "x"}},
		}),
		"prompt token ids negative": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []any{json.Number("-1")}},
		}),
		"prompt token ids fractional": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []any{json.Number("1.5")}},
		}),
		"prompt token ids overflow": chatBody(fwkrh.PayloadMap{
			reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []any{json.Number("4294967296")}},
		}),
		"raw payload": chatBody(fwkrh.RawPayload(`{}`)),
		"completions text prompt": {
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hi"}},
			Payload:     fwkrh.PayloadMap{"prompt": "hi"},
		},
		"completions text prompt with kv ids": {
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hi"}},
			Payload: fwkrh.PayloadMap{
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
