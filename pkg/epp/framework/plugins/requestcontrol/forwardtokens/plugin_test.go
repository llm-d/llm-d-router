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

package forwardtokens

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func tokenized(ids ...uint32) *requesthandling.TokenizedRequest {
	return &requesthandling.TokenizedRequest{Prompts: []requesthandling.PromptTokens{{TokenIDs: ids}}}
}

// chat returns a chat-completions body, the shape the plugin acts on.
func chat(payload requesthandling.RequestPayload, tr *requesthandling.TokenizedRequest) *requesthandling.InferenceRequestBody {
	return &requesthandling.InferenceRequestBody{
		ChatCompletions: &requesthandling.ChatCompletionsRequest{}, Payload: payload, TokenizedRequest: tr,
	}
}

func run(t *testing.T, p *Plugin, body *requesthandling.InferenceRequestBody) {
	t.Helper()
	require.NoError(t, p.PreRequest(context.Background(), &scheduling.InferenceRequest{Body: body}, nil))
}

func TestPluginFactory(t *testing.T) {
	p, err := PluginFactory("fwd", json.NewDecoder(strings.NewReader(`{"minTokens": 512}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, 512, p.(*Plugin).minTokens)
	assert.Equal(t, PluginType, p.TypedName().Type)

	p, err = PluginFactory("fwd", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, 0, p.(*Plugin).minTokens)

	_, err = PluginFactory("fwd", json.NewDecoder(strings.NewReader(`{"minTokens": -1}`)), nil)
	assert.Error(t, err)
	_, err = PluginFactory("fwd", json.NewDecoder(strings.NewReader(`{"unknown": 1}`)), nil)
	assert.NoError(t, err) // plain decoder: strictness is the framework loader's job
}

func TestForwardsTokenIDs(t *testing.T) {
	p, err := New("fwd", Config{})
	require.NoError(t, err)
	payload := requesthandling.PayloadMap{"model": "m"}
	body := chat(payload, tokenized(1, 2, 3))

	run(t, p, body)

	assert.True(t, body.Mutated)
	params := payload["kv_transfer_params"].(map[string]any)
	assert.Equal(t, []uint32{1, 2, 3}, params["prompt_token_ids"])
	out, err := json.Marshal(payload)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"m","kv_transfer_params":{"prompt_token_ids":[1,2,3]}}`, string(out))
}

func TestPreservesExistingKVTransferParams(t *testing.T) {
	p, _ := New("fwd", Config{})
	for name, existing := range map[string]any{
		"decoded map": map[string]any{"max_load_tokens": json.Number("0")},
		"raw json":    json.RawMessage(`{"max_load_tokens":0}`),
	} {
		t.Run(name, func(t *testing.T) {
			payload := requesthandling.PayloadMap{"kv_transfer_params": existing}
			body := chat(payload, tokenized(7))
			run(t, p, body)
			params := payload["kv_transfer_params"].(map[string]any)
			assert.Contains(t, params, "max_load_tokens")
			assert.Equal(t, []uint32{7}, params["prompt_token_ids"])
		})
	}
}

func TestKeepsClientTokenIDs(t *testing.T) {
	p, _ := New("fwd", Config{})
	payload := requesthandling.PayloadMap{"kv_transfer_params": map[string]any{"prompt_token_ids": []any{9}}}
	body := chat(payload, tokenized(1, 2))
	run(t, p, body)
	assert.False(t, body.Mutated)
	assert.Equal(t, []any{9}, payload["kv_transfer_params"].(map[string]any)["prompt_token_ids"])
}

func TestSkips(t *testing.T) {
	p, _ := New("fwd", Config{MinTokens: 3})
	mm := tokenized(1, 2, 3, 4)
	mm.Prompts[0].MultiModalFeatures = []requesthandling.MultiModalFeature{{}}
	multi := &requesthandling.TokenizedRequest{Prompts: []requesthandling.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}, {TokenIDs: []uint32{4, 5, 6}}}}

	tests := map[string]*requesthandling.InferenceRequestBody{
		"not tokenized":     chat(requesthandling.PayloadMap{}, nil),
		"below minTokens":   chat(requesthandling.PayloadMap{}, tokenized(1, 2)),
		"multimodal":        chat(requesthandling.PayloadMap{}, mm),
		"multi-prompt":      chat(requesthandling.PayloadMap{}, multi),
		"raw payload":       chat(requesthandling.RawPayload(`{}`), tokenized(1, 2, 3)),
		"params not object": chat(requesthandling.PayloadMap{"kv_transfer_params": "x"}, tokenized(1, 2, 3)),
		"completions": {Completions: &requesthandling.CompletionsRequest{},
			Payload: requesthandling.PayloadMap{}, TokenizedRequest: tokenized(1, 2, 3)},
		"generate": {Generate: &requesthandling.GenerateRequest{},
			Payload: requesthandling.PayloadMap{}, TokenizedRequest: tokenized(1, 2, 3)},
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			run(t, p, body)
			assert.False(t, body.Mutated)
			if m, ok := body.Payload.(requesthandling.PayloadMap); ok {
				if v, ok := m["kv_transfer_params"].(map[string]any); ok {
					assert.NotContains(t, v, "prompt_token_ids")
				}
			}
		})
	}
	require.NoError(t, p.PreRequest(context.Background(), nil, nil))
	require.NoError(t, p.PreRequest(context.Background(), &scheduling.InferenceRequest{}, nil))
}

// vLLM serves Messages and Responses through its chat path, so both get the IDs too.
func TestForwardsForEveryChatShapedAPI(t *testing.T) {
	p, _ := New("fwd", Config{})
	for name, body := range map[string]*requesthandling.InferenceRequestBody{
		"chat completions": {ChatCompletions: &requesthandling.ChatCompletionsRequest{}},
		"messages":         {Messages: &requesthandling.MessagesRequest{}},
		"responses":        {Responses: &requesthandling.ResponsesRequest{}},
	} {
		t.Run(name, func(t *testing.T) {
			payload := requesthandling.PayloadMap{}
			body.Payload, body.TokenizedRequest = payload, tokenized(4, 5)
			run(t, p, body)
			assert.True(t, body.Mutated)
			assert.Equal(t, []uint32{4, 5}, payload["kv_transfer_params"].(map[string]any)["prompt_token_ids"])
		})
	}
}
