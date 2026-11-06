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

	commonrequest "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
)

func tokenized(ids ...uint32) *fwkrequest.TokenizedRequest {
	return &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: ids}}}
}

// chat returns a chat-completions body, the shape the plugin acts on.
func chat(payload fwkrequest.RequestPayload, tr *fwkrequest.TokenizedRequest) *fwkrequest.InferenceRequestBody {
	return &fwkrequest.InferenceRequestBody{
		ChatCompletions: &fwkrequest.ChatCompletionsRequest{}, Payload: payload, TokenizedRequest: tr,
	}
}

func run(t *testing.T, p *Plugin, body *fwkrequest.InferenceRequestBody) {
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
	payload := fwkrequest.PayloadMap{"model": "m"}
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
			payload := fwkrequest.PayloadMap{"kv_transfer_params": existing}
			body := chat(payload, tokenized(7))
			run(t, p, body)
			params := payload["kv_transfer_params"].(map[string]any)
			assert.Contains(t, params, "max_load_tokens")
			assert.Equal(t, []uint32{7}, params["prompt_token_ids"])
		})
	}
}

func TestKeepsClientTokenIDs(t *testing.T) {
	p, _ := New("fwd", Config{TrustCallerTokenIDs: true})
	payload := fwkrequest.PayloadMap{"kv_transfer_params": map[string]any{"prompt_token_ids": []any{9}}}
	body := chat(payload, tokenized(1, 2))
	run(t, p, body)
	assert.False(t, body.Mutated)
	assert.Equal(t, []any{9}, payload["kv_transfer_params"].(map[string]any)["prompt_token_ids"])
}

// Parsers leave kv_transfer_params as json.RawMessage; a long client-supplied
// ID array must be kept without being materialized into per-token boxed values.
func TestKeepsClientTokenIDsRaw(t *testing.T) {
	p, _ := New("fwd", Config{TrustCallerTokenIDs: true})
	raw := json.RawMessage(`{"prompt_token_ids": [7, 8, 9], "do_remote_prefill": true}`)
	payload := fwkrequest.PayloadMap{"kv_transfer_params": raw}
	body := chat(payload, tokenized(1, 2))
	run(t, p, body)
	assert.False(t, body.Mutated)
	// The raw bytes pass through untouched.
	assert.JSONEq(t, `{"prompt_token_ids": [7, 8, 9], "do_remote_prefill": true}`,
		string(payload["kv_transfer_params"].(json.RawMessage)))
}

// An explicit null kv_transfer_params.prompt_token_ids counts as unset: the
// forwarded IDs are written in its place, in both wire forms.
func TestNullPromptTokenIDsIsForwarded(t *testing.T) {
	p, _ := New("fwd", Config{})
	for name, payload := range map[string]fwkrequest.PayloadMap{
		"raw json":    {"kv_transfer_params": json.RawMessage(`{"prompt_token_ids": null}`)},
		"decoded map": {"kv_transfer_params": map[string]any{"prompt_token_ids": nil}},
	} {
		t.Run(name, func(t *testing.T) {
			body := chat(payload, tokenized(1, 2))
			run(t, p, body)
			assert.True(t, body.Mutated)
			// Both wire forms end up as a decoded map with the forwarded IDs.
			v, isMap := payload["kv_transfer_params"].(map[string]any)
			if assert.True(t, isMap, "raw kv_transfer_params should have been replaced") {
				assert.Equal(t, []uint32{1, 2}, v[commonrequest.FieldPromptTokenIDs])
			}
		})
	}
}

// An untrusted caller-supplied prompt_token_ids is stripped and forwarding is
// skipped: the render returned the caller's list, so the engine must render
// from messages instead.
func TestUntrustedCallerTokenIDsStripped(t *testing.T) {
	p, _ := New("fwd", Config{})
	payload := fwkrequest.PayloadMap{"kv_transfer_params": map[string]any{"prompt_token_ids": []any{9}}}
	body := chat(payload, tokenized(1, 2))
	run(t, p, body)
	// The caller's IDs were removed from kv_transfer_params.
	params := payload["kv_transfer_params"].(map[string]any)
	assert.NotContains(t, params, "prompt_token_ids")
}

// The strip must fire on the json.RawMessage wire form the parsers actually
// produce for chat-shaped APIs, not just the decoded map.
func TestUntrustedCallerTokenIDsStrippedRaw(t *testing.T) {
	p, _ := New("fwd", Config{})
	payload := fwkrequest.PayloadMap{
		"kv_transfer_params": json.RawMessage(`{"prompt_token_ids": [9], "do_remote_prefill": true}`),
	}
	body := chat(payload, tokenized(1, 2))
	run(t, p, body)
	// The caller's IDs must be gone; other kv params preserved.
	raw := payload["kv_transfer_params"].(json.RawMessage)
	var params map[string]any
	require.NoError(t, json.Unmarshal(raw, &params))
	assert.NotContains(t, params, "prompt_token_ids")
	assert.Equal(t, true, params["do_remote_prefill"])
}

// Background at the top level and file references nested in input must be
// rejected, matching RejectStatefulResponsesFields' coverage.
func TestStatefulResponsesBackgroundAndFiles(t *testing.T) {
	p, _ := New("fwd", Config{})
	for name, payload := range map[string]fwkrequest.PayloadMap{
		"background top-level": {"input": "hi", "background": true},
		"file_id nested": {"input": []any{map[string]any{
			"type":   "function_call_output",
			"output": []any{map[string]any{"type": "input_file", "file_id": "file-1"}},
		}}},
	} {
		t.Run(name, func(t *testing.T) {
			body := &fwkrequest.InferenceRequestBody{
				Responses: &fwkrequest.ResponsesRequest{Input: payload["input"]},
				Payload:   payload, TokenizedRequest: tokenized(1, 2),
			}
			run(t, p, body)
			assert.False(t, body.Mutated)
		})
	}
}

// Background as an empty string is not a background request under any
// coercion pydantic applies; it is treated as absent and forwarded.
func TestStatefulResponsesBackgroundEmptyString(t *testing.T) {
	p, _ := New("fwd", Config{})
	payload := fwkrequest.PayloadMap{"input": "hi", "background": ""}
	body := &fwkrequest.InferenceRequestBody{
		Responses: &fwkrequest.ResponsesRequest{Input: "hi"},
		Payload:   payload, TokenizedRequest: tokenized(1, 2),
	}
	run(t, p, body)
	// Not stateful: forwarding proceeds.
	assert.True(t, body.Mutated)
}

// With trustCallerTokenIDs, the caller's list is kept as sent.
func TestTrustedCallerTokenIDsKept(t *testing.T) {
	p, _ := New("fwd", Config{TrustCallerTokenIDs: true})
	payload := fwkrequest.PayloadMap{"kv_transfer_params": map[string]any{"prompt_token_ids": []any{9}}}
	body := chat(payload, tokenized(1, 2))
	run(t, p, body)
	assert.False(t, body.Mutated)
}

// The token-producer's legacy Messages/Responses translations do not guarantee
// token parity, so their IDs must never be forwarded.
func TestSkipsLegacyTranslated(t *testing.T) {
	p, _ := New("fwd", Config{})
	payload := fwkrequest.PayloadMap{}
	body := chat(payload, tokenized(1, 2))
	body.TokenizedRequest.LegacyTranslated = true
	run(t, p, body)
	assert.False(t, body.Mutated)
	assert.NotContains(t, payload, "kv_transfer_params")
}

// echo and return_prompt_text need vLLM to render the messages itself.
func TestSkipsEchoAndReturnPromptText(t *testing.T) {
	for name, field := range map[string]string{
		"echo":               "echo",
		"return_prompt_text": "return_prompt_text",
	} {
		t.Run(name, func(t *testing.T) {
			p, _ := New("fwd", Config{})
			payload := fwkrequest.PayloadMap{field: true}
			body := chat(payload, tokenized(1, 2))
			run(t, p, body)
			assert.False(t, body.Mutated)
			assert.NotContains(t, payload, "kv_transfer_params")
		})
	}
}

func TestSkips(t *testing.T) {
	p, _ := New("fwd", Config{MinTokens: 3})
	mm := tokenized(1, 2, 3, 4)
	mm.Prompts[0].MultiModalFeatures = []fwkrequest.MultiModalFeature{{}}
	multi := &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}, {TokenIDs: []uint32{4, 5, 6}}}}

	tests := map[string]*fwkrequest.InferenceRequestBody{
		"not tokenized":     chat(fwkrequest.PayloadMap{}, nil),
		"below minTokens":   chat(fwkrequest.PayloadMap{}, tokenized(1, 2)),
		"multimodal":        chat(fwkrequest.PayloadMap{}, mm),
		"multi-prompt":      chat(fwkrequest.PayloadMap{}, multi),
		"raw payload":       chat(fwkrequest.RawPayload(`{}`), tokenized(1, 2, 3)),
		"params not object": chat(fwkrequest.PayloadMap{"kv_transfer_params": "x"}, tokenized(1, 2, 3)),
		"generate": {Generate: &fwkrequest.GenerateRequest{},
			Payload: fwkrequest.PayloadMap{}, TokenizedRequest: tokenized(1, 2, 3)},
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			run(t, p, body)
			assert.False(t, body.Mutated)
			if m, ok := body.Payload.(fwkrequest.PayloadMap); ok {
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
	for name, body := range map[string]*fwkrequest.InferenceRequestBody{
		"chat completions": {ChatCompletions: &fwkrequest.ChatCompletionsRequest{}},
		"messages":         {Messages: &fwkrequest.MessagesRequest{}},
		"responses":        {Responses: &fwkrequest.ResponsesRequest{}},
	} {
		t.Run(name, func(t *testing.T) {
			payload := fwkrequest.PayloadMap{}
			body.Payload, body.TokenizedRequest = payload, tokenized(4, 5)
			run(t, p, body)
			assert.True(t, body.Mutated)
			assert.Equal(t, []uint32{4, 5}, payload["kv_transfer_params"].(map[string]any)["prompt_token_ids"])
		})
	}
}

// Only the vllm backend produces engine tokens; the estimate backend's byte-packed IDs must not be
// forwarded, so the dependency check refuses that producer at startup.
func TestConsumesRequiresVLLMBackend(t *testing.T) {
	p, _ := New("fwd", Config{})
	dep, ok := p.Consumes().Required[tokenproducer.TokenizedPromptDataKey].(plugin.ConsumedData)
	require.True(t, ok)
	require.NotNil(t, dep.Matches)
	assert.NoError(t, dep.Matches(map[string]string{"backend": tokenproducer.BackendVLLM}))
	assert.Error(t, dep.Matches(map[string]string{"backend": tokenproducer.BackendEstimate}))
	assert.Error(t, dep.Matches(map[string]string{}))
}

// A stateful Responses request is rendered from history the worker stores, so the router's IDs
// would leave it out.
func TestSkipsStatefulResponses(t *testing.T) {
	p, _ := New("fwd", Config{})
	for name, payload := range map[string]fwkrequest.PayloadMap{
		"previous_response_id": {"input": "hi", "previous_response_id": "resp-1"},
		"conversation":         {"input": "hi", "conversation": "conv-1"},
	} {
		t.Run(name, func(t *testing.T) {
			responses := &fwkrequest.ResponsesRequest{Input: "hi"}
			if name == "previous_response_id" {
				responses.PreviousResponseID = "resp-1"
			} else {
				responses.Conversation = "conv-1"
			}
			body := &fwkrequest.InferenceRequestBody{Responses: responses,
				Payload: payload, TokenizedRequest: tokenized(1, 2)}
			run(t, p, body)
			assert.False(t, body.Mutated)
			assert.NotContains(t, payload, "kv_transfer_params")
		})
	}
	// An explicit null is unset, so the request is stateless and forwarded.
	payload := fwkrequest.PayloadMap{"input": "hi", "previous_response_id": nil}
	body := &fwkrequest.InferenceRequestBody{Responses: &fwkrequest.ResponsesRequest{},
		Payload: payload, TokenizedRequest: tokenized(1, 2)}
	run(t, p, body)
	assert.True(t, body.Mutated)
}

func completions(payload fwkrequest.PayloadMap, prompt fwkrequest.Prompt) *fwkrequest.InferenceRequestBody {
	return &fwkrequest.InferenceRequestBody{Completions: &fwkrequest.CompletionsRequest{Prompt: prompt},
		Payload: payload, TokenizedRequest: tokenized(4, 5, 6)}
}

// vLLM's completions path takes a token array as the prompt without tokenizing it.
func TestForwardsCompletionsPrompt(t *testing.T) {
	p, _ := New("fwd", Config{})
	payload := fwkrequest.PayloadMap{"model": "m", "prompt": "hello", "echo": false}
	body := completions(payload, fwkrequest.Prompt{Raw: "hello"})
	run(t, p, body)
	assert.True(t, body.Mutated)
	assert.Equal(t, []uint32{4, 5, 6}, payload["prompt"])
	assert.NotContains(t, payload, "kv_transfer_params")
}

func TestSkipsCompletions(t *testing.T) {
	p, _ := New("fwd", Config{})
	tests := map[string]*fwkrequest.InferenceRequestBody{
		"already token IDs": completions(fwkrequest.PayloadMap{"prompt": []any{1, 2}},
			fwkrequest.Prompt{TokenIDs: [][]uint32{{1, 2}}}),
		"echo":          completions(fwkrequest.PayloadMap{"prompt": "hi", "echo": true}, fwkrequest.Prompt{Raw: "hi"}),
		"echo raw json": completions(fwkrequest.PayloadMap{"prompt": "hi", "echo": json.RawMessage(`true`)}, fwkrequest.Prompt{Raw: "hi"}),
		"suffix":        completions(fwkrequest.PayloadMap{"prompt": "hi", "suffix": "end"}, fwkrequest.Prompt{Raw: "hi"}),
		"multi-prompt": {Completions: &fwkrequest.CompletionsRequest{}, Payload: fwkrequest.PayloadMap{"prompt": []any{"a", "b"}},
			TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: []uint32{1}}, {TokenIDs: []uint32{2}}}}},
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			before := body.Payload.(fwkrequest.PayloadMap)["prompt"]
			run(t, p, body)
			assert.False(t, body.Mutated)
			assert.Equal(t, before, body.Payload.(fwkrequest.PayloadMap)["prompt"])
		})
	}
}

// The EPP's dependency check refuses the plugin alongside an estimate token-producer, which is
// also what it auto-creates when no token-producer is configured.
func TestStartupRejectsEstimateTokenProducer(t *testing.T) {
	fwd, _ := New("fwd", Config{})
	estimate, err := tokenproducer.PluginFactory(tokenproducer.PluginType, nil, plugin.NewEppHandle(context.Background(), nil))
	require.NoError(t, err)
	_, err = datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{fwd, estimate})
	require.Error(t, err)
	assert.Contains(t, err.Error(), `"backend"="estimate"`)
}
