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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

func TestResponsesPayloadWire_StringInput(t *testing.T) {
	r := &fwkrh.ResponsesRequest{Input: "hello there"}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[{"role":"user","content":"hello there"}]}`, string(body))
}

func TestResponsesPayloadWire_InstructionsAndMessages(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Instructions: "be terse",
		Input: []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"type": "message", "role": "assistant", "content": "assistant-reply"},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[
		{"role":"system","content":"be terse"},
		{"role":"user","content":"hi"},
		{"role":"assistant","content":"assistant-reply"}
	]}`, string(body))
}

func TestResponsesPayloadWire_SkipsComplexInputItems(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": "before"},
			map[string]any{"type": "function_call", "call_id": "call_1", "name": "run", "arguments": "{}"},
			map[string]any{"type": "reasoning", "content": "thinking..."},
			map[string]any{"role": "user", "content": "after"},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[
		{"role":"user","content":"before"},
		{"role":"user","content":"after"}
	]}`, string(body))
}

func TestResponsesPayloadWire_ArrayContentTextParts(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "hi"},
			}},
			map[string]any{"type": "message", "role": "assistant", "content": []any{
				map[string]any{"type": "output_text", "text": "hello back"},
			}},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	// A single text part collapses to plain string content.
	assert.JSONEq(t, `{"messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":"hello back"}
	]}`, string(body))
}

func TestResponsesPayloadWire_ArrayContentMultipleTextParts(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "part one"},
				map[string]any{"type": "input_text", "text": "part two"},
			}},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[
		{"role":"user","content":[
			{"type":"text","text":"part one"},
			{"type":"text","text":"part two"}
		]}
	]}`, string(body))
}

//nolint:goconst // "role"/"content"/"type" JSON keys read clearly inline; not worth naming
func TestResponsesPayloadWire_ArrayContentImagePart(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			// A single image part converts on its own, using Structured
			// rather than collapsing to Raw (that collapse is text-only).
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_image", "image_url": "http://example.com/y.png"},
			}},
			// A text part alongside an image part keeps both.
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_image", "image_url": "http://example.com/x.png"},
				map[string]any{"type": "input_text", "text": "describe this"},
			}},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://example.com/y.png"}}]},
		{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"http://example.com/x.png"}},
			{"type":"text","text":"describe this"}
		]}
	]}`, string(body))
}

//nolint:goconst // "role"/"content"/"type" JSON keys read clearly inline; not worth naming
func TestResponsesPayloadWire_ArrayContentSkipsUnrecognizedParts(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			// A text part alongside an unrecognized part keeps only the text part.
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "refusal", "refusal": "cannot help with that"},
				map[string]any{"type": "input_text", "text": "describe this"},
			}},
			// An item whose content is entirely unrecognized parts has no
			// renderable content and is skipped, like other unhandled shapes.
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "refusal", "refusal": "cannot help with that"},
			}},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[{"role":"user","content":"describe this"}]}`, string(body))
}

func TestResponsesPayloadWire_Tools(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: "hi",
		Tools: []any{
			map[string]any{
				"type":        "function",
				"name":        "get_weather",
				"description": "look up the weather",
				"parameters":  map[string]any{"type": "object"},
			},
			// already nested passes through unchanged
			map[string]any{
				"type":     "function",
				"function": map[string]any{"name": "nested"},
			},
			// a non-function tool type passes through unchanged
			map[string]any{"type": "web_search"},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"messages":[{"role":"user","content":"hi"}],
		"tools":[
			{"type":"function","function":{"name":"get_weather","description":"look up the weather","parameters":{"type":"object"}}},
			{"type":"function","function":{"name":"nested"}},
			{"type":"web_search"}
		]
	}`, string(body))
}

func TestResponsesPayloadWire_EmptyInputErrors(t *testing.T) {
	_, err := responsesPayload(&fwkrh.ResponsesRequest{})
	assert.ErrorContains(t, err, "no renderable input")

	_, err = responsesPayload(&fwkrh.ResponsesRequest{
		Input: []any{map[string]any{"type": "function_call", "call_id": "call_1"}},
	})
	assert.ErrorContains(t, err, "no renderable input")

	_, err = responsesPayload(&fwkrh.ResponsesRequest{
		Input: []any{map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "refusal", "refusal": "cannot help with that"},
		}}},
	})
	assert.ErrorContains(t, err, "no renderable input")
}

func TestProduce_ResponsesPopulatesTokenizedRequest(t *testing.T) {
	var gotPayload fwkrh.RequestPayload
	tok := &mockTokenizer{
		renderChatFunc: func(payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			gotPayload = payload
			return []uint32{5, 6, 7}, nil, nil
		},
	}
	p := newTestPlugin(tok)
	p.backend = renderBackend{tk: tok, modelName: "configured-model"}

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Responses: &fwkrh.ResponsesRequest{Input: "hi", CacheSalt: "tenant-a"},
			Payload:   fwkrh.PayloadMap{},
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []uint32{5, 6, 7}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
	assert.Equal(t, "tenant-a", req.Body.TokenizedRequest.CacheSalt)

	pm, ok := gotPayload.(fwkrh.PayloadMap)
	require.True(t, ok)
	assert.Equal(t, "configured-model", pm["model"])
}

func TestProduce_ResponsesTokenizerError(t *testing.T) {
	tok := &mockTokenizer{
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return nil, nil, assert.AnError
		},
	}
	p := newTestPlugin(tok)
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Responses: &fwkrh.ResponsesRequest{Input: "hi"},
			Payload:   fwkrh.PayloadMap{},
		},
	}
	err := p.Produce(context.Background(), req, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tokenization failed")
	assert.Nil(t, req.Body.TokenizedRequest)
}

func TestProduce_ResponsesNoRenderableInputErrors(t *testing.T) {
	tok := &mockTokenizer{
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Fatal("must not call RenderChat with no renderable input")
			return nil, nil, nil
		},
	}
	p := newTestPlugin(tok)
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Responses: &fwkrh.ResponsesRequest{
				Input: []any{map[string]any{"type": "function_call", "call_id": "call_1"}},
			},
			Payload: fwkrh.PayloadMap{},
		},
	}
	// The director logs and continues on a producer error (see
	// Director.HandleRequest); Plugin.Produce itself still surfaces it so the
	// caller can tell tokenization was skipped rather than run successfully.
	err := p.Produce(context.Background(), req, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no renderable input")
	assert.Nil(t, req.Body.TokenizedRequest)
}
