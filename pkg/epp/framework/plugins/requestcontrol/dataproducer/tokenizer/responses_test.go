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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
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

// TestResponsesPayloadWire_ComplexInputItemsFailClosed covers input mixing
// convertible items with item types this code cannot represent: dropping
// the unsupported items would tokenize a prompt shorter than the one vLLM
// serves, so the whole conversion fails instead.
func TestResponsesPayloadWire_ComplexInputItemsFailClosed(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": "before"},
			map[string]any{"type": "web_search_call", "id": "ws_1", "status": "completed"},
			map[string]any{"role": "user", "content": "after"},
		},
	}
	_, err := responsesPayload(r)
	assert.ErrorContains(t, err, `"web_search_call"`)
	assert.ErrorContains(t, err, "not supported")
}

// TestResponsesPayloadWire_ToolCallTurn covers the second turn of a
// tool-calling loop as the OpenAI SDK sends it: the function_call becomes an
// assistant message carrying tool_calls, and the function_call_output the
// tool message answering it.
func TestResponsesPayloadWire_ToolCallTurn(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": "what is the weather"},
			map[string]any{"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": `{"city":"Paris"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[
		{"role":"user","content":"what is the weather"},
		{"role":"assistant","tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}
		]},
		{"role":"tool","content":"sunny","tool_call_id":"call_1"}
	]}`, string(body))
}

// TestResponsesPayloadWire_AssistantTurnFolds covers a turn spelled as
// reasoning, message and two function_call items, followed by both outputs:
// vLLM renders the four items as one assistant message, so the translation
// folds them the same way, and an output given as content parts converts
// like message content.
func TestResponsesPayloadWire_AssistantTurnFolds(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": "plan a trip"},
			map[string]any{"type": "reasoning", "id": "rs_1", "summary": []any{
				map[string]any{"type": "summary_text", "text": "compare two cities"},
			}},
			map[string]any{"type": "message", "role": "assistant", "content": []any{
				map[string]any{"type": "output_text", "text": "Checking both."},
			}},
			map[string]any{"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": `{"city":"Paris"}`},
			map[string]any{"type": "function_call", "call_id": "call_2", "name": "get_weather", "arguments": `{"city":"Rome"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
			map[string]any{"type": "function_call_output", "call_id": "call_2", "output": []any{
				map[string]any{"type": "input_text", "text": "rain"},
			}},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[
		{"role":"user","content":"plan a trip"},
		{"role":"assistant","reasoning":"compare two cities","content":"Checking both.","tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},
			{"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Rome\"}"}}
		]},
		{"role":"tool","content":"sunny","tool_call_id":"call_1"},
		{"role":"tool","content":"rain","tool_call_id":"call_2"}
	]}`, string(body))
}

// TestResponsesPayloadWire_ReasoningPlacement covers the two reasoning
// rules: a content part wins over a summary part, and a reasoning item that
// follows an assistant message already carrying reasoning starts a new
// assistant message instead of overwriting it.
func TestResponsesPayloadWire_ReasoningPlacement(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": "q"},
			map[string]any{"type": "reasoning", "id": "rs_1",
				"content": []any{map[string]any{"type": "reasoning_text", "text": "full thought"}},
				"summary": []any{map[string]any{"type": "summary_text", "text": "short"}},
			},
			map[string]any{"type": "message", "role": "assistant", "content": "a"},
			map[string]any{"type": "reasoning", "id": "rs_2", "summary": []any{
				map[string]any{"type": "summary_text", "text": "second thought"},
			}},
			map[string]any{"type": "function_call", "call_id": "call_1", "name": "run", "arguments": "{}"},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[
		{"role":"user","content":"q"},
		{"role":"assistant","reasoning":"full thought","content":"a"},
		{"role":"assistant","reasoning":"second thought","tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"run","arguments":"{}"}}
		]}
	]}`, string(body))
}

// TestResponsesPayloadWire_ItemsFailClosed covers the item shapes the
// translation refuses: a function_call missing a required field or carrying
// a namespace, a reasoning item with encrypted content, an output without
// its call_id, and an item type this code does not know.
func TestResponsesPayloadWire_ItemsFailClosed(t *testing.T) {
	for _, tc := range []struct {
		name string
		item map[string]any
		want string
	}{
		{"function_call without arguments", map[string]any{"type": "function_call", "call_id": "call_1", "name": "run"}, "needs call_id, name and arguments"},
		{"function_call without name", map[string]any{"type": "function_call", "call_id": "call_1", "arguments": "{}"}, "needs call_id, name and arguments"},
		{"namespaced function_call", map[string]any{"type": "function_call", "call_id": "call_1", "name": "run", "arguments": "{}", "namespace": "ns"}, "namespace"},
		{"encrypted reasoning", map[string]any{"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA"}, "encrypted_content"},
		{"reasoning part without text", map[string]any{"type": "reasoning", "id": "rs_1", "summary": []any{map[string]any{"type": "summary_text"}}}, "has no text field"},
		{"function_call_output without call_id", map[string]any{"type": "function_call_output", "output": "sunny"}, "has no call_id"},
		{"function_call_output with an audio part", map[string]any{"type": "function_call_output", "call_id": "call_1", "output": []any{map[string]any{"type": "input_audio"}}}, `"input_audio"`},
		{"unknown item type", map[string]any{"type": "web_search_call", "id": "ws_1"}, `"web_search_call"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := responsesPayload(&fwkrh.ResponsesRequest{Input: []any{
				map[string]any{"role": "user", "content": "q"},
				tc.item,
			}})
			assert.ErrorContains(t, err, tc.want)
		})
	}
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

// TestResponsesPayloadWire_ImageContent covers input_image conversion: its
// URL is a bare string on the part, unlike chat completions' nested
// {"image_url": {"url": ...}} shape, and it converts alongside text parts in
// the same message rather than failing the conversion.
func TestResponsesPayloadWire_ImageContent(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_image", "image_url": "http://example.com/x.png"},
				map[string]any{"type": "input_text", "text": "describe this"},
			}},
			// an image-only message converts too, collapsing to Structured
			// with a single block rather than Raw (Raw only collapses text).
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_image", "image_url": "http://example.com/y.png"},
			}},
		},
	}
	got, err := responsesPayload(r)
	require.NoError(t, err)
	body, err := got.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[
		{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"http://example.com/x.png"}},
			{"type":"text","text":"describe this"}
		]},
		{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"http://example.com/y.png"}}
		]}
	]}`, string(body))
}

// TestResponsesPayloadWire_ArrayContentNonTextPartsFailClosed covers a
// content part this code cannot represent at all (for example input_audio):
// dropping it would tokenize a prompt shorter than the one vLLM serves, so
// the conversion fails even when a text part sits alongside it in the same
// message.
func TestResponsesPayloadWire_ArrayContentNonTextPartsFailClosed(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_audio", "data": "AAAA", "format": "wav"},
				map[string]any{"type": "input_text", "text": "describe this"},
			}},
		},
	}
	_, err := responsesPayload(r)
	assert.ErrorContains(t, err, `"input_audio"`)
	assert.ErrorContains(t, err, "not supported")
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

	_, err = responsesPayload(&fwkrh.ResponsesRequest{Input: []any{}})
	assert.ErrorContains(t, err, "no renderable input")

	// An unsupported item or content part fails closed with its own error
	// rather than "no renderable input": see
	// TestResponsesPayloadWire_ComplexInputItemsFailClosed and
	// TestResponsesPayloadWire_ArrayContentNonTextPartsFailClosed.
	_, err = responsesPayload(&fwkrh.ResponsesRequest{
		Input: []any{map[string]any{"type": "function_call", "call_id": "call_1"}},
	})
	assert.ErrorContains(t, err, `"function_call"`)

	_, err = responsesPayload(&fwkrh.ResponsesRequest{
		Input: []any{map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "input_audio", "data": "AAAA", "format": "wav"},
		}}},
	})
	assert.ErrorContains(t, err, `"input_audio"`)
}

// legacyResponsesForced forces the legacy chat-completions translation path,
// skipping discovery, for tests written against that path's behavior.
func legacyResponsesForced() *legacyResponsesMode {
	return &legacyResponsesMode{mode: responsesRenderModeLegacy}
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
	p.backend = renderBackend{tk: tok, modelName: "configured-model", legacyResponses: legacyResponsesForced()}

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

// TestProduce_ResponsesUnsupportedInputErrors exercises the legacy
// translation path's own validation; the native path has no such check and
// forwards whatever content it is given.
func TestProduce_ResponsesUnsupportedInputErrors(t *testing.T) {
	tok := &mockTokenizer{
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Fatal("must not call RenderChat with unsupported input")
			return nil, nil, nil
		},
	}
	p := newTestPlugin(tok)
	p.backend = renderBackend{tk: tok, legacyResponses: legacyResponsesForced()}
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Responses: &fwkrh.ResponsesRequest{
				Input: []any{map[string]any{"type": "web_search_call", "id": "ws_1"}},
			},
			Payload: fwkrh.PayloadMap{},
		},
	}
	// The director logs and continues on a producer error (see
	// Director.HandleRequest); Plugin.Produce itself still surfaces it so the
	// caller can tell tokenization was skipped rather than run successfully.
	err := p.Produce(context.Background(), req, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `"web_search_call"`)
	assert.Nil(t, req.Body.TokenizedRequest)
}

func TestResponsesRenderMode(t *testing.T) {
	for _, tc := range []struct {
		name, params string
		legacy       bool
	}{
		{"model only defaults to auto", `{"modelName":"configured-model"}`, false},
		{"empty config defaults to auto", `{"modelName":"configured-model","vllm":{}}`, false},
		{"empty mode defaults to auto", `{"modelName":"configured-model","vllm":{"responsesRenderMode":""}}`, false},
		{"explicit auto", `{"modelName":"configured-model","vllm":{"responsesRenderMode":"auto"}}`, false},
		{"explicit legacy", `{"modelName":"configured-model","vllm":{"responsesRenderMode":"legacy"}}`, true},
		{"explicit native", `{"modelName":"configured-model","vllm":{"responsesRenderMode":"native"}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var warnings []string
			logger := funcr.New(func(_, args string) {
				mu.Lock()
				defer mu.Unlock()
				warnings = append(warnings, args)
			}, funcr.Options{})
			ctx, cancel := context.WithCancel(log.IntoContext(context.Background(), logger))
			cancel()
			got, err := PluginFactory("responses", plugin.StrictDecoder(json.RawMessage(tc.params)), plugin.NewEppHandle(ctx, nil))
			require.NoError(t, err)
			p := got.(*Plugin)
			auto := p.backend.(renderBackend).legacyResponses != nil && !tc.legacy

			const raw = `{"model":"adapter","max_tokens":8,"input":"hi"}`
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				body, readErr := io.ReadAll(r.Body)
				require.NoError(t, readErr)
				if auto && calls == 1 {
					require.Equal(t, responsesRenderPath, r.URL.Path)
					require.JSONEq(t, `{"model":"configured-model","max_output_tokens":1,"input":"warmup"}`, string(body))
					_, _ = io.WriteString(w, `{"token_ids":[1]}`)
					return
				}
				if tc.legacy {
					require.Equal(t, chatRenderPath, r.URL.Path)
					require.JSONEq(t, `{"model":"configured-model","messages":[{"role":"user","content":"hi"}]}`, string(body))
				} else {
					require.Equal(t, responsesRenderPath, r.URL.Path)
					require.JSONEq(t, raw, string(body))
				}
				_, _ = io.WriteString(w, `{"token_ids":[1,2,3],"features":{"mm_hashes":{"image":["hash"]},"mm_placeholders":{"image":[{"offset":1,"length":2}]}}}`)
			}))
			defer srv.Close()
			backend := p.backend.(renderBackend)
			backend.tk = newHTTPRenderer(t, srv)
			p.backend = backend

			for range 2 {
				parsed, perr := openai.NewOpenAIParser().ParseRequest(context.Background(), []byte(raw), map[string]string{":path": "/v1/responses"})
				require.NoError(t, perr)
				req := &scheduling.InferenceRequest{Body: parsed.Body}
				require.NoError(t, p.Produce(context.Background(), req, nil))
				require.NotNil(t, req.Body.TokenizedRequest)
				assert.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
				assert.Equal(t, []fwkrh.MultiModalFeature{{Modality: reqcommon.ModalityImage, Hash: "hash", Offset: 1, Length: 2}},
					req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures)
			}
			wantCalls := 2
			if auto {
				wantCalls++
			}
			require.Equal(t, wantCalls, calls)
			mu.Lock()
			defer mu.Unlock()
			if tc.legacy {
				require.Len(t, warnings, 1)
				require.Contains(t, warnings[0], "deprecated")
				require.Contains(t, warnings[0], "responsesRenderMode")
				require.Contains(t, warnings[0], "native")
				require.Contains(t, warnings[0], "token parity")
			} else {
				require.Empty(t, warnings)
			}
		})
	}
}

func TestResponsesRenderModeRejectsInvalidValue(t *testing.T) {
	for _, mode := range []string{"unknown", "NATIVE", "legacy "} {
		t.Run(mode, func(t *testing.T) {
			params := `{"modelName":"m","vllm":{"responsesRenderMode":"` + mode + `"}}`
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			p, err := PluginFactory("responses", plugin.StrictDecoder(json.RawMessage(params)), plugin.NewEppHandle(ctx, nil))
			require.ErrorContains(t, err, "responsesRenderMode")
			require.ErrorContains(t, err, `"legacy"`)
			require.ErrorContains(t, err, `"native"`)
			require.Nil(t, p)
		})
	}
}

// TestResponsesRenderModeChatOnlyRenderer covers a render endpoint that only
// implements /v1/chat/completions/render: explicit legacy mode succeeds,
// explicit native mode surfaces the 404 rather than silently falling back.
func TestResponsesRenderModeChatOnlyRenderer(t *testing.T) {
	for _, mode := range []string{responsesRenderModeNative, responsesRenderModeLegacy} {
		t.Run("mode="+mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			params := fmt.Sprintf(`{"modelName":"configured-model","vllm":{"responsesRenderMode":%q}}`, mode)
			got, err := PluginFactory("responses", plugin.StrictDecoder(json.RawMessage(params)), plugin.NewEppHandle(ctx, nil))
			require.NoError(t, err)
			p := got.(*Plugin)
			var paths []string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				paths = append(paths, r.URL.Path)
				if r.URL.Path != chatRenderPath {
					http.NotFound(w, r)
					return
				}
				_, _ = io.WriteString(w, `{"token_ids":[1,2,3]}`)
			}))
			defer srv.Close()
			backend := p.backend.(renderBackend)
			backend.tk = newHTTPRenderer(t, srv)
			p.backend = backend
			const raw = `{"model":"adapter","max_tokens":8,"input":"hi"}`
			parsed, err := openai.NewOpenAIParser().ParseRequest(context.Background(), []byte(raw), map[string]string{":path": "/v1/responses"})
			require.NoError(t, err)
			req := &scheduling.InferenceRequest{Body: parsed.Body}
			err = p.Produce(context.Background(), req, nil)
			if mode == responsesRenderModeLegacy {
				require.NoError(t, err)
				require.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
				require.Equal(t, []string{chatRenderPath}, paths)
			} else {
				var statusErr *renderStatusError
				require.ErrorAs(t, err, &statusErr)
				require.Equal(t, http.StatusNotFound, statusErr.StatusCode)
				require.Nil(t, req.Body.TokenizedRequest)
				require.Equal(t, []string{responsesRenderPath}, paths)
			}
		})
	}
}

// TestResponsesAutoDiscoveryFallsBackOnUnsupportedStatus covers every status
// vLLM uses to signal /v1/responses/render is unavailable: 404 and 405 when
// the route itself is absent, and 501 when the route is registered but the
// model has no Responses render handler. Auto discovery falls back to the
// legacy chat-completions translation on all three rather than failing
// tokenization.
func TestResponsesAutoDiscoveryFallsBackOnUnsupportedStatus(t *testing.T) {
	for _, status := range []int{http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusNotImplemented} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var paths []string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				paths = append(paths, r.URL.Path)
				if r.URL.Path == responsesRenderPath {
					w.WriteHeader(status)
					return
				}
				require.Equal(t, chatRenderPath, r.URL.Path)
				_, _ = io.WriteString(w, `{"token_ids":[1,2,3]}`)
			}))
			defer srv.Close()
			renderer := newHTTPRenderer(t, srv)
			p := newTestPlugin(renderer)
			backend := p.backend.(renderBackend)
			backend.legacyResponses = &legacyResponsesMode{name: "test", discovery: make(chan struct{}, 1)}
			backend.modelName = "configured-model"
			p.backend = backend
			req := &scheduling.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Responses: &fwkrh.ResponsesRequest{Input: "hi"},
					Payload:   fwkrh.PayloadMap{},
				},
			}
			require.NoError(t, p.Produce(context.Background(), req, nil))
			require.NotNil(t, req.Body.TokenizedRequest)
			assert.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
			// One chat call confirms the legacy path renders during discovery,
			// a second serves the request itself.
			assert.Equal(t, []string{responsesRenderPath, chatRenderPath, chatRenderPath}, paths)
		})
	}
}

func TestLegacyResponsesToolFields(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		want      map[string]float64
	}{
		{"dropped empty fields", `{"input":"hi","tools":[],"tool_choice":"auto","parallel_tool_calls":false}`, map[string]float64{
			"tools/dropped": 1, "tool_choice/dropped": 1, "parallel_tool_calls/dropped": 1,
		}},
		{"converted tools", `{"input":"hi","tools":[{"type":"function","name":"private_tool"}]}`, map[string]float64{
			"tools/preserved": 1,
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := setupSpanRecorder(t)
			before := renderToolFieldCounts(t)
			renderer := newRenderServer(t, vllmConfig{}, func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, chatRenderPath, r.URL.Path)
				_, _ = io.WriteString(w, `{"token_ids":[1]}`)
			})
			parsed, err := openai.NewOpenAIParser().ParseRequest(t.Context(), []byte(tc.raw), map[string]string{":path": "/v1/responses"})
			require.NoError(t, err)
			p := newTestPlugin(renderer)
			p.backend = renderBackend{tk: renderer, modelName: "m", legacyResponses: legacyResponsesForced()}
			require.NoError(t, p.Produce(toolcalling.WithTelemetryEnabled(t.Context()), &scheduling.InferenceRequest{Body: parsed.Body}, nil))
			after := renderToolFieldCounts(t)
			for key, count := range before {
				after[key] -= count
			}
			for key, count := range after {
				if count == 0 {
					delete(after, key)
				}
			}
			require.Equal(t, tc.want, after)
			require.Len(t, recorder.Ended(), 2)
			for _, span := range recorder.Ended() {
				assert.Equal(t, "responses", spanAttrs(span)[semconv.LLMDToolCallingAPISurfaceKey].AsString())
			}
		})
	}
}

func TestLiveLegacyDiscoveryProbesUnobserved(t *testing.T) {
	for _, surface := range []string{"messages", "responses"} {
		t.Run(surface, func(t *testing.T) {
			recorder := setupSpanRecorder(t)
			var paths []string
			renderer := newRenderServer(t, vllmConfig{}, func(w http.ResponseWriter, r *http.Request) {
				paths = append(paths, r.URL.Path)
				if r.URL.Path != chatRenderPath {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				_, _ = io.WriteString(w, `{"token_ids":[1]}`)
			})
			backend := renderBackend{tk: renderer, modelName: "m"}
			body := &fwkrh.InferenceRequestBody{}
			var nativePath string
			if surface == "messages" {
				nativePath = messagesRenderPath
				backend.legacyMessages = &legacyMessagesMode{discovery: make(chan struct{}, 1)}
				body.Messages = &fwkrh.MessagesRequest{Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "hi"}}}}
				body.Payload = fwkrh.PayloadMap{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}
			} else {
				nativePath = responsesRenderPath
				backend.legacyResponses = &legacyResponsesMode{discovery: make(chan struct{}, 1)}
				body.Responses = &fwkrh.ResponsesRequest{Input: "hi"}
				body.Payload = fwkrh.PayloadMap{"input": "hi"}
			}
			p := newTestPlugin(renderer)
			p.backend = backend
			require.NoError(t, p.Produce(t.Context(), &scheduling.InferenceRequest{Body: body}, nil))
			assert.Equal(t, []string{nativePath, chatRenderPath, chatRenderPath}, paths)
			assert.Equal(t, float64(0), renderFailures(renderer.pluginName, renderReasonStatus))
			count, _ := renderHistogram(t, renderer.pluginName, renderResultSuccess)
			assert.Equal(t, uint64(1), count)
			assert.Len(t, recorder.Ended(), 2, "tokenize span and request render span")
		})
	}
}
