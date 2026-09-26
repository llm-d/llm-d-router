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

func TestResponsesPayloadWire_ArrayContentSkipsNonTextParts(t *testing.T) {
	r := &fwkrh.ResponsesRequest{
		Input: []any{
			// A text part alongside a non-text part keeps only the text part.
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_image", "image_url": "http://example.com/x.png"},
				map[string]any{"type": "input_text", "text": "describe this"},
			}},
			// An item whose content is entirely non-text parts has no
			// renderable text and is skipped, like other unhandled shapes.
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_image", "image_url": "http://example.com/y.png"},
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
			map[string]any{"type": "input_image", "image_url": "http://example.com/x.png"},
		}}},
	})
	assert.ErrorContains(t, err, "no renderable input")
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

// TestProduce_ResponsesNoRenderableInputErrors exercises the legacy
// translation path's own validation; the native path has no such check and
// forwards whatever content it is given.
func TestProduce_ResponsesNoRenderableInputErrors(t *testing.T) {
	tok := &mockTokenizer{
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			t.Fatal("must not call RenderChat with no renderable input")
			return nil, nil, nil
		},
	}
	p := newTestPlugin(tok)
	p.backend = renderBackend{tk: tok, legacyResponses: legacyResponsesForced()}
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
					require.JSONEq(t, `{"model":"configured-model","max_tokens":1,"input":"warmup"}`, string(body))
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
				assert.Equal(t, []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "hash", Offset: 1, Length: 2}},
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
