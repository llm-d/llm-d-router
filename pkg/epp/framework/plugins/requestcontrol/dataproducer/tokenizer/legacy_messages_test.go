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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/sglanghttp"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmhttp"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

func TestMessagesRenderMode(t *testing.T) {
	for _, tc := range []struct {
		name, params string
		legacy       bool
	}{
		{"model only defaults to legacy", `{"modelName":"configured-model"}`, true},
		{"empty config defaults to legacy", `{"modelName":"configured-model","vllm":{}}`, true},
		{"empty mode defaults to legacy", `{"modelName":"configured-model","vllm":{"messagesRenderMode":""}}`, true},
		{"explicit legacy", `{"modelName":"configured-model","vllm":{"messagesRenderMode":"legacy"}}`, true},
		{"explicit native", `{"modelName":"configured-model","vllm":{"messagesRenderMode":"native"}}`, false},
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
			got, err := PluginFactory("messages", plugin.StrictDecoder(json.RawMessage(tc.params)), plugin.NewEppHandle(ctx, nil))
			require.NoError(t, err)
			p := got.(*Plugin)

			const raw = ` {"model":"adapter","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"cache_salt":"tenant-a","unknown":{"z":1,"a":2}} `
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				body, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				require.Equal(t, "Bearer secret", r.Header.Get("Authorization"))
				if tc.legacy {
					require.Equal(t, chatRenderPath, r.URL.Path)
					require.JSONEq(t, `{"model":"configured-model","messages":[{"role":"user","content":"hi"}]}`, string(body))
				} else {
					require.Equal(t, messagesRenderPath, r.URL.Path)
					require.Equal(t, raw, string(body))
				}
				_, _ = io.WriteString(w, `{"token_ids":[1,2,3],"features":{"mm_hashes":{"image":["hash"]},"mm_placeholders":{"image":[{"offset":1,"length":2}]}}}`)
			}))
			defer srv.Close()
			backend := p.backend.(renderBackend)
			backend.tk = newHTTPRenderer(t, srv)
			p.backend = backend

			for range 2 {
				parsed, err := anthropic.NewAnthropicParser().ParseRequest(context.Background(), []byte(raw), map[string]string{":path": "/v1/messages"})
				require.NoError(t, err)
				projection, err := json.Marshal(parsed.Body.Messages)
				require.NoError(t, err)
				req := &scheduling.InferenceRequest{Body: parsed.Body, Headers: map[string]string{"authorization": "Bearer secret"}}
				require.NoError(t, p.Produce(context.Background(), req, nil))
				require.Equal(t, &fwkrh.TokenizedRequest{
					CacheSalt: "tenant-a",
					Prompts: []fwkrh.PromptTokens{{
						TokenIDs:           []uint32{1, 2, 3},
						MultiModalFeatures: []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "hash", Offset: 1, Length: 2}},
					}},
				}, req.Body.TokenizedRequest)
				require.Equal(t, fwkrh.RawPayload(raw), req.Body.WirePayload())
				require.False(t, req.Body.Mutated)
				unchanged, err := json.Marshal(req.Body.Messages)
				require.NoError(t, err)
				require.Equal(t, projection, unchanged)
			}
			require.Equal(t, 2, calls)
			mu.Lock()
			defer mu.Unlock()
			if tc.legacy {
				require.Len(t, warnings, 1)
				require.Contains(t, warnings[0], "deprecated")
				require.Contains(t, warnings[0], "messagesRenderMode")
				require.Contains(t, warnings[0], "native")
				require.Contains(t, warnings[0], "token parity")
			} else {
				require.Empty(t, warnings)
			}
		})
	}
}

func TestMessagesRenderModeRejectsInvalidValue(t *testing.T) {
	for _, mode := range []string{"auto", "NATIVE", "legacy "} {
		t.Run(mode, func(t *testing.T) {
			params := `{"modelName":"m","vllm":{"messagesRenderMode":"` + mode + `"}}`
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			p, err := PluginFactory("messages", plugin.StrictDecoder(json.RawMessage(params)), plugin.NewEppHandle(ctx, nil))
			require.ErrorContains(t, err, "messagesRenderMode")
			require.ErrorContains(t, err, `"legacy"`)
			require.ErrorContains(t, err, `"native"`)
			require.Nil(t, p)
		})
	}
}

func TestMessagesRenderModeEstimateDoesNotWarn(t *testing.T) {
	var logs strings.Builder
	ctx := log.IntoContext(context.Background(), funcr.New(func(_, args string) {
		logs.WriteString(args)
	}, funcr.Options{}))
	p, err := PluginFactory("estimate", plugin.StrictDecoder(json.RawMessage(`{"estimate":{}}`)), plugin.NewEppHandle(ctx, nil))
	require.NoError(t, err)
	require.IsType(t, estimateBackend{}, p.(*Plugin).backend)
	require.Empty(t, logs.String())
}

func TestMessagesRenderModeDoesNotFallback(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		mode, path := "native", messagesRenderPath
		if legacy {
			mode, path = "legacy", chatRenderPath
		}
		for _, tc := range []struct {
			name    string
			status  int
			body    string
			wantErr bool
		}{
			{"missing endpoint", http.StatusNotFound, "not found", true},
			{"unauthorized", http.StatusUnauthorized, "unauthorized", true},
			{"forbidden", http.StatusForbidden, "forbidden", true},
			{"server error", http.StatusInternalServerError, "error", true},
			{"invalid response", http.StatusOK, "not JSON", true},
			{"empty tokens", http.StatusOK, `{"token_ids":[]}`, false},
		} {
			t.Run(mode+"/"+tc.name, func(t *testing.T) {
				calls := 0
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls++
					require.Equal(t, path, r.URL.Path)
					w.WriteHeader(tc.status)
					_, _ = io.WriteString(w, tc.body)
				}))
				defer srv.Close()
				parsed, err := anthropic.NewAnthropicParser().ParseRequest(context.Background(),
					[]byte(`{"model":"adapter","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`), map[string]string{":path": "/v1/messages"})
				require.NoError(t, err)
				p := newTestPlugin(newHTTPRenderer(t, srv))
				p.backend = renderBackend{tk: newHTTPRenderer(t, srv), modelName: "configured-model", legacyMessages: legacy}
				req := &scheduling.InferenceRequest{Body: parsed.Body}
				err = p.Produce(context.Background(), req, nil)
				if tc.wantErr {
					require.Error(t, err)
				} else {
					require.NoError(t, err)
				}
				require.Nil(t, req.Body.TokenizedRequest)
				require.Equal(t, 1, calls)
			})
		}
	}
}

func TestMessagesRenderModeLeavesOtherProtocolsUnchanged(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		mode := "native"
		if legacy {
			mode = "legacy"
		}
		for _, tc := range []struct {
			name, path, raw, renderPath string
			parser                      fwkrh.Parser
			direct                      bool
		}{
			{"chat", "/v1/chat/completions", ` {"model":"adapter","messages":[{"role":"user","content":"hi"}],"unknown":{"z":1,"a":2}} `, chatRenderPath, openai.NewOpenAIParser(), false},
			{"completions", "/v1/completions", ` {"model":"adapter","prompt":[1,2,3],"truncate_prompt_tokens":3} `, completionsRenderPath, openai.NewOpenAIParser(), false},
			{"vllm generate", "/inference/v1/generate", `{"model":"adapter","token_ids":[1,2,3],"sampling_params":{"max_tokens":1}}`, "", vllmhttp.NewVllmHTTPParser(), false},
			{"sglang generate", "/generate", `{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":1}}`, "", sglanghttp.NewSGLangHTTPParser(), false},
			{"direct messages", messagesRenderPath, `{"model":"adapter","unknown":{"z":1,"a":2}}`, "", anthropic.NewAnthropicParser(), true},
			{"direct chat", chatRenderPath, `{"model":"adapter","unknown":{"z":1,"a":2}}`, "", openai.NewOpenAIParser(), true},
			{"direct completions", completionsRenderPath, `{"model":"adapter","unknown":{"z":1,"a":2}}`, "", openai.NewOpenAIParser(), true},
		} {
			t.Run(mode+"/"+tc.name, func(t *testing.T) {
				calls := 0
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls++
					require.Equal(t, tc.renderPath, r.URL.Path)
					raw, err := io.ReadAll(r.Body)
					require.NoError(t, err)
					require.Equal(t, tc.raw, string(raw))
					if r.URL.Path == completionsRenderPath {
						_, _ = io.WriteString(w, `[{"token_ids":[1,2,3]}]`)
					} else {
						_, _ = io.WriteString(w, `{"token_ids":[1,2,3]}`)
					}
				}))
				defer srv.Close()
				p := newTestPlugin(newHTTPRenderer(t, srv))
				p.backend = renderBackend{tk: newHTTPRenderer(t, srv), modelName: "configured-model", legacyMessages: legacy}
				parsed, err := tc.parser.ParseRequest(context.Background(), []byte(tc.raw), map[string]string{":path": tc.path})
				require.NoError(t, err)
				before, err := json.Marshal(parsed.Body.Payload)
				require.NoError(t, err)
				req := &scheduling.InferenceRequest{Body: parsed.Body}
				require.NoError(t, p.Produce(context.Background(), req, nil))
				if tc.direct {
					require.Nil(t, req.Body.TokenizedRequest)
				} else {
					require.Equal(t, []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}}, req.Body.TokenizedRequest.Prompts)
				}
				after, err := json.Marshal(req.Body.Payload)
				require.NoError(t, err)
				require.Equal(t, before, after)
				require.False(t, req.Body.Mutated)
				if tc.renderPath == "" {
					require.Zero(t, calls)
				} else {
					require.Equal(t, 1, calls)
				}
			})
		}
	}
}

func TestLegacyMessagesPreservesPrepopulatedTokens(t *testing.T) {
	p := newTestPlugin(&mockTokenizer{})
	p.backend = renderBackend{tk: &mockTokenizer{}, legacyMessages: true}
	tokens := &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}}}
	req := &scheduling.InferenceRequest{Body: &fwkrh.InferenceRequestBody{
		Messages:         &fwkrh.MessagesRequest{CacheSalt: "tenant-a"},
		Payload:          fwkrh.RawPayload(`{"cache_salt":"tenant-a"}`),
		TokenizedRequest: tokens,
	}}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.Same(t, tokens, req.Body.TokenizedRequest)
	require.Equal(t, "tenant-a", tokens.CacheSalt)
}

func TestLegacyMessagesToRenderChatRequest_RawSystem(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		System:   fwkrh.AnthropicContent{Raw: "You are helpful."},
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hello"}}},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 2)
	assert.Equal(t, "system", result.Conversation[0].Role)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "You are helpful."}, result.Conversation[0].Content)
	assert.Equal(t, "user", result.Conversation[1].Role)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "Hello"}, result.Conversation[1].Content)
}

func TestLegacyMessagesToRenderChatRequest_Tools(t *testing.T) {
	tools := []fwkrh.AnthropicTool{
		{Name: "get_weather", Description: "Get the weather", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)},
	}
	msg := &fwkrh.MessagesRequest{
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "What is the weather today?"}}},
		Tools:    tools,
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Tools, 1)
	assert.Equal(t, map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "get_weather",
			"description": "Get the weather",
			"parameters":  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		},
	}, result.Tools[0])
}

func TestLegacyMessagesToRenderChatRequest_ToolDefaults(t *testing.T) {
	strict, deferLoading := true, false
	msg := &fwkrh.MessagesRequest{
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hi"}}},
		Tools: []fwkrh.AnthropicTool{
			{Name: "no_schema"},
			{Name: "flags", InputSchema: json.RawMessage(`null`), Strict: &strict, DeferLoading: &deferLoading},
		},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Tools, 2)
	assert.Equal(t, map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":       "no_schema",
			"parameters": json.RawMessage(`{"type":"object"}`),
		},
	}, result.Tools[0])
	assert.Equal(t, map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":          "flags",
			"parameters":    json.RawMessage(`{"type":"object"}`),
			"strict":        true,
			"defer_loading": false,
		},
	}, result.Tools[1])
}

func TestLegacyMessagesToRenderChatRequest_StructuredSystem(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		System: fwkrh.AnthropicContent{
			Structured: []fwkrh.AnthropicContentBlock{
				{Type: "text", Text: "System line 1."},
				{Type: "text", Text: "System line 2."},
			},
		},
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hi"}}},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 2)
	assert.Equal(t, "system", result.Conversation[0].Role)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "System line 1.System line 2."}, result.Conversation[0].Content)
}

func TestLegacyMessagesToRenderChatRequest_SystemBillingHeaderStripped(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		System: fwkrh.AnthropicContent{
			Structured: []fwkrh.AnthropicContentBlock{
				{Type: "text", Text: "x-anthropic-billing-header: 7b3f2c"},
				{Type: "text", Text: "Real system prompt."},
			},
		},
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hi"}}},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 2)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "Real system prompt."}, result.Conversation[0].Content)
}

func TestLegacyMessagesToRenderChatRequest_NoSystem(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hi"}}},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 1)
	assert.Equal(t, "user", result.Conversation[0].Role)
}

func TestLegacyMessagesToRenderChatRequest_StructuredMessage(t *testing.T) {
	tests := []struct {
		name     string
		messages []fwkrh.AnthropicMessage
		wantConv []tokenizerTypes.Conversation
	}{
		{
			name: "text-only structured content",
			messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{
					Structured: []fwkrh.AnthropicContentBlock{
						{Type: "text", Text: "Hello"},
						{Type: "text", Text: "World"},
					},
				}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "user", Content: &tokenizerTypes.Content{
					Structured: []tokenizerTypes.ContentBlock{
						{Type: "text", Text: "Hello"},
						{Type: "text", Text: "World"},
					},
				}},
			},
		},
		{
			name: "image returns data URI",
			messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{
					Structured: []fwkrh.AnthropicContentBlock{
						{Type: "text", Text: "Describe this"},
						{Type: "image", Source: &fwkrh.AnthropicImageSource{Type: "base64", MediaType: "image/png", Data: "abc123"}},
					},
				}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "user", Content: &tokenizerTypes.Content{
					Structured: []tokenizerTypes.ContentBlock{
						{Type: "text", Text: "Describe this"},
						{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "data:image/png;base64,abc123"}},
					},
				}},
			},
		},
		{
			name: "image returns https URL",
			messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{
					Structured: []fwkrh.AnthropicContentBlock{
						{Type: "text", Text: "Describe this"},
						{Type: "image", Source: &fwkrh.AnthropicImageSource{Type: "url", URL: "https://example.com/img.jpg"}},
					},
				}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "user", Content: &tokenizerTypes.Content{
					Structured: []tokenizerTypes.ContentBlock{
						{Type: "text", Text: "Describe this"},
						{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "https://example.com/img.jpg"}},
					},
				}},
			},
		},
		{
			name: "image with no media type defaults to jpeg",
			messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{
					Structured: []fwkrh.AnthropicContentBlock{
						{Type: "image", Source: &fwkrh.AnthropicImageSource{Type: "base64", Data: "abc123"}},
					},
				}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "user", Content: &tokenizerTypes.Content{
					Structured: []tokenizerTypes.ContentBlock{
						{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "data:image/jpeg;base64,abc123"}},
					},
				}},
			},
		},
		{
			name: "image source with neither URL nor data is dropped",
			messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{
					Structured: []fwkrh.AnthropicContentBlock{
						{Type: "text", Text: "Describe this"},
						{Type: "image", Source: &fwkrh.AnthropicImageSource{Type: "base64"}},
					},
				}},
			},
			wantConv: []tokenizerTypes.Conversation{
				{Role: "user", Content: &tokenizerTypes.Content{Raw: "Describe this"}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := &fwkrh.MessagesRequest{Messages: tt.messages}
			result := legacyMessagesToRenderChatRequest(msg)
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

func TestLegacyProduceMessages(t *testing.T) {
	wantTokens := []uint32{100, 200, 300}
	var gotPayload fwkrh.RequestPayload
	tok := &mockTokenizer{
		renderChatFunc: func(payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			gotPayload = payload
			return wantTokens, nil, nil
		},
	}
	p := newTestPlugin(tok)
	p.backend = renderBackend{tk: tok, modelName: "configured-model", legacyMessages: true}

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{
				"system":   "Be helpful.",
				"messages": []any{map[string]any{"role": "user", "content": "Hi"}},
			},
			Messages: &fwkrh.MessagesRequest{
				System:   fwkrh.AnthropicContent{Raw: "Be helpful."},
				Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hi"}}},
			},
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []fwkrh.PromptTokens{{TokenIDs: wantTokens}}, req.Body.TokenizedRequest.Prompts)

	pm, ok := gotPayload.AsMap()
	require.True(t, ok, "RenderChat payload must be a map")
	assert.NotContains(t, pm, "system", "raw Anthropic top-level system must not reach /render")
	msgs, ok := pm["messages"].([]any)
	require.True(t, ok, "payload must carry the /render chat messages array")
	require.Len(t, msgs, 2)
	assertLegacyRolesInOrder(t, msgs, "system", "user")
}

func assertLegacyRolesInOrder(t *testing.T, msgs []any, roles ...string) {
	t.Helper()
	require.Len(t, msgs, len(roles))
	for i, want := range roles {
		raw, ok := msgs[i].(json.RawMessage)
		require.True(t, ok, "message %d must be pre-encoded JSON", i)
		var m map[string]any
		require.NoError(t, json.Unmarshal(raw, &m), "message %d must be valid JSON", i)
		assert.Equal(t, want, m["role"], "message %d role", i)
	}
}

func TestLegacyPythonDumps(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"object with separators", "{\"city\":\"Z\u00fcrich\",\"n\":5}", `{"city": "Z\u00fcrich", "n": 5}`},
		{"nested arrays and objects", `{"z":1,"a":[{"y":1,"b":2},true,null,"x"]}`, `{"z": 1, "a": [{"y": 1, "b": 2}, true, null, "x"]}`},
		{"empty object", `{}`, `{}`},
		{"null", `null`, `null`},
		{"escapes", `{"s":"a\"b\\c\nd e","f":"/"}`, `{"s": "a\"b\\c\nd e", "f": "/"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := legacyPythonDumps(json.RawMessage(tt.in))
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestLegacyPythonDumpsNonASCIIEscaped(t *testing.T) {
	got, err := legacyPythonDumps(json.RawMessage("{\"e\":\"Z\u00fcrich \U0001f600\"}"))
	require.NoError(t, err)
	assert.Equal(t, `{"e": "Z\u00fcrich \ud83d\ude00"}`, got)
}

func TestLegacyPythonArguments(t *testing.T) {
	assert.Equal(t, "{}", legacyPythonArguments(nil))
	assert.Equal(t, "{}", legacyPythonArguments(json.RawMessage(`null`)))
	assert.Equal(t, "{}", legacyPythonArguments(json.RawMessage(`{}`)))
	assert.Equal(t, `{"a": 1}`, legacyPythonArguments(json.RawMessage(`{"a":1}`)))
}

func TestLegacyMessagesToRenderChatRequest_ToolUseAndThinking(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		Messages: []fwkrh.AnthropicMessage{
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Book a table"}},
			{Role: "assistant", Content: fwkrh.AnthropicContent{
				Structured: []fwkrh.AnthropicContentBlock{
					{Type: "thinking", Thinking: "The user wants dinner."},
					{Type: "redacted_thinking"},
					{Type: "text", Text: "Sure."},
					{Type: "tool_use", ID: "toolu_01", Name: "book_table", Input: json.RawMessage(`{"guests":2,"time":"19:00"}`)},
					{Type: "tool_use", ID: "toolu_02", Name: "notify", Input: json.RawMessage(`null`)},
				},
			}},
		},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 2)
	assistant := result.Conversation[1]
	assert.Equal(t, "assistant", assistant.Role)
	assert.Equal(t, "The user wants dinner.", assistant.Reasoning)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "Sure."}, assistant.Content)
	require.Len(t, assistant.ToolCalls, 2)
	assert.Equal(t, map[string]any{
		"id":   "toolu_01",
		"type": "function",
		"function": map[string]any{
			"name":      "book_table",
			"arguments": `{"guests": 2, "time": "19:00"}`,
		},
	}, assistant.ToolCalls[0])
	assert.Equal(t, map[string]any{
		"id":   "toolu_02",
		"type": "function",
		"function": map[string]any{
			"name":      "notify",
			"arguments": "{}",
		},
	}, assistant.ToolCalls[1])
}

func TestLegacyMessagesToRenderChatRequest_AssistantToolOnlyOmitsContent(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		Messages: []fwkrh.AnthropicMessage{
			{Role: "assistant", Content: fwkrh.AnthropicContent{
				Structured: []fwkrh.AnthropicContentBlock{
					{Type: "tool_use", ID: "toolu_01", Name: "run", Input: json.RawMessage(`{"cmd":"ls"}`)},
				},
			}},
		},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 1)
	assert.Nil(t, result.Conversation[0].Content)
	require.Len(t, result.Conversation[0].ToolCalls, 1)
}

func TestLegacyMessagesToRenderChatRequest_ToolResult(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		Messages: []fwkrh.AnthropicMessage{
			{Role: "user", Content: fwkrh.AnthropicContent{
				Structured: []fwkrh.AnthropicContentBlock{
					{Type: "tool_result", ToolUseID: "toolu_01", Content: fwkrh.AnthropicContent{
						Structured: []fwkrh.AnthropicContentBlock{
							{Type: "text", Text: "stdout line 1"},
							{Type: "text", Text: "stdout line 2"},
							{Type: "image", Source: &fwkrh.AnthropicImageSource{Type: "base64", MediaType: "image/png", Data: "abc"}},
						},
					}},
					{Type: "text", Text: "What do you see?"},
				},
			}},
			{Role: "user", Content: fwkrh.AnthropicContent{
				Structured: []fwkrh.AnthropicContentBlock{
					{Type: "tool_result", ToolUseID: "toolu_02", Content: fwkrh.AnthropicContent{Raw: "plain string result"}},
				},
			}},
		},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 4)

	tool1 := result.Conversation[0]
	assert.Equal(t, "tool", tool1.Role)
	assert.Equal(t, "toolu_01", tool1.ToolCallID)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "stdout line 1\nstdout line 2"}, tool1.Content)

	images := result.Conversation[1]
	assert.Equal(t, "user", images.Role)
	assert.Equal(t, &tokenizerTypes.Content{
		Structured: []tokenizerTypes.ContentBlock{
			{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "data:image/png;base64,abc"}},
		},
	}, images.Content)

	user := result.Conversation[2]
	assert.Equal(t, "user", user.Role)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "What do you see?"}, user.Content)

	tool2 := result.Conversation[3]
	assert.Equal(t, "tool", tool2.Role)
	assert.Equal(t, "toolu_02", tool2.ToolCallID)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "plain string result"}, tool2.Content)
}

func TestLegacyMessagesToRenderChatRequest_ToolResultOnlyUserDropped(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		Messages: []fwkrh.AnthropicMessage{
			{Role: "user", Content: fwkrh.AnthropicContent{
				Structured: []fwkrh.AnthropicContentBlock{
					{Type: "tool_result", ToolUseID: "toolu_01", Content: fwkrh.AnthropicContent{Raw: "result"}},
				},
			}},
		},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 1)
	assert.Equal(t, "tool", result.Conversation[0].Role)
}

func TestLegacyMessagesToRenderChatRequest_FullAgenticTurn(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		System: fwkrh.AnthropicContent{Raw: "You can use tools."},
		Tools: []fwkrh.AnthropicTool{{
			Name:        "get_weather",
			Description: "Get the weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		}},
		Messages: []fwkrh.AnthropicMessage{
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Weather in Zurich?"}},
			{Role: "assistant", Content: fwkrh.AnthropicContent{
				Structured: []fwkrh.AnthropicContentBlock{
					{Type: "tool_use", ID: "toolu_01", Name: "get_weather", Input: json.RawMessage(`{"city":"Zurich"}`)},
				},
			}},
			{Role: "user", Content: fwkrh.AnthropicContent{
				Structured: []fwkrh.AnthropicContentBlock{
					{Type: "tool_result", ToolUseID: "toolu_01", Content: fwkrh.AnthropicContent{Raw: "Sunny, 22C"}},
				},
			}},
		},
	}

	result := legacyMessagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 4)
	assert.Equal(t, "system", result.Conversation[0].Role)
	assert.Equal(t, "user", result.Conversation[1].Role)
	assert.Equal(t, "assistant", result.Conversation[2].Role)
	assert.Nil(t, result.Conversation[2].Content)
	assert.Equal(t, "tool", result.Conversation[3].Role)
	assert.Equal(t, "toolu_01", result.Conversation[3].ToolCallID)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "Sunny, 22C"}, result.Conversation[3].Content)
	require.Len(t, result.Tools, 1)
}

func TestLegacyProduceMessagesToolSchemaOrder(t *testing.T) {
	var gotPayload fwkrh.RequestPayload
	tok := &mockTokenizer{
		renderChatFunc: func(payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			gotPayload = payload
			return []uint32{1}, nil, nil
		},
	}
	p := newTestPlugin(tok)
	p.backend = renderBackend{tk: tok, modelName: "configured-model", legacyMessages: true}

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Messages: &fwkrh.MessagesRequest{
				Tools: []fwkrh.AnthropicTool{{
					Name:        "get_weather",
					InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
				}},
				Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "hi"}}},
			},
		},
	}
	require.NoError(t, p.Produce(context.Background(), req, nil))

	rendered, err := json.Marshal(gotPayload)
	require.NoError(t, err)
	assert.Contains(t, string(rendered),
		`"parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`,
		"input_schema key order must be preserved verbatim")
}
