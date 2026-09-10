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
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

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
		{"model only defaults to auto", `{"modelName":"configured-model"}`, false},
		{"empty config defaults to auto", `{"modelName":"configured-model","vllm":{}}`, false},
		{"empty mode defaults to auto", `{"modelName":"configured-model","vllm":{"messagesRenderMode":""}}`, false},
		{"explicit auto", `{"modelName":"configured-model","vllm":{"messagesRenderMode":"auto"}}`, false},
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
			auto := p.backend.(renderBackend).legacyMessages != nil && !tc.legacy

			const raw = ` {"model":"adapter","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"cache_salt":"tenant-a","unknown":{"z":1,"a":2}} `
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				body, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				require.Equal(t, "Bearer secret", r.Header.Get("Authorization"))
				if auto && calls == 1 {
					require.Equal(t, messagesRenderPath, r.URL.Path)
					require.JSONEq(t, `{"model":"configured-model","max_tokens":1,"messages":[{"role":"user","content":"warmup"}]}`, string(body))
					_, _ = io.WriteString(w, `{"token_ids":[1]}`)
					return
				}
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
	for _, mode := range []string{"unknown", "NATIVE", "legacy "} {
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

func TestMessagesRenderModeChatOnlyRenderer(t *testing.T) {
	for _, mode := range []string{messagesRenderModeNative, messagesRenderModeLegacy} {
		t.Run("mode="+mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			params := fmt.Sprintf(`{"modelName":"configured-model","vllm":{"messagesRenderMode":%q}}`, mode)
			got, err := PluginFactory("messages", plugin.StrictDecoder(json.RawMessage(params)), plugin.NewEppHandle(ctx, nil))
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
			const raw = `{"model":"adapter","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`
			parsed, err := anthropic.NewAnthropicParser().ParseRequest(context.Background(), []byte(raw), map[string]string{":path": "/v1/messages"})
			require.NoError(t, err)
			req := &scheduling.InferenceRequest{Body: parsed.Body}
			err = p.Produce(context.Background(), req, nil)
			if mode == messagesRenderModeLegacy {
				require.NoError(t, err)
				require.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
				require.Equal(t, []string{chatRenderPath}, paths)
			} else {
				var statusErr *renderStatusError
				require.ErrorAs(t, err, &statusErr)
				require.Equal(t, http.StatusNotFound, statusErr.StatusCode)
				require.Nil(t, req.Body.TokenizedRequest)
				require.Equal(t, []string{messagesRenderPath}, paths)
			}
			require.Equal(t, fwkrh.RawPayload(raw), req.Body.WirePayload())
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
				renderer := newHTTPRenderer(t, srv)
				p := newTestPlugin(renderer)
				selection, err := configureLegacyMessages(context.Background(), "test", mode)
				require.NoError(t, err)
				p.backend = renderBackend{tk: renderer, modelName: "configured-model", legacyMessages: selection}
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
	for _, mode := range []string{"auto", "native", "legacy"} {
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
				renderer := newHTTPRenderer(t, srv)
				p := newTestPlugin(renderer)
				selection, err := configureLegacyMessages(context.Background(), "test", mode)
				require.NoError(t, err)
				p.backend = renderBackend{tk: renderer, modelName: "configured-model", legacyMessages: selection}
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
	p.backend = renderBackend{tk: &mockTokenizer{}, legacyMessages: &legacyMessagesMode{mode: messagesRenderModeLegacy}}
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

func TestLegacyMessagesPayloadWire(t *testing.T) {
	const raw = `{"messages":[{"role":"user","content":"<hi>"},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"run","input":{"z":1e0,"a":2}}]}],"tools":[{"name":"run","input_schema":{"z":1,"a":2}}]}`
	const want = `{"messages":[{"role":"user","content":"\u003chi\u003e"},{"role":"assistant","tool_calls":[{"function":{"arguments":"{\"z\": 1e0, \"a\": 2}","name":"run"},"id":"call_1","type":"function"}]}],"tools":[{"function":{"name":"run","parameters":{"z":1,"a":2}},"type":"function"}]}`
	var msg fwkrh.MessagesRequest
	require.NoError(t, json.Unmarshal([]byte(raw), &msg))
	got, err := legacyMessagesPayload(&msg).Marshal()
	require.NoError(t, err)
	require.Equal(t, want, string(got))
}

func TestAutoMessagesRendering(t *testing.T) {
	for _, tc := range []struct {
		name         string
		nativeStatus int
		nativeBody   string
		chatStatus   int
		chatBody     string
		userStatus   int
		legacy       bool
		discoveryErr bool
	}{
		{name: "native"},
		{name: "chat only", nativeStatus: 404, legacy: true},
		{name: "method unavailable", nativeStatus: 405, legacy: true},
		{name: "neither endpoint", nativeStatus: 404, chatStatus: 404, discoveryErr: true},
		{name: "unknown main model", nativeStatus: 404, nativeBody: `{"error":{"type":"NotFoundError","param":"model"}}`, chatStatus: 404, discoveryErr: true},
		{name: "unauthorized", nativeStatus: 401, discoveryErr: true},
		{name: "forbidden", nativeStatus: 403, discoveryErr: true},
		{name: "invalid probe", nativeStatus: 400, discoveryErr: true},
		{name: "validation error", nativeStatus: 422, discoveryErr: true},
		{name: "rate limited", nativeStatus: 429, discoveryErr: true},
		{name: "server error", nativeStatus: 500, discoveryErr: true},
		{name: "not implemented", nativeStatus: 501, discoveryErr: true},
		{name: "malformed response", nativeBody: "not JSON", discoveryErr: true},
		{name: "empty response", nativeBody: `{"token_ids":[]}`, discoveryErr: true},
		{name: "chat unauthorized", nativeStatus: 404, chatStatus: 401, discoveryErr: true},
		{name: "chat malformed response", nativeStatus: 404, chatBody: "not JSON", discoveryErr: true},
		{name: "chat empty response", nativeStatus: 404, chatBody: `{"token_ids":[]}`, discoveryErr: true},
		{name: "unknown request adapter", userStatus: 404},
		{name: "request server error", userStatus: 500},
	} {
		for _, mode := range []string{"", "auto"} {
			t.Run(tc.name+"/mode="+mode, func(t *testing.T) {
				const raw = ` {"model":"adapter","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"output_config":{"effort":"high"},"unknown":{"z":1,"a":2}} `
				var calls []string
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					require.Equal(t, http.MethodPost, r.Method)
					require.Equal(t, "Bearer secret", r.Header.Get("Authorization"))
					body, err := io.ReadAll(r.Body)
					require.NoError(t, err)
					var payload struct {
						Model    string `json:"model"`
						Messages []struct {
							Content string `json:"content"`
						} `json:"messages"`
					}
					require.NoError(t, json.Unmarshal(body, &payload))
					require.Len(t, payload.Messages, 1)
					status, response := tc.userStatus, `{"token_ids":[1,2,3]}`
					if payload.Messages[0].Content == "warmup" {
						calls = append(calls, "probe "+r.URL.Path)
						require.JSONEq(t, `{"model":"configured-model","max_tokens":1,"messages":[{"role":"user","content":"warmup"}]}`, string(body))
						status = tc.nativeStatus
						if r.URL.Path == chatRenderPath {
							status = tc.chatStatus
							if tc.chatBody != "" {
								response = tc.chatBody
							}
						} else {
							require.Equal(t, messagesRenderPath, r.URL.Path)
							if tc.nativeBody != "" {
								response = tc.nativeBody
							}
						}
					} else {
						calls = append(calls, "request "+r.URL.Path)
						if tc.legacy {
							require.Equal(t, chatRenderPath, r.URL.Path)
							require.Equal(t, "configured-model", payload.Model)
						} else {
							require.Equal(t, messagesRenderPath, r.URL.Path)
							require.Equal(t, raw, string(body))
						}
					}
					if status != 0 {
						w.WriteHeader(status)
					}
					_, _ = io.WriteString(w, response)
				}))
				defer srv.Close()
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				p, err := NewPlugin(ctx, "messages", &tokenizerPluginConfig{
					ModelName: "configured-model", VLLM: &vllmConfig{URL: srv.URL, MessagesRenderMode: mode},
				})
				require.NoError(t, err)
				var wantCalls []string
				for i := range 2 {
					if i == 0 || tc.discoveryErr {
						wantCalls = append(wantCalls, "probe "+messagesRenderPath)
						if tc.nativeStatus == 404 || tc.nativeStatus == 405 {
							wantCalls = append(wantCalls, "probe "+chatRenderPath)
						}
					}
					if !tc.discoveryErr {
						path := messagesRenderPath
						if tc.legacy {
							path = chatRenderPath
						}
						wantCalls = append(wantCalls, "request "+path)
					}
					parsed, err := anthropic.NewAnthropicParser().ParseRequest(context.Background(), []byte(raw), map[string]string{":path": "/v1/messages"})
					require.NoError(t, err)
					req := &scheduling.InferenceRequest{Body: parsed.Body, Headers: map[string]string{"authorization": "Bearer secret"}}
					err = p.Produce(context.Background(), req, nil)
					if tc.discoveryErr || tc.userStatus != 0 {
						require.Error(t, err)
						require.Nil(t, req.Body.TokenizedRequest)
					} else {
						require.NoError(t, err)
						require.Equal(t, []uint32{1, 2, 3}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)
					}
					require.Equal(t, fwkrh.RawPayload(raw), req.Body.WirePayload())
				}
				require.Equal(t, wantCalls, calls)
			})
		}
	}
}

func TestAutoMessagesDiscoveryConcurrent(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			started, release := make(chan struct{}), make(chan struct{})
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.URL.Path == messagesRenderPath {
					close(started)
					<-release
					if legacy {
						http.NotFound(w, r)
						return
					}
				} else {
					require.Equal(t, chatRenderPath, r.URL.Path)
				}
				_, _ = io.WriteString(w, `{"token_ids":[1]}`)
			}))
			defer srv.Close()
			var warnings strings.Builder
			ctx := log.IntoContext(context.Background(), funcr.New(func(_, args string) { warnings.WriteString(args) }, funcr.Options{}))
			selection, err := configureLegacyMessages(ctx, "test", "auto")
			require.NoError(t, err)
			renderer := newHTTPRenderer(t, srv)
			var wg sync.WaitGroup
			for range 16 {
				wg.Go(func() {
					got, err := selection.useLegacy(ctx, renderer, "main")
					assert.NoError(t, err)
					assert.Equal(t, legacy, got)
				})
			}
			<-started
			cancelled, cancel := context.WithCancel(ctx)
			cancel()
			_, err = selection.useLegacy(cancelled, renderer, "main")
			assert.ErrorIs(t, err, context.Canceled)
			close(release)
			wg.Wait()
			if legacy {
				require.EqualValues(t, 2, calls.Load())
				require.Equal(t, 1, strings.Count(warnings.String(), "deprecated"))
			} else {
				require.EqualValues(t, 1, calls.Load())
				require.Empty(t, warnings.String())
			}
		})
	}
}

func TestAutoMessagesDiscoveryRetry(t *testing.T) {
	for _, failure := range []string{"authentication", "timeout", "transport"} {
		t.Run(failure, func(t *testing.T) {
			selection, err := configureLegacyMessages(context.Background(), "test", "auto")
			require.NoError(t, err)
			renderer, err := newVLLMHTTPRenderer(&vllmConfig{Timeout: "1ms", MMTimeout: "1ms"})
			require.NoError(t, err)
			calls := 0
			renderer.client.Transport = roundTripperFunc(func(r *http.Request) (*http.Response, error) {
				calls++
				require.Equal(t, messagesRenderPath, r.URL.Path)
				if calls == 1 {
					switch failure {
					case "authentication":
						require.Empty(t, r.Header.Get("Authorization"))
						return &http.Response{StatusCode: 401, Body: io.NopCloser(strings.NewReader("unauthorized")), Header: http.Header{}}, nil
					case "timeout":
						<-r.Context().Done()
						return nil, r.Context().Err()
					default:
						return nil, io.ErrUnexpectedEOF
					}
				}
				require.Equal(t, "Bearer secret", r.Header.Get("Authorization"))
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"token_ids":[1]}`)), Header: http.Header{}}, nil
			})
			_, err = selection.useLegacy(context.Background(), renderer, "main")
			require.Error(t, err)
			for range 2 {
				legacy, err := selection.useLegacy(withAuthHeader(context.Background(), "Bearer secret"), renderer, "main")
				require.NoError(t, err)
				require.False(t, legacy)
			}
			require.Equal(t, 2, calls)
		})
	}
}

func TestAutoMessagesDiscoveryWarmup(t *testing.T) {
	var calls []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.URL.Path)
		_, _ = io.WriteString(w, `{"token_ids":[1]}`)
	}))
	defer srv.Close()
	selection, err := configureLegacyMessages(context.Background(), "test", "auto")
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	backend := renderBackend{tk: newHTTPRenderer(t, srv), modelName: "main", legacyMessages: selection}
	backend.warmup(ctx)
	require.NoError(t, ctx.Err())
	legacy, err := selection.useLegacy(ctx, backend.tk, "main")
	require.NoError(t, err)
	require.False(t, legacy)
	require.Equal(t, []string{messagesRenderPath, chatRenderPath, chatRenderPath}, calls)
}

func TestMessagesDiscoveryLive(t *testing.T) {
	e := renderEndpointForTest(t)
	target, err := url.Parse(e.url)
	require.NoError(t, err)
	for _, tc := range []struct {
		name                                               string
		hide, invalidMain, invalidRequest, unauthenticated bool
	}{
		{name: "native"},
		{name: "chat-only", hide: true},
		{name: "unknown-main-model", invalidMain: true},
		{name: "unknown-request-adapter", invalidRequest: true},
		{name: "authentication-retry", unauthenticated: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.unauthenticated && e.auth == "" {
				t.Skip("set VLLM_RENDER_TEST_AUTHORIZATION for authentication checks")
			}
			proxy := httputil.NewSingleHostReverseProxy(target)
			var paths []string
			var mu sync.Mutex
			endpoint := e
			if tc.invalidMain {
				endpoint.model = "missing-main-model"
			}
			p := endpoint.plugin(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				paths = append(paths, r.URL.Path)
				mu.Unlock()
				require.NotEqual(t, "/tokenize", r.URL.Path)
				if tc.hide && r.URL.Path == messagesRenderPath {
					http.NotFound(w, r)
					return
				}
				proxy.ServeHTTP(w, r)
			}))
			if tc.unauthenticated {
				backend := p.backend.(renderBackend)
				_, err := backend.legacyMessages.useLegacy(t.Context(), backend.tk, e.model)
				require.True(t, isRenderAuthError(err), "%v", err)
			}
			model := e.model
			if tc.invalidRequest {
				model = "missing-request-adapter"
			}
			raw := []byte(fmt.Sprintf(` {"model":%q,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]} `, model))
			var want []fwkrh.PromptTokens
			if !tc.invalidMain && !tc.invalidRequest {
				want = e.render(t, messagesRenderPath, raw)
			}
			for range 2 {
				parsed, err := anthropic.NewAnthropicParser().ParseRequest(t.Context(), raw, map[string]string{":path": "/v1/messages"})
				require.NoError(t, err)
				req := &scheduling.InferenceRequest{Body: parsed.Body, Headers: map[string]string{"authorization": e.auth}}
				err = p.Produce(t.Context(), req, nil)
				if tc.invalidMain || tc.invalidRequest {
					var status *renderStatusError
					require.ErrorAs(t, err, &status)
					require.Equal(t, http.StatusNotFound, status.StatusCode)
					require.Nil(t, req.Body.TokenizedRequest)
				} else {
					require.NoError(t, err)
					require.Equal(t, want, req.Body.TokenizedRequest.Prompts)
				}
				require.Equal(t, fwkrh.RawPayload(raw), req.Body.WirePayload())
			}
			wantPaths := []string{messagesRenderPath, messagesRenderPath, messagesRenderPath}
			if tc.hide {
				wantPaths = []string{messagesRenderPath, chatRenderPath, chatRenderPath, chatRenderPath}
			}
			if tc.invalidMain {
				wantPaths = []string{messagesRenderPath, chatRenderPath, messagesRenderPath, chatRenderPath}
			}
			if tc.unauthenticated {
				wantPaths = append([]string{messagesRenderPath}, wantPaths...)
			}
			mu.Lock()
			defer mu.Unlock()
			require.Equal(t, wantPaths, paths)
		})
	}
}

func TestLegacyMessagesRenderLive(t *testing.T) {
	e := renderEndpointForTest(t)
	target, err := url.Parse(e.url)
	require.NoError(t, err)
	for _, tc := range []struct{ name, messages, chat string }{
		{"text", `"messages":[{"role":"user","content":"hi"}]`, `"messages":[{"role":"user","content":"hi"}]`},
		{"system", `"system":"Be brief","messages":[{"role":"user","content":"hi"}]`, `"messages":[{"role":"system","content":"Be brief"},{"role":"user","content":"hi"}]`},
		{"structured-text", `"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]`, `"messages":[{"role":"user","content":"hi"}]`},
		{"tools", `"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"greet","input_schema":{"type":"object","properties":{"z":{"type":"number"},"a":{"type":"string"}}}}]`, `"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"greet","parameters":{"type":"object","properties":{"z":{"type":"number"},"a":{"type":"string"}}}}}]`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			proxy := httputil.NewSingleHostReverseProxy(target)
			var paths []string
			var mu sync.Mutex
			p := e.plugin(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				paths = append(paths, r.URL.Path)
				mu.Unlock()
				proxy.ServeHTTP(w, r)
			}))
			raw := []byte(fmt.Sprintf(`{"model":%q,"max_tokens":8,%s}`, e.model, tc.messages))
			parsed, err := anthropic.NewAnthropicParser().ParseRequest(t.Context(), raw, map[string]string{":path": "/v1/messages"})
			require.NoError(t, err)
			req := &scheduling.InferenceRequest{Body: parsed.Body, Headers: map[string]string{"authorization": e.auth}}
			require.NoError(t, p.Produce(t.Context(), req, nil))
			require.Equal(t, e.render(t, chatRenderPath, []byte(fmt.Sprintf(`{"model":%q,%s}`, e.model, tc.chat))), req.Body.TokenizedRequest.Prompts)
			require.Equal(t, fwkrh.RawPayload(raw), parsed.Body.WirePayload())
			mu.Lock()
			defer mu.Unlock()
			require.Equal(t, []string{messagesRenderPath, chatRenderPath, chatRenderPath}, paths, "requires a renderer without /v1/messages/render")
		})
	}
}

func BenchmarkLegacyMessagesPayload(b *testing.B) {
	for _, count := range []int{1, 32, 256} {
		b.Run(fmt.Sprintf("messages=%d", count), func(b *testing.B) {
			msg := &fwkrh.MessagesRequest{Messages: make([]fwkrh.AnthropicMessage, count)}
			for i := range msg.Messages {
				msg.Messages[i] = fwkrh.AnthropicMessage{Role: "user", Content: fwkrh.AnthropicContent{Raw: strings.Repeat("text ", 64)}}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := legacyMessagesPayload(msg).Marshal(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestLegacyMessagesToRenderChatRequest_RawSystem(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		System:   fwkrh.AnthropicContent{Raw: "You are helpful."},
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hello"}}},
	}

	result := messagesToRenderChatRequest(msg)

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

	result := messagesToRenderChatRequest(msg)

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

	result := messagesToRenderChatRequest(msg)

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

	result := messagesToRenderChatRequest(msg)

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

	result := messagesToRenderChatRequest(msg)

	require.Len(t, result.Conversation, 2)
	assert.Equal(t, &tokenizerTypes.Content{Raw: "Real system prompt."}, result.Conversation[0].Content)
}

func TestLegacyMessagesToRenderChatRequest_NoSystem(t *testing.T) {
	msg := &fwkrh.MessagesRequest{
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Hi"}}},
	}

	result := messagesToRenderChatRequest(msg)

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
			result := messagesToRenderChatRequest(msg)
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
	p.backend = renderBackend{tk: tok, modelName: "configured-model", legacyMessages: &legacyMessagesMode{mode: messagesRenderModeLegacy}}

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
	wire, err := pm.Marshal()
	require.NoError(t, err)
	require.JSONEq(t, `{"model":"configured-model","messages":[{"role":"system","content":"Be helpful."},{"role":"user","content":"Hi"}]}`, string(wire))
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
			got, err := pythonDumps(json.RawMessage(tt.in))
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestLegacyPythonDumpsNonASCIIEscaped(t *testing.T) {
	got, err := pythonDumps(json.RawMessage("{\"e\":\"Z\u00fcrich \U0001f600\"}"))
	require.NoError(t, err)
	assert.Equal(t, `{"e": "Z\u00fcrich \ud83d\ude00"}`, got)
}

func TestLegacyPythonArguments(t *testing.T) {
	assert.Equal(t, "{}", pythonArguments(nil))
	assert.Equal(t, "{}", pythonArguments(json.RawMessage(`null`)))
	assert.Equal(t, "{}", pythonArguments(json.RawMessage(`{}`)))
	assert.Equal(t, `{"a": 1}`, pythonArguments(json.RawMessage(`{"a":1}`)))
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

	result := messagesToRenderChatRequest(msg)

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

	result := messagesToRenderChatRequest(msg)

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

	result := messagesToRenderChatRequest(msg)

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

	result := messagesToRenderChatRequest(msg)

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

	result := messagesToRenderChatRequest(msg)

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
	p.backend = renderBackend{tk: tok, modelName: "configured-model", legacyMessages: &legacyMessagesMode{mode: messagesRenderModeLegacy}}

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
