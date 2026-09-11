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
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/stretchr/testify/require"
	"google.golang.org/genproto/googleapis/api/httpbody"
	"google.golang.org/protobuf/proto"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/sglanghttp"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vertexai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc"
	pb "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmhttp"
)

type liveRenderEndpoint struct {
	url, model, alias, auth string
}

func renderEndpointForTest(t *testing.T) liveRenderEndpoint {
	t.Helper()
	e := liveRenderEndpoint{
		url:   strings.TrimRight(os.Getenv("VLLM_RENDER_TEST_URL"), "/"),
		model: os.Getenv("VLLM_RENDER_TEST_MODEL"),
		alias: os.Getenv("VLLM_RENDER_TEST_ALIAS"),
		auth:  os.Getenv("VLLM_RENDER_TEST_AUTHORIZATION"),
	}
	if e.url == "" {
		t.Skip("set VLLM_RENDER_TEST_URL to run real-renderer checks")
	}
	require.NotEmpty(t, e.model, "set VLLM_RENDER_TEST_MODEL")
	require.NotEmpty(t, e.alias, "set VLLM_RENDER_TEST_ALIAS to another accepted model name")
	return e
}

func (e liveRenderEndpoint) post(t *testing.T, path string, raw []byte) []byte {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, e.url+path, bytes.NewReader(raw))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", e.auth)
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode, "%s", body)
	return body
}

func (e liveRenderEndpoint) render(t *testing.T, path string, raw []byte) []fwkrh.PromptTokens {
	t.Helper()
	body := e.post(t, path, raw)
	var rendered []renderResponse
	if path == completionsRenderPath {
		require.NoError(t, json.Unmarshal(body, &rendered))
	} else {
		var result renderResponse
		require.NoError(t, json.Unmarshal(body, &result))
		rendered = []renderResponse{result}
	}
	require.NotEmpty(t, rendered)
	prompts := make([]fwkrh.PromptTokens, len(rendered))
	for i, result := range rendered {
		require.NotEmpty(t, result.TokenIDs)
		prompts[i] = fwkrh.PromptTokens{TokenIDs: result.TokenIDs, MultiModalFeatures: convertMMFeaturesToUpstream(toKVCacheMM(result.Features))}
	}
	return prompts
}

func TestRenderServingLive(t *testing.T) {
	e := renderEndpointForTest(t)
	if os.Getenv("VLLM_RENDER_TEST_SERVE") != "1" {
		t.Skip("set VLLM_RENDER_TEST_SERVE=1 to issue bounded inference requests")
	}
	for _, tc := range []struct{ name, path, fields string }{
		{"chat/text", "/v1/chat/completions", `"messages":[{"role":"user","content":"Say hi."}]`},
		{"chat/tools", "/v1/chat/completions", `"messages":[{"role":"user","content":"Say hi."}],"tools":[{"type":"function","function":{"name":"greet","parameters":{"type":"object","properties":{"z":{"type":"number"},"a":{"type":"string"}}}}}]`},
		{"completions/text", "/v1/completions", `"prompt":"Say hi."`},
		{"completions/text-batch", "/v1/completions", `"prompt":["Say hi.","Say bye."]`},
		{"completions/token-batch", "/v1/completions", `"prompt":[[1,2,3,4],[5,6,7,8]],"truncate_prompt_tokens":2,"add_special_tokens":false`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(fmt.Sprintf(`{"model":%q,"max_tokens":1,"temperature":0,"return_token_ids":true,%s}`, e.model, tc.fields))
			parsed, err := openai.NewOpenAIParser().ParseRequest(t.Context(), raw, map[string]string{":path": tc.path})
			require.NoError(t, err)
			target, err := url.Parse(e.url)
			require.NoError(t, err)
			p := e.plugin(t, httputil.NewSingleHostReverseProxy(target))
			req := &scheduling.InferenceRequest{Body: parsed.Body, Headers: map[string]string{"authorization": e.auth}}
			require.NoError(t, p.Produce(t.Context(), req, nil))
			require.Equal(t, e.render(t, tc.path+"/render", raw), req.Body.TokenizedRequest.Prompts)
			var served struct {
				PromptTokenIDs []uint32 `json:"prompt_token_ids"`
				Choices        []struct {
					Index          int      `json:"index"`
					PromptTokenIDs []uint32 `json:"prompt_token_ids"`
				} `json:"choices"`
			}
			require.NoError(t, json.Unmarshal(e.post(t, tc.path, wireBytes(t, parsed.Body.WirePayload())), &served))
			if tc.path == "/v1/chat/completions" {
				require.Equal(t, req.Body.TokenizedRequest.Prompts[0].TokenIDs, served.PromptTokenIDs)
			} else {
				require.Len(t, served.Choices, len(req.Body.TokenizedRequest.Prompts))
				for _, choice := range served.Choices {
					require.Equal(t, req.Body.TokenizedRequest.Prompts[choice.Index].TokenIDs, choice.PromptTokenIDs)
				}
			}
		})
	}
	t.Run("vllm-generate", func(t *testing.T) {
		rendered := e.render(t, completionsRenderPath, []byte(fmt.Sprintf(`{"model":%q,"prompt":"Say hi.","add_special_tokens":false}`, e.model)))
		tokens, err := json.Marshal(rendered[0].TokenIDs)
		require.NoError(t, err)
		raw := []byte(fmt.Sprintf(`{"model":%q,"token_ids":%s,"sampling_params":{"max_tokens":1,"temperature":0}}`, e.model, tokens))
		parsed, err := vllmhttp.NewVllmHTTPParser().ParseRequest(t.Context(), raw, map[string]string{":path": "/inference/v1/generate"})
		require.NoError(t, err)
		before := wireBytes(t, parsed.Body.WirePayload())
		p := e.plugin(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { t.Error("Generate must not render"); w.WriteHeader(500) }))
		req := &scheduling.InferenceRequest{Body: parsed.Body}
		require.NoError(t, p.Produce(t.Context(), req, nil))
		require.Equal(t, rendered, req.Body.TokenizedRequest.Prompts)
		wire := wireBytes(t, parsed.Body.WirePayload())
		require.Equal(t, before, wire)
		require.JSONEq(t, string(raw), string(wire))
		var served struct {
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
			} `json:"usage"`
		}
		require.NoError(t, json.Unmarshal(e.post(t, "/inference/v1/generate", wire), &served))
		require.Equal(t, len(rendered[0].TokenIDs), served.Usage.PromptTokens)
	})
}

func (e liveRenderEndpoint) plugin(t *testing.T, handler http.Handler) *Plugin {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	p, err := NewPlugin(ctx, "live", &tokenizerPluginConfig{ModelName: e.model, VLLM: &vllmConfig{URL: srv.URL}})
	require.NoError(t, err)
	return p
}

func TestNativeRenderLive(t *testing.T) {
	e := renderEndpointForTest(t)
	const rewritten = "model-rewrite"
	for _, tc := range []struct{ name, path, fields string }{
		{"chat/text", "/v1/chat/completions", `"messages":[{"role":"user","content":"hi"}]`},
		{"chat/system", "/v1/chat/completions", `"messages":[{"role":"system","content":"Be brief"},{"role":"user","content":"hi"}]`},
		{"chat/multiturn", "/v1/chat/completions", `"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"continue"}]`},
		{"chat/tools", "/v1/chat/completions", `"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{"z":{"type":"number"},"a":{"type":"string"}}}}}],"tool_choice":"auto"`},
		{"chat/tool-history", "/v1/chat/completions", `"messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"run","arguments":"{\"z\":1,\"a\":\"x\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"done"}]`},
		{"chat/template-controls", "/v1/chat/completions", `"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false},"stream":true`},
		{"completions/text", "/v1/completions", `"prompt":"hello world"`},
		{"completions/batch", "/v1/completions", `"prompt":["hello world","second prompt"]`},
		{"completions/tokens", "/v1/completions", `"prompt":[1,2,3,4],"add_special_tokens":false`},
		{"completions/token-batch-truncation", "/v1/completions", `"prompt":[[1,2,3,4],[5,6,7,8]],"truncate_prompt_tokens":2,"add_special_tokens":false`},
		{"completions/text-truncation", "/v1/completions", `"prompt":"one two three four five six seven eight","truncate_prompt_tokens":3`},
		{"messages/text", "/v1/messages", `"messages":[{"role":"user","content":"hi"}]`},
		{"messages/system", "/v1/messages", `"system":[{"type":"text","text":"Be brief"}],"messages":[{"role":"user","content":"hi"}]`},
		{"messages/inline-system", "/v1/messages", `"messages":[{"role":"system","content":"Be brief"},{"role":"user","content":"hi"}]`},
		{"messages/tools", "/v1/messages", `"messages":[{"role":"user","content":"run"}],"tools":[{"name":"run","input_schema":{"type":"object","properties":{"z":{"type":"number"},"a":{"type":"string"}}}}]`},
		{"messages/tool-history", "/v1/messages", `"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"thinking","thinking":"use the tool","signature":"sig"},{"type":"tool_use","id":"call_1","name":"run","input":{"z":1,"a":"x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"done"}]}]`},
		{"messages/effort-and-template", "/v1/messages", `"messages":[{"role":"user","content":"hi"}],"output_config":{"effort":"high"},"chat_template_kwargs":{"enable_thinking":false}`},
		{"messages/empty-text", "/v1/messages", `"messages":[{"role":"user","content":""}]`},
	} {
		for _, variant := range []string{"original", rewritten, "direct-render"} {
			t.Run(tc.name+"/"+variant, func(t *testing.T) {
				model := e.model
				if variant == rewritten {
					model = e.alias
				}
				body := []byte(fmt.Sprintf(` {"model":%q,"max_tokens":8,"cache_salt":"live-test",%s} `, e.model, tc.fields))
				reference := []byte(fmt.Sprintf(` {"model":%q,"max_tokens":8,"cache_salt":"live-test",%s} `, model, tc.fields))
				path := tc.path
				if variant == "direct-render" {
					path += "/render"
				}
				var parser fwkrh.Parser = openai.NewOpenAIParser()
				if tc.path == "/v1/messages" {
					parser = anthropic.NewAnthropicParser()
				}
				parsed, err := parser.ParseRequest(t.Context(), body, map[string]string{":path": path})
				require.NoError(t, err)
				if variant == rewritten {
					parsed.Body.Payload, err = parser.(fwkrh.ModelNameRewriter).RewriteModelName(parsed.Body.Payload.(fwkrh.MarshalablePayload), model)
					require.NoError(t, err)
					parsed.Body.Mutated = true
				}
				want := e.render(t, tc.path+"/render", reference)
				target, err := url.Parse(e.url)
				require.NoError(t, err)
				proxy := httputil.NewSingleHostReverseProxy(target)
				p := e.plugin(t, proxy)
				req := &scheduling.InferenceRequest{Body: parsed.Body, Headers: map[string]string{"authorization": e.auth}}
				require.NoError(t, p.Produce(t.Context(), req, nil))
				if variant == "direct-render" {
					require.Nil(t, req.Body.TokenizedRequest)
				} else {
					require.Equal(t, want, req.Body.TokenizedRequest.Prompts)
					require.Equal(t, "live-test", req.Body.TokenizedRequest.CacheSalt)
				}
				if variant == rewritten {
					parsed.Body.MutatePayloadMap(func(payload fwkrh.PayloadMap) { payload["vllm_xargs"] = map[string]any{"kv_cache_report_mode": "full"} })
				}
				wire := wireBytes(t, parsed.Body.WirePayload())
				if variant == rewritten {
					assertNativeFieldsUnchanged(t, reference, wire)
				} else {
					require.Equal(t, body, wire)
				}
				require.Equal(t, want, e.render(t, tc.path+"/render", wire))
			})
		}
	}
}

func wireBytes(t *testing.T, payload fwkrh.RequestPayload) []byte {
	t.Helper()
	if raw, ok := payload.(fwkrh.RawPayload); ok {
		return raw
	}
	raw, err := payload.(fwkrh.Marshaler).Marshal()
	require.NoError(t, err)
	return raw
}

func liveGRPCFrame(t *testing.T, msg proto.Message) []byte {
	t.Helper()
	raw, err := proto.Marshal(msg)
	require.NoError(t, err)
	frame := make([]byte, 5+len(raw))
	binary.BigEndian.PutUint32(frame[1:5], uint32(len(raw)))
	copy(frame[5:], raw)
	return frame
}

func TestRenderLiveProtocolHandoff(t *testing.T) {
	e := renderEndpointForTest(t)
	raw := []byte(fmt.Sprintf(`{"model":%q,"prompt":"hello world","add_special_tokens":false}`, e.model))
	want := e.render(t, completionsRenderPath, raw)
	tokenJSON, err := json.Marshal(want[0].TokenIDs)
	require.NoError(t, err)
	for _, tc := range []struct {
		name, path string
		parser     fwkrh.Parser
		body       []byte
	}{
		{"sglang", "/generate", sglanghttp.NewSGLangHTTPParser(), []byte(fmt.Sprintf(`{"input_ids":%s,"extra_key":"live-test","sampling_params":{"max_new_tokens":1}}`, tokenJSON))},
		{"vllm-http", "/inference/v1/generate", vllmhttp.NewVllmHTTPParser(), []byte(fmt.Sprintf(`{"model":%q,"token_ids":%s,"cache_salt":"live-test","sampling_params":{"max_tokens":1}}`, e.model, tokenJSON))},
		{"vllm-grpc", "/vllm.grpc.engine.VllmEngine/Generate", vllmgrpc.NewVllmGRPCParser(), liveGRPCFrame(t, &pb.GenerateRequest{Input: &pb.GenerateRequest_Tokenized{Tokenized: &pb.TokenizedInput{InputIds: want[0].TokenIDs}}})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parsed, err := tc.parser.ParseRequest(t.Context(), tc.body, map[string]string{":path": tc.path})
			require.NoError(t, err)
			p := e.plugin(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				t.Error("supplied tokens must not call the renderer")
				w.WriteHeader(500)
			}))
			req := &scheduling.InferenceRequest{Body: parsed.Body}
			require.NoError(t, p.Produce(t.Context(), req, nil))
			require.Equal(t, want, req.Body.TokenizedRequest.Prompts)
			require.False(t, req.Body.Mutated)
		})
	}
	for _, vertex := range []bool{false, true} {
		t.Run(fmt.Sprintf("grpc-text/vertex=%t", vertex), func(t *testing.T) {
			var parser fwkrh.Parser = vllmgrpc.NewVllmGRPCParser()
			path := "/vllm.grpc.engine.VllmEngine/Generate"
			body := liveGRPCFrame(t, &pb.GenerateRequest{Input: &pb.GenerateRequest_Text{Text: "hello world"}})
			reference := []byte(fmt.Sprintf(`{"model":%q,"prompt":"hello world"}`, e.model))
			renderPath := completionsRenderPath
			if vertex {
				parser = vertexai.NewVertexAIParser()
				path = "/google.cloud.aiplatform.v1beta1.PredictionService/ChatCompletions"
				reference = []byte(fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hello world"}]}`, e.model))
				body = liveGRPCFrame(t, &aiplatformpb.ChatCompletionsRequest{HttpBody: &httpbody.HttpBody{Data: reference}})
				renderPath = chatRenderPath
			}
			parsed, err := parser.ParseRequest(t.Context(), body, map[string]string{":path": path})
			require.NoError(t, err)
			target, err := url.Parse(e.url)
			require.NoError(t, err)
			p := e.plugin(t, httputil.NewSingleHostReverseProxy(target))
			req := &scheduling.InferenceRequest{Body: parsed.Body, Headers: map[string]string{"authorization": e.auth}}
			require.NoError(t, p.Produce(t.Context(), req, nil))
			require.Equal(t, e.render(t, renderPath, reference), req.Body.TokenizedRequest.Prompts)
			require.False(t, parsed.Body.Mutated)
		})
	}
}
