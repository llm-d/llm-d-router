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
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

func TestNativeRenderPreservesRequest(t *testing.T) {
	for _, tt := range []struct {
		name, path, body string
		parser           fwkrh.Parser
	}{
		{"chat", "/v1/chat/completions", ` {"model":"adapter","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"z":9007199254740993,"a":1e0}}}],"extra":{"z":1,"a":2}} `, openai.NewOpenAIParser()},
		{"messages", "/v1/messages", ` {"model":"adapter","max_tokens":10,"system":[{"type":"text","text":"x-anthropic-billing-header: unchanged"}],"messages":[{"role":"user","content":[{"type":"future","payload":{"z":1,"a":2}}]}],"tools":[{"name":"f","input_schema":{"properties":{"z":{},"a":{}}}}],"output_config":{"effort":"high"},"chat_template_kwargs":{"enable_thinking":false}} `, anthropic.NewAnthropicParser()},
		{"agentic messages", "/v1/messages", `{"model":"adapter","max_tokens":8,"system":"x-anthropic-billing-header: keep","messages":[{"role":"user","content":"run"},{"role":"system","content":"inline"},{"role":"assistant","content":[{"type":"thinking","thinking":"reason","signature":"sig"},{"type":"tool_use","id":"call_1","name":"run","input":{"z":9007199254740993,"a":1e0}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"ok"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}],"tools":[{"name":"run","input_schema":{"properties":{"z":{"type":"number"},"a":{"type":"number"}}}}],"output_config":{"effort":"high"},"chat_template_kwargs":{"z":1,"enable_thinking":false}}`, anthropic.NewAnthropicParser()},
		{"completions", "/v1/completions", ` {"model":"adapter","prompt":"hi","extra":{"z":9007199254740993,"a":1e0}} `, openai.NewOpenAIParser()},
		{"completion tokens", "/v1/completions", ` {"model":"adapter","prompt":[1,2,3,4],"truncate_prompt_tokens":2,"add_special_tokens":false} `, openai.NewOpenAIParser()},
		{"nested completion tokens", "/v1/completions", ` {"model":"adapter","prompt":[[1,2],[3,4]],"max_tokens":4,"truncate_prompt_tokens":-1} `, openai.NewOpenAIParser()},
		{"one string array", "/v1/completions", ` {"model":"adapter","prompt":["hello"]} `, openai.NewOpenAIParser()},
		{"string array", "/v1/completions", ` {"model":"adapter","prompt":["hello","world"]} `, openai.NewOpenAIParser()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var got []byte
			var path string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, http.MethodPost, r.Method)
				require.Equal(t, "application/json", r.Header.Get("Content-Type"))
				require.Equal(t, "Bearer native-render", r.Header.Get("Authorization"))
				path = r.URL.Path
				got, _ = io.ReadAll(r.Body)
				if tt.path == "/v1/completions" {
					_, _ = io.WriteString(w, `[{"token_ids":[3,4]}]`)
				} else {
					_, _ = io.WriteString(w, `{"token_ids":[3,4]}`)
				}
			}))
			defer srv.Close()
			parsed, err := tt.parser.ParseRequest(context.Background(), []byte(tt.body), map[string]string{":path": tt.path})
			require.NoError(t, err)
			p := newTestPlugin(newHTTPRenderer(t, srv))
			req := &scheduling.InferenceRequest{Body: parsed.Body, TargetModel: "adapter", Headers: map[string]string{"authorization": "Bearer native-render"}}
			require.NoError(t, p.Produce(context.Background(), req, nil))
			require.Equal(t, tt.path+"/render", path)
			require.Equal(t, tt.body, string(got))
			require.Equal(t, [][]uint32{{3, 4}}, [][]uint32{req.Body.TokenizedRequest.Prompts[0].TokenIDs})

			unrewritten, err := tt.parser.ParseRequest(context.Background(), []byte(tt.body), map[string]string{":path": tt.path})
			require.NoError(t, err)
			unrewritten.Body.MutatePayloadMap(func(payload fwkrh.PayloadMap) {
				payload["vllm_xargs"] = map[string]any{"kv_cache_report_mode": "full"}
			})
			lateBody, err := unrewritten.Body.WirePayload().(fwkrh.Marshaler).Marshal()
			require.NoError(t, err)
			assertNativeFieldsUnchanged(t, []byte(tt.body), lateBody)

			// A resolved model and late routing metadata must not reorder content.
			parsed.Body.TokenizedRequest = nil
			rewriter := tt.parser.(fwkrh.ModelNameRewriter)
			parsed.Body.Payload, err = rewriter.RewriteModelName(parsed.Body.Payload.(fwkrh.MarshalablePayload), "resolved-adapter")
			require.NoError(t, err)
			parsed.Body.Mutated = true
			require.NoError(t, p.Produce(context.Background(), req, nil))
			var rendered map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(got, &rendered))
			require.Equal(t, `"resolved-adapter"`, string(rendered["model"]))
			assertNativeFieldsUnchanged(t, []byte(tt.body), got, "model")

			parsed.Body.MutatePayloadMap(func(payload fwkrh.PayloadMap) {
				payload["vllm_xargs"] = map[string]any{"kv_cache_report_mode": "full"}
			})
			forwarded, err := parsed.Body.WirePayload().(fwkrh.Marshaler).Marshal()
			require.NoError(t, err)
			assertNativeFieldsUnchanged(t, got, forwarded)
		})
	}
}

func assertNativeFieldsUnchanged(t *testing.T, before, after []byte, except ...string) {
	t.Helper()
	var original, final map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(before, &original))
	require.NoError(t, json.Unmarshal(after, &final))
	for _, key := range except {
		delete(original, key)
	}
	for key, value := range original {
		var want, got bytes.Buffer
		require.NoError(t, json.Compact(&want, value))
		require.NoError(t, json.Compact(&got, final[key]))
		require.Equal(t, want.String(), got.String(), "field %s", key)
	}
}

func TestNativeRenderFailureDoesNotConvertOrEstimate(t *testing.T) {
	for _, tt := range []struct {
		name     string
		status   int
		response string
	}{
		{"missing native messages endpoint", http.StatusNotFound, "not found"},
		{"invalid response", http.StatusOK, "not JSON"},
		{"empty tokens", http.StatusOK, `{"token_ids":[]}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				require.Equal(t, "/v1/messages/render", r.URL.Path)
				w.WriteHeader(tt.status)
				_, _ = io.WriteString(w, tt.response)
			}))
			defer srv.Close()
			parsed, err := anthropic.NewAnthropicParser().ParseRequest(context.Background(), []byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`), map[string]string{":path": "/v1/messages"})
			require.NoError(t, err)
			req := &scheduling.InferenceRequest{Body: parsed.Body}
			err = newTestPlugin(newHTTPRenderer(t, srv)).Produce(context.Background(), req, nil)
			if tt.name == "empty tokens" {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
			}
			require.Nil(t, req.Body.TokenizedRequest)
			require.Equal(t, 1, calls)
		})
	}
}

func TestDirectRenderRequestsPassThrough(t *testing.T) {
	for _, tt := range []struct {
		path   string
		parser fwkrh.Parser
	}{
		{"/v1/completions/render", openai.NewOpenAIParser()},
		{"/v1/chat/completions/render", openai.NewOpenAIParser()},
		{"/v1/messages/render", anthropic.NewAnthropicParser()},
	} {
		t.Run(tt.path, func(t *testing.T) {
			// Validation belongs to the destination endpoint.
			raw := []byte(`{"unknown":{"z":1,"a":2}}`)
			parsed, err := tt.parser.ParseRequest(context.Background(), raw, map[string]string{":path": tt.path})
			require.NoError(t, err)
			require.True(t, parsed.SkipResponseProcessing)
			require.Equal(t, fwkrh.RawPayload(raw), parsed.Body.Payload)
			req := &scheduling.InferenceRequest{Body: parsed.Body}
			require.NoError(t, newTestPlugin(&mockTokenizer{}).Produce(context.Background(), req, nil))
			require.Nil(t, req.Body.TokenizedRequest)
		})
	}
}

func TestNativeRenderDoesNotReconstructNonHTTPInput(t *testing.T) {
	srv, _ := httpFixture(t, nil, renderResponse{})
	defer srv.Close()
	for _, body := range []*fwkrh.InferenceRequestBody{
		{ChatCompletions: &fwkrh.ChatCompletionsRequest{}},
		{Messages: &fwkrh.MessagesRequest{}},
		{Completions: &fwkrh.CompletionsRequest{}},
	} {
		err := newTestPlugin(newHTTPRenderer(t, srv)).Produce(context.Background(), &scheduling.InferenceRequest{Body: body}, nil)
		require.ErrorContains(t, err, "requires an HTTP JSON payload")
		require.Nil(t, body.TokenizedRequest)
	}
}
