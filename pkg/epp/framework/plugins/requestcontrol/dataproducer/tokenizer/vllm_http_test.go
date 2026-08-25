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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
	"github.com/llm-d/llm-d-router/test/utils"
)

const testHTTPModel = "test-model"

func newHTTPRenderer(t *testing.T, srv *httptest.Server) *vllmHTTPRenderer {
	t.Helper()
	r, err := newVLLMHTTPRenderer(&vllmConfig{URL: srv.URL}, testHTTPModel)
	require.NoError(t, err)
	return r
}

// httpFixture mimics vLLM's /render endpoints and captures request bodies.
func httpFixture(t *testing.T, completionsResp []renderResponse, chatResp renderResponse) (*httptest.Server, *httpCaptured) {
	t.Helper()
	cap := &httpCaptured{}
	mux := http.NewServeMux()
	mux.HandleFunc(completionsRenderPath, func(w http.ResponseWriter, r *http.Request) {
		cap.completions, _ = io.ReadAll(r.Body)
		_ = json.NewEncoder(w).Encode(completionsResp)
	})
	mux.HandleFunc(chatRenderPath, func(w http.ResponseWriter, r *http.Request) {
		cap.chat, _ = io.ReadAll(r.Body)
		_ = json.NewEncoder(w).Encode(chatResp)
	})
	return httptest.NewServer(mux), cap
}

type httpCaptured struct{ completions, chat []byte }

func TestVLLMHTTPRenderer_Render(t *testing.T) {
	srv, cap := httpFixture(t,
		[]renderResponse{{TokenIDs: []uint32{1, 2, 3}}}, renderResponse{})
	defer srv.Close()

	r := newHTTPRenderer(t, srv)
	allTokenIDs, offsets, err := r.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
	require.NoError(t, err)
	assert.Equal(t, [][]uint32{{1, 2, 3}}, allTokenIDs)
	assert.Nil(t, offsets)

	var sent map[string]any
	require.NoError(t, json.Unmarshal(cap.completions, &sent))
	assert.Equal(t, testHTTPModel, sent["model"])
	assert.Equal(t, "hello", sent["prompt"])
}

func TestProduce_CompletionsVLLMHTTPUsesRawPayload(t *testing.T) {
	srv, cap := httpFixture(t,
		[]renderResponse{{TokenIDs: []uint32{4, 5}}}, renderResponse{})
	defer srv.Close()

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "hello"},
			},
			Payload: fwkrh.PayloadMap{
				"prompt":      "hello",
				"dummy_field": "kept",
			},
		},
	}

	p := newTestPlugin(newHTTPRenderer(t, srv))
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []uint32{4, 5}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)

	var sent map[string]any
	require.NoError(t, json.Unmarshal(cap.completions, &sent))
	assert.Equal(t, "kept", sent["dummy_field"])
	assert.Equal(t, testHTTPModel, sent["model"])
}

// TestVLLMHTTPRenderer_RenderChat_Multimodal covers the chat endpoint: the raw
// payload is forwarded directly and multimodal features are converted from wire
// format to kvcache map shape.
func TestVLLMHTTPRenderer_RenderChat_Multimodal(t *testing.T) {
	srv, cap := httpFixture(t, nil, renderResponse{
		TokenIDs: []uint32{1, 2, 3, 4, 5},
		Features: &renderMMFeatures{
			MMHashes: map[string][]string{"image": {"abc123"}},
			MMPlaceholders: map[string][]renderPlaceholder{
				"image": {{Offset: 2, Length: 3}},
			},
		},
	})
	defer srv.Close()

	r := newHTTPRenderer(t, srv)
	payload := fwkrh.PayloadMap{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,xx"}},
					map[string]any{"type": "text", "text": "describe"},
				},
			},
		},
		"add_generation_prompt": true,
	}
	tokenIDs, mm, err := r.RenderChat(context.Background(), payload)
	require.NoError(t, err)
	assert.Equal(t, []uint32{1, 2, 3, 4, 5}, tokenIDs)
	require.NotNil(t, mm)
	assert.Equal(t, []string{"abc123"}, mm.MMHashes["image"])
	require.Len(t, mm.MMPlaceholders["image"], 1)
	assert.Equal(t, 2, mm.MMPlaceholders["image"][0].Offset)
	assert.Equal(t, 3, mm.MMPlaceholders["image"][0].Length)

	var sent map[string]any
	require.NoError(t, json.Unmarshal(cap.chat, &sent))
	assert.Equal(t, testHTTPModel, sent["model"])
	assert.Equal(t, true, sent["add_generation_prompt"])
	msgs, ok := sent["messages"].([]any)
	require.True(t, ok)
	require.Len(t, msgs, 1)
	parts, ok := msgs[0].(map[string]any)["content"].([]any)
	require.True(t, ok, "structured content must be forwarded as an array of parts")
	require.Len(t, parts, 2)
	assert.Equal(t, "image_url", parts[0].(map[string]any)["type"])
	assert.Equal(t, "text", parts[1].(map[string]any)["type"])
}

func TestVLLMHTTPRenderer_RenderChat_ForwardsMMContentBlocks(t *testing.T) {
	srv, cap := httpFixture(t, nil, renderResponse{TokenIDs: []uint32{6, 7}})
	defer srv.Close()

	payload := fwkrh.PayloadMap{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":        "input_audio",
						"input_audio": map[string]any{"data": "AAAA", "format": "wav"},
					},
					map[string]any{
						"type":      "video_url",
						"video_url": map[string]any{"url": "https://example.test/video.mp4"},
					},
				},
			},
		},
	}

	r := newHTTPRenderer(t, srv)
	tokenIDs, _, err := r.RenderChat(context.Background(), payload)
	require.NoError(t, err)
	assert.Equal(t, []uint32{6, 7}, tokenIDs)

	var sent map[string]any
	require.NoError(t, json.Unmarshal(cap.chat, &sent))
	msgs, ok := sent["messages"].([]any)
	require.True(t, ok)
	require.Len(t, msgs, 1)
	parts, ok := msgs[0].(map[string]any)["content"].([]any)
	require.True(t, ok)
	require.Len(t, parts, 2)

	audio, ok := parts[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "input_audio", audio["type"])
	inputAudio, ok := audio["input_audio"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "wav", inputAudio["format"])

	video, ok := parts[1].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "video_url", video["type"])
	videoURL, ok := video["video_url"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "https://example.test/video.mp4", videoURL["url"])
}

func TestProduce_ChatCompletionsVLLMHTTPUsesRawPayload(t *testing.T) {
	srv, cap := httpFixture(t, nil, renderResponse{TokenIDs: []uint32{9, 10}})
	defer srv.Close()

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hi"}}},
			},
			Payload: fwkrh.PayloadMap{
				"messages": []any{map[string]any{"role": "user", "content": "hi"}},
				"model":    "caller-supplied-model",
				"dummy":    "kept",
				"reasoning": map[string]any{
					"effort": "high",
				},
			},
		},
	}

	p := newTestPlugin(newHTTPRenderer(t, srv))
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []uint32{9, 10}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)

	var sent map[string]any
	require.NoError(t, json.Unmarshal(cap.chat, &sent))
	assert.Equal(t, "kept", sent["dummy"])
	assert.Equal(t, map[string]any{"effort": "high"}, sent["reasoning"])
	assert.Equal(t, testHTTPModel, sent["model"])
}

// TestProduce_MessagesVLLMHTTPFullAgenticTurn covers the complete production
// path from a parsed Anthropic request through the rebuilt OpenAI-shaped
// payload sent to vLLM's chat render endpoint.
func TestProduce_MessagesVLLMHTTPFullAgenticTurn(t *testing.T) {
	srv, captured := httpFixture(t, nil, renderResponse{TokenIDs: []uint32{11, 12, 13}})
	defer srv.Close()

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Messages: &fwkrh.MessagesRequest{
				System: fwkrh.AnthropicContent{Raw: "You can use tools."},
				Tools: []fwkrh.AnthropicTool{{
					Name:        "get_weather",
					Description: "Get the weather",
					InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
				}},
				Messages: []fwkrh.AnthropicMessage{
					{Role: "user", Content: fwkrh.AnthropicContent{Raw: "Weather in Zurich?"}},
					{Role: "assistant", Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
						{Type: "thinking", Thinking: "I should check."},
						{Type: "tool_use", ID: "toolu_01", Name: "get_weather", Input: json.RawMessage(`{"city":"Zurich"}`)},
					}}},
					{Role: "user", Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
						{Type: "tool_result", ToolUseID: "toolu_01", Content: fwkrh.AnthropicContent{Raw: "Sunny, 22C"}},
					}}},
				},
			},
		},
	}

	p := newTestPlugin(newHTTPRenderer(t, srv))
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.NotNil(t, req.Body.TokenizedRequest)
	assert.Equal(t, []uint32{11, 12, 13}, req.Body.TokenizedRequest.Prompts[0].TokenIDs)

	var sent struct {
		Model    string           `json:"model"`
		Messages []map[string]any `json:"messages"`
		Tools    []map[string]any `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(captured.chat, &sent))
	assert.Equal(t, testHTTPModel, sent.Model)
	require.Len(t, sent.Tools, 1)
	assert.Equal(t, "function", sent.Tools[0]["type"])
	function, ok := sent.Tools[0]["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "get_weather", function["name"])
	parameters, ok := function["parameters"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "object", parameters["type"])

	require.Len(t, sent.Messages, 4)
	assert.Equal(t, "system", sent.Messages[0]["role"])
	assert.Equal(t, "user", sent.Messages[1]["role"])
	assistant := sent.Messages[2]
	assert.Equal(t, "assistant", assistant["role"])
	assert.NotContains(t, assistant, "content")
	assert.Equal(t, "I should check.", assistant["reasoning"])
	toolCalls, ok := assistant["tool_calls"].([]any)
	require.True(t, ok)
	require.Len(t, toolCalls, 1)
	toolCall, ok := toolCalls[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "toolu_01", toolCall["id"])
	toolFunction, ok := toolCall["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, `{"city": "Zurich"}`, toolFunction["arguments"])

	toolResult := sent.Messages[3]
	assert.Equal(t, "tool", toolResult["role"])
	assert.Equal(t, "toolu_01", toolResult["tool_call_id"])
	assert.Equal(t, "Sunny, 22C", toolResult["content"])
}

func TestVLLMHTTPRenderer_RenderMultiPrompt(t *testing.T) {
	srv, _ := httpFixture(t,
		[]renderResponse{
			{TokenIDs: []uint32{1, 2, 3}},
			{TokenIDs: []uint32{4, 5}},
		}, renderResponse{})
	defer srv.Close()

	r := newHTTPRenderer(t, srv)
	allTokenIDs, offsets, err := r.Render(context.Background(), fwkrh.PayloadMap{"prompt": []string{"alpha", "beta"}})
	require.NoError(t, err)
	assert.Equal(t, [][]uint32{{1, 2, 3}, {4, 5}}, allTokenIDs)
	assert.Nil(t, offsets)
}

func TestVLLMHTTPRenderer_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	r := newHTTPRenderer(t, srv)
	_, _, err := r.Render(context.Background(), fwkrh.PayloadMap{"prompt": "x"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "500")
}

func TestPluginFactory_RejectsBothBackends(t *testing.T) {
	params := `{
		"modelName": "m",
		"udsTokenizerConfig": {"socketFile": "/tmp/foo.sock"},
		"vllm": {"url": "http://localhost:8000"}
	}`
	handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
	p, err := PluginFactory("test", plugin.StrictDecoder(json.RawMessage(params)), handle)
	require.Error(t, err)
	assert.Nil(t, p)
	assert.Contains(t, err.Error(), "only one of")
}

func TestPluginFactory_HTTPBackend_BadTimeout(t *testing.T) {
	params := `{
		"modelName": "m",
		"vllm": {"timeout": "nope"}
	}`
	handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
	p, err := PluginFactory("test", plugin.StrictDecoder(json.RawMessage(params)), handle)
	require.Error(t, err)
	assert.Nil(t, p)
	assert.Contains(t, err.Error(), "invalid 'timeout'")
}

// The render transport must inject W3C trace context so the vLLM pod shares
// the same trace id as the EPP request.
func TestVLLMHTTPRenderer_RenderPropagatesTraceContext(t *testing.T) {
	prevProp := otel.GetTextMapPropagator()
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	t.Cleanup(func() { otel.SetTextMapPropagator(prevProp) })

	var gotTraceparent string
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotTraceparent = r.Header.Get("traceparent")
		close(done)
		_ = json.NewEncoder(w).Encode([]renderResponse{{TokenIDs: []uint32{1}}})
	}))
	defer srv.Close()

	r := newHTTPRenderer(t, srv)

	traceID, err := trace.TraceIDFromHex("0123456789abcdef0123456789abcdef")
	require.NoError(t, err)
	spanID, err := trace.SpanIDFromHex("0123456789abcdef")
	require.NoError(t, err)
	sc := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: trace.FlagsSampled,
		Remote:     true,
	})
	ctx := trace.ContextWithSpanContext(context.Background(), sc)

	_, _, err = r.Render(ctx, fwkrh.PayloadMap{"prompt": "hello"})
	require.NoError(t, err)
	<-done

	if gotTraceparent == "" {
		t.Fatal("expected traceparent header to be injected into outbound render request, got none")
	}
	if !strings.Contains(gotTraceparent, traceID.String()) {
		t.Fatalf("expected outbound traceparent to carry trace ID %s, got %q", traceID, gotTraceparent)
	}
}

// TestVLLMHTTPRenderer_ChatTimeoutRawMessages asserts that pre-marshaled
// (Anthropic-rebuilt) messages with array content still select the multimodal
// timeout.
func TestVLLMHTTPRenderer_ChatTimeoutRawMessages(t *testing.T) {
	r := &vllmHTTPRenderer{timeout: 5 * time.Second, mmTimeout: 30 * time.Second}

	textOnly := fwkrh.PayloadMap{"messages": []any{
		json.RawMessage(`{"role":"user","content":"hi"}`),
	}}
	assert.Equal(t, 5*time.Second, r.chatTimeout(textOnly))

	multimodal := fwkrh.PayloadMap{"messages": []any{
		json.RawMessage(`{"role":"user","content":"hi"}`),
		json.RawMessage(`{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}}]}`),
	}}
	assert.Equal(t, 30*time.Second, r.chatTimeout(multimodal))
}

// TestBuildChatRenderRequest_MessageFields asserts the wire shape of rebuilt
// messages: tool_calls and reasoning ride along, tool messages carry
// tool_call_id, and content is omitted when absent.
func TestBuildChatRenderRequest_MessageFields(t *testing.T) {
	req := &tokenizerTypes.RenderChatRequest{
		Conversation: []tokenizerTypes.Conversation{
			{Role: "assistant", Content: nil, Reasoning: "hmm", ToolCalls: []any{map[string]any{
				"id":   "t1",
				"type": "function",
				"function": map[string]any{
					"name":      "run",
					"arguments": `{"cmd": "ls"}`,
				},
			}}},
			{Role: "tool", ToolCallID: "t1", Content: &tokenizerTypes.Content{Raw: "out"}},
		},
	}

	data, err := json.Marshal(buildChatRenderRequest(req))
	require.NoError(t, err)

	var msgs []map[string]any
	require.NoError(t, json.Unmarshal(data, &struct {
		Messages *[]map[string]any `json:"messages"`
	}{&msgs}))
	require.Len(t, msgs, 2)

	assert.NotContains(t, msgs[0], "content", "assistant with only tool_calls omits content")
	assert.Equal(t, "hmm", msgs[0]["reasoning"])
	assert.Equal(t, []any{map[string]any{
		"id":   "t1",
		"type": "function",
		"function": map[string]any{
			"name":      "run",
			"arguments": `{"cmd": "ls"}`,
		},
	}}, msgs[0]["tool_calls"])

	assert.Equal(t, "tool", msgs[1]["role"])
	assert.Equal(t, "t1", msgs[1]["tool_call_id"])
	assert.Equal(t, "out", msgs[1]["content"])
}

// TestVLLMHTTPRenderer_RenderSpanName asserts the outbound render span is
// named after the render route instead of the transport default "HTTP POST".
func TestVLLMHTTPRenderer_RenderSpanName(t *testing.T) {
	prevTP := otel.GetTracerProvider()
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	otel.SetTracerProvider(tp)
	t.Cleanup(func() {
		otel.SetTracerProvider(prevTP)
		_ = tp.Shutdown(context.Background())
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[{"token_ids":[1]}]`))
	}))
	defer srv.Close()

	// The transport resolves its tracer from the global provider at
	// construction, so the renderer must be built after the swap above.
	r := newHTTPRenderer(t, srv)

	ctx, span := tp.Tracer("test").Start(context.Background(), "parent")
	_, _, err := r.Render(ctx, fwkrh.PayloadMap{"prompt": "hello"})
	span.End()
	require.NoError(t, err)

	var clientSpans []tracetest.SpanStub
	for _, s := range exporter.GetSpans() {
		if s.SpanKind == trace.SpanKindClient {
			clientSpans = append(clientSpans, s)
		}
	}
	require.NotEmpty(t, clientSpans, "expected a client span for the render call")
	for _, s := range clientSpans {
		assert.Equal(t, "tokenize_render /v1/completions/render", s.Name)
	}
}
