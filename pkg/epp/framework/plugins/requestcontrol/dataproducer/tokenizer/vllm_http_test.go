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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/test/utils"
)

const testHTTPModel = "test-model"

func newHTTPRenderer(t *testing.T, srv *httptest.Server) *vllmHTTPRenderer {
	t.Helper()
	r, err := newVLLMHTTPRenderer(&vllmConfig{HTTP: srv.URL}, testHTTPModel)
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
	tokenIDs, offsets, err := r.Render(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, []uint32{1, 2, 3}, tokenIDs)
	assert.Nil(t, offsets)

	var sent map[string]any
	require.NoError(t, json.Unmarshal(cap.completions, &sent))
	assert.Equal(t, testHTTPModel, sent["model"])
	assert.Equal(t, "hello", sent["prompt"])
}

// TestVLLMHTTPRenderer_RenderChat_Multimodal covers the chat endpoint, the
// OpenAI-shape request projection from kvcache RenderChatRequest, and the
// wire→kvcache conversion of multimodal features.
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
	req := &tokenizerTypes.RenderChatRequest{
		Conversation: []tokenizerTypes.Conversation{{
			Role: "user",
			Content: tokenizerTypes.Content{Structured: []tokenizerTypes.ContentBlock{
				{Type: "image_url", ImageURL: tokenizerTypes.ImageBlock{URL: "data:image/png;base64,xx"}},
				{Type: "text", Text: "describe"},
			}},
		}},
		AddGenerationPrompt: true,
	}
	tokenIDs, mm, err := r.RenderChat(context.Background(), req)
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
	require.True(t, ok, "structured content must be sent as an array of parts")
	require.Len(t, parts, 2)
	assert.Equal(t, "image_url", parts[0].(map[string]any)["type"])
	assert.Equal(t, "text", parts[1].(map[string]any)["type"])
}

func TestVLLMHTTPRenderer_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	r := newHTTPRenderer(t, srv)
	_, _, err := r.Render(context.Background(), "x")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "500")
}

func TestPluginFactory_RejectsBothBackends(t *testing.T) {
	params := `{
		"modelName": "m",
		"udsTokenizerConfig": {"socketFile": "/tmp/foo.sock"},
		"vllm": {"http": "http://localhost:8000"}
	}`
	handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
	p, err := PluginFactory("test", json.RawMessage(params), handle)
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
	p, err := PluginFactory("test", json.RawMessage(params), handle)
	require.Error(t, err)
	assert.Nil(t, p)
	assert.Contains(t, err.Error(), "invalid 'timeout'")
}
