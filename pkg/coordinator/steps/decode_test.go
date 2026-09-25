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

package steps

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestDecodeStep_NonStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != testChatCompletionsPath {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get(gateway.EPPProfileHeader) != gateway.PhaseDecode {
			t.Fatalf("expected EPP-Profile: decode, got %q", r.Header.Get(gateway.EPPProfileHeader))
		}

		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)

		if parsed["model"] != "llama-3" {
			t.Fatalf("expected model llama-3, got %v", parsed["model"])
		}
		if parsed["stream"] != false {
			t.Fatalf("expected stream=false, got %v", parsed["stream"])
		}

		// Verify kv_transfer_params injected with do_remote_prefill
		kvParams, ok := parsed["kv_transfer_params"].(map[string]any)
		if !ok {
			t.Fatal("expected kv_transfer_params in decode body")
		}
		if kvParams["block_id"] != "xyz" {
			t.Errorf("kv_transfer_params.block_id = %v, want xyz", kvParams["block_id"])
		}
		if kvParams["peer_host"] != "10.0.0.5" {
			t.Errorf("kv_transfer_params.peer_host = %v, want 10.0.0.5", kvParams["peer_host"])
		}
		if kvParams["do_remote_decode"] != false {
			t.Errorf("kv_transfer_params.do_remote_decode = %v, want false", kvParams["do_remote_decode"])
		}
		if kvParams["do_remote_prefill"] != true {
			t.Errorf("kv_transfer_params.do_remote_prefill = %v, want true", kvParams["do_remote_prefill"])
		}

		// Verify no tokens field (dead field, never consumed downstream)
		if _, ok := parsed["tokens"]; ok {
			t.Fatal("decode request should not have a tokens field")
		}

		// Verify uuid was injected into the image_url content part
		messages := parsed["messages"].([]any)
		msg := messages[0].(map[string]any)
		content := msg["content"].([]any)
		imgPart := content[0].(map[string]any)
		if imgPart["uuid"] != testImageHash {
			t.Fatalf("expected uuid=hash-a in image_url part, got %v", imgPart["uuid"])
		}
		// Verify image_url is preserved alongside the injected uuid
		imgURL, ok := imgPart["image_url"].(map[string]any)
		if !ok {
			t.Fatalf("expected image_url map, got %T", imgPart["image_url"])
		}
		if imgURL["url"] != "https://example.com/cat.jpg" {
			t.Fatalf("expected image_url.url preserved, got %v", imgURL["url"])
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "I see a cat."}},
			},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL})
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-1",
		OriginalPath: testChatCompletionsPath,
		Model:        "llama-3",
		Stream:       false,
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: testImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		KVTransferParams: map[string]any{"block_id": "xyz", "peer_host": "10.0.0.5", "peer_port": 7777},
		Body: map[string]any{
			"model":  "llama-3",
			"stream": false,
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "https://example.com/cat.jpg"},
						},
					},
				},
			},
		},
		ResponseWriter: recorder,
	}

	err = step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := recorder.Result()
	if result.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", result.StatusCode)
	}

	respBody, _ := io.ReadAll(result.Body)
	if !strings.Contains(string(respBody), "I see a cat.") {
		t.Fatalf("expected response to contain 'I see a cat.', got: %s", string(respBody))
	}
}

func TestDecodeStep_Responses_NonStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != reqcommon.PathResponses {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)

		kvParams, ok := parsed["kv_transfer_params"].(map[string]any)
		if !ok {
			t.Fatal("expected kv_transfer_params in decode body")
		}
		if kvParams["block_id"] != "xyz" {
			t.Errorf("kv_transfer_params.block_id = %v, want xyz", kvParams["block_id"])
		}

		// Verify no tokens field (dead field, never consumed downstream)
		if _, ok := parsed["tokens"]; ok {
			t.Fatal("decode request should not have a tokens field")
		}

		input := parsed["input"].([]any)
		item := input[0].(map[string]any)
		content := item["content"].([]any)
		imgPart := content[0].(map[string]any)
		if imgPart["uuid"] != testImageHash {
			t.Fatalf("expected uuid=hash-a in input_image part, got %v", imgPart["uuid"])
		}
		if imgPart["image_url"] != "https://example.com/cat.jpg" {
			t.Fatalf("expected image_url preserved, got %v", imgPart["image_url"])
		}

		_ = json.NewEncoder(w).Encode(map[string]any{"output": []map[string]any{}})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL})
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-responses",
		OriginalPath: reqcommon.PathResponses,
		Model:        "llama-3",
		Stream:       false,
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: testImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		KVTransferParams: map[string]any{"block_id": "xyz", "peer_host": "10.0.0.5", "peer_port": 7777},
		Body: map[string]any{
			"model": "llama-3",
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "input_image",
							"image_url": "https://example.com/cat.jpg",
						},
					},
				},
			},
		},
		ResponseWriter: recorder,
	}

	err = step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if recorder.Result().StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", recorder.Result().StatusCode)
	}
}

// See gateway.DetectFormat's doc comment for why injectUUIDs gates on path
// rather than field presence. A chat-completions request carrying a stray
// top-level "input" array must not have that array's image part stamped
// with a uuid.
func TestDecodeStep_IgnoresStrayInputOnChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)

		messages := parsed["messages"].([]any)
		msgPart := messages[0].(map[string]any)["content"].([]any)[0].(map[string]any)
		if msgPart["uuid"] != testImageHash {
			t.Fatalf("expected uuid=%s on the messages image part, got %v", testImageHash, msgPart["uuid"])
		}

		input := parsed["input"].([]any)
		inputPart := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)
		if _, ok := inputPart["uuid"]; ok {
			t.Fatalf("expected no uuid stamped on the stray input array's part, got %v", inputPart["uuid"])
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ok"}}},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL})
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-stray-input",
		OriginalPath: testChatCompletionsPath,
		Model:        "llama-3",
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: testImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		KVTransferParams: map[string]any{"block_id": "xyz"},
		Body: map[string]any{
			"model": "llama-3",
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/cat.jpg"}},
					},
				},
			},
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_image", "image_url": "https://example.com/dog.jpg"},
					},
				},
			},
		},
		ResponseWriter: recorder,
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if recorder.Result().StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", recorder.Result().StatusCode)
	}
}

func TestDecodeStep_CompletionsFormat_NoRenderedTokens(t *testing.T) {
	var parsed map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &parsed)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": "ok"}}})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-compl",
		OriginalPath:     reqcommon.PathCompletions,
		Model:            "test-model",
		TokenIDs:         nil,
		KVTransferParams: map[string]any{},
		Body:             map[string]any{"model": "test-model", "prompt": "Hello"},
		ResponseWriter:   recorder,
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if parsed["prompt"] != "Hello" {
		t.Fatalf("expected original prompt to pass through, got %v", parsed["prompt"])
	}
}

// TestDecodeStep_CompletionsFormat_RewritesPromptAndTopLevelKV verifies that for
// the /v1/completions format the decode step rewrites prompt to the rendered
// token IDs and places kv_transfer_params at the top level.
func TestDecodeStep_CompletionsFormat_RewritesPromptAndTopLevelKV(t *testing.T) {
	var parsed map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &parsed)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": "ok"}}})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-compl-rendered",
		OriginalPath:     reqcommon.PathCompletions,
		Model:            "test-model",
		TokenIDs:         []int{1, 2345},
		KVTransferParams: map[string]any{"block_id": "block-1"},
		Body:             map[string]any{"model": "test-model", "prompt": "Hello"},
		ResponseWriter:   recorder,
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tokenIDs, _ := parsed["prompt"].([]any); len(tokenIDs) != 2 {
		t.Fatalf("expected prompt rewritten to rendered token_ids, got %v", parsed["prompt"])
	}
	if _, ok := parsed[reqcommon.FieldKVTransferParams]; !ok {
		t.Fatal("expected top-level kv_transfer_params for /v1/completions")
	}
}

// TestDecodeStep_GenerateFormat_ToplevelKV verifies that for the
// /inference/v1/generate format the decode step places kv_transfer_params at the
// top level of the request body, and preserves the client's sampling_params so
// the decode generation honors the requested max_tokens.
func TestDecodeStep_GenerateFormat_ToplevelKV(t *testing.T) {
	var parsed map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &parsed)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": "ok"}}})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.NIXL})
	if err != nil {
		t.Fatal(err)
	}

	wantBlockID := "block-gen-1"
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-gen",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "test-model",
		TokenIDs:         []int{1, 2, 3, 4, 5},
		KVTransferParams: map[string]any{"block_id": wantBlockID, "peer_host": "10.0.0.42", "peer_port": 7777},
		Body: map[string]any{
			"model":           "test-model",
			"token_ids":       []int{1, 2, 3, 4, 5},
			"sampling_params": map[string]any{"max_tokens": 50},
		},
		ResponseWriter: recorder,
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	sampling, ok := parsed["sampling_params"].(map[string]any)
	if !ok {
		t.Fatal("expected sampling_params in decode body")
	}
	// Client sampling fields are preserved (decode honors the real max_tokens).
	if sampling["max_tokens"] != float64(50) {
		t.Fatalf("expected sampling_params.max_tokens=50 preserved, got %v", sampling["max_tokens"])
	}
	// The transfer params are no longer nested under extra_args.
	if _, ok := sampling["extra_args"]; ok {
		t.Fatalf("expected no sampling_params.extra_args in generate format, got %v", sampling["extra_args"])
	}
	kvParams, ok := parsed["kv_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected top-level kv_transfer_params in generate format")
	}
	if kvParams["block_id"] != wantBlockID {
		t.Errorf("kv_transfer_params.block_id = %v, want %v", kvParams["block_id"], wantBlockID)
	}
	if kvParams["do_remote_prefill"] != true {
		t.Errorf("kv_transfer_params.do_remote_prefill = %v, want true", kvParams["do_remote_prefill"])
	}
}

// TestDecodeStep_UnreachableFormat_ReturnsError verifies that request paths
// for formats prepareDecodeBody's switch does not handle explicitly
// (APITypeMessages, APITypeSGLangGenerate) fail through
// its default case, reporting an error instead of sending an unprepared
// body upstream.
func TestDecodeStep_UnreachableFormat_ReturnsError(t *testing.T) {
	for _, path := range []string{reqcommon.PathMessages, reqcommon.PathSGLangGenerate} {
		t.Run(path, func(t *testing.T) {
			step, err := NewDecodeStep(gateway.New(config.GatewayConfig{}), map[string]any{ParamKVConnector: kv.NIXL})
			if err != nil {
				t.Fatal(err)
			}

			reqCtx := &pipeline.RequestContext{
				OriginalPath:     path,
				Body:             map[string]any{"model": testModelName},
				KVTransferParams: map[string]any{"block_id": "block-1"},
			}
			if err := step.(*DecodeStep).prepareDecodeBody(context.Background(), reqCtx); err == nil {
				t.Fatal("expected an error for an unreachable request format")
			}
		})
	}
}

func TestDecodeStep_Streaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)

		if parsed["stream"] != true {
			t.Fatalf("expected stream=true")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)

		events := []string{
			`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
			`data: {"choices":[{"delta":{"content":" world"}}]}`,
			`data: [DONE]`,
		}
		for _, event := range events {
			fmt.Fprintf(w, "%s\n\n", event)
			flusher.Flush()
		}
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, _ := NewDecodeStep(gwClient, map[string]any{})

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-1",
		OriginalPath: testChatCompletionsPath,
		Model:        "test",
		Stream:       true,
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1"},
		},
		KVTransferParams: map[string]any{},
		Body:             map[string]any{"model": "test", "stream": true},
		ResponseWriter:   recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := recorder.Result()
	if result.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %s", result.Header.Get("Content-Type"))
	}

	respBody, _ := io.ReadAll(result.Body)
	body := string(respBody)
	if !strings.Contains(body, `"content":"Hello"`) {
		t.Fatalf("expected Hello event, got: %s", body)
	}
	if !strings.Contains(body, "[DONE]") {
		t.Fatalf("expected [DONE] event, got: %s", body)
	}
}

func TestDecodeStep_GatewayError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("upstream unavailable"))
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, _ := NewDecodeStep(gwClient, map[string]any{})

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-1",
		OriginalPath: testChatCompletionsPath,
		Model:        "test",
		Stream:       false,
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1"},
		},
		KVTransferParams: map[string]any{},
		Body:             map[string]any{"model": "test", "stream": false},
		ResponseWriter:   recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	var streamed *pipeline.UpstreamStreamedError
	if !errors.As(err, &streamed) {
		t.Fatalf("expected *pipeline.UpstreamStreamedError, got %T (%v)", err, err)
	}
	if streamed.StatusCode != http.StatusBadGateway {
		t.Fatalf("expected StatusCode=502 on the streamed error, got %d", streamed.StatusCode)
	}

	result := recorder.Result()
	if result.StatusCode != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", result.StatusCode)
	}

	respBody, _ := io.ReadAll(result.Body)
	if !strings.Contains(string(respBody), "upstream unavailable") {
		t.Fatalf("expected error body forwarded, got: %s", string(respBody))
	}
}

// TestDecodeStep_NilClientTransport builds a step around a gateway.Client whose
// Transport() returns nil. gateway.NewWithTransport documents that as valid and
// leaves the default-transport fallback to http.Client; the timedRoundTripper
// wrapper must reproduce the same fallback so the step does not panic.
func TestDecodeStep_NilClientTransport(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": "ok"}}})
	}))
	defer server.Close()

	gwClient := gateway.NewWithTransport(nil, server.URL)
	step, err := NewDecodeStep(gwClient, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-1",
		OriginalPath:     testChatCompletionsPath,
		Model:            "test",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body:             map[string]any{"model": "test", "stream": false},
		ResponseWriter:   recorder,
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := recorder.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("expected 200, got %d", got)
	}
}

func TestDecodeStep_TransportError(t *testing.T) {
	// Start a server, capture its URL, then close it: subsequent connects fail
	// before any HTTP response arrives. This exercises the ErrorHandler branch
	// of newDecodeProxy, not the ModifyResponse branch used by TestDecodeStep_GatewayError.
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}))
	serverURL := server.URL
	server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: serverURL})
	step, _ := NewDecodeStep(gwClient, map[string]any{})

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-1",
		OriginalPath:     testChatCompletionsPath,
		Model:            "test",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body:             map[string]any{"model": "test", "stream": false},
		ResponseWriter:   recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	var streamed *pipeline.UpstreamStreamedError
	if !errors.As(err, &streamed) {
		t.Fatalf("expected *pipeline.UpstreamStreamedError, got %T (%v)", err, err)
	}
	if streamed.StatusCode != 0 {
		t.Fatalf("transport error must carry StatusCode=0, got %d", streamed.StatusCode)
	}
	if streamed.Cause == nil {
		t.Fatalf("transport error must carry Cause, got nil")
	}

	result := recorder.Result()
	if result.StatusCode != http.StatusBadGateway {
		t.Fatalf("expected ErrorHandler-written 502, got %d", result.StatusCode)
	}
}
