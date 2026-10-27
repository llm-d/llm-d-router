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

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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
		if r.Header.Get(reqcommon.EPPProfileHeaderKey) != gateway.PhaseDecode {
			t.Fatalf("expected x-llm-d-epp-profile: decode, got %q", r.Header.Get(reqcommon.EPPProfileHeaderKey))
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

		// The uuid lands on the image part only: hashIdx must not advance over
		// the text parts, or every image would carry another image's hash.
		messages := parsed["messages"].([]any)
		msg := messages[0].(map[string]any)
		content := msg["content"].([]any)
		imgPart := content[1].(map[string]any)
		if imgPart["uuid"] != testImageHash {
			t.Errorf("expected uuid=hash-a in image_url part, got %v", imgPart["uuid"])
		}
		for _, idx := range []int{0, 2} {
			if uuid, ok := content[idx].(map[string]any)["uuid"]; ok {
				t.Errorf("text part %d must carry no uuid, got %v", idx, uuid)
			}
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
			{Modality: reqcommon.ModalityImage, Hash: testImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		KVTransferParams: map[string]any{"block_id": "xyz", "peer_host": "10.0.0.5", "peer_port": 7777},
		Body: map[string]any{
			"model":  "llama-3",
			"stream": false,
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "describe this"},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "https://example.com/cat.jpg"},
						},
						map[string]any{"type": "text", "text": "in one word"},
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

		// The uuid lands on the image part only: hashIdx must not advance over
		// the input_text parts, or every image would carry another image's hash.
		input := parsed["input"].([]any)
		item := input[0].(map[string]any)
		content := item["content"].([]any)
		imgPart := content[1].(map[string]any)
		if imgPart["uuid"] != testImageHash {
			t.Errorf("expected uuid=hash-a in input_image part, got %v", imgPart["uuid"])
		}
		if imgPart["image_url"] != "https://example.com/cat.jpg" {
			t.Errorf("expected image_url preserved, got %v", imgPart["image_url"])
		}
		for _, idx := range []int{0, 2} {
			if uuid, ok := content[idx].(map[string]any)["uuid"]; ok {
				t.Errorf("input_text part %d must carry no uuid, got %v", idx, uuid)
			}
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
			{Modality: reqcommon.ModalityImage, Hash: testImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		KVTransferParams: map[string]any{"block_id": "xyz", "peer_host": "10.0.0.5", "peer_port": 7777},
		Body: map[string]any{
			"model": "llama-3",
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "describe this"},
						map[string]any{
							"type":      "input_image",
							"image_url": "https://example.com/cat.jpg",
						},
						map[string]any{"type": "input_text", "text": "in one word"},
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

func TestDecodeStep_Messages_NonStreaming(t *testing.T) {
	var parsed map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != reqcommon.PathMessages {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &parsed)
		_ = json.NewEncoder(w).Encode(map[string]any{"type": "message", "content": []map[string]any{}})
	}))
	defer server.Close()

	step, err := NewDecodeStep(gateway.New(config.GatewayConfig{Address: server.URL}), map[string]any{ParamKVConnector: kv.NIXL})
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-messages",
		OriginalPath: reqcommon.PathMessages,
		Model:        "llama-3",
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: testImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		KVTransferParams: map[string]any{"block_id": "xyz", "peer_host": "10.0.0.5", "peer_port": 7777},
		Body: map[string]any{
			"model":      "llama-3",
			"max_tokens": 50,
			"messages": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "text", "text": "describe this"},
					map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": "https://example.com/cat.jpg"}},
				}},
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

	kvParams, ok := parsed["kv_transfer_params"].(map[string]any)
	if !ok || kvParams["block_id"] != "xyz" {
		t.Fatalf("expected the prefill's kv_transfer_params at the top level, got %v", parsed["kv_transfer_params"])
	}
	if parsed["max_tokens"] != float64(50) {
		t.Fatalf("expected the client's max_tokens preserved, got %v", parsed["max_tokens"])
	}
	// vLLM's Anthropic content block defines no uuid, so decode stamps none and
	// the worker hashes the image itself.
	content := parsed["messages"].([]any)[0].(map[string]any)["content"].([]any)
	image := content[1].(map[string]any)
	if uuid, ok := image["uuid"]; ok {
		t.Fatalf("expected no uuid on a Messages image block, got %v", uuid)
	}
	if source := image["source"].(map[string]any); source["url"] != "https://example.com/cat.jpg" {
		t.Fatalf("expected the image source preserved, got %v", source)
	}
}

// A chat-completions request carrying a stray top-level "input" array must not
// have that array's image part stamped with a uuid.
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
			{Modality: reqcommon.ModalityImage, Hash: testImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
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

func TestDecodeStep_InjectUUIDsIntoRawPromptItems(t *testing.T) {
	for _, tc := range []struct {
		name     string
		path     string
		field    string
		modality reqcommon.Modality
		items    string
	}{
		{
			name:     "chat completions",
			path:     testChatCompletionsPath,
			field:    "messages",
			modality: reqcommon.ModalityImage,
			items:    `[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]`,
		},
		{
			name:     "responses",
			path:     reqcommon.PathResponses,
			field:    "input",
			modality: reqcommon.ModalityImage,
			items:    `[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]`,
		},
		// The modalities this branch adds walk the same raw-JSON round trip, so
		// each part shape that names one is stamped here too.
		{
			name:     "chat completions audio_url",
			path:     testChatCompletionsPath,
			field:    "messages",
			modality: reqcommon.ModalityAudio,
			items:    `[{"role":"user","content":[{"type":"audio_url","audio_url":{"url":"https://example.com/a.wav"}}]}]`,
		},
		{
			name:     "chat completions input_audio",
			path:     testChatCompletionsPath,
			field:    "messages",
			modality: reqcommon.ModalityAudio,
			items:    `[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}]}]`,
		},
		{
			name:     "chat completions video_url",
			path:     testChatCompletionsPath,
			field:    "messages",
			modality: reqcommon.ModalityVideo,
			items:    `[{"role":"user","content":[{"type":"video_url","video_url":{"url":"https://example.com/v.mp4"}}]}]`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reqCtx := &pipeline.RequestContext{
				OriginalPath: tc.path,
				MultimodalEntries: []pipeline.MultimodalEntry{
					{Modality: tc.modality, Hash: "hash-raw"},
				},
				Body: map[string]any{tc.field: json.RawMessage(tc.items)},
			}
			(&DecodeStep{}).injectUUIDs(reqCtx, logr.Discard())

			items, ok := reqCtx.Body[tc.field].([]any)
			if !ok {
				t.Fatalf("%s is %T, want the edited array stored back", tc.field, reqCtx.Body[tc.field])
			}
			part := items[0].(map[string]any)["content"].([]any)[0].(map[string]any)
			if part["uuid"] != "hash-raw" {
				t.Fatalf("expected uuid=hash-raw in %s part, got %v", tc.modality, part["uuid"])
			}
		})
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

// TestDecodeStep_TokensInFormat_RewritesChatRequest verifies the tokens-in
// decode leg (use_openai_format=false): a chat-completions client is sent to
// the generate route with the rendered token IDs, output_mode=text so the
// worker answers with detokenized text, and its top-level max_tokens moved
// under sampling_params, which the generate API requires and does not default.
func TestDecodeStep_TokensInFormat_RewritesChatRequest(t *testing.T) {
	var parsed map[string]any
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &parsed)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"output_mode": reqcommon.OutputModeText,
			"choices":     []map[string]any{{"token_ids": []int{9707}, "text": " Hello"}},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamKVConnector:    kv.NIXL,
		"use_openai_format": false,
	})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-tokens-in",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            testModelName,
		TokenIDs:         []int{1, 2, 3},
		KVTransferParams: map[string]any{"block_id": "block-1"},
		Body: map[string]any{
			"model":      testModelName,
			"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
			"max_tokens": 64,
		},
		ResponseWriter: httptest.NewRecorder(),
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gotPath != reqcommon.PathVLLMGenerate {
		t.Fatalf("decode posted to %q, want %q", gotPath, reqcommon.PathVLLMGenerate)
	}
	if parsed[reqcommon.FieldOutputMode] != reqcommon.OutputModeText {
		t.Fatalf("output_mode = %v, want %q", parsed[reqcommon.FieldOutputMode], reqcommon.OutputModeText)
	}
	tokenIDs, ok := parsed["token_ids"].([]any)
	if !ok || len(tokenIDs) != 3 {
		t.Fatalf("token_ids = %v, want the 3 rendered ids", parsed["token_ids"])
	}
	// The chat body's own shape must not reach the generate route.
	if _, ok := parsed["messages"]; ok {
		t.Fatal("messages reached the generate request")
	}
	sampling, ok := parsed[reqcommon.FieldSamplingParams].(map[string]any)
	if !ok {
		t.Fatalf("sampling_params = %v, want an object", parsed[reqcommon.FieldSamplingParams])
	}
	if sampling[reqcommon.FieldMaxTokens] != float64(64) {
		t.Fatalf("sampling_params.max_tokens = %v, want the client's 64", sampling[reqcommon.FieldMaxTokens])
	}
	if _, ok := parsed[reqcommon.FieldKVTransferParams].(map[string]any); !ok {
		t.Fatal("expected top-level kv_transfer_params on the generate leg")
	}
}

// TestDecodeStep_TokensInFormat_KeepsGenerateClientFields verifies that a
// client already on the generate route keeps the fields it sent (sampling_params
// and the generate-only extras) when the tokens-in decode leg rebuilds its body.
func TestDecodeStep_TokensInFormat_KeepsGenerateClientFields(t *testing.T) {
	var parsed map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &parsed)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"token_ids": []int{1}}}})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamKVConnector:    kv.NIXL,
		"use_openai_format": false,
	})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-gen-native",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            testModelName,
		TokenIDs:         []int{1, 2, 3, 4, 5},
		KVTransferParams: map[string]any{"block_id": "block-1"},
		Body: map[string]any{
			"model":           testModelName,
			"token_ids":       []int{1, 2, 3, 4, 5},
			"cache_salt":      "tenant-a",
			"sampling_params": map[string]any{"max_tokens": 50, "min_tokens": 3},
		},
		ResponseWriter: httptest.NewRecorder(),
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if parsed[reqcommon.FieldOutputMode] != reqcommon.OutputModeText {
		t.Fatalf("output_mode = %v, want %q", parsed[reqcommon.FieldOutputMode], reqcommon.OutputModeText)
	}
	if parsed["cache_salt"] != "tenant-a" {
		t.Fatalf("cache_salt = %v, want the client's value preserved", parsed["cache_salt"])
	}
	sampling, ok := parsed[reqcommon.FieldSamplingParams].(map[string]any)
	if !ok {
		t.Fatalf("sampling_params = %v, want an object", parsed[reqcommon.FieldSamplingParams])
	}
	// decode generates the real output, so the client's own caps stand.
	if sampling[reqcommon.FieldMaxTokens] != float64(50) {
		t.Fatalf("sampling_params.max_tokens = %v, want 50", sampling[reqcommon.FieldMaxTokens])
	}
	if sampling[reqcommon.FieldMinTokens] != float64(3) {
		t.Fatalf("sampling_params.min_tokens = %v, want 3", sampling[reqcommon.FieldMinTokens])
	}
}

// TestDecodeStep_WrapsChatCompletion verifies the end of the wrapping path: a
// non-streaming chat-completions client whose leg was rewritten to the generate
// route receives a chat.completion envelope rather than the generate one.
func TestDecodeStep_WrapsChatCompletion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"request_id":"r1","output_mode":"derender","model":"test-model",
			"kv_transfer_params":{"remote_host":"10.0.0.1"},
			"choices":[{"index":0,"finish_reason":"stop","token_ids":[17],
			"message":{"role":"assistant","content":"2 + 2 = 4"}}]}`))
	}))
	defer server.Close()

	step, err := NewDecodeStep(gateway.New(config.GatewayConfig{Address: server.URL}), map[string]any{
		ParamKVConnector:          kv.NIXL,
		"use_openai_format":       false,
		reqcommon.FieldOutputMode: reqcommon.OutputModeDerender,
		ParamWrapChatCompletion:   true,
	})
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-wrap",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            testModelName,
		TokenIDs:         []int{1, 2, 3},
		KVTransferParams: map[string]any{"block_id": "block-1"},
		Body:             map[string]any{"model": testModelName, "messages": []any{}},
		ResponseWriter:   recorder,
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &got); err != nil {
		t.Fatalf("client body is not JSON: %v (%q)", err, recorder.Body.String())
	}
	if got[reqcommon.FieldObject] != reqcommon.ObjectChatCompletion {
		t.Fatalf("object = %v, want %q", got[reqcommon.FieldObject], reqcommon.ObjectChatCompletion)
	}
	if _, present := got[reqcommon.FieldKVTransferParams]; present {
		t.Error("kv_transfer_params reached the client")
	}
	choice, _ := got[reqcommon.FieldChoices].([]any)[0].(map[string]any)
	msg, _ := choice[reqcommon.FieldMessage].(map[string]any)
	if msg[reqcommon.FieldContent] != "2 + 2 = 4" {
		t.Fatalf("content = %v, want the generated text", msg[reqcommon.FieldContent])
	}
	if got[reqcommon.HeaderContentType] != nil {
		t.Error("content-type leaked into the body")
	}
	if ct := recorder.Header().Get(reqcommon.HeaderContentType); ct != reqcommon.ContentTypeJSON {
		t.Errorf("content-type = %q, want %q", ct, reqcommon.ContentTypeJSON)
	}
}

// TestDecodeStep_WrapChatCompletionSkipped verifies the wrap is confined to the
// case it can serve. A streaming request is proxied instead, since the wrap
// rewrites one document and does not assemble chat.completion.chunk frames; a
// client on another route keeps the envelope its own API declares.
func TestDecodeStep_WrapChatCompletionSkipped(t *testing.T) {
	const generateBody = `{"request_id":"r1","output_mode":"derender",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(generateBody))
	}))
	defer server.Close()

	for _, tc := range []struct {
		name   string
		path   string
		stream bool
	}{
		{name: "streaming chat completions", path: reqcommon.PathChatCompletions, stream: true},
		{name: "generate client", path: reqcommon.PathVLLMGenerate},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step, err := NewDecodeStep(gateway.New(config.GatewayConfig{Address: server.URL}), map[string]any{
				ParamKVConnector:          kv.NIXL,
				"use_openai_format":       false,
				reqcommon.FieldOutputMode: reqcommon.OutputModeText,
				ParamWrapChatCompletion:   true,
			})
			if err != nil {
				t.Fatal(err)
			}

			recorder := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-skip",
				OriginalPath:     tc.path,
				Model:            testModelName,
				Stream:           tc.stream,
				TokenIDs:         []int{1, 2, 3},
				KVTransferParams: map[string]any{"block_id": "block-1"},
				Body:             map[string]any{"model": testModelName},
				ResponseWriter:   recorder,
			}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if body := recorder.Body.String(); body != generateBody {
				t.Fatalf("client body = %q, want the generate response unchanged", body)
			}
		})
	}
}

// TestDecodeStep_UnreachableFormat_ReturnsError verifies that a request path
// for a format prepareDecodeBody's switch does not handle explicitly
// (APITypeSGLangGenerate) fails through its default case, reporting an error
// instead of sending an unprepared body upstream.
func TestDecodeStep_UnreachableFormat_ReturnsError(t *testing.T) {
	step, err := NewDecodeStep(gateway.New(config.GatewayConfig{}), map[string]any{ParamKVConnector: kv.NIXL})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		OriginalPath:     reqcommon.PathSGLangGenerate,
		Body:             map[string]any{"model": testModelName},
		KVTransferParams: map[string]any{"block_id": "block-1"},
	}
	if _, err := step.(*DecodeStep).prepareDecodeBody(context.Background(), reqCtx, reqcommon.APITypeSGLangGenerate); err == nil {
		t.Fatal("expected an error for an unreachable request format")
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
			{Modality: reqcommon.ModalityImage, Hash: "h1"},
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
			{Modality: reqcommon.ModalityImage, Hash: "h1"},
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

// ---- injectUUIDs across audio, video, and input_audio ---------------------

// Entry hashes reused across the injectUUIDs tests, extracted so goconst does
// not flag their repetition.
const (
	testHashImage = "H-img"
	testHashAudio = "H-aud"
	testHashVideo = "H-vid"
)

// Every recognized media content-part type receives the uuid of its
// (modality, local index) entry; non-media parts are left alone.
func TestInjectUUIDs_TagsAllMediaParts(t *testing.T) {
	step := &DecodeStep{}
	imagePart := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "u-img"}}
	audioURLPart := map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "u-aud"}}
	inputAudioPart := map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "d", "format": "wav"}}
	videoPart := map[string]any{"type": "video_url", "video_url": map[string]any{"url": "u-vid"}}
	textPart := map[string]any{"type": "text", "text": "hi"}

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role":    "user",
					"content": []any{textPart, imagePart, audioURLPart, videoPart, inputAudioPart},
				},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: testHashImage},
			{Modality: reqcommon.ModalityAudio, Hash: "H-aud-url"},
			{Modality: reqcommon.ModalityVideo, Hash: "H-vid"},
			{Modality: reqcommon.ModalityAudio, Hash: "H-input-audio"},
		},
	}
	step.injectUUIDs(reqCtx, logr.Discard())

	if got := imagePart["uuid"]; got != testHashImage {
		t.Errorf("image uuid = %v, want H-img", got)
	}
	if got := audioURLPart["uuid"]; got != "H-aud-url" {
		t.Errorf("audio_url uuid = %v, want H-aud-url", got)
	}
	if got := inputAudioPart["uuid"]; got != "H-input-audio" {
		t.Errorf("input_audio uuid = %v, want H-input-audio", got)
	}
	if got := videoPart["uuid"]; got != "H-vid" {
		t.Errorf("video uuid = %v, want H-vid", got)
	}
	if _, ok := textPart["uuid"]; ok {
		t.Errorf("text part must not be tagged: %+v", textPart)
	}
}

// Two audio parts in the same request receive the hashes of the two audio
// entries, in walker order.
func TestInjectUUIDs_TagsRepeatedModalityInOrder(t *testing.T) {
	step := &DecodeStep{}
	aud0 := map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "u0"}}
	aud1 := map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "u1"}}
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": []any{aud0, aud1}},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityAudio, Hash: "H0"},
			{Modality: reqcommon.ModalityAudio, Hash: "H1"},
		},
	}
	step.injectUUIDs(reqCtx, logr.Discard())
	if got := aud0["uuid"]; got != "H0" {
		t.Errorf("audio[0] uuid = %v, want H0", got)
	}
	if got := aud1["uuid"]; got != "H1" {
		t.Errorf("audio[1] uuid = %v, want H1", got)
	}
}

// Every media-typed part consumes a slot, whether or not it still carries usable
// media: replace-media-urls already rejected the payload-less ones (see
// collectMediaRefs), so this walk must count the same parts in the same order.
// Here the part at position 0 gets the position-0 hash even when it is the
// broken one, and the surplus part is left untagged.
func TestInjectUUIDs_CountsEveryMediaPart(t *testing.T) {
	step := &DecodeStep{}
	brokenImg := map[string]any{"type": "image_url", "image_url": map[string]any{"url": nil}}
	laterImg := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "u-img"}}
	brokenInputAudio := map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "", "format": "wav"}}
	laterInputAudio := map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "AA==", "format": "wav"}}

	reqCtx := &pipeline.RequestContext{
		OriginalPath: testChatCompletionsPath,
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role":    "user",
					"content": []any{brokenImg, laterImg, brokenInputAudio, laterInputAudio},
				},
			},
		},
		// One entry per modality against two parts each, so the second part of
		// each modality is the surplus one.
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: testHashImage},
			{Modality: reqcommon.ModalityAudio, Hash: testHashAudio},
		},
	}
	step.injectUUIDs(reqCtx, logr.Discard())

	if got := brokenImg["uuid"]; got != testHashImage {
		t.Errorf("image[0] uuid = %v, want %s: a broken part still holds its position", got, testHashImage)
	}
	if got, tagged := laterImg["uuid"]; tagged {
		t.Errorf("image[1] has no entry and must stay untagged, got uuid %v", got)
	}
	if got := brokenInputAudio["uuid"]; got != testHashAudio {
		t.Errorf("audio[0] uuid = %v, want %s: a broken part still holds its position", got, testHashAudio)
	}
	if got, tagged := laterInputAudio["uuid"]; tagged {
		t.Errorf("audio[1] has no entry and must stay untagged, got uuid %v", got)
	}
}

// A modality with more well-formed parts than entries: the surplus part is left
// without a uuid and the request proceeds. Degrading rather than failing is
// deliberate -- uuid is only the decode backend's prefix-cache key, so the
// request stays correct and the cost is the worker re-processing that media. The
// encode fanout fails on the same mismatch because it would pair an entry with
// another part's bytes.
//
// Asserting the key's absence, not a non-matching value, is what makes a "tagged
// with the wrong hash" regression fail here. The image part is a control: an
// audio overflow must not shift another modality's positions.
func TestInjectUUIDs_ExtraPartForModalityStaysUntagged(t *testing.T) {
	step := &DecodeStep{}
	imagePart := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "u-img"}}
	aud0 := map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "u0"}}
	aud1 := map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "u1"}}

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": []any{imagePart, aud0, aud1}},
			},
		},
		// Two audio parts, one audio entry.
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: testHashImage},
			{Modality: reqcommon.ModalityAudio, Hash: testHashAudio},
		},
	}

	sink := &logCaptureSink{}
	step.injectUUIDs(reqCtx, logr.New(sink))

	if got := aud0["uuid"]; got != testHashAudio {
		t.Errorf("audio[0] uuid = %v, want H-aud", got)
	}
	if got, tagged := aud1["uuid"]; tagged {
		t.Errorf("audio[1] has no entry and must stay untagged, got uuid %v", got)
	}
	if got := imagePart["uuid"]; got != testHashImage {
		t.Errorf("image uuid = %v, want H-img; an audio overflow must not shift image positions", got)
	}

	if len(sink.infos) != 1 {
		t.Fatalf("expected 1 log line for the miss, got %d (errors: %d)", len(sink.infos), len(sink.errors))
	}
	if sink.infos[0].level != logutil.DEBUG {
		t.Errorf("miss logged at V(%d), want V(%d)", sink.infos[0].level, logutil.DEBUG)
	}
	if len(sink.errors) != 0 {
		t.Errorf("a surplus part is not an error condition, got %d error logs", len(sink.errors))
	}
}

// The opposite mismatch, which the stamping walk cannot see: it iterates parts,
// so an entry with no part to stamp used to pass unrecorded. See
// injectMediaPartUUIDs for why that direction is the worse one.
//
// The whole branch is a log line, so the level is what gets pinned: DEBUG, as
// for the surplus-part branch. The image entry has no part at all and the audio
// entry one of its two; the video part is a control that a fully paired modality
// stays quiet.
func TestInjectUUIDs_EntryWithNoMediaPartIsLogged(t *testing.T) {
	step := &DecodeStep{}
	audioPart := map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "u-aud"}}
	videoPart := map[string]any{"type": "video_url", "video_url": map[string]any{"url": "u-vid"}}

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": []any{audioPart, videoPart}},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			// One image entry and no image part; two audio entries and one
			// audio part; one video entry and one video part.
			{Modality: reqcommon.ModalityImage, Hash: testHashImage},
			{Modality: reqcommon.ModalityAudio, Hash: testHashAudio},
			{Modality: reqcommon.ModalityAudio, Hash: testHashAudio + "-2"},
			{Modality: reqcommon.ModalityVideo, Hash: testHashVideo},
		},
	}

	sink := &logCaptureSink{}
	step.injectUUIDs(reqCtx, logr.New(sink))

	if got := audioPart["uuid"]; got != testHashAudio {
		t.Errorf("audio[0] uuid = %v, want %v; a surplus entry must not stop the parts that do pair", got, testHashAudio)
	}
	if got := videoPart["uuid"]; got != testHashVideo {
		t.Errorf("video uuid = %v, want %v", got, testHashVideo)
	}

	// Map iteration order is unspecified, so count the lines rather than
	// indexing them: image and audio each report, video does not.
	if len(sink.infos) != 2 {
		t.Fatalf("expected 2 log lines, one per modality with a surplus entry, got %d", len(sink.infos))
	}
	for _, got := range sink.infos {
		if got.msg != "MultimodalEntry with no media part" {
			t.Errorf("unexpected log line %q", got.msg)
		}
		if got.level != logutil.DEBUG {
			t.Errorf("surplus entry logged at V(%d), want V(%d)", got.level, logutil.DEBUG)
		}
	}
	if len(sink.errors) != 0 {
		t.Errorf("a surplus entry degrades rather than fails, got %d error logs", len(sink.errors))
	}
}

// TestDecodeStep_Responses_StampsFunctionCallOutputImage verifies the uuid
// stamping walk sees the same image set replace-media-urls built entries from,
// including an image under a function_call_output's output. An unstamped part
// makes the worker compute its own hash, so the entry primed into the encoder
// cache for it is never looked up.
func TestDecodeStep_Responses_StampsFunctionCallOutputImage(t *testing.T) {
	const outputImageHash = "hash-b"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Errorf("decode body did not parse: %v", err)
			return
		}

		input, ok := parsed["input"].([]any)
		if !ok || len(input) != 2 {
			t.Errorf("expected 2 input items, got %v", parsed["input"])
			return
		}
		contentPart := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)
		if contentPart["uuid"] != testImageHash {
			t.Errorf("content image uuid = %v, want %s", contentPart["uuid"], testImageHash)
		}
		outputPart := input[1].(map[string]any)["output"].([]any)[0].(map[string]any)
		if outputPart["uuid"] != outputImageHash {
			t.Errorf("output image uuid = %v, want %s", outputPart["uuid"], outputImageHash)
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
		RequestID:    "req-responses-output",
		OriginalPath: reqcommon.PathResponses,
		Model:        "llama-3",
		TokenIDs:     []int{1, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: testImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
			{Modality: reqcommon.ModalityImage, Hash: outputImageHash, Placeholder: pipeline.PlaceholderRange{Offset: 2, Length: 1}},
		},
		KVTransferParams: map[string]any{"block_id": "xyz"},
		Body: map[string]any{
			"model": "llama-3",
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": reqcommon.PartTypeInputImage, "image_url": "https://example.com/cat.jpg"},
					},
				},
				map[string]any{
					"type":    "function_call_output",
					"call_id": "call-1",
					"output": []any{
						map[string]any{"type": reqcommon.PartTypeInputImage, "image_url": "https://example.com/dog.jpg"},
					},
				},
			},
		},
		ResponseWriter: recorder,
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// A parts/entries count mismatch stamps the pairs it can and leaves the rest
// untouched. Both walks read the same body, so neither direction is reachable
// through a configured pipeline; the behavior is pinned because the two
// directions fail differently. Surplus parts reach the worker with no uuid, so
// it hashes the image itself and never reads the entry primed into the encoder
// cache. Surplus entries leave mm_hashes naming an image the decode body does
// not identify.
func TestInjectMediaPartUUIDs_CountMismatch(t *testing.T) {
	newInput := func(parts int) []any {
		content := make([]any, 0, parts)
		for i := 0; i < parts; i++ {
			content = append(content, map[string]any{
				"type":      reqcommon.PartTypeInputImage,
				"image_url": fmt.Sprintf("data:image/jpeg;base64,img-%d", i),
			})
		}
		return []any{map[string]any{"role": "user", "content": content}}
	}
	newEntries := func(n int) []pipeline.MultimodalEntry {
		entries := make([]pipeline.MultimodalEntry, 0, n)
		for i := 0; i < n; i++ {
			entries = append(entries, pipeline.MultimodalEntry{Modality: reqcommon.ModalityImage, Hash: fmt.Sprintf("hash-%d", i)})
		}
		return entries
	}

	for _, tc := range []struct {
		name    string
		parts   int
		entries int
		// want is the expected uuid per part, nil where none must be stamped.
		want []any
	}{
		{name: "counts agree", parts: 2, entries: 2, want: []any{"hash-0", "hash-1"}},
		{name: "more parts than entries", parts: 3, entries: 2, want: []any{"hash-0", "hash-1", nil}},
		{name: "more entries than parts", parts: 1, entries: 3, want: []any{"hash-0"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := newInput(tc.parts)
			injectMediaPartUUIDs(input, reqcommon.APITypeResponses, newEntries(tc.entries), logr.Discard())

			content := input[0].(map[string]any)["content"].([]any)
			for i, want := range tc.want {
				if got := content[i].(map[string]any)["uuid"]; got != want {
					t.Errorf("part %d uuid = %v, want %v", i, got, want)
				}
			}
		})
	}
}
