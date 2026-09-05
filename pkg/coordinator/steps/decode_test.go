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

		// Verify tokens field present for chat completions format
		tokens, ok := parsed["tokens"].(map[string]any)
		if !ok {
			t.Fatal("expected tokens field in chat/completions decode request")
		}
		tokenIDs, _ := tokens["token_ids"].([]any)
		if len(tokenIDs) != 5 {
			t.Fatalf("expected 5 token_ids in tokens field, got %d", len(tokenIDs))
		}

		// Verify uuid was injected into the image_url content part
		messages := parsed["messages"].([]any)
		msg := messages[0].(map[string]any)
		content := msg["content"].([]any)
		imgPart := content[0].(map[string]any)
		if imgPart["uuid"] != "hash-a" {
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
			{Index: 0, Modality: ModalityImage, Hash: "hash-a", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
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
		OriginalPath:     gateway.PathCompletions,
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

// TestDecodeStep_GenerateFormat_NestsKVInExtraArgs verifies that for the
// /inference/v1/generate format the decode step places kv_transfer_params
// inside sampling_params.extra_args (the only place the engine reads them)
// rather than at the top level, and preserves the client's sampling_params so
// the decode generation honors the requested max_tokens.
func TestDecodeStep_GenerateFormat_NestsKVInExtraArgs(t *testing.T) {
	var parsed map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &parsed)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": "ok"}}})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{"use_openai_format": false, ParamKVConnector: kv.NIXL})
	if err != nil {
		t.Fatal(err)
	}

	wantBlockID := "block-gen-1"
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-gen",
		OriginalPath:     gateway.DefaultGeneratePath,
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

	// kv_transfer_params must not be at the top level in generate format.
	if _, ok := parsed["kv_transfer_params"]; ok {
		t.Fatal("generate format should not have top-level kv_transfer_params")
	}

	sampling, ok := parsed["sampling_params"].(map[string]any)
	if !ok {
		t.Fatal("expected sampling_params in decode body")
	}
	// Client sampling fields are preserved (decode honors the real max_tokens).
	if sampling["max_tokens"] != float64(50) {
		t.Fatalf("expected sampling_params.max_tokens=50 preserved, got %v", sampling["max_tokens"])
	}
	extraArgs, ok := sampling["extra_args"].(map[string]any)
	if !ok {
		t.Fatal("expected sampling_params.extra_args in generate format")
	}
	kvParams, ok := extraArgs["kv_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected kv_transfer_params in sampling_params.extra_args")
	}
	if kvParams["block_id"] != wantBlockID {
		t.Errorf("kv_transfer_params.block_id = %v, want %v", kvParams["block_id"], wantBlockID)
	}
	if kvParams["do_remote_prefill"] != true {
		t.Errorf("kv_transfer_params.do_remote_prefill = %v, want true", kvParams["do_remote_prefill"])
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
			{Index: 0, Modality: ModalityImage, Hash: "h1"},
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
			{Index: 0, Modality: ModalityImage, Hash: "h1"},
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

// TestInjectUUIDs_TagsAllMediaParts asserts every recognized media
// content-part type receives a uuid tag matching its (modality, local index)
// entry in MultimodalEntries. Non-media parts (text, unknown types) are
// left alone.
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
			{Index: 0, Modality: ModalityImage, Hash: "H-img"},
			{Index: 1, Modality: ModalityAudio, Hash: "H-aud-url"},
			{Index: 2, Modality: ModalityVideo, Hash: "H-vid"},
			{Index: 3, Modality: ModalityAudio, Hash: "H-input-audio"},
		},
	}
	step.injectUUIDs(reqCtx)

	if got := imagePart["uuid"]; got != "H-img" {
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

// TestInjectUUIDs_TagsRepeatedModalityInOrder asserts that two audio parts
// in the same request receive the hashes of the two audio entries, in
// walker order.
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
			{Index: 0, Modality: ModalityAudio, Hash: "H0"},
			{Index: 1, Modality: ModalityAudio, Hash: "H1"},
		},
	}
	step.injectUUIDs(reqCtx)
	if got := aud0["uuid"]; got != "H0" {
		t.Errorf("audio[0] uuid = %v, want H0", got)
	}
	if got := aud1["uuid"]; got != "H1" {
		t.Errorf("audio[1] uuid = %v, want H1", got)
	}
}

// TestInjectUUIDs_SkipsMalformedParts asserts that content parts
// replace_media_urls silently drops (missing/null inner map, non-string
// url, empty input_audio data) also do not consume a slot here. The one
// well-formed part of each modality must receive its entry's hash even
// when a malformed part appears earlier in the same message.
func TestInjectUUIDs_SkipsMalformedParts(t *testing.T) {
	step := &DecodeStep{}
	malformedImg := map[string]any{"type": "image_url", "image_url": map[string]any{"url": nil}}
	goodImg := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "u-img"}}
	malformedInputAudio := map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "", "format": "wav"}}
	goodInputAudio := map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "AA==", "format": "wav"}}

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role":    "user",
					"content": []any{malformedImg, goodImg, malformedInputAudio, goodInputAudio},
				},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Modality: ModalityImage, Hash: "H-img"},
			{Index: 1, Modality: ModalityAudio, Hash: "H-aud"},
		},
	}
	step.injectUUIDs(reqCtx)

	if _, tagged := malformedImg["uuid"]; tagged {
		t.Errorf("malformed image_url must not be tagged: %+v", malformedImg)
	}
	if got := goodImg["uuid"]; got != "H-img" {
		t.Errorf("good image_url uuid = %v, want H-img", got)
	}
	if _, tagged := malformedInputAudio["uuid"]; tagged {
		t.Errorf("malformed input_audio must not be tagged: %+v", malformedInputAudio)
	}
	if got := goodInputAudio["uuid"]; got != "H-aud" {
		t.Errorf("good input_audio uuid = %v, want H-aud", got)
	}
}
