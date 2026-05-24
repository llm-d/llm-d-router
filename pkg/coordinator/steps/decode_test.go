package steps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestDecodeStep_NonStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != testChatCompletionsPath {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get(gateway.EPPPhaseHeader) != gateway.PhaseDecode {
			t.Fatalf("expected EPP-Phase: decode, got %q", r.Header.Get(gateway.EPPPhaseHeader))
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

	step, err := NewDecodeStep(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	step.(*DecodeStep).SetGatewayClient(gwClient)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-1",
		OriginalPath: testChatCompletionsPath,
		Model:        "llama-3",
		Stream:       false,
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-a", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
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

	step, _ := NewDecodeStep(map[string]any{})
	step.(*DecodeStep).SetGatewayClient(gwClient)

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

	step, _ := NewDecodeStep(map[string]any{})
	step.(*DecodeStep).SetGatewayClient(gwClient)

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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
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
