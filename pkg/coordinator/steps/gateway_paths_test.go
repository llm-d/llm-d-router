package steps

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestGatewayPaths_EncodePrefillDecode(t *testing.T) {
	var mu sync.Mutex
	receivedPaths := []string{}

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		receivedPaths = append(receivedPaths, r.URL.Path)
		mu.Unlock()

		switch r.URL.Path {
		case "/encode/inference/v1/generate":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
			})
		case "/prefill/inference/v1/generate":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "b1"},
			})
		case "/decode/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}},
			})
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "unexpected path", 404)
		}
	}))
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	// --- Encode step ---
	encodeStep, _ := NewEncodeStep(map[string]any{"gateway_path": "/inference/v1/generate"})
	encodeStep.(*EncodeStep).SetGatewayClient(gwClient)

	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-path-test",
		OriginalPath: "/v1/chat/completions",
		Model:        "test-model",
		Stream:       false,
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		KVTransferParams: make(map[string]any),
		Body: map[string]any{
			"model":  "test-model",
			"stream": false,
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "https://example.com/img.jpg"},
						},
					},
				},
			},
		},
	}

	err := encodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	// --- Prefill step ---
	prefillStep, _ := NewPrefillStep(map[string]any{"gateway_path": "/inference/v1/generate"})
	prefillStep.(*PrefillStep).SetGatewayClient(gwClient)

	err = prefillStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	// --- Decode step ---
	decodeStep, _ := NewDecodeStep(map[string]any{})
	decodeStep.(*DecodeStep).SetGatewayClient(gwClient)

	recorder := httptest.NewRecorder()
	reqCtx.ResponseWriter = recorder
	reqCtx.Flusher = recorder

	err = decodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	// --- Validate paths ---
	mu.Lock()
	defer mu.Unlock()

	expectedPaths := []string{
		"/encode/inference/v1/generate",
		"/prefill/inference/v1/generate",
		"/decode/v1/chat/completions",
	}

	if len(receivedPaths) != len(expectedPaths) {
		t.Fatalf("expected %d requests, got %d: %v", len(expectedPaths), len(receivedPaths), receivedPaths)
	}

	for i, expected := range expectedPaths {
		if receivedPaths[i] != expected {
			t.Errorf("request %d: expected path %q, got %q", i, expected, receivedPaths[i])
		}
	}
}

func TestGatewayPaths_DecodeWithCompletionsEndpoint(t *testing.T) {
	var receivedPath string

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{}}})
	}))
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	decodeStep, _ := NewDecodeStep(map[string]any{})
	decodeStep.(*DecodeStep).SetGatewayClient(gwClient)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-2",
		OriginalPath:     "/v1/completions",
		Model:            "test",
		Stream:           false,
		KVTransferParams: map[string]any{"k": "v"},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1"},
		},
		Body:           map[string]any{"model": "test", "stream": false},
		ResponseWriter: recorder,
		Flusher:        recorder,
	}

	err := decodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	if receivedPath != "/decode/v1/completions" {
		t.Fatalf("expected /decode/v1/completions, got %s", receivedPath)
	}
}
