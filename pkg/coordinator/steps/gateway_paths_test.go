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
	"io"
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
	receivedPhases := []string{}

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		phase := r.Header.Get(gateway.EPPPhaseHeader)
		mu.Lock()
		receivedPhases = append(receivedPhases, phase)
		mu.Unlock()

		if r.URL.Path != gateway.PathChatCompletions {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "unexpected path", 404)
			return
		}

		switch phase {
		case gateway.PhaseEncode:
			body, _ := io.ReadAll(r.Body)
			var parsed map[string]any
			_ = json.Unmarshal(body, &parsed)
			tokens, _ := parsed["tokens"].(map[string]any)
			features, _ := tokens["features"].(map[string]any)
			mmHashes, _ := features["mm_hashes"].(map[string]any)
			imageHashes, _ := mmHashes[ModalityImage].([]any)
			hash, _ := imageHashes[0].(string)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{
					hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
				},
			})
		case gateway.PhasePrefill:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "b1", "peer_host": "10.0.0.2", "peer_port": 5502},
			})
		case gateway.PhaseDecode:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}},
			})
		default:
			t.Errorf("unexpected EPP-Phase: %s", phase)
			http.Error(w, "unexpected phase", 404)
		}
	}))
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	// --- Encode step ---
	encodeStep, _ := NewEncodeStep(gwClient, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-path-test",
		OriginalPath: gateway.PathChatCompletions,
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
	prefillStep, _ := NewPrefillStep(gwClient, map[string]any{})

	err = prefillStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	// --- Decode step ---
	decodeStep, _ := NewDecodeStep(gwClient, map[string]any{})

	recorder := httptest.NewRecorder()
	reqCtx.ResponseWriter = recorder

	err = decodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	// --- Validate EPP-Phase headers ---
	mu.Lock()
	defer mu.Unlock()

	expectedPhases := []string{
		gateway.PhaseEncode,
		gateway.PhasePrefill,
		gateway.PhaseDecode,
	}

	if len(receivedPhases) != len(expectedPhases) {
		t.Fatalf("expected %d requests, got %d: %v", len(expectedPhases), len(receivedPhases), receivedPhases)
	}

	for i, expected := range expectedPhases {
		if receivedPhases[i] != expected {
			t.Errorf("request %d: expected EPP-Phase %q, got %q", i, expected, receivedPhases[i])
		}
	}
}

func TestGatewayPaths_CompletionsPreservedWhenOpenAIFormatDisabled(t *testing.T) {
	var mu sync.Mutex
	receivedPaths := []string{}

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		receivedPaths = append(receivedPaths, r.URL.Path)
		mu.Unlock()

		phase := r.Header.Get(gateway.EPPPhaseHeader)
		switch phase {
		case gateway.PhaseEncode:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{
					"h1": map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
				},
			})
		case gateway.PhasePrefill:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "b1"},
			})
		case gateway.PhaseDecode:
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{}}})
		}
	}))
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-openai-false",
		OriginalPath: gateway.PathCompletions,
		Model:        "test-model",
		Stream:       false,
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		KVTransferParams: make(map[string]any),
		Body:             map[string]any{"model": "test-model", "stream": false, "prompt": "hello"},
	}

	encodeStep, _ := NewEncodeStep(gwClient, map[string]any{"use_openai_format": false})
	if err := encodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	prefillStep, _ := NewPrefillStep(gwClient, map[string]any{"use_openai_format": false})
	if err := prefillStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	recorder := httptest.NewRecorder()
	reqCtx.ResponseWriter = recorder

	decodeStep, _ := NewDecodeStep(gwClient, map[string]any{"use_openai_format": false})
	if err := decodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	for i, path := range receivedPaths {
		if path != gateway.PathCompletions {
			t.Errorf("request %d: expected /v1/completions, got %s", i, path)
		}
	}
}

func TestGatewayPaths_DecodeWithCompletionsEndpoint(t *testing.T) {
	var receivedPath string
	var receivedPhase string

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		receivedPhase = r.Header.Get(gateway.EPPPhaseHeader)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{}}})
	}))
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	decodeStep, _ := NewDecodeStep(gwClient, map[string]any{})

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-2",
		OriginalPath:     gateway.PathCompletions,
		Model:            "test",
		Stream:           false,
		TokenIDs:         []int{1, 2345, 6789},
		KVTransferParams: map[string]any{"k": "v"},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1"},
		},
		Body:           map[string]any{"model": "test", "stream": false},
		ResponseWriter: recorder,
	}

	err := decodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	if receivedPath != gateway.PathCompletions {
		t.Fatalf("expected /v1/completions, got %s", receivedPath)
	}
	if receivedPhase != gateway.PhaseDecode {
		t.Fatalf("expected EPP-Phase: decode, got %q", receivedPhase)
	}
}
