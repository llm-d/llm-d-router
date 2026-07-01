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
	"sync/atomic"
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/connectors/ec"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestEncodeStep_ParallelFanOut(t *testing.T) {
	var requestCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)

		if r.Header.Get(gateway.EPPPhaseHeader) != gateway.PhaseEncode {
			t.Errorf("expected EPP-Phase: encode, got %q", r.Header.Get(gateway.EPPPhaseHeader))
		}

		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)

		// Verify model is present (required by /inference/v1/generate validator)
		if parsed["model"] != testModelName {
			t.Errorf("expected model=%s in encode request, got %v", testModelName, parsed["model"])
		}

		// Verify token_ids present
		tokenIDs, ok := parsed["token_ids"].([]any)
		if !ok || len(tokenIDs) == 0 {
			t.Errorf("expected token_ids in encode request")
		}

		// Verify features structure
		features, ok := parsed["features"].(map[string]any)
		if !ok {
			t.Errorf("expected features in encode request")
		}
		mmHashes, _ := features["mm_hashes"].(map[string]any)
		imageHashes, _ := mmHashes[ModalityImage].([]any)
		if len(imageHashes) != 1 {
			t.Errorf("expected 1 hash per encode request, got %d", len(imageHashes))
		}
		kwargsData, _ := features["kwargs_data"].(map[string]any)
		imageKwargs, _ := kwargsData[ModalityImage].([]any)
		if len(imageKwargs) != 1 {
			t.Errorf("expected 1 kwargs_data per encode request, got %d", len(imageKwargs))
		}

		// Echo the per-image hash back as the ec_transfer_params key
		hash, _ := imageHashes[0].(string)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ec_transfer_params": map[string]any{
				hash: map[string]any{
					"peer_host":               "10.0.0.1",
					"peer_port":               5501,
					"size_bytes":              2359296,
					"nixl_agent_metadata_b64": "TklYTA==",
				},
			},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, err := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		"max_parallel":      4,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-1",
		Model:     testModelName,
		TokenIDs:  []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-a", KwargsData: "dGVuc29yLWE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Index: 1, Hash: "hash-b", KwargsData: "dGVuc29yLWI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
			{Index: 2, Hash: "hash-c", KwargsData: "dGVuc29yLWM=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
		},
	}

	err = step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if int(requestCount.Load()) != 3 {
		t.Fatalf("expected 3 gateway requests, got %d", requestCount.Load())
	}
	if len(reqCtx.ECTransferParams) != 3 {
		t.Fatalf("expected 3 ec_transfer_params entries, got %d", len(reqCtx.ECTransferParams))
	}

	seen := make(map[string]bool)
	for i, entry := range reqCtx.ECTransferParams {
		if len(entry) != 1 {
			t.Fatalf("entry %d: expected single-key map, got %d keys: %v", i, len(entry), entry)
		}
		for hash, param := range entry {
			seen[hash] = true
			paramMap, ok := param.(map[string]any)
			if !ok {
				t.Fatalf("entry %s: not a map: %T", hash, param)
			}
			if paramMap["peer_host"] != "10.0.0.1" {
				t.Fatalf("entry %s: unexpected peer_host: %v", hash, paramMap["peer_host"])
			}
		}
	}
	for _, want := range []string{"hash-a", "hash-b", "hash-c"} {
		if !seen[want] {
			t.Errorf("missing key %q in merged ECTransferParams: %v", want, reqCtx.ECTransferParams)
		}
	}
}

// TestEncodeStep_SkipsInvalidECTransferParams verifies that an encoder
// response whose ec_transfer_params is present but unusable (non-object,
// explicit null, or empty object) is skipped rather than failing the encode,
// matching the sidecar EC-NIXL proxy. Each case must succeed and record no
// transfer params. The missing-field case is covered by
// TestEncodeStep_EncoderReturnsNoECParams.
func TestEncodeStep_SkipsInvalidECTransferParams(t *testing.T) {
	cases := []struct {
		name  string
		value any
	}{
		{name: "NonObjectString", value: "not-an-object"},
		{name: "NonObjectArray", value: []any{1, 2}},
		{name: "ExplicitNull", value: nil},
		{name: "EmptyObject", value: map[string]any{}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{"ec_transfer_params": tc.value})
			}))
			defer server.Close()

			step, err := NewEncodeStep(gateway.New(config.GatewayConfig{Address: server.URL}), map[string]any{
				"use_openai_format": false,
				ParamECConnector:    ec.NIXL,
			})
			if err != nil {
				t.Fatal(err)
			}

			reqCtx := &pipeline.RequestContext{
				RequestID: "req-1",
				Model:     testModelName,
				TokenIDs:  []int{1, 32000, 32000, 2345},
				MultimodalEntries: []pipeline.MultimodalEntry{
					{Index: 0, Hash: "hash-a", KwargsData: "dGVuc29yLWE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
				},
			}

			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("invalid ec_transfer_params should be skipped, not fail the encode: %v", err)
			}
			if len(reqCtx.ECTransferParams) != 0 {
				t.Fatalf("expected no ec_transfer_params recorded, got %v", reqCtx.ECTransferParams)
			}
		})
	}
}

func TestEncodeStep_PartialFailure(t *testing.T) {
	var count atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := count.Add(1)
		body, _ := io.ReadAll(r.Body)
		if n == 2 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("encode failed"))
			return
		}
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)
		features, _ := parsed["features"].(map[string]any)
		mmHashes, _ := features["mm_hashes"].(map[string]any)
		imageHashes, _ := mmHashes[ModalityImage].([]any)
		hash, _ := imageHashes[0].(string)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ec_transfer_params": map[string]any{
				hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
			},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, _ := NewEncodeStep(gwClient, map[string]any{"max_parallel": 1, "use_openai_format": false})

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-2",
		Model:     "test",
		TokenIDs:  []int{1, 32000, 32000, 32000},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Index: 1, Hash: "h2", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Index: 2, Hash: "h3", KwargsData: "dDM=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error when one encode fails")
	}
}

func TestEncodeStep_ChatCompletionsFormat(t *testing.T) {
	var receivedBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(gateway.EPPPhaseHeader) != gateway.PhaseEncode {
			t.Fatalf("expected EPP-Phase: encode, got %q", r.Header.Get(gateway.EPPPhaseHeader))
		}

		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &receivedBody)

		// Extract hash from tokens.features
		tokens, _ := receivedBody["tokens"].(map[string]any)
		features, _ := tokens["features"].(map[string]any)
		mmHashes, _ := features["mm_hashes"].(map[string]any)
		imageHashes, _ := mmHashes[ModalityImage].([]any)
		hash, _ := imageHashes[0].(string)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ec_transfer_params": map[string]any{
				hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
			},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewEncodeStep(gwClient, map[string]any{
		ParamECConnector: ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-chat",
		OriginalPath: gateway.PathChatCompletions,
		Model:        testModelName,
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		Body: map[string]any{
			"model":  testModelName,
			"stream": false,
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "describe"},
						map[string]any{"type": imageURLPartType, imageURLPartType: map[string]any{"url": "data:image/jpeg;base64,abc"}},
					},
				},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-x", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
	}

	err = step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify model present
	if receivedBody["model"] != testModelName {
		t.Fatalf("expected model from body, got %v", receivedBody["model"])
	}

	// Verify messages contains only image (no text) in per-image body
	messages, ok := receivedBody["messages"].([]any)
	if !ok {
		t.Fatal("expected messages in chat/completions format")
	}
	msg := messages[0].(map[string]any)
	content := msg["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content part (image only), got %d", len(content))
	}
	part := content[0].(map[string]any)
	if part["type"] != imageURLPartType {
		t.Fatalf("expected %s content part, got %v", imageURLPartType, part["type"])
	}

	// Verify tokens nested field
	tokens, ok := receivedBody["tokens"].(map[string]any)
	if !ok {
		t.Fatal("expected tokens field in chat/completions format")
	}
	tokenIDs, _ := tokens["token_ids"].([]any)
	if len(tokenIDs) != 4 { // BOS + 3 placeholders
		t.Fatalf("expected 4 token_ids in tokens, got %d", len(tokenIDs))
	}
	tokensFeatures, ok := tokens["features"].(map[string]any)
	if !ok {
		t.Fatal("expected features in tokens field")
	}
	// tokens.features should NOT have kwargs_data
	if _, ok := tokensFeatures["kwargs_data"]; ok {
		t.Fatal("tokens.features should not have kwargs_data in chat format")
	}
	if _, ok := tokensFeatures["mm_hashes"]; !ok {
		t.Fatal("tokens.features should have mm_hashes")
	}

	// Verify no top-level token_ids or features
	if _, ok := receivedBody["token_ids"]; ok {
		t.Fatal("chat format should not have top-level token_ids")
	}
	if _, ok := receivedBody["features"]; ok {
		t.Fatal("chat format should not have top-level features")
	}
}

// TestEncodeStep_TextOnly verifies that Execute returns immediately without any
// gateway calls when MultimodalEntries is empty (text-only request). ECTransferParams
// must remain nil so the prefill step emits no ec_transfer_params field.
func TestEncodeStep_TextOnly(t *testing.T) {
	gatewayCallCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gatewayCallCount++
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewEncodeStep(gwClient, map[string]any{ParamECConnector: ec.NIXL})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		RequestID:         "req-text-only",
		Model:             "test-model",
		TokenIDs:          []int{1, 42, 43, 2},
		MultimodalEntries: nil,
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gatewayCallCount != 0 {
		t.Fatalf("expected no gateway calls for text-only request, got %d", gatewayCallCount)
	}
	if reqCtx.ECTransferParams != nil {
		t.Fatalf("expected nil ECTransferParams for text-only request, got %v", reqCtx.ECTransferParams)
	}
}

// TestEncodeStep_EncoderReturnsNoECParams verifies the all-missing degradation path:
// when every encoder response omits ec_transfer_params, MergeEncodeResponse skips each
// entry and ECTransferParams stays nil, so the prefill step forwards the request without
// the field. The encode step must not error -- missing metadata is warn-and-continue.
func TestEncodeStep_EncoderReturnsNoECParams(t *testing.T) {
	var requestCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)
		// 2xx with no ec_transfer_params field.
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": ""}}},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-no-ec",
		Model:     "test-model",
		TokenIDs:  []int{1, 32000, 32000, 2},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-a", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
			{Index: 1, Hash: "hash-b", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("missing ec_transfer_params must not fail the encode step: %v", err)
	}
	if int(requestCount.Load()) != 2 {
		t.Fatalf("expected 2 gateway requests, got %d", requestCount.Load())
	}
	if len(reqCtx.ECTransferParams) != 0 {
		t.Fatalf("expected empty ECTransferParams when all encoders return no ec params, got %v", reqCtx.ECTransferParams)
	}
}

func TestEncodeStep_BuildsCorrectTokenIDs(t *testing.T) {
	var receivedTokenIDs []any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)
		receivedTokenIDs, _ = parsed["token_ids"].([]any)
		features, _ := parsed["features"].(map[string]any)
		mmHashes, _ := features["mm_hashes"].(map[string]any)
		imageHashes, _ := mmHashes[ModalityImage].([]any)
		hash, _ := imageHashes[0].(string)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ec_transfer_params": map[string]any{
				hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
			},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, _ := NewEncodeStep(gwClient, map[string]any{"use_openai_format": false})

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-tok",
		Model:     "test",
		TokenIDs:  []int{1, 32000, 32000, 32000, 2345, 6789},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should be BOS(1) + 3 placeholder tokens(32000)
	if len(receivedTokenIDs) != 4 {
		t.Fatalf("expected 4 token_ids (BOS + 3 placeholders), got %d", len(receivedTokenIDs))
	}
	if receivedTokenIDs[0] != float64(1) {
		t.Fatalf("expected BOS=1, got %v", receivedTokenIDs[0])
	}
	for i := 1; i < 4; i++ {
		if receivedTokenIDs[i] != float64(32000) {
			t.Fatalf("expected placeholder=32000 at index %d, got %v", i, receivedTokenIDs[i])
		}
	}
}
