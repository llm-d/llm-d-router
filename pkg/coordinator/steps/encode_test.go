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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestEncodeStep_ParallelFanOut(t *testing.T) {
	var requestCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)

		if r.Header.Get(gateway.EPPProfileHeader) != gateway.PhaseEncode {
			t.Errorf("expected EPP-Profile: encode, got %q", r.Header.Get(gateway.EPPProfileHeader))
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
		if r.Header.Get(gateway.EPPProfileHeader) != gateway.PhaseEncode {
			t.Fatalf("expected EPP-Profile: encode, got %q", r.Header.Get(gateway.EPPProfileHeader))
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

// TestEncodeStep_ChatCompletionsFormat_CapsMaxCompletionTokens is a
// The encode chat sub-request is built fresh from the request context and does
// not carry the client's sampling fields, so max_completion_tokens is not
// propagated and is never injected: max_tokens=1 alone caps output.
func TestEncodeStep_ChatCompletionsFormat_OmitsMaxCompletionTokens(t *testing.T) {
	var receivedBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &receivedBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ec_transfer_params": map[string]any{"hash-b": map[string]any{"peer_port": 5501}},
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
		RequestID:    "req-chat-max-completion-tokens",
		OriginalPath: gateway.PathChatCompletions,
		Model:        testModelName,
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		Body: map[string]any{
			"model":                 testModelName,
			"max_completion_tokens": 100,
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": imageURLPartType, imageURLPartType: map[string]any{"url": "data:image/jpeg;base64,abc"}},
					},
				},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-b", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if receivedBody["max_tokens"] != float64(1) {
		t.Fatalf("expected encode sub-request max_tokens capped to 1, got %v", receivedBody["max_tokens"])
	}
	if _, ok := receivedBody["max_completion_tokens"]; ok {
		t.Fatalf("expected encode sub-request to omit max_completion_tokens, got %v", receivedBody["max_completion_tokens"])
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

// TestEncodeStep_SkipsForGenerate verifies that Execute makes no gateway calls
// and leaves ECTransferParams nil for a /inference/v1/generate request even when
// multimodal entries are present: the prefill worker runs the vision encoder
// inline, so the encode fan-out and EC handoff are skipped.
func TestEncodeStep_SkipsForGenerate(t *testing.T) {
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
		RequestID:    "req-generate",
		Model:        "test-model",
		OriginalPath: gateway.DefaultGeneratePath,
		TokenIDs:     []int{1, 32000, 32000, 2},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-a", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gatewayCallCount != 0 {
		t.Fatalf("expected no gateway calls for generate request, got %d", gatewayCallCount)
	}
	if reqCtx.ECTransferParams != nil {
		t.Fatalf("expected nil ECTransferParams for generate request, got %v", reqCtx.ECTransferParams)
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

// TestEncodeStep_GenerateFormat_CapsSingleToken verifies the generate-format
// encoder sub-request caps output to a single token: sampling_params carries
// max_tokens=1 and strips min_tokens (it defaults to 0, keeping min_tokens <=
// max_tokens).
func TestEncodeStep_GenerateFormat_CapsSingleToken(t *testing.T) {
	var samplingParams map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)
		samplingParams, _ = parsed["sampling_params"].(map[string]any)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ec_transfer_params": map[string]any{"h1": map[string]any{"peer_port": 5501}},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, _ := NewEncodeStep(gwClient, map[string]any{"use_openai_format": false})

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-gen-cap",
		Model:     "test",
		TokenIDs:  []int{1, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if samplingParams["max_tokens"] != float64(1) {
		t.Fatalf("expected sampling_params.max_tokens=1, got %v", samplingParams["max_tokens"])
	}
	if _, ok := samplingParams["min_tokens"]; ok {
		t.Fatalf("expected sampling_params.min_tokens to be stripped, got %v", samplingParams["min_tokens"])
	}
}

// ---- multimodal encoder fanout ---------------------------------------------

// TestCollectMediaParts_MixedModalities asserts the walker returns per-modality
// lists of parts in walker order. Order within each modality is the original
// request's discovery order for that modality.
func TestCollectMediaParts_MixedModalities(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "describe"},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "u1"}},
					map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "u2"}},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "u3"}},
					map[string]any{"type": "video_url", "video_url": map[string]any{"url": "u4"}},
					map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "d5", "format": "wav"}},
				},
			},
		},
	}
	partsByMod := collectMediaParts(body)
	if got := len(partsByMod[ModalityImage]); got != 2 {
		t.Errorf("image parts = %d, want 2", got)
	}
	if got := len(partsByMod[ModalityAudio]); got != 2 {
		t.Errorf("audio parts = %d, want 2 (audio_url + input_audio)", got)
	}
	if got := len(partsByMod[ModalityVideo]); got != 1 {
		t.Errorf("video parts = %d, want 1", got)
	}
	// audio_url comes before input_audio (walker order in request).
	if url, _ := partsByMod[ModalityAudio][0]["audio_url"].(map[string]any); url["url"] != "u2" {
		t.Errorf("audio[0] not the audio_url part: %+v", partsByMod[ModalityAudio][0])
	}
	if data, _ := partsByMod[ModalityAudio][1]["input_audio"].(map[string]any); data["data"] != "d5" {
		t.Errorf("audio[1] not the input_audio part: %+v", partsByMod[ModalityAudio][1])
	}
}

// TestCollectMediaParts_SkipsMalformedParts asserts that content parts
// replace_media_urls silently skips (missing/null inner map, non-string
// url, empty input_audio data) are also skipped here, so per-modality
// indexing stays aligned with MultimodalEntries. Without this alignment,
// a valid entry would pair with a malformed part in encode fanout.
func TestCollectMediaParts_SkipsMalformedParts(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					// malformed: inner map missing -> replace_media_urls skips
					map[string]any{"type": "image_url"},
					// malformed: url is nil -> replace_media_urls skips
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": nil}},
					// well-formed
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "u-good"}},
					// malformed audio_url: url is a number
					map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": 42}},
					// well-formed audio_url
					map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "au-good"}},
					// malformed input_audio: empty data
					map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "", "format": "wav"}},
					// malformed input_audio: no inner map
					map[string]any{"type": "input_audio"},
					// well-formed input_audio
					map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "AA==", "format": "wav"}},
				},
			},
		},
	}
	partsByMod := collectMediaParts(body)
	if got := len(partsByMod[ModalityImage]); got != 1 {
		t.Fatalf("image parts = %d, want 1 (malformed skipped)", got)
	}
	if got := len(partsByMod[ModalityAudio]); got != 2 {
		t.Fatalf("audio parts = %d, want 2 (audio_url + input_audio, malformed skipped)", got)
	}
	if url, _ := partsByMod[ModalityImage][0]["image_url"].(map[string]any); url["url"] != "u-good" {
		t.Errorf("image[0] = %+v, want the well-formed part", partsByMod[ModalityImage][0])
	}
	if url, _ := partsByMod[ModalityAudio][0]["audio_url"].(map[string]any); url["url"] != "au-good" {
		t.Errorf("audio[0] = %+v, want the well-formed audio_url", partsByMod[ModalityAudio][0])
	}
	if data, _ := partsByMod[ModalityAudio][1]["input_audio"].(map[string]any); data["data"] != "AA==" {
		t.Errorf("audio[1] = %+v, want the well-formed input_audio", partsByMod[ModalityAudio][1])
	}
}

// TestBuildSingleMediaContent_PerModality asserts each modality's content
// part is emitted in its native OpenAI shape.
func TestBuildSingleMediaContent_PerModality(t *testing.T) {
	partsByMod := map[string][]map[string]any{
		ModalityImage: {
			{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,IMG"}},
		},
		ModalityAudio: {
			{"type": "audio_url", "audio_url": map[string]any{"url": "data:audio/wav;base64,AUD"}},
			{"type": "input_audio", "input_audio": map[string]any{"data": "IA==", "format": "wav"}},
		},
		ModalityVideo: {
			{"type": "video_url", "video_url": map[string]any{"url": "data:video/mp4;base64,VID"}},
		},
	}

	for _, tc := range []struct {
		name     string
		modality string
		localIdx int
		wantType string
		wantKey  string
	}{
		{"image", ModalityImage, 0, "image_url", "image_url"},
		{"audio_url", ModalityAudio, 0, "audio_url", "audio_url"},
		{"input_audio", ModalityAudio, 1, "input_audio", "input_audio"},
		{"video_url", ModalityVideo, 0, "video_url", "video_url"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := buildSingleMediaContent(partsByMod, tc.modality, tc.localIdx)
			if got["type"] != tc.wantType {
				t.Errorf("type = %v, want %q", got["type"], tc.wantType)
			}
			if _, ok := got[tc.wantKey]; !ok {
				t.Errorf("inner key %q missing: %+v", tc.wantKey, got)
			}
		})
	}
}

// TestBuildSingleMediaContent_OutOfRangeFallback asserts a bug-safe empty
// URL part of the matching modality is returned when localIdx is out of
// range. The path is not expected to run under correct entry<->part
// pairing; the fallback must at least keep the emitted sub-request
// self-consistent (audio entry -> audio part shape).
func TestBuildSingleMediaContent_OutOfRangeFallback(t *testing.T) {
	partsByMod := map[string][]map[string]any{
		ModalityImage: {
			{"type": "image_url", "image_url": map[string]any{"url": "u0"}},
		},
	}
	for _, tc := range []struct {
		name     string
		modality string
		wantType string
	}{
		{"image", ModalityImage, imageURLPartType},
		{"audio", ModalityAudio, audioURLPartType},
		{"video", ModalityVideo, videoURLPartType},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// 99 is out of range for every modality in partsByMod, so the
			// fallback path runs regardless of which modality is under test.
			got := buildSingleMediaContent(partsByMod, tc.modality, 99)
			if got["type"] != tc.wantType {
				t.Errorf("type = %v, want %q", got["type"], tc.wantType)
			}
			inner, ok := got[tc.wantType].(map[string]any)
			if !ok {
				t.Fatalf("inner key %q missing or wrong type: %+v", tc.wantType, got)
			}
			if inner["url"] != "" {
				t.Errorf("fallback url = %v, want empty", inner["url"])
			}
		})
	}
}

// TestEncodeStep_MixedModalityFanout drives the full encode step with a
// mixed-modality request. Each entry produces one fanout sub-request; each
// sub-request carries the correct modality-keyed feature map and the
// matching content part in messages[0].content.
func TestEncodeStep_MixedModalityFanout(t *testing.T) {
	var seq atomic.Int32
	captured := make(map[string]map[string]any) // request-body seq -> parsed body
	var mu sync.Mutex
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read encoder body: %v", err)
		}
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)
		i := int(seq.Add(1) - 1)
		mu.Lock()
		captured[fmt.Sprintf("req-%d", i)] = parsed
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
	}))
	defer encoderBackend.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: encoderBackend.URL})
	encodeStep, err := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": true,
	})
	if err != nil {
		t.Fatalf("NewEncodeStep: %v", err)
	}
	reqCtx := &pipeline.RequestContext{
		RequestID:    "mixed-fanout",
		Model:        "llama-3",
		OriginalPath: gateway.PathChatCompletions,
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		Body: map[string]any{
			"model":  "llama-3",
			"stream": false,
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,IMG"}},
						map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "data:audio/wav;base64,AUD"}},
						map[string]any{"type": "video_url", "video_url": map[string]any{"url": "data:video/mp4;base64,VID"}},
					},
				},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Modality: ModalityImage, Hash: "img-hash", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
			{Index: 1, Modality: ModalityAudio, Hash: "aud-hash", Placeholder: pipeline.PlaceholderRange{Offset: 2, Length: 1}},
			{Index: 2, Modality: ModalityVideo, Hash: "vid-hash", Placeholder: pipeline.PlaceholderRange{Offset: 3, Length: 1}},
		},
		KVTransferParams: make(map[string]any),
	}

	if err := encodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	if seq.Load() != 3 {
		t.Fatalf("expected 3 fanout requests, got %d", seq.Load())
	}
	// Collect the modality keys we saw across the three requests, one per
	// entry, keyed by its modality.
	sawKeys := map[string]bool{}
	sawTypes := map[string]bool{}
	for _, body := range captured {
		tokens, _ := body["tokens"].(map[string]any)
		features, _ := tokens["features"].(map[string]any)
		hashes, _ := features["mm_hashes"].(map[string]any)
		for k := range hashes {
			sawKeys[k] = true
		}
		msgs, _ := body["messages"].([]any)
		content, _ := msgs[0].(map[string]any)["content"].([]any)
		if part, ok := content[0].(map[string]any); ok {
			if pt, ok := part["type"].(string); ok {
				sawTypes[pt] = true
			}
		}
	}
	for _, m := range []string{ModalityImage, ModalityAudio, ModalityVideo} {
		if !sawKeys[m] {
			t.Errorf("no fanout request carried mm_hashes[%q]", m)
		}
	}
	for _, pt := range []string{"image_url", "audio_url", "video_url"} {
		if !sawTypes[pt] {
			t.Errorf("no fanout content used type=%q", pt)
		}
	}
}
