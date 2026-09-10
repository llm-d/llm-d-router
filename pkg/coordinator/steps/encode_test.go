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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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
			{Modality: ModalityImage, Hash: "hash-a", KwargsData: "dGVuc29yLWE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: ModalityImage, Hash: "hash-b", KwargsData: "dGVuc29yLWI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
			{Modality: ModalityImage, Hash: "hash-c", KwargsData: "dGVuc29yLWM=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
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
					{Modality: ModalityImage, Hash: "hash-a", KwargsData: "dGVuc29yLWE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
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
			{Modality: ModalityImage, Hash: "h1", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: ModalityImage, Hash: "h2", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: ModalityImage, Hash: "h3", KwargsData: "dDM=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
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
			{Modality: ModalityImage, Hash: "hash-x", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
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

// TestEncodeStep_ChatCompletionsFormat_CapsMaxCompletionTokens verifies the
// encode chat sub-request carries max_completion_tokens=1 unconditionally
// (via capSingleTokenOutput/reqcommon.PrimeSingleTokenRequest), even though the
// sub-request is built fresh from the request context and never copies the
// client's own max_completion_tokens value.
func TestEncodeStep_ChatCompletionsFormat_CapsMaxCompletionTokens(t *testing.T) {
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
			{Modality: ModalityImage, Hash: "hash-b", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if receivedBody["max_tokens"] != float64(1) {
		t.Fatalf("expected encode sub-request max_tokens capped to 1, got %v", receivedBody["max_tokens"])
	}
	if receivedBody["max_completion_tokens"] != float64(1) {
		t.Fatalf("expected encode sub-request max_completion_tokens capped to 1, got %v", receivedBody["max_completion_tokens"])
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
			{Modality: ModalityImage, Hash: "hash-a", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
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
			{Modality: ModalityImage, Hash: "hash-a", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
			{Modality: ModalityImage, Hash: "hash-b", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
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
			{Modality: ModalityImage, Hash: "h1", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
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
			{Modality: ModalityImage, Hash: "h1", KwargsData: "dGVzdA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
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
			got, err := buildSingleMediaContent(partsByMod, tc.modality, tc.localIdx)
			if err != nil {
				t.Fatalf("in-range lookup returned error: %v", err)
			}
			if got["type"] != tc.wantType {
				t.Errorf("type = %v, want %q", got["type"], tc.wantType)
			}
			if _, ok := got[tc.wantKey]; !ok {
				t.Errorf("inner key %q missing: %+v", tc.wantKey, got)
			}
		})
	}
}

// TestBuildSingleMediaContent_OutOfRangeErrors asserts an out-of-range
// localIdx is an error rather than a stand-in content part. The path only
// runs when entries and parts got out of line upstream, which is a
// coordinator bug; there is nothing worth sending the encoder in that
// case, and the error names the modality so the miss is traceable.
func TestBuildSingleMediaContent_OutOfRangeErrors(t *testing.T) {
	partsByMod := map[string][]map[string]any{
		ModalityImage: {
			{"type": "image_url", "image_url": map[string]any{"url": "u0"}},
		},
	}
	for _, tc := range []struct {
		name     string
		modality string
		localIdx int
	}{
		{"image past end", ModalityImage, 99},
		{"audio absent", ModalityAudio, 0},
		{"video absent", ModalityVideo, 0},
		{"negative index", ModalityImage, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := buildSingleMediaContent(partsByMod, tc.modality, tc.localIdx)
			if err == nil {
				t.Fatalf("expected error, got content %+v", got)
			}
			if got != nil {
				t.Errorf("expected nil content alongside the error, got %+v", got)
			}
			if !strings.Contains(err.Error(), tc.modality) {
				t.Errorf("error %q should name the modality %q", err, tc.modality)
			}
		})
	}
}

// TestEncodeStep_MissingMediaPartFails drives the full step with one more
// entry than the request has media parts, the shape of an entry<->part
// pairing bug. The fanout must fail instead of sending the encoder a
// sub-request with no media in it.
func TestEncodeStep_MissingMediaPartFails(t *testing.T) {
	// The well-formed entry may or may not reach the encoder before the
	// broken one fails the group, so this only has to answer plausibly.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewEncodeStep(gwClient, map[string]any{"use_openai_format": true})
	if err != nil {
		t.Fatalf("NewEncodeStep: %v", err)
	}
	reqCtx := &pipeline.RequestContext{
		RequestID:    "missing-part",
		Model:        "llama-3",
		OriginalPath: gateway.PathChatCompletions,
		TokenIDs:     []int{1, 32000, 2345},
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,IMG"}},
				}},
			},
		},
		// Two entries, one part: the second has nothing to pair with.
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: ModalityImage, Hash: "h0", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
			{Modality: ModalityImage, Hash: "h1", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
		},
		KVTransferParams: make(map[string]any),
	}

	err = step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected an error when an entry has no media part")
	}
	if !strings.Contains(err.Error(), "no image media part") {
		t.Errorf("error should name the missing part, got %v", err)
	}
	if errors.Is(err, pipeline.ErrBadRequest) {
		t.Error("a coordinator pairing bug is not a client error; expected no ErrBadRequest")
	}
}

// fanoutPairing is what one encode sub-request says about the entry it was
// built for: the single modality key under mm_hashes, the hash filed under
// it, the content part's type, and the payload that part carries.
type fanoutPairing struct {
	modality string
	hash     string
	partType string
	payload  string
}

// readFanoutPairing extracts the pairing a single encode sub-request body
// asserts. A sub-request carries exactly one entry, so mm_hashes must hold
// exactly one modality key with exactly one hash, and content exactly one
// part. Anything else is itself a failure.
func readFanoutPairing(t *testing.T, body map[string]any) fanoutPairing {
	t.Helper()
	tokens, _ := body["tokens"].(map[string]any)
	features, _ := tokens["features"].(map[string]any)
	hashes, _ := features["mm_hashes"].(map[string]any)
	if len(hashes) != 1 {
		t.Fatalf("sub-request must carry exactly one modality key, got %v", hashes)
	}
	var got fanoutPairing
	for mod, raw := range hashes {
		list, _ := raw.([]any)
		if len(list) != 1 {
			t.Fatalf("mm_hashes[%q] must carry exactly one hash, got %v", mod, raw)
		}
		got.modality = mod
		got.hash, _ = list[0].(string)
	}

	msgs, _ := body["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("sub-request must carry exactly one message, got %v", msgs)
	}
	content, _ := msgs[0].(map[string]any)["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("sub-request must carry exactly one content part, got %v", content)
	}
	part, _ := content[0].(map[string]any)
	got.partType, _ = part["type"].(string)
	inner, _ := part[got.partType].(map[string]any)
	// URL-based parts carry the payload under "url", input_audio under "data".
	if url, ok := inner["url"].(string); ok {
		got.payload = url
	} else {
		got.payload, _ = inner["data"].(string)
	}
	return got
}

// captureFanout runs the encode step against a recording backend and returns
// the pairing each sub-request carried, keyed by hash. Keying by hash is what
// makes a mispairing visible: the hash names the entry the sub-request was
// built for, so the part beside it must be that entry's part.
func captureFanout(t *testing.T, reqCtx *pipeline.RequestContext) map[string]fanoutPairing {
	t.Helper()
	var mu sync.Mutex
	pairings := make(map[string]fanoutPairing)
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read encoder body: %v", err)
		}
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)
		got := readFanoutPairing(t, parsed)
		mu.Lock()
		pairings[got.hash] = got
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
	if err := encodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	return pairings
}

// assertPairings checks every expected hash reached the encoder beside its own
// modality key, part type, and payload, and that nothing extra arrived.
func assertPairings(t *testing.T, got map[string]fanoutPairing, want []fanoutPairing) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("expected %d distinct fanout sub-requests, got %d: %+v", len(want), len(got), got)
	}
	for _, w := range want {
		g, ok := got[w.hash]
		if !ok {
			t.Errorf("no fanout sub-request carried hash %q", w.hash)
			continue
		}
		if g != w {
			t.Errorf("hash %q paired with %+v, want %+v", w.hash, g, w)
		}
	}
}

// TestEncodeStep_MixedModalityFanout drives the full encode step with a
// mixed-modality request. Each entry produces one fanout sub-request, and the
// assertion is per sub-request: the hash, the modality key it sits under, and
// the media bytes beside it must all belong to the same entry. Checking only
// that every modality and every part type appeared somewhere across the
// sub-requests would pass on any permutation of them, which is the failure
// this guards (see mediaPartIsWellFormed).
func TestEncodeStep_MixedModalityFanout(t *testing.T) {
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
			{Modality: ModalityImage, Hash: "img-hash", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
			{Modality: ModalityAudio, Hash: "aud-hash", Placeholder: pipeline.PlaceholderRange{Offset: 2, Length: 1}},
			{Modality: ModalityVideo, Hash: "vid-hash", Placeholder: pipeline.PlaceholderRange{Offset: 3, Length: 1}},
		},
		KVTransferParams: make(map[string]any),
	}

	assertPairings(t, captureFanout(t, reqCtx), []fanoutPairing{
		{modality: ModalityImage, hash: "img-hash", partType: "image_url", payload: "data:image/jpeg;base64,IMG"},
		{modality: ModalityAudio, hash: "aud-hash", partType: "audio_url", payload: "data:audio/wav;base64,AUD"},
		{modality: ModalityVideo, hash: "vid-hash", partType: "video_url", payload: "data:video/mp4;base64,VID"},
	})
}

// TestEncodeStep_WithinModalityFanoutPairing covers the pairing case a
// cross-modality test cannot reach: two audio entries in one request, one
// carried as audio_url and one as input_audio. Both share the audio modality
// key, so only the per-modality local index distinguishes them, and a request
// mixing the two part types is the case where a local-index regression sends
// one entry's hash with the other entry's bytes.
func TestEncodeStep_WithinModalityFanoutPairing(t *testing.T) {
	reqCtx := &pipeline.RequestContext{
		RequestID:    "within-modality-fanout",
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
						// Walker order: inline audio, then image, then audio URL.
						// The audio entries are non-adjacent on purpose.
						map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "INLINE-AUD", "format": "wav"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,IMG"}},
						map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "data:audio/wav;base64,URL-AUD"}},
					},
				},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: ModalityAudio, Hash: "aud-inline-hash", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
			{Modality: ModalityImage, Hash: "img-hash", Placeholder: pipeline.PlaceholderRange{Offset: 2, Length: 1}},
			{Modality: ModalityAudio, Hash: "aud-url-hash", Placeholder: pipeline.PlaceholderRange{Offset: 3, Length: 1}},
		},
		KVTransferParams: make(map[string]any),
	}

	assertPairings(t, captureFanout(t, reqCtx), []fanoutPairing{
		{modality: ModalityAudio, hash: "aud-inline-hash", partType: "input_audio", payload: "INLINE-AUD"},
		{modality: ModalityImage, hash: "img-hash", partType: "image_url", payload: "data:image/jpeg;base64,IMG"},
		{modality: ModalityAudio, hash: "aud-url-hash", partType: "audio_url", payload: "data:audio/wav;base64,URL-AUD"},
	})
}
