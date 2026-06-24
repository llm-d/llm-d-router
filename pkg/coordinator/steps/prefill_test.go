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
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/connectors/ec"
	"github.com/llm-d/coordinator/pkg/connectors/kv"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestPrefillStep_SendsCorrectGenerateRequest(t *testing.T) {
	var prefillBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/inference/v1/generate" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get(gateway.EPPPhaseHeader) != gateway.PhasePrefill {
			t.Fatalf("expected EPP-Phase: prefill, got %q", r.Header.Get(gateway.EPPPhaseHeader))
		}

		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &prefillBody)

		_ = json.NewEncoder(w).Encode(map[string]any{
			"kv_transfer_params": map[string]any{
				"block_id":  "block-xyz",
				"peer_host": "10.0.0.5",
				"peer_port": 6001,
			},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, err := NewPrefillStep(map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	step.(*PrefillStep).SetGatewayClient(gwClient)

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-1",
		Model:     "llama-3",
		TokenIDs:  []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-a", KwargsData: "dGVuc29yLWE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Index: 1, Hash: "hash-b", KwargsData: "dGVuc29yLWI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
		},
		ECTransferParams: []map[string]any{
			{"hash-a": map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."}},
			{"hash-b": map[string]any{"peer_port": 5502, "size_bytes": 1228800, "nixl_agent_metadata_b64": "QWdlbnQ..."}},
		},
		KVTransferParams: make(map[string]any),
	}

	err = step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify request_id
	if prefillBody["request_id"] != "req-1" {
		t.Fatalf("expected request_id=req-1, got %v", prefillBody["request_id"])
	}

	// Verify model
	if prefillBody["model"] != "llama-3" {
		t.Fatalf("expected model=llama-3, got %v", prefillBody["model"])
	}

	// Verify token_ids
	tokenIDs, ok := prefillBody["token_ids"].([]any)
	if !ok || len(tokenIDs) != 8 {
		t.Fatalf("expected 8 token_ids, got %v", prefillBody["token_ids"])
	}

	// Verify features
	features, ok := prefillBody["features"].(map[string]any)
	if !ok {
		t.Fatal("expected features in prefill request")
	}
	mmHashes, _ := features["mm_hashes"].(map[string]any)
	imageHashes, _ := mmHashes[ModalityImage].([]any)
	if len(imageHashes) != 2 {
		t.Fatalf("expected 2 mm_hashes, got %d", len(imageHashes))
	}
	if imageHashes[0] != "hash-a" || imageHashes[1] != "hash-b" {
		t.Fatalf("unexpected mm_hashes: %v", imageHashes)
	}

	// Verify kwargs_data carries per-image base64 tensors
	kwargsData, ok := features["kwargs_data"].(map[string]any)
	if !ok {
		t.Fatalf("expected kwargs_data map in prefill, got %T", features["kwargs_data"])
	}
	imageKwargs, _ := kwargsData[ModalityImage].([]any)
	if len(imageKwargs) != 2 || imageKwargs[0] != "dGVuc29yLWE=" || imageKwargs[1] != "dGVuc29yLWI=" {
		t.Fatalf("expected kwargs_data.image=[dGVuc29yLWE=,dGVuc29yLWI=], got %v", imageKwargs)
	}

	// Verify ec_transfer_params is a flat map keyed by mm_hash
	ecParams, ok := prefillBody["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected ec_transfer_params in prefill request")
	}
	if len(ecParams) != 2 {
		t.Fatalf("expected 2 ec_transfer_params entries, got %d: %v", len(ecParams), ecParams)
	}
	for _, want := range []string{"hash-a", "hash-b"} {
		if _, ok := ecParams[want]; !ok {
			t.Errorf("missing hash %q in ec_transfer_params: %v", want, ecParams)
		}
	}

	// Verify sampling_params with extra_args workaround
	samplingParams, ok := prefillBody["sampling_params"].(map[string]any)
	if !ok {
		t.Fatal("expected sampling_params in body")
	}
	if samplingParams["max_tokens"] != float64(1) {
		t.Fatalf("expected sampling_params.max_tokens=1, got %v", samplingParams["max_tokens"])
	}
	extraArgs, ok := samplingParams["extra_args"].(map[string]any)
	if !ok {
		t.Fatal("expected sampling_params.extra_args in generate format")
	}
	kvParams, ok := extraArgs["kv_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected kv_transfer_params in extra_args")
	}
	if kvParams["do_remote_decode"] != true {
		t.Fatalf("expected kv_transfer_params.do_remote_decode=true, got %v", kvParams["do_remote_decode"])
	}

	// Verify no top-level kv_transfer_params in generate format
	if _, ok := prefillBody["kv_transfer_params"]; ok {
		t.Fatal("generate format should not have top-level kv_transfer_params")
	}

	// Verify response populated KVTransferParams
	if reqCtx.KVTransferParams["block_id"] != "block-xyz" {
		t.Fatalf("expected block_id=block-xyz, got %v", reqCtx.KVTransferParams["block_id"])
	}
}

func TestPrefillStep_CompletionsFormat(t *testing.T) {
	var prefillBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != gateway.PathCompletions {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get(gateway.EPPPhaseHeader) != gateway.PhasePrefill {
			t.Fatalf("expected EPP-Phase: prefill, got %q", r.Header.Get(gateway.EPPPhaseHeader))
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &prefillBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"kv_transfer_params": map[string]any{"block_id": "block-1", "peer_host": "10.0.0.5", "peer_port": 6001},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewPrefillStep(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	step.(*PrefillStep).SetGatewayClient(gwClient)

	reqCtx := &pipeline.RequestContext{
		RequestID:         "req-compl",
		OriginalPath:      gateway.PathCompletions,
		Model:             "test-model",
		TokenIDs:          []int{1, 2345, 6789},
		MultimodalEntries: nil,
		KVTransferParams:  make(map[string]any),
	}

	err = step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, ok := prefillBody["token_ids"]; ok {
		t.Fatal("completions format should not have token_ids field")
	}
	prompt, ok := prefillBody["prompt"].([]any)
	if !ok || len(prompt) != 3 {
		t.Fatalf("expected prompt with 3 token_ids, got %v", prefillBody["prompt"])
	}
	if prefillBody["request_id"] != "req-compl" {
		t.Fatalf("expected request_id, got %v", prefillBody["request_id"])
	}
	// Completions format has top-level kv_transfer_params
	kvParams, ok := prefillBody["kv_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected kv_transfer_params in completions format")
	}
	if kvParams["do_remote_decode"] != true {
		t.Fatalf("expected do_remote_decode=true, got %v", kvParams["do_remote_decode"])
	}
}

func TestPrefillStep_CompletionsFormat_NoRenderedTokens(t *testing.T) {
	var prefillBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &prefillBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"kv_transfer_params": map[string]any{"block_id": "block-1", "peer_host": "10.0.0.5", "peer_port": 6001},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewPrefillStep(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	step.(*PrefillStep).SetGatewayClient(gwClient)

	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-compl",
		OriginalPath:     gateway.PathCompletions,
		Model:            "test-model",
		TokenIDs:         nil,
		Body:             map[string]any{"prompt": "Hello"},
		KVTransferParams: make(map[string]any),
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if prefillBody["prompt"] != "Hello" {
		t.Fatalf("expected original prompt to pass through, got %v", prefillBody["prompt"])
	}
}

func TestPrefillStep_ChatCompletionsFormat(t *testing.T) {
	var prefillBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != gateway.PathChatCompletions {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get(gateway.EPPPhaseHeader) != gateway.PhasePrefill {
			t.Fatalf("expected EPP-Phase: prefill, got %q", r.Header.Get(gateway.EPPPhaseHeader))
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &prefillBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"kv_transfer_params": map[string]any{"block_id": "block-2", "peer_host": "10.0.0.5", "peer_port": 6001},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewPrefillStep(map[string]any{
		ParamECConnector: ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	step.(*PrefillStep).SetGatewayClient(gwClient)

	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-chat",
		OriginalPath: gateway.PathChatCompletions,
		Model:        "test-model",
		TokenIDs:     []int{1, 32000, 32000, 32000, 2345},
		Body: map[string]any{
			"model":  "test-model",
			"stream": false,
			"messages": []any{
				map[string]any{"role": "user", "content": "hello"},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-a", KwargsData: "dGVuc29y", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
		},
		ECTransferParams: []map[string]any{
			{"hash-a": map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."}},
		},
		KVTransferParams: make(map[string]any),
	}

	err = step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if prefillBody["model"] != "test-model" {
		t.Fatalf("expected model from original body, got %v", prefillBody["model"])
	}
	if _, ok := prefillBody["messages"]; !ok {
		t.Fatal("expected messages from original body in chat format")
	}

	// Verify tokens nested field
	tokens, ok := prefillBody["tokens"].(map[string]any)
	if !ok {
		t.Fatal("expected tokens field in chat format")
	}
	tokenIDs, _ := tokens["token_ids"].([]any)
	if len(tokenIDs) != 5 {
		t.Fatalf("expected 5 token_ids in tokens, got %d", len(tokenIDs))
	}
	tokensFeatures, ok := tokens["features"].(map[string]any)
	if !ok {
		t.Fatal("expected features in tokens field")
	}
	// tokens.features should NOT have kwargs_data
	if _, ok := tokensFeatures["kwargs_data"]; ok {
		t.Fatal("tokens.features should not have kwargs_data")
	}
	if _, ok := tokensFeatures["mm_hashes"]; !ok {
		t.Fatal("tokens.features should have mm_hashes")
	}

	// Verify ec_transfer_params is forwarded in chat format
	ecParams, ok := prefillBody["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected ec_transfer_params in chat format prefill body")
	}
	ecEntry, ok := ecParams["hash-a"].(map[string]any)
	if !ok || len(ecEntry) == 0 {
		t.Errorf("ec_transfer_params[hash-a] missing or empty: %v", ecParams["hash-a"])
	}

	// Verify top-level kv_transfer_params
	if _, ok := prefillBody["kv_transfer_params"]; !ok {
		t.Fatal("expected kv_transfer_params in chat format")
	}
	// Verify no top-level token_ids (should be in tokens field)
	if _, ok := prefillBody["token_ids"]; ok {
		t.Fatal("chat format should not have top-level token_ids")
	}
	if _, ok := prefillBody["request_id"]; ok {
		t.Fatal("chat format should not have request_id (uses original body)")
	}
}

// TestSharedStorage_OmitsECTransferParams_InPrefillBody verifies that the
// ec-shared-storage EC connector never emits an ec_transfer_params field on
// the prefill body, in every prefill wire format (chat-completions,
// completions, generate). A regression that set the field unconditionally
// would send "ec_transfer_params": null and silently break ec-shared-storage
// deployments, where the consumer reads embeddings from shared storage.
func TestSharedStorage_OmitsECTransferParams_InPrefillBody(t *testing.T) {
	cases := []struct {
		name         string
		useOpenAI    bool
		originalPath string
		body         map[string]any
	}{
		{
			name:         "ChatCompletions",
			useOpenAI:    true,
			originalPath: gateway.PathChatCompletions,
			body: map[string]any{
				"model":    "m",
				"messages": []any{map[string]any{"role": "user", "content": "hi"}},
			},
		},
		{name: "Completions", useOpenAI: true, originalPath: gateway.PathCompletions},
		{name: "Generate", useOpenAI: false, originalPath: gateway.PathChatCompletions},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var raw []byte
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ = io.ReadAll(r.Body)
				_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": map[string]any{}})
			}))
			defer server.Close()

			step, err := NewPrefillStep(map[string]any{
				"use_openai_format": tc.useOpenAI,
				ParamECConnector:    ec.SharedStorage,
			})
			if err != nil {
				t.Fatalf("NewPrefillStep: %v", err)
			}
			step.(*PrefillStep).SetGatewayClient(gateway.New(config.GatewayConfig{Address: server.URL}))

			reqCtx := &pipeline.RequestContext{
				RequestID:        "req",
				OriginalPath:     tc.originalPath,
				Model:            "m",
				TokenIDs:         []int{1, 2, 3},
				Body:             tc.body,
				KVTransferParams: make(map[string]any),
			}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("Execute: %v", err)
			}

			var parsed map[string]any
			if err := json.Unmarshal(raw, &parsed); err != nil {
				t.Fatalf("unmarshal prefill body: %v", err)
			}
			if _, ok := parsed["ec_transfer_params"]; ok {
				t.Errorf("ec-shared-storage must not set ec_transfer_params; body=%s", raw)
			}
		})
	}
}

// TestPrefillStep_ConflictingECParams_RejectsRequest verifies the load-bearing
// "reject the request" contract end to end: when two encode responses carry
// conflicting descriptors for the same mm_hash, PrefillStep.Execute must fail
// and never reach the gateway. The unit-level conflict test
// (TestNIXL_MergeAndPrepare_DuplicateHashes_Conflict) covers the connector in
// isolation; this confirms the error propagates through the step boundary.
func TestPrefillStep_ConflictingECParams_RejectsRequest(t *testing.T) {
	gatewayHit := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gatewayHit = true
		_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": map[string]any{}})
	}))
	defer server.Close()

	step, err := NewPrefillStep(map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	step.(*PrefillStep).SetGatewayClient(gateway.New(config.GatewayConfig{Address: server.URL}))

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-conflict",
		Model:     "test-model",
		TokenIDs:  []int{1, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash-a", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
		},
		ECTransferParams: []map[string]any{
			{"hash-a": map[string]any{"peer_port": 5501}},
			{"hash-a": map[string]any{"peer_port": 5599}},
		},
		KVTransferParams: make(map[string]any),
	}

	if err := step.Execute(context.Background(), reqCtx); err == nil {
		t.Fatal("expected Execute to fail on conflicting ec_transfer_params descriptors")
	}
	if gatewayHit {
		t.Error("conflicting descriptors must reject the request before contacting the gateway")
	}
}

func TestPrefillStep_GatewayError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("overloaded"))
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, _ := NewPrefillStep(map[string]any{})
	step.(*PrefillStep).SetGatewayClient(gwClient)

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-1",
		Model:     "test",
		TokenIDs:  []int{1, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "h1", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
		},
		ECTransferParams: []map[string]any{
			{"h1": map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."}},
		},
		KVTransferParams: make(map[string]any),
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for 503 response")
	}
}

// TestPrefillStep_CoercesInvalidKVTransferParams verifies that a prefill
// response whose kv_transfer_params is not a usable JSON object (non-object
// type, explicit null, empty object, or absent) is coerced to no transfer
// params rather than failing the prefill step, mirroring the EC NIXL
// connector's ecParamsFromResponse. Each case must succeed and leave
// KVTransferParams empty.
func TestPrefillStep_CoercesInvalidKVTransferParams(t *testing.T) {
	cases := []struct {
		name string
		body map[string]any
	}{
		{name: "NonObjectString", body: map[string]any{"kv_transfer_params": "not-an-object"}},
		{name: "NonObjectArray", body: map[string]any{"kv_transfer_params": []any{1, 2}}},
		{name: "ExplicitNull", body: map[string]any{"kv_transfer_params": nil}},
		{name: "EmptyObject", body: map[string]any{"kv_transfer_params": map[string]any{}}},
		{name: "FieldAbsent", body: map[string]any{"other": "field"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(tc.body)
			}))
			defer server.Close()

			step, err := NewPrefillStep(map[string]any{
				"use_openai_format": false,
				ParamKVConnector:    kv.NIXL,
				ParamECConnector:    ec.NIXL,
			})
			if err != nil {
				t.Fatal(err)
			}
			step.(*PrefillStep).SetGatewayClient(gateway.New(config.GatewayConfig{Address: server.URL}))

			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-1",
				Model:            "test-model",
				TokenIDs:         []int{1, 2345},
				KVTransferParams: make(map[string]any),
			}

			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("invalid kv_transfer_params should be coerced, not fail the prefill: %v", err)
			}
			if len(reqCtx.KVTransferParams) != 0 {
				t.Fatalf("expected no kv_transfer_params recorded, got %v", reqCtx.KVTransferParams)
			}
		})
	}
}
