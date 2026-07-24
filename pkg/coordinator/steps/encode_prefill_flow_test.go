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

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestEncodeToPrefill_ECTransferParamsFlow(t *testing.T) {
	var prefillBody map[string]any

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		phase := r.Header.Get(gateway.EPPPhaseHeader)
		switch phase {
		case gateway.PhaseEncode:
			body, _ := io.ReadAll(r.Body)
			var parsed map[string]any
			_ = json.Unmarshal(body, &parsed)
			features, _ := parsed["features"].(map[string]any)
			mmHashes, _ := features["mm_hashes"].(map[string]any)
			imageHashes, _ := mmHashes[ModalityImage].([]any)
			hash, _ := imageHashes[0].(string)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{
					hash: map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."},
				},
			})

		case gateway.PhasePrefill:
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &prefillBody)

			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "b1", "peer_host": "10.0.0.2", "peer_port": 5502},
			})

		default:
			http.Error(w, "unexpected phase: "+phase, 404)
		}
	}))
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	reqCtx := &pipeline.RequestContext{
		RequestID: "encode-prefill-flow",
		Model:     "llama-3",
		TokenIDs:  []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "img-hash-1", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Index: 1, Hash: "img-hash-2", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
		},
		KVTransferParams: make(map[string]any),
	}

	// Run encode step
	encodeStep, _ := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})

	err := encodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	// Verify encode populated ECTransferParams (ordered list)
	if len(reqCtx.ECTransferParams) != 2 {
		t.Fatalf("expected 2 ec_transfer_params, got %d", len(reqCtx.ECTransferParams))
	}

	// Run prefill step
	prefillStep, _ := NewPrefillStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})

	err = prefillStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	// Validate prefill request body
	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}

	// Verify features has kwargs_data with per-image base64 tensors
	features, _ := prefillBody["features"].(map[string]any)
	kwargsData, ok := features["kwargs_data"].(map[string]any)
	if !ok {
		t.Fatalf("expected kwargs_data map in prefill, got %T", features["kwargs_data"])
	}
	imageKwargs, _ := kwargsData[ModalityImage].([]any)
	if len(imageKwargs) != 2 || imageKwargs[0] != "dDE=" || imageKwargs[1] != "dDI=" {
		t.Fatalf("expected kwargs_data.image=[dDE=,dDI=], got %v", imageKwargs)
	}

	// Verify mm_hashes in features
	mmHashes, _ := features["mm_hashes"].(map[string]any)
	imageHashes, _ := mmHashes[ModalityImage].([]any)
	if len(imageHashes) != 2 {
		t.Fatalf("expected 2 mm_hashes in prefill features, got %d", len(imageHashes))
	}

	// Verify sampling_params with extra_args workaround
	samplingParams, _ := prefillBody["sampling_params"].(map[string]any)
	if samplingParams["max_tokens"] != float64(1) {
		t.Fatalf("expected sampling_params.max_tokens=1, got %v", samplingParams["max_tokens"])
	}
	extraArgs, ok := samplingParams["extra_args"].(map[string]any)
	if !ok {
		t.Fatal("expected extra_args in sampling_params for generate format")
	}
	kvParams, ok := extraArgs["kv_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected kv_transfer_params in extra_args")
	}
	if kvParams["do_remote_decode"] != true {
		t.Fatalf("expected do_remote_decode=true, got %v", kvParams["do_remote_decode"])
	}

	// Verify ec_transfer_params is a flat map keyed by mm_hash, nested in extra_args
	ecParams, ok := extraArgs["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected ec_transfer_params in extra_args, got %T", extraArgs["ec_transfer_params"])
	}
	if len(ecParams) != 2 {
		t.Fatalf("expected 2 ec_transfer_params entries, got %d: %v", len(ecParams), ecParams)
	}
	for _, want := range []string{"img-hash-1", "img-hash-2"} {
		entry, ok := ecParams[want].(map[string]any)
		if !ok || len(entry) == 0 {
			t.Errorf("ec_transfer_params[%q] missing or empty: %v", want, ecParams[want])
		}
	}

	// Verify response populated KVTransferParams
	if reqCtx.KVTransferParams["block_id"] != "b1" {
		t.Fatalf("expected KVTransferParams.block_id=b1, got %v", reqCtx.KVTransferParams["block_id"])
	}
}

// TestEncodeToPrefill_PartialECResponse verifies that when only some encoders
// return ec_transfer_params (others return 200 OK with no EC field), the
// flow does not panic, prefill is still sent, and ec_transfer_params on the
// prefill request contains only the hashes that were reported.
func TestEncodeToPrefill_PartialECResponse(t *testing.T) {
	var prefillBody map[string]any

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		phase := r.Header.Get(gateway.EPPPhaseHeader)
		switch phase {
		case gateway.PhaseEncode:
			body, _ := io.ReadAll(r.Body)
			var parsed map[string]any
			_ = json.Unmarshal(body, &parsed)
			features, _ := parsed["features"].(map[string]any)
			mmHashes, _ := features["mm_hashes"].(map[string]any)
			imageHashes, _ := mmHashes[ModalityImage].([]any)
			hash, _ := imageHashes[0].(string)

			// Only image 1 gets EC params; image 2 returns no ec_transfer_params.
			if hash == "img-hash-1" {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"ec_transfer_params": map[string]any{
						hash: map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."},
					},
				})
			} else {
				_ = json.NewEncoder(w).Encode(map[string]any{})
			}

		case gateway.PhasePrefill:
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &prefillBody)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "b1"},
			})

		default:
			http.Error(w, "unexpected phase: "+phase, 404)
		}
	}))
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	reqCtx := &pipeline.RequestContext{
		RequestID: "partial-ec-flow",
		Model:     "llama-3",
		TokenIDs:  []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0, Hash: "img-hash-1", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Index: 1, Hash: "img-hash-2", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
		},
		KVTransferParams: make(map[string]any),
	}

	encodeStep, _ := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err := encodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	// Only image 1 contributed to ECTransferParams; image 2's empty response was skipped.
	if len(reqCtx.ECTransferParams) != 1 {
		t.Fatalf("expected 1 ECTransferParams entry after partial response, got %d: %v",
			len(reqCtx.ECTransferParams), reqCtx.ECTransferParams)
	}

	prefillStep, _ := NewPrefillStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err := prefillStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}

	samplingParams, _ := prefillBody["sampling_params"].(map[string]any)
	extraArgs, _ := samplingParams["extra_args"].(map[string]any)
	ecParams, ok := extraArgs["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected ec_transfer_params in extra_args (with reported hash only), got %T", extraArgs["ec_transfer_params"])
	}
	if len(ecParams) != 1 {
		t.Fatalf("expected 1 ec_transfer_params entry, got %d: %v", len(ecParams), ecParams)
	}
	if _, ok := ecParams["img-hash-1"]; !ok {
		t.Errorf("expected img-hash-1 in ec_transfer_params, got %v", ecParams)
	}
	if _, ok := ecParams["img-hash-2"]; ok {
		t.Errorf("unexpected img-hash-2 in ec_transfer_params (encoder did not report it): %v", ecParams)
	}
}
