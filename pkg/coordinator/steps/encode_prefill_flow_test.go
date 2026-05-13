package steps

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestEncodeToPrefill_ECTransferParamsFlow(t *testing.T) {
	var prefillBody map[string]any

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/encode"):
			json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{
					"peer_host": "10.0.0.1",
					"peer_port": 5501,
				},
			})

		case strings.HasPrefix(r.URL.Path, "/prefill"):
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &prefillBody)

			json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "b1"},
			})

		default:
			http.Error(w, "unexpected path: "+r.URL.Path, 404)
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
	encodeStep, _ := NewEncodeStep(map[string]any{"gateway_path": "/inference/v1/generate"})
	encodeStep.(*EncodeStep).SetGatewayClient(gwClient)

	err := encodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	// Verify encode populated ECTransferParams (ordered list)
	if len(reqCtx.ECTransferParams) != 2 {
		t.Fatalf("expected 2 ec_transfer_params, got %d", len(reqCtx.ECTransferParams))
	}

	// Run prefill step
	prefillStep, _ := NewPrefillStep(map[string]any{"gateway_path": "/inference/v1/generate"})
	prefillStep.(*PrefillStep).SetGatewayClient(gwClient)

	err = prefillStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	// Validate prefill request body
	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}

	// Verify ec_transfer_params formatted as {"image": [params_0, params_1]}
	ecParams, ok := prefillBody["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected ec_transfer_params map, got %T", prefillBody["ec_transfer_params"])
	}
	imageEC, ok := ecParams["image"].([]any)
	if !ok || len(imageEC) != 2 {
		t.Fatalf("expected ec_transfer_params.image with 2 entries, got %v", ecParams)
	}
	entry0, _ := imageEC[0].(map[string]any)
	if entry0["peer_host"] != "10.0.0.1" {
		t.Fatalf("ec_transfer_params[0].peer_host = %v, want 10.0.0.1", entry0["peer_host"])
	}

	// Verify features has kwargs_data=null
	features, _ := prefillBody["features"].(map[string]any)
	if features["kwargs_data"] != nil {
		t.Fatalf("expected kwargs_data=null in prefill, got %v", features["kwargs_data"])
	}

	// Verify mm_hashes in features
	mmHashes, _ := features["mm_hashes"].(map[string]any)
	imageHashes, _ := mmHashes["image"].([]any)
	if len(imageHashes) != 2 {
		t.Fatalf("expected 2 mm_hashes in prefill features, got %d", len(imageHashes))
	}

	// Verify sampling_params
	samplingParams, _ := prefillBody["sampling_params"].(map[string]any)
	if samplingParams["max_tokens"] != float64(1) {
		t.Fatalf("expected sampling_params.max_tokens=1, got %v", samplingParams["max_tokens"])
	}

	// Verify response populated KVTransferParams
	if reqCtx.KVTransferParams["block_id"] != "b1" {
		t.Fatalf("expected KVTransferParams.block_id=b1, got %v", reqCtx.KVTransferParams["block_id"])
	}
}
