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

		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)

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
		imageHashes, _ := mmHashes["image"].([]any)
		if len(imageHashes) != 1 {
			t.Errorf("expected 1 hash per encode request, got %d", len(imageHashes))
		}
		kwargsData, _ := features["kwargs_data"].(map[string]any)
		imageKwargs, _ := kwargsData["image"].([]any)
		if len(imageKwargs) != 1 {
			t.Errorf("expected 1 kwargs_data per encode request, got %d", len(imageKwargs))
		}

		// Echo the per-image hash back as the ec_transfer_params key, matching
		// the real encoder shape: {mm_hash: {peer_host, peer_port, ...}}.
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

	step, err := NewEncodeStep(map[string]any{
		"gateway_path":   gateway.DefaultGeneratePath,
		"max_parallel":   4,
		ParamECConnector: ec.NIXLv2,
	})
	if err != nil {
		t.Fatal(err)
	}
	step.(*EncodeStep).SetGatewayClient(gwClient)

	reqCtx := &pipeline.RequestContext{
		RequestID: "req-1",
		Model:     "test-model",
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
		imageHashes, _ := mmHashes["image"].([]any)
		hash, _ := imageHashes[0].(string)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ec_transfer_params": map[string]any{
				hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
			},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	step, _ := NewEncodeStep(map[string]any{"max_parallel": 1})
	step.(*EncodeStep).SetGatewayClient(gwClient)

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

func TestEncodeStep_BuildsCorrectTokenIDs(t *testing.T) {
	var receivedTokenIDs []any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)
		receivedTokenIDs, _ = parsed["token_ids"].([]any)
		features, _ := parsed["features"].(map[string]any)
		mmHashes, _ := features["mm_hashes"].(map[string]any)
		imageHashes, _ := mmHashes["image"].([]any)
		hash, _ := imageHashes[0].(string)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ec_transfer_params": map[string]any{
				hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
			},
		})
	}))
	defer server.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, _ := NewEncodeStep(map[string]any{})
	step.(*EncodeStep).SetGatewayClient(gwClient)

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
