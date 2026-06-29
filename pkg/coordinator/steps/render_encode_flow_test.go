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
	"github.com/llm-d/coordinator/pkg/connectors/ec"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestRenderToEncode_FeaturesFlow(t *testing.T) {
	renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{ModalityImage: {"hash-img0", "hash-img1"}},
				"mm_placeholders": map[string][]any{ModalityImage: {map[string]any{"offset": 1, "length": 3}, map[string]any{"offset": 4, "length": 3}}},
				"kwargs_data":     map[string][]string{ModalityImage: {"dGVuc29yQQ==", "dGVuc29yQg=="}},
			},
		})
	}))
	defer renderServer.Close()

	var mu sync.Mutex
	receivedBodies := []map[string]any{}

	gwServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)

		mu.Lock()
		receivedBodies = append(receivedBodies, parsed)
		mu.Unlock()

		// Echo per-image hash back as the ec_transfer_params key (nixl shape).
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
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	// Run render step
	renderStep, _ := NewRenderStep(nil, map[string]any{})
	renderStep.(*RenderStep).SetServiceAddress(renderServer.URL)

	reqCtx := &pipeline.RequestContext{
		RequestID:    "render-encode-flow",
		OriginalPath: gateway.PathChatCompletions,
		Model:        "test-model",
		Body:         map[string]any{"model": "test-model"},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0},
			{Index: 1},
		},
	}

	err := renderStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	// Verify render populated entries
	if reqCtx.MultimodalEntries[0].Hash != "hash-img0" {
		t.Fatalf("render did not set hash for entry 0: %s", reqCtx.MultimodalEntries[0].Hash)
	}
	if reqCtx.MultimodalEntries[0].KwargsData != "dGVuc29yQQ==" {
		t.Fatalf("render did not set KwargsData for entry 0")
	}
	if reqCtx.MultimodalEntries[0].Placeholder.Length != 3 {
		t.Fatalf("render did not set Placeholder for entry 0")
	}

	// Run encode step (nixl EC connector merges per-hash ec_transfer_params)
	encodeStep, _ := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})

	err = encodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	// Verify 2 encode requests sent
	mu.Lock()
	defer mu.Unlock()
	if len(receivedBodies) != 2 {
		t.Fatalf("expected 2 encode requests, got %d", len(receivedBodies))
	}

	// Each request should have token_ids and features with single-entry per-modality data
	for i, body := range receivedBodies {
		tokenIDs, _ := body["token_ids"].([]any)
		if len(tokenIDs) != 4 { // BOS + 3 placeholders
			t.Fatalf("request %d: expected 4 token_ids, got %d", i, len(tokenIDs))
		}

		features, _ := body["features"].(map[string]any)
		mmHashes, _ := features["mm_hashes"].(map[string]any)
		hashes, _ := mmHashes[ModalityImage].([]any)
		if len(hashes) != 1 {
			t.Fatalf("request %d: expected 1 hash, got %d", i, len(hashes))
		}

		kwargs, _ := features["kwargs_data"].(map[string]any)
		imageKwargs, _ := kwargs[ModalityImage].([]any)
		if len(imageKwargs) != 1 {
			t.Fatalf("request %d: expected 1 kwargs_data entry, got %d", i, len(imageKwargs))
		}
	}

	// Verify ECTransferParams populated
	if len(reqCtx.ECTransferParams) != 2 {
		t.Fatalf("expected 2 ec_transfer_params, got %d", len(reqCtx.ECTransferParams))
	}
}
