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

package pipeline_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

func TestFullPipeline_AllConnectorCombinations(t *testing.T) {
	cases := []struct {
		kvConnector     string
		ecConnector     string
		wantECInPrefill bool // ec_transfer_params should be present in prefill body
	}{
		{kv.NIXL, ec.NIXL, true},
		{kv.NIXL, ec.SharedStorage, false},
		{kv.SharedStorage, ec.NIXL, true},
		{kv.SharedStorage, ec.SharedStorage, false},
	}

	for _, tc := range cases {
		t.Run(tc.kvConnector+"+"+tc.ecConnector, func(t *testing.T) {
			renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"token_ids": []int{1, 32000, 32000, 32000, 2345, 6789},
					"features": map[string]any{
						"mm_hashes":       map[string][]string{steps.ModalityImage: {"vllm-hash-img0"}},
						"mm_placeholders": map[string][]any{steps.ModalityImage: {map[string]any{"offset": 1, "length": 3}}},
						"kwargs_data":     map[string][]string{steps.ModalityImage: {"dGVuc29yLWRhdGE="}},
					},
				})
			}))
			defer renderServer.Close()

			var mu sync.Mutex
			var capturedPrefillBody map[string]any

			gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				phase := r.Header.Get(gateway.EPPProfileHeader)
				switch phase {
				case gateway.PhaseEncode:
					body, _ := io.ReadAll(r.Body)
					var parsed map[string]any
					_ = json.Unmarshal(body, &parsed)
					// Generate format: features at top level
					features, _ := parsed["features"].(map[string]any)
					mmHashes, _ := features["mm_hashes"].(map[string]any)
					imageHashes, _ := mmHashes[steps.ModalityImage].([]any)
					hash, _ := imageHashes[0].(string)
					_ = json.NewEncoder(w).Encode(map[string]any{
						"ec_transfer_params": map[string]any{
							hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
						},
					})
				case gateway.PhasePrefill:
					body, _ := io.ReadAll(r.Body)
					var parsed map[string]any
					_ = json.Unmarshal(body, &parsed)
					mu.Lock()
					capturedPrefillBody = parsed
					mu.Unlock()
					_ = json.NewEncoder(w).Encode(map[string]any{
						"kv_transfer_params": map[string]any{
							"block_id":  "abc123",
							"peer_host": "10.0.0.2",
							"peer_port": 5502,
						},
					})
				case gateway.PhaseDecode:
					_ = json.NewEncoder(w).Encode(map[string]any{
						"choices": []map[string]any{
							{"message": map[string]any{"role": "assistant", "content": "Hello!"}},
						},
					})
				default:
					http.Error(w, "not found", 404)
				}
			}))
			defer gatewayServer.Close()

			gwClient := gateway.New(config.GatewayConfig{Address: gatewayServer.URL, MaxIdleConnsPerHost: 10})

			stepConfigs := []config.StepConfig{
				{Type: "replace-media-urls", Params: map[string]any{"download_timeout": "5s"}},
				{Type: "render", Params: map[string]any{"endpoint": reqcommon.PathChatCompletions + "/render"}},
				{Type: "encode", Params: map[string]any{"use_openai_format": false, steps.ParamECConnector: tc.ecConnector}},
				{Type: "prefill", Params: map[string]any{"use_openai_format": false, steps.ParamKVConnector: tc.kvConnector, steps.ParamECConnector: tc.ecConnector}},
				{Type: "decode", Params: map[string]any{steps.ParamKVConnector: tc.kvConnector}},
			}

			pipelineSteps := make([]pipeline.Step, 0, len(stepConfigs))
			for _, sc := range stepConfigs {
				step, err := pipeline.Build(sc.Type, gwClient, sc.Params)
				if err != nil {
					t.Fatalf("building step %s: %v", sc.Type, err)
				}
				if ra, ok := step.(renderAware); ok {
					ra.SetServiceAddress(renderServer.URL)
				}
				pipelineSteps = append(pipelineSteps, step)
			}

			requestBody := `{
				"model": "test-model", "stream": false,
				"messages": [{"role": "user", "content": [
					{"type": "text", "text": "What is in this image?"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZS1pbWFnZS1kYXRh"}}
				]}]
			}`

			recorder := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "test-" + tc.kvConnector + "+" + tc.ecConnector,
				OriginalPath:     reqcommon.PathChatCompletions,
				OriginalBody:     []byte(requestBody),
				Model:            "test-model",
				KVTransferParams: make(map[string]any),
				ResponseWriter:   recorder,
			}
			_ = json.Unmarshal([]byte(requestBody), &reqCtx.Body)

			if err := pipeline.New(pipelineSteps).Execute(t.Context(), reqCtx); err != nil {
				t.Fatalf("pipeline failed: %v", err)
			}

			respBody, _ := io.ReadAll(recorder.Result().Body)
			if !strings.Contains(string(respBody), "Hello!") {
				t.Fatalf("expected 'Hello!' in response, got: %s", respBody)
			}

			if tc.wantECInPrefill {
				if len(reqCtx.ECTransferParams) == 0 {
					t.Error("expected ECTransferParams to be populated")
				}
			} else {
				if len(reqCtx.ECTransferParams) != 0 {
					t.Errorf("expected ECTransferParams to be empty, got %d entries", len(reqCtx.ECTransferParams))
				}
			}
			if len(reqCtx.KVTransferParams) == 0 {
				t.Error("expected KVTransferParams to be populated")
			}

			mu.Lock()
			captured := capturedPrefillBody
			mu.Unlock()
			if captured == nil {
				t.Fatal("prefill was not called")
			}
			// Generate format carries transfer params at the top level of the body.
			_, hasEC := captured["ec_transfer_params"]
			if tc.wantECInPrefill && !hasEC {
				t.Error("expected top-level ec_transfer_params in prefill body")
			}
			if !tc.wantECInPrefill && hasEC {
				t.Error("unexpected top-level ec_transfer_params in prefill body")
			}
		})
	}
}

func TestFullPipeline_Integration(t *testing.T) {
	renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 2345, 6789},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{steps.ModalityImage: {"vllm-hash-img0"}},
				"mm_placeholders": map[string][]any{steps.ModalityImage: {map[string]any{"offset": 1, "length": 3}}},
				"kwargs_data":     map[string][]string{steps.ModalityImage: {"dGVuc29yLWRhdGE="}},
			},
		})
	}))
	defer renderServer.Close()

	var mu sync.Mutex
	var capturedDecodeBody map[string]any

	gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		phase := r.Header.Get(gateway.EPPProfileHeader)
		switch phase {
		case gateway.PhaseEncode:
			body, _ := io.ReadAll(r.Body)
			var parsed map[string]any
			_ = json.Unmarshal(body, &parsed)
			features, _ := parsed["features"].(map[string]any)
			mmHashes, _ := features["mm_hashes"].(map[string]any)
			imageHashes, _ := mmHashes[steps.ModalityImage].([]any)
			hash, _ := imageHashes[0].(string)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{
					hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
				},
			})
		case gateway.PhasePrefill:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{
					"block_id":  "abc123",
					"peer_host": "10.0.0.2",
					"peer_port": 5502,
				},
			})
		case gateway.PhaseDecode:
			body, _ := io.ReadAll(r.Body)
			var parsed map[string]any
			_ = json.Unmarshal(body, &parsed)
			mu.Lock()
			capturedDecodeBody = parsed
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{
					{"message": map[string]any{"role": "assistant", "content": "Hello!"}},
				},
			})
		default:
			http.Error(w, "not found", 404)
		}
	}))
	defer gatewayServer.Close()

	gwClient := gateway.New(config.GatewayConfig{
		Address:             gatewayServer.URL,
		MaxIdleConnsPerHost: 10,
	})

	stepConfigs := []config.StepConfig{
		{Type: "replace-media-urls", Params: map[string]any{"download_timeout": "5s"}},
		{Type: "render", Params: map[string]any{"endpoint": reqcommon.PathChatCompletions + "/render"}},
		{Type: "encode", Params: map[string]any{"use_openai_format": false, steps.ParamECConnector: ec.NIXL}},
		{Type: "prefill", Params: map[string]any{"use_openai_format": false, steps.ParamECConnector: ec.NIXL}},
		{Type: "decode"},
	}

	pipelineSteps := make([]pipeline.Step, 0, len(stepConfigs))
	for _, sc := range stepConfigs {
		step, err := pipeline.Build(sc.Type, gwClient, sc.Params)
		if err != nil {
			t.Fatalf("building step %s: %v", sc.Type, err)
		}

		if ra, ok := step.(renderAware); ok {
			ra.SetServiceAddress(renderServer.URL)
		}

		pipelineSteps = append(pipelineSteps, step)
	}

	p := pipeline.New(pipelineSteps)

	requestBody := `{
		"model": "test-model",
		"stream": false,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "What is in this image?"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZS1pbWFnZS1kYXRh"}}
				]
			}
		]
	}`

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "test-123",
		OriginalPath:     reqcommon.PathChatCompletions,
		OriginalBody:     []byte(requestBody),
		Stream:           false,
		Model:            "test-model",
		KVTransferParams: make(map[string]any),
		ResponseWriter:   recorder,
	}

	_ = json.Unmarshal([]byte(requestBody), &reqCtx.Body)

	err := p.Execute(t.Context(), reqCtx)
	if err != nil {
		t.Fatalf("pipeline execution failed: %v", err)
	}

	result := recorder.Result()
	if result.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", result.StatusCode)
	}

	respBody, _ := io.ReadAll(result.Body)
	if !strings.Contains(string(respBody), "Hello!") {
		t.Fatalf("expected response to contain 'Hello!', got: %s", string(respBody))
	}

	if len(reqCtx.ECTransferParams) == 0 {
		t.Fatal("expected ECTransferParams to be populated")
	}
	if len(reqCtx.KVTransferParams) == 0 {
		t.Fatal("expected KVTransferParams to be populated")
	}

	// The decode step's body format must track the request's original path
	// (/v1/chat/completions) rather than the pipeline's use_openai_format
	// setting, which decode ignores. A regression here would nest
	// kv_transfer_params under sampling_params.extra_args instead, the
	// generate-shaped body a chat-completions endpoint does not read.
	mu.Lock()
	decodeBody := capturedDecodeBody
	mu.Unlock()
	if decodeBody == nil {
		t.Fatal("decode was not called")
	}
	if _, ok := decodeBody["kv_transfer_params"]; !ok {
		t.Error("expected top-level kv_transfer_params in decode body for /v1/chat/completions")
	}
	if sp, ok := decodeBody["sampling_params"].(map[string]any); ok {
		if ea, ok := sp["extra_args"].(map[string]any); ok {
			if _, ok := ea["kv_transfer_params"]; ok {
				t.Error("kv_transfer_params must not be nested under sampling_params.extra_args for chat completions")
			}
		}
	}
}

// TestFullPipeline_MetricsSmoke drives the full chat pipeline under mixed
// text and image traffic and asserts the pipeline-amplification series land
// in a fresh coordinator registry -- the same Register call the /metrics
// endpoint uses. This is the smoke tier of the PR test plan: per-observation
// correctness is covered by unit tests in steps and metrics; this test
// proves the series show up after real end-to-end request processing.
func TestFullPipeline_MetricsSmoke(t *testing.T) {
	reg := newMetricsRegistry(t)

	// The render mock counts image_url parts in the (already inlined)
	// request body so its features always match the request's media count.
	renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed struct {
			Messages []struct {
				Content []struct {
					Type string `json:"type"`
				} `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(body, &parsed)
		images := 0
		for _, m := range parsed.Messages {
			for _, part := range m.Content {
				if part.Type == "image_url" {
					images++
				}
			}
		}

		hashes := make([]string, images)
		placeholders := make([]any, images)
		kwargs := make([]string, images)
		for i := range hashes {
			hashes[i] = fmt.Sprintf("vllm-hash-img%d", i)
			placeholders[i] = map[string]any{"offset": 1, "length": 3}
			kwargs[i] = "dGVuc29yLWRhdGE="
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 2345, 6789},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{steps.ModalityImage: hashes},
				"mm_placeholders": map[string][]any{steps.ModalityImage: placeholders},
				"kwargs_data":     map[string][]string{steps.ModalityImage: kwargs},
			},
		})
	}))
	defer renderServer.Close()

	gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get(gateway.EPPProfileHeader) {
		case gateway.PhaseEncode:
			body, _ := io.ReadAll(r.Body)
			var parsed struct {
				Features struct {
					MMHashes struct {
						Image []any `json:"image"`
					} `json:"mm_hashes"`
				} `json:"features"`
			}
			_ = json.Unmarshal(body, &parsed)
			transfer := map[string]any{}
			for _, h := range parsed.Features.MMHashes.Image {
				if hash, ok := h.(string); ok {
					transfer[hash] = map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501}
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"ec_transfer_params": transfer})
		case gateway.PhasePrefill:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "abc123", "peer_host": "10.0.0.2", "peer_port": 5502},
			})
		case gateway.PhaseDecode:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "Hello!"}}},
			})
		default:
			http.Error(w, "not found", 404)
		}
	}))
	defer gatewayServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gatewayServer.URL})

	stepConfigs := []config.StepConfig{
		{Type: "replace-media-urls", Params: map[string]any{"download_timeout": "5s"}},
		{Type: "render", Params: map[string]any{"endpoint": reqcommon.PathChatCompletions + "/render"}},
		{Type: "encode", Params: map[string]any{"use_openai_format": false, steps.ParamECConnector: ec.NIXL}},
		{Type: "prefill", Params: map[string]any{"use_openai_format": false, steps.ParamECConnector: ec.NIXL}},
		{Type: "decode"},
	}
	pipelineSteps := make([]pipeline.Step, 0, len(stepConfigs))
	for _, sc := range stepConfigs {
		step, err := pipeline.Build(sc.Type, gwClient, sc.Params)
		if err != nil {
			t.Fatalf("building step %s: %v", sc.Type, err)
		}
		if ra, ok := step.(renderAware); ok {
			ra.SetServiceAddress(renderServer.URL)
		}
		pipelineSteps = append(pipelineSteps, step)
	}
	p := pipeline.New(pipelineSteps)

	run := func(body string) error {
		recorder := httptest.NewRecorder()
		reqCtx := &pipeline.RequestContext{
			RequestID:        "smoke",
			OriginalPath:     reqcommon.PathChatCompletions,
			OriginalBody:     []byte(body),
			Stream:           false,
			Model:            "test-model",
			KVTransferParams: make(map[string]any),
			ResponseWriter:   recorder,
		}
		if err := json.Unmarshal([]byte(body), &reqCtx.Body); err != nil {
			t.Fatalf("unmarshaling request body: %v", err)
		}
		return p.Execute(t.Context(), reqCtx)
	}

	// Mixed traffic: text-only, a data-URI image, and a loopback image URL
	// that the SSRF guard blocks at dial so the third pipeline run fails
	// after the media inventory and download observations.
	if err := run(`{"model":"test-model","stream":false,"messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]}`); err != nil {
		t.Fatalf("text-only request failed: %v", err)
	}
	if err := run(`{"model":"test-model","stream":false,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,ZmFrZS1pbWFnZS1kYXRh"}}]}]}`); err != nil {
		t.Fatalf("data-URI request failed: %v", err)
	}
	if err := run(`{"model":"test-model","stream":false,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://127.0.0.1:1/blocked.png"}}]}]}`); err == nil {
		t.Fatal("expected the loopback image URL to be blocked by the SSRF guard")
	}

	// Every pipeline execution observes encode_subrequests once, including
	// the failed one: 0 (text) + 1 (data URI) + 0 (blocked before encode).
	require.Equal(t, uint64(3), histogramCount(t, reg, "llm_d_coordinator_encode_subrequests", nil))
	require.InDelta(t, 1.0, histogramSum(t, reg, "llm_d_coordinator_encode_subrequests", nil), 1e-9)

	// media_items carries one image observation per request: 0 + 1 + 1.
	require.Equal(t, uint64(3), histogramCount(t, reg, "llm_d_coordinator_media_items", map[string]string{"media_type": "image"}))
	require.InDelta(t, 2.0, histogramSum(t, reg, "llm_d_coordinator_media_items", map[string]string{"media_type": "image"}), 1e-9)

	// Only the loopback fetch started a download (dialed and refused): the
	// data URI and the text-only request never reach the HTTP client, so no
	// success series may exist at all.
	require.Equal(t, uint64(1), histogramCount(t, reg, "llm_d_coordinator_media_download_duration_seconds", map[string]string{"result": "error"}))
	requireHistogramAbsent(t, reg, "llm_d_coordinator_media_download_duration_seconds", map[string]string{"result": "success"})
}
