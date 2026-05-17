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

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/connector"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	"github.com/llm-d/coordinator/pkg/steps"
)

func TestFullPipeline_AllConnectorCombinations(t *testing.T) {
	cases := []struct {
		kvConnector     string
		ecConnector     string
		wantECInPrefill bool // ec_transfer_params should be present in prefill body
	}{
		{connector.KVNIXLv2, connector.ECNIXLv2, true},
		{connector.KVNIXLv2, connector.ECSharedStorage, false},
		{connector.KVSharedStorage, connector.ECNIXLv2, true},
		{connector.KVSharedStorage, connector.ECSharedStorage, false},
	}

	for _, tc := range cases {
		t.Run(tc.kvConnector+"+"+tc.ecConnector, func(t *testing.T) {
			imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "image/png")
				_, _ = w.Write([]byte("fake-image-data"))
			}))
			defer imageServer.Close()

			renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"token_ids": []int{1, 32000, 32000, 32000, 2345, 6789},
					"features": map[string]any{
						"mm_hashes":       map[string][]string{"image": {"vllm-hash-img0"}},
						"mm_placeholders": map[string][]any{"image": {map[string]any{"offset": 1, "length": 3}}},
						"kwargs_data":     map[string][]string{"image": {"dGVuc29yLWRhdGE="}},
					},
				})
			}))
			defer renderServer.Close()

			var mu sync.Mutex
			var capturedPrefillBody map[string]any

			gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case strings.HasPrefix(r.URL.Path, "/encode"):
					body, _ := io.ReadAll(r.Body)
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
				case strings.HasPrefix(r.URL.Path, "/prefill"):
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
				case strings.HasPrefix(r.URL.Path, "/decode"):
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
				{Type: "render", Params: map[string]any{"endpoint": "/v1/chat/completions/render"}},
				{Type: "encode", Params: map[string]any{steps.ParamGatewayPath: gateway.DefaultGeneratePath, steps.ParamECConnector: tc.ecConnector}},
				{Type: "prefill", Params: map[string]any{steps.ParamGatewayPath: gateway.DefaultGeneratePath, steps.ParamKVConnector: tc.kvConnector, steps.ParamECConnector: tc.ecConnector}},
				{Type: "decode", Params: map[string]any{steps.ParamKVConnector: tc.kvConnector}},
			}

			pipelineSteps := make([]pipeline.Step, 0, len(stepConfigs))
			for _, sc := range stepConfigs {
				step, err := pipeline.Build(sc.Type, sc.Params)
				if err != nil {
					t.Fatalf("building step %s: %v", sc.Type, err)
				}
				type gatewayAware interface{ SetGatewayClient(*gateway.Client) }
				if ga, ok := step.(gatewayAware); ok {
					ga.SetGatewayClient(gwClient)
				}
				type renderAware interface{ SetServiceAddress(string) }
				if ra, ok := step.(renderAware); ok {
					ra.SetServiceAddress(renderServer.URL)
				}
				pipelineSteps = append(pipelineSteps, step)
			}

			requestBody := fmt.Sprintf(`{
				"model": "test-model", "stream": false,
				"messages": [{"role": "user", "content": [
					{"type": "text", "text": "What is in this image?"},
					{"type": "image_url", "image_url": {"url": "%s/image.png"}}
				]}]
			}`, imageServer.URL)

			recorder := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "test-" + tc.kvConnector + "+" + tc.ecConnector,
				OriginalPath:     "/v1/chat/completions",
				OriginalBody:     []byte(requestBody),
				Model:            "test-model",
				KVTransferParams: make(map[string]any),
				ResponseWriter:   recorder,
				Flusher:          recorder,
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
			_, hasEC := captured["ec_transfer_params"]
			if tc.wantECInPrefill && !hasEC {
				t.Error("expected ec_transfer_params in prefill body")
			}
			if !tc.wantECInPrefill && hasEC {
				t.Error("unexpected ec_transfer_params in prefill body")
			}
		})
	}
}

func TestFullPipeline_Integration(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("fake-image-data"))
	}))
	defer imageServer.Close()

	renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 2345, 6789},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{"image": {"vllm-hash-img0"}},
				"mm_placeholders": map[string][]any{"image": {map[string]any{"offset": 1, "length": 3}}},
				"kwargs_data":     map[string][]string{"image": {"dGVuc29yLWRhdGE="}},
			},
		})
	}))
	defer renderServer.Close()

	gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/encode"):
			body, _ := io.ReadAll(r.Body)
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
		case strings.HasPrefix(r.URL.Path, "/prefill"):
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{
					"block_id":  "abc123",
					"peer_host": "10.0.0.2",
					"peer_port": 5502,
				},
			})
		case strings.HasPrefix(r.URL.Path, "/decode"):
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
		{Type: "render", Params: map[string]any{"endpoint": "/v1/chat/completions/render"}},
		{Type: "encode", Params: map[string]any{steps.ParamGatewayPath: gateway.DefaultGeneratePath, steps.ParamECConnector: connector.ECNIXLv2}},
		{Type: "prefill", Params: map[string]any{steps.ParamGatewayPath: gateway.DefaultGeneratePath, steps.ParamECConnector: connector.ECNIXLv2}},
		{Type: "decode", Params: map[string]any{}},
	}

	pipelineSteps := make([]pipeline.Step, 0, len(stepConfigs))
	for _, sc := range stepConfigs {
		step, err := pipeline.Build(sc.Type, sc.Params)
		if err != nil {
			t.Fatalf("building step %s: %v", sc.Type, err)
		}

		type gatewayAware interface{ SetGatewayClient(*gateway.Client) }
		if ga, ok := step.(gatewayAware); ok {
			ga.SetGatewayClient(gwClient)
		}
		type renderAware interface{ SetServiceAddress(string) }
		if ra, ok := step.(renderAware); ok {
			ra.SetServiceAddress(renderServer.URL)
		}

		pipelineSteps = append(pipelineSteps, step)
	}

	p := pipeline.New(pipelineSteps)

	requestBody := fmt.Sprintf(`{
		"model": "test-model",
		"stream": false,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "What is in this image?"},
					{"type": "image_url", "image_url": {"url": "%s/image.png"}}
				]
			}
		]
	}`, imageServer.URL)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "test-123",
		OriginalPath:     "/v1/chat/completions",
		OriginalBody:     []byte(requestBody),
		Stream:           false,
		Model:            "test-model",
		KVTransferParams: make(map[string]any),
		ResponseWriter:   recorder,
		Flusher:          recorder,
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
}
