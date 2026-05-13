package pipeline_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	_ "github.com/llm-d/coordinator/pkg/steps"
)

func TestFullPipeline_Integration(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Write([]byte("fake-image-data"))
	}))
	defer imageServer.Close()

	renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
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
			json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{
					"peer_host": "10.0.0.1",
					"peer_port": 5501,
				},
			})
		case strings.HasPrefix(r.URL.Path, "/prefill"):
			json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{
					"block_id":  "abc123",
					"peer_host": "10.0.0.2",
					"peer_port": 5502,
				},
			})
		case strings.HasPrefix(r.URL.Path, "/decode"):
			json.NewEncoder(w).Encode(map[string]any{
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
		{Type: "encode", Params: map[string]any{"gateway_path": "/inference/v1/generate"}},
		{Type: "prefill", Params: map[string]any{"gateway_path": "/inference/v1/generate"}},
		{Type: "decode", Params: map[string]any{}},
	}

	var pipelineSteps []pipeline.Step
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

	json.Unmarshal([]byte(requestBody), &reqCtx.Body)

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
