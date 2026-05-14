package pipeline_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	_ "github.com/llm-d/coordinator/pkg/steps"
)

func TestTextOnlyRequest_SkipsMediaDownloadAndEncode(t *testing.T) {
	var encodeCalled atomic.Bool
	var renderCalled atomic.Bool
	var prefillCalled atomic.Bool

	renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		renderCalled.Store(true)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 2345, 6789},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{"image": {}},
				"mm_placeholders": map[string][]any{"image": {}},
				"kwargs_data":     map[string][]string{"image": {}},
			},
		})
	}))
	defer renderServer.Close()

	gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/encode"):
			encodeCalled.Store(true)
			t.Error("encode should not be called for text-only request")
		case strings.HasPrefix(r.URL.Path, "/prefill"):
			prefillCalled.Store(true)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "b1"},
			})
		case strings.HasPrefix(r.URL.Path, "/decode"):
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{
					{"message": map[string]any{"role": "assistant", "content": "Hi there!"}},
				},
			})
		default:
			http.Error(w, "unexpected", 500)
		}
	}))
	defer gatewayServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gatewayServer.URL})

	stepConfigs := []config.StepConfig{
		{Type: "replace-media-urls", Params: map[string]any{}},
		{Type: "render", Params: map[string]any{}},
		{Type: "encode", Params: map[string]any{}},
		{Type: "prefill", Params: map[string]any{}},
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

	recorder := httptest.NewRecorder()
	body := map[string]any{
		"model":  "llama-3",
		"stream": false,
		"messages": []any{
			map[string]any{"role": "user", "content": "Hello, how are you?"},
		},
	}

	reqCtx := &pipeline.RequestContext{
		RequestID:        "text-only-test",
		OriginalPath:     "/v1/chat/completions",
		OriginalBody:     mustJSON(body),
		Body:             body,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: make(map[string]any),
		ResponseWriter:   recorder,
		Flusher:          recorder,
	}

	err := p.Execute(t.Context(), reqCtx)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	if encodeCalled.Load() {
		t.Fatal("encode was called for text-only request")
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected 0 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
	if !renderCalled.Load() {
		t.Fatal("render should be called for all requests")
	}
	if !prefillCalled.Load() {
		t.Fatal("prefill should be called for text-only request")
	}

	result := recorder.Result()
	respBody, _ := io.ReadAll(result.Body)
	if !strings.Contains(string(respBody), "Hi there!") {
		t.Fatalf("expected decode response, got: %s", string(respBody))
	}
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}
