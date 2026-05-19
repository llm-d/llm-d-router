package steps

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestConditionalDecodeStep_CacheHit(t *testing.T) {
	var receivedPath string
	var receivedBody map[string]any
	var receivedPreferHeader string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		receivedPreferHeader = r.Header.Get("Prefer")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &receivedBody)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "cached response"}},
			},
		})
	}))
	defer srv.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: srv.URL})
	step, err := NewConditionalDecodeStep(nil)
	if err != nil {
		t.Fatal(err)
	}
	step.(*ConditionalDecodeStep).SetGatewayClient(gwClient)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:      "req-1",
		OriginalPath:   "/v1/chat/completions",
		Body:           map[string]any{"model": "test-model", "stream": false, "messages": []any{}},
		ResponseWriter: recorder,
		Flusher:        recorder,
	}

	err = step.Execute(context.Background(), reqCtx)
	if !errors.Is(err, pipeline.ErrPipelineDone) {
		t.Fatalf("expected ErrPipelineDone, got %v", err)
	}

	if receivedPath != decodePath {
		t.Fatalf("expected path %s, got %s", decodePath, receivedPath)
	}
	if receivedBody["model"] != "test-model" {
		t.Fatalf("expected model test-model in request body, got %v", receivedBody["model"])
	}
	if receivedPreferHeader != "if-available" {
		t.Fatalf("expected Prefer: if-available header, got %q", receivedPreferHeader)
	}

	result := recorder.Result()
	if result.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", result.StatusCode)
	}

	respBody, _ := io.ReadAll(result.Body)
	if !strings.Contains(string(respBody), "cached response") {
		t.Fatalf("expected 'cached response' in body, got: %s", string(respBody))
	}
}

func TestConditionalDecodeStep_CacheHit_Streaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)

		events := []string{
			`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
			`data: {"choices":[{"delta":{"content":" world"}}]}`,
			`data: [DONE]`,
		}
		for _, event := range events {
			fmt.Fprintf(w, "%s\n\n", event)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: srv.URL})
	step, _ := NewConditionalDecodeStep(nil)
	step.(*ConditionalDecodeStep).SetGatewayClient(gwClient)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:      "req-1",
		OriginalPath:   "/v1/chat/completions",
		Body:           map[string]any{"model": "test", "stream": true},
		ResponseWriter: recorder,
		Flusher:        recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	if !errors.Is(err, pipeline.ErrPipelineDone) {
		t.Fatalf("expected ErrPipelineDone, got %v", err)
	}

	result := recorder.Result()
	if result.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %s", result.Header.Get("Content-Type"))
	}

	respBody, _ := io.ReadAll(result.Body)
	body := string(respBody)
	if !strings.Contains(body, `"content":"Hello"`) {
		t.Fatalf("expected Hello event, got: %s", body)
	}
	if !strings.Contains(body, "[DONE]") {
		t.Fatalf("expected [DONE] event, got: %s", body)
	}
}

func TestConditionalDecodeStep_CacheMiss_412(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusPreconditionFailed)
		_, _ = w.Write([]byte("no cache available"))
	}))
	defer srv.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: srv.URL})
	step, _ := NewConditionalDecodeStep(nil)
	step.(*ConditionalDecodeStep).SetGatewayClient(gwClient)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:      "req-1",
		OriginalPath:   "/v1/chat/completions",
		Body:           map[string]any{"model": "test"},
		ResponseWriter: recorder,
		Flusher:        recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("expected nil error on cache miss, got %v", err)
	}

	result := recorder.Result()
	if result.StatusCode != http.StatusOK {
		t.Fatalf("expected recorder default 200 (nothing written), got %d", result.StatusCode)
	}
	respBody, _ := io.ReadAll(result.Body)
	if len(respBody) != 0 {
		t.Fatalf("expected empty response body on cache miss, got: %s", string(respBody))
	}
}

func TestConditionalDecodeStep_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer srv.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: srv.URL})
	step, _ := NewConditionalDecodeStep(nil)
	step.(*ConditionalDecodeStep).SetGatewayClient(gwClient)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:      "req-1",
		OriginalPath:   "/v1/chat/completions",
		Body:           map[string]any{"model": "test"},
		ResponseWriter: recorder,
		Flusher:        recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	if !errors.Is(err, pipeline.ErrPipelineDone) {
		t.Fatalf("expected ErrPipelineDone, got %v", err)
	}

	result := recorder.Result()
	if result.StatusCode != http.StatusInternalServerError {
		t.Fatalf("expected 500 forwarded, got %d", result.StatusCode)
	}

	respBody, _ := io.ReadAll(result.Body)
	if !strings.Contains(string(respBody), "internal error") {
		t.Fatalf("expected error body forwarded, got: %s", string(respBody))
	}
}
