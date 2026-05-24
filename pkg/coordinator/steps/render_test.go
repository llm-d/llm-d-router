package steps

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestRenderStep_ParsesFullResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != gateway.PathChatCompletions+"/render" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Fatalf("expected application/json content type")
		}

		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)
		if parsed["model"] != "gpt-4o" {
			t.Fatalf("expected model gpt-4o, got %v", parsed["model"])
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{ModalityImage: {"vllm-hash-a", "vllm-hash-b"}},
				"mm_placeholders": map[string][]any{ModalityImage: {map[string]any{"offset": 1, "length": 3}, map[string]any{"offset": 4, "length": 3}}},
				"kwargs_data":     map[string][]string{ModalityImage: {"dGVuc29yLWE=", "dGVuc29yLWI="}},
			},
		})
	}))
	defer server.Close()

	step, err := NewRenderStep(map[string]any{"endpoint": gateway.PathChatCompletions + "/render"})
	if err != nil {
		t.Fatal(err)
	}
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		Body:  map[string]any{"model": "gpt-4o", "messages": []any{}},
		Model: "gpt-4o",
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0},
			{Index: 1},
		},
	}

	err = step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify token_ids were stored
	if len(reqCtx.TokenIDs) != 9 {
		t.Fatalf("expected 9 token_ids, got %d", len(reqCtx.TokenIDs))
	}
	if reqCtx.TokenIDs[0] != 1 {
		t.Fatalf("expected BOS=1, got %d", reqCtx.TokenIDs[0])
	}

	// Verify hashes from render response
	if reqCtx.MultimodalEntries[0].Hash != "vllm-hash-a" {
		t.Fatalf("expected hash vllm-hash-a, got %s", reqCtx.MultimodalEntries[0].Hash)
	}
	if reqCtx.MultimodalEntries[1].Hash != "vllm-hash-b" {
		t.Fatalf("expected hash vllm-hash-b, got %s", reqCtx.MultimodalEntries[1].Hash)
	}

	// Verify kwargs_data
	if reqCtx.MultimodalEntries[0].KwargsData != "dGVuc29yLWE=" {
		t.Fatalf("expected kwargs_data for entry 0, got %s", reqCtx.MultimodalEntries[0].KwargsData)
	}
	if reqCtx.MultimodalEntries[1].KwargsData != "dGVuc29yLWI=" {
		t.Fatalf("expected kwargs_data for entry 1, got %s", reqCtx.MultimodalEntries[1].KwargsData)
	}

	// Verify placeholders
	if reqCtx.MultimodalEntries[0].Placeholder.Offset != 1 || reqCtx.MultimodalEntries[0].Placeholder.Length != 3 {
		t.Fatalf("unexpected placeholder for entry 0: %+v", reqCtx.MultimodalEntries[0].Placeholder)
	}
	if reqCtx.MultimodalEntries[1].Placeholder.Offset != 4 || reqCtx.MultimodalEntries[1].Placeholder.Length != 3 {
		t.Fatalf("unexpected placeholder for entry 1: %+v", reqCtx.MultimodalEntries[1].Placeholder)
	}
}

func TestRenderStep_RunsEvenWithNoMultimodal(t *testing.T) {
	var called bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 2345, 6789},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{ModalityImage: {}},
				"mm_placeholders": map[string][]any{ModalityImage: {}},
				"kwargs_data":     map[string][]string{ModalityImage: {}},
			},
		})
	}))
	defer server.Close()

	step, _ := NewRenderStep(map[string]any{})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		Body:              map[string]any{"model": "test"},
		MultimodalEntries: nil,
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("render should be called even without multimodal entries")
	}
	if len(reqCtx.TokenIDs) != 3 {
		t.Fatalf("expected 3 token_ids, got %d", len(reqCtx.TokenIDs))
	}
}

func TestRenderStep_CompletionsTokenArray_SkipsRender(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("render service should not be called for token array prompt")
	}))
	defer server.Close()

	step, _ := NewRenderStep(map[string]any{})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.PathCompletions,
		Body: map[string]any{
			"model":  "test",
			"prompt": []any{float64(1), float64(2345), float64(6789)},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.TokenIDs) != 3 {
		t.Fatalf("expected 3 token_ids, got %d", len(reqCtx.TokenIDs))
	}
	if reqCtx.TokenIDs[0] != 1 || reqCtx.TokenIDs[1] != 2345 || reqCtx.TokenIDs[2] != 6789 {
		t.Fatalf("unexpected token_ids: %v", reqCtx.TokenIDs)
	}
}

func TestRenderStep_CompletionsTextPrompt_CallsRender(t *testing.T) {
	var receivedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 2345, 6789},
		})
	}))
	defer server.Close()

	step, _ := NewRenderStep(map[string]any{})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.PathCompletions,
		Body: map[string]any{
			"model":  "test",
			"prompt": "Hello, world!",
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if receivedPath != gateway.PathCompletions+"/render" {
		t.Fatalf("expected %s/render, got %s", gateway.PathCompletions, receivedPath)
	}
	if len(reqCtx.TokenIDs) != 3 {
		t.Fatalf("expected 3 token_ids, got %d", len(reqCtx.TokenIDs))
	}
	promptTokens, ok := reqCtx.Body["prompt"].([]int)
	if !ok {
		t.Fatalf("expected prompt to be replaced with []int, got %T", reqCtx.Body["prompt"])
	}
	if len(promptTokens) != 3 || promptTokens[0] != 1 {
		t.Fatalf("unexpected prompt tokens: %v", promptTokens)
	}
}

func TestRenderStep_ServiceError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer server.Close()

	step, _ := NewRenderStep(map[string]any{})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		Body:              map[string]any{"model": "test"},
		MultimodalEntries: []pipeline.MultimodalEntry{{Index: 0}},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
}
