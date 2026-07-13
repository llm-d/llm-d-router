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
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
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

	step, err := NewRenderStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.PathChatCompletions,
		Body:         map[string]any{"model": "gpt-4o", "messages": []any{}},
		Model:        "gpt-4o",
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

	step, _ := NewRenderStep(nil, map[string]any{})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath:      gateway.PathChatCompletions,
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

	step, _ := NewRenderStep(nil, map[string]any{})
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
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"token_ids": []int{1, 2345, 6789}},
		})
	}))
	defer server.Close()

	step, _ := NewRenderStep(nil, map[string]any{})
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

func TestRenderStep_RejectsTooManyTotalTokens_ChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{ModalityImage: {}},
				"mm_placeholders": map[string][]any{ModalityImage: {}},
				"kwargs_data":     map[string][]string{ModalityImage: {}},
			},
		})
	}))
	defer server.Close()

	step, _ := NewRenderStep(nil, map[string]any{"max_total_tokens": 5})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.PathChatCompletions,
		Body:         map[string]any{"model": "test"},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for exceeding max_total_tokens")
	}
	if !strings.Contains(err.Error(), "too many total tokens") {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(err.Error(), "got 10") || !strings.Contains(err.Error(), "max 5") {
		t.Fatalf("error should include counts: %v", err)
	}
}

func TestRenderStep_RejectsTooManyTotalTokens_CompletionsString(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"token_ids": []int{1, 2, 3, 4, 5, 6, 7}},
		})
	}))
	defer server.Close()

	step, _ := NewRenderStep(nil, map[string]any{"max_total_tokens": 4})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.PathCompletions,
		Body:         map[string]any{"model": "test", "prompt": "some text"},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for exceeding max_total_tokens")
	}
	if !strings.Contains(err.Error(), "too many total tokens") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRenderStep_RejectsTooManyTotalTokens_CompletionsTokenArray(t *testing.T) {
	step, _ := NewRenderStep(nil, map[string]any{"max_total_tokens": 2})
	step.(*RenderStep).SetServiceAddress("http://unused")

	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.PathCompletions,
		Body:         map[string]any{"model": "test", "prompt": []any{float64(1), float64(2), float64(3)}},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for exceeding max_total_tokens on token-array prompt")
	}
	if !strings.Contains(err.Error(), "too many total tokens") {
		t.Fatalf("unexpected error: %v", err)
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("token-limit rejection should be a client error: %v", err)
	}
}

func TestRenderStep_UpstreamErrorCarriesStatus(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusInternalServerError} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(status)
		}))

		step, _ := NewRenderStep(nil, nil)
		step.(*RenderStep).SetServiceAddress(server.URL)

		reqCtx := &pipeline.RequestContext{
			OriginalPath: gateway.PathChatCompletions,
			Body:         map[string]any{"model": "test"},
		}

		err := step.Execute(context.Background(), reqCtx)
		server.Close()
		if err == nil {
			t.Fatalf("expected error when render service returns %d", status)
		}
		if errors.Is(err, pipeline.ErrBadRequest) {
			t.Fatalf("upstream failure must not be a coordinator-side bad request: %v", err)
		}
		var upstream *pipeline.UpstreamError
		if !errors.As(err, &upstream) {
			t.Fatalf("expected an UpstreamError, got %v", err)
		}
		if upstream.StatusCode != status {
			t.Fatalf("expected status %d, got %d", status, upstream.StatusCode)
		}
		if upstream.Step != RenderStepName {
			t.Fatalf("expected step %q, got %q", RenderStepName, upstream.Step)
		}
	}
}

func TestRenderStep_RejectsTooManyPlaceholderTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 100, 100, 100, 100, 100, 100, 100, 200},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{ModalityImage: {"h0", "h1"}},
				"mm_placeholders": map[string][]any{ModalityImage: {map[string]any{"offset": 1, "length": 4}, map[string]any{"offset": 5, "length": 3}}},
				"kwargs_data":     map[string][]string{ModalityImage: {"AAAA", "AAAA"}},
			},
		})
	}))
	defer server.Close()

	step, _ := NewRenderStep(nil, map[string]any{"max_total_placeholder_tokens": 5})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.PathChatCompletions,
		Body:         map[string]any{"model": "test"},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0},
			{Index: 1},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for exceeding max_total_placeholder_tokens")
	}
	if !strings.Contains(err.Error(), "too many placeholder tokens") {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(err.Error(), "got 7") || !strings.Contains(err.Error(), "max 5") {
		t.Fatalf("error should include counts: %v", err)
	}
}

func TestRenderStep_AllowsAtPlaceholderLimit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 100, 100, 100, 200},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{ModalityImage: {"h0"}},
				"mm_placeholders": map[string][]any{ModalityImage: {map[string]any{"offset": 1, "length": 3}}},
				"kwargs_data":     map[string][]string{ModalityImage: {"AAAA"}},
			},
		})
	}))
	defer server.Close()

	step, _ := NewRenderStep(nil, map[string]any{"max_total_placeholder_tokens": 3})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath:      gateway.PathChatCompletions,
		Body:              map[string]any{"model": "test"},
		MultimodalEntries: []pipeline.MultimodalEntry{{Index: 0}},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error at limit: %v", err)
	}
}

func TestRenderStep_RejectsNegativeLimits(t *testing.T) {
	if _, err := NewRenderStep(nil, map[string]any{"max_total_tokens": -1}); err == nil {
		t.Fatal("expected error for negative max_total_tokens")
	}
	if _, err := NewRenderStep(nil, map[string]any{"max_total_placeholder_tokens": -1}); err == nil {
		t.Fatal("expected error for negative max_total_placeholder_tokens")
	}
}

func TestRenderStep_ServiceError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer server.Close()

	step, _ := NewRenderStep(nil, map[string]any{})
	step.(*RenderStep).SetServiceAddress(server.URL)

	reqCtx := &pipeline.RequestContext{
		OriginalPath:      gateway.PathChatCompletions,
		Body:              map[string]any{"model": "test"},
		MultimodalEntries: []pipeline.MultimodalEntry{{Index: 0}},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
}
