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
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
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
			{Modality: ModalityImage},
			{Modality: ModalityImage},
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

// TestRenderStep_ChatCompletions_MultipleModalities covers the chat-completions
// path with entries in three modalities (image, audio, video). The render
// server returns per-modality slices, and the step must fill each entry
// with the hash, placeholder, and kwargs from the slot that matches its
// modality and its per-modality position. Without this test the multi-
// modality bounds check and per-modality walker in render.go have no
// coverage on the chat-completions path.
func TestRenderStep_ChatCompletions_MultipleModalities(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 51000, 51000, 71000, 71000, 71000},
			"features": map[string]any{
				"mm_hashes": map[string][]string{
					ModalityImage: {"img-hash"},
					ModalityAudio: {"aud-hash"},
					ModalityVideo: {"vid-hash"},
				},
				"mm_placeholders": map[string][]any{
					ModalityImage: {map[string]any{"offset": 1, "length": 3}},
					ModalityAudio: {map[string]any{"offset": 4, "length": 2}},
					ModalityVideo: {map[string]any{"offset": 6, "length": 3}},
				},
				"kwargs_data": map[string][]string{
					ModalityImage: {"aW1n"},
					ModalityAudio: {"YXVk"},
					ModalityVideo: {"dmlk"},
				},
			},
		})
	}))
	defer server.Close()

	step, err := NewRenderStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	step.(*RenderStep).SetServiceAddress(server.URL)

	// Entries are pre-populated in walker order (image, audio, video), the
	// same order replace_media_urls would produce for a request that mixes
	// the three modalities. Render fills in Hash/Placeholder/KwargsData
	// per entry from the matching per-modality slot.
	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.PathChatCompletions,
		Body:         map[string]any{"model": "test-model", "messages": []any{}},
		Model:        "test-model",
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: ModalityImage},
			{Modality: ModalityAudio},
			{Modality: ModalityVideo},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := []struct {
		hash     string
		kwargs   string
		modality string
		offset   int
		length   int
	}{
		{"img-hash", "aW1n", ModalityImage, 1, 3},
		{"aud-hash", "YXVk", ModalityAudio, 4, 2},
		{"vid-hash", "dmlk", ModalityVideo, 6, 3},
	}
	for i, w := range want {
		got := reqCtx.MultimodalEntries[i]
		if got.Hash != w.hash || got.KwargsData != w.kwargs ||
			got.Modality != w.modality ||
			got.Placeholder.Offset != w.offset || got.Placeholder.Length != w.length {
			t.Errorf("entry %d = %+v, want hash=%q kwargs=%q modality=%q offset=%d length=%d",
				i, got, w.hash, w.kwargs, w.modality, w.offset, w.length)
		}
	}
}

// TestRenderStep_ChatCompletions_RejectsWrongModalitySplit covers the two
// response checks on this path, and the difference between them.
//
// The aggregate check sums every per-modality slice and compares the total to
// the entry count, so it catches a response that returns the wrong number of
// items overall. It cannot see a response that returns the right number sorted
// into the wrong modalities: the per-entry guard is what catches that, by
// requiring each entry to find a slot in its own modality's slice.
//
// Every case but the last keeps all three totals equal to the entry count, so
// only the per-entry guard can reject them. They misfile a different field each,
// because that guard tests mm_hashes, mm_placeholders, and kwargs_data
// separately and one misfiled field would otherwise leave two clauses unrun.
//
// The short_modality_slice case is the one that pins the comparison itself. In
// the other cases the entry's modality is absent from the response, so a slice
// length of zero is enough to reject them, and a guard weakened to check only
// for an empty slice would still pass. There the slice exists and is one item
// short, which is the shape a weakened guard would answer by handing the entry
// the previous entry's hash instead of failing.
//
// A malformed render response is the service's fault, not the caller's, so
// these are plain errors rather than ErrBadRequest and surface as 5xx.
func TestRenderStep_ChatCompletions_RejectsWrongModalitySplit(t *testing.T) {
	placeholder := func(offset, length int) any {
		return map[string]any{"offset": offset, "length": length}
	}
	imageAudio := []pipeline.MultimodalEntry{{Modality: ModalityImage}, {Modality: ModalityAudio}}

	for _, tc := range []struct {
		name     string
		entries  []pipeline.MultimodalEntry
		features map[string]any
		wantMsg  string
	}{
		{
			// Both hashes tagged audio. Totals are 2, so the aggregate check
			// passes, but the image entry finds an absent (nil) image slice.
			name:    "hashes_under_wrong_modality",
			entries: imageAudio,
			features: map[string]any{
				"mm_hashes":       map[string][]string{ModalityAudio: {"aud-hash", "img-hash"}},
				"mm_placeholders": map[string][]any{ModalityAudio: {placeholder(1, 3), placeholder(4, 2)}},
				"kwargs_data":     map[string][]string{ModalityAudio: {"YXVk", "aW1n"}},
			},
			wantMsg: ModalityImage,
		},
		{
			// Two image entries but one image hash, and the audio slice absorbs
			// the extra. Totals are 3, and the image slice is present and one
			// short, so rejecting requires comparing idx against its length.
			name: "short_modality_slice",
			entries: []pipeline.MultimodalEntry{
				{Modality: ModalityImage}, {Modality: ModalityImage}, {Modality: ModalityAudio},
			},
			features: map[string]any{
				"mm_hashes": map[string][]string{
					ModalityImage: {"img-hash"},
					ModalityAudio: {"aud-hash", "extra-hash"},
				},
				"mm_placeholders": map[string][]any{
					ModalityImage: {placeholder(1, 3)},
					ModalityAudio: {placeholder(4, 2), placeholder(6, 2)},
				},
				"kwargs_data": map[string][]string{
					ModalityImage: {"aW1n"},
					ModalityAudio: {"YXVk", "ZXh0"},
				},
			},
			wantMsg: ModalityImage,
		},
		{
			// Hashes and placeholders split correctly; kwargs_data puts both
			// items under image, so the audio entry runs out on kwargs alone.
			name:    "kwargs_split_disagrees",
			entries: imageAudio,
			features: map[string]any{
				"mm_hashes": map[string][]string{
					ModalityImage: {"img-hash"},
					ModalityAudio: {"aud-hash"},
				},
				"mm_placeholders": map[string][]any{
					ModalityImage: {placeholder(1, 3)},
					ModalityAudio: {placeholder(4, 2)},
				},
				"kwargs_data": map[string][]string{ModalityImage: {"aW1n", "YXVk"}},
			},
			wantMsg: ModalityAudio,
		},
		{
			// Three hashes for two entries: the aggregate check rejects this
			// before the per-entry walk starts.
			name:    "total_count_mismatch",
			entries: imageAudio,
			features: map[string]any{
				"mm_hashes": map[string][]string{
					ModalityImage: {"img-hash", "extra-hash"},
					ModalityAudio: {"aud-hash"},
				},
				"mm_placeholders": map[string][]any{
					ModalityImage: {placeholder(1, 3)},
					ModalityAudio: {placeholder(4, 2)},
				},
				"kwargs_data": map[string][]string{
					ModalityImage: {"aW1n"},
					ModalityAudio: {"YXVk"},
				},
			},
			wantMsg: "mm_hashes",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"token_ids": []int{1, 32000, 32000, 32000, 51000, 51000},
					"features":  tc.features,
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
				Body:         map[string]any{"model": "test-model", "messages": []any{}},
				Model:        "test-model",
				// Cloned: Execute fills entries in place as it walks, and the
				// two-entry fixture is shared between cases.
				MultimodalEntries: slices.Clone(tc.entries),
			}

			err = step.Execute(context.Background(), reqCtx)
			if err == nil {
				t.Fatalf("expected an error, got entries %+v", reqCtx.MultimodalEntries)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("error %q should name %q", err, tc.wantMsg)
			}
			if errors.Is(err, pipeline.ErrBadRequest) {
				t.Errorf("a malformed render response is not a client error, got %v", err)
			}
		})
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
			{Modality: ModalityImage},
			{Modality: ModalityImage},
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
		MultimodalEntries: []pipeline.MultimodalEntry{{Modality: ModalityImage}},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error at limit: %v", err)
	}
}

func TestRenderStep_PlaceholderLimitOverflow(t *testing.T) {
	step, err := NewRenderStep(nil, map[string]any{"max_total_placeholder_tokens": 5})
	if err != nil {
		t.Fatalf("NewRenderStep: %v", err)
	}
	rs := step.(*RenderStep)

	// Two lengths whose sum overflows int and wraps negative. Without the
	// overflow guard, total > max is false and the limit is silently bypassed.
	entries := []pipeline.MultimodalEntry{
		{Modality: ModalityImage, Placeholder: pipeline.PlaceholderRange{Length: math.MaxInt}},
		{Modality: ModalityImage, Placeholder: pipeline.PlaceholderRange{Length: math.MaxInt}},
	}
	err = rs.checkPlaceholderLimit(entries)
	if err == nil {
		t.Fatal("expected error for overflowing placeholder length sum")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
	if !strings.Contains(err.Error(), "too many placeholder tokens") {
		t.Fatalf("unexpected error: %v", err)
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
		MultimodalEntries: []pipeline.MultimodalEntry{{Modality: ModalityImage}},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
}

func TestRenderStep_GenerateFormat_TextOnly(t *testing.T) {
	step, err := NewRenderStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.DefaultGeneratePath,
		Body: map[string]any{
			"model":     "test-model",
			"token_ids": []any{float64(1), float64(2345), float64(6789)},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.TokenIDs) != 3 {
		t.Fatalf("expected 3 token IDs, got %d: %v", len(reqCtx.TokenIDs), reqCtx.TokenIDs)
	}
	if reqCtx.TokenIDs[0] != 1 || reqCtx.TokenIDs[1] != 2345 || reqCtx.TokenIDs[2] != 6789 {
		t.Fatalf("unexpected token IDs: %v", reqCtx.TokenIDs)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestRenderStep_GenerateFormat_Multimodal(t *testing.T) {
	step, err := NewRenderStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.DefaultGeneratePath,
		Body: map[string]any{
			"model":     "test-model",
			"token_ids": []any{float64(1), float64(32000), float64(32000), float64(32000), float64(2)},
			"features": map[string]any{
				"mm_hashes": map[string]any{"image": []any{"abc123"}},
				"mm_placeholders": map[string]any{"image": []any{
					map[string]any{"offset": float64(1), "length": float64(3)},
				}},
				"kwargs_data": map[string]any{"image": []any{"dGVuc29y"}},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.TokenIDs) != 5 {
		t.Fatalf("expected 5 token IDs, got %d", len(reqCtx.TokenIDs))
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}
	e := reqCtx.MultimodalEntries[0]
	if e.Hash != "abc123" || e.Placeholder.Offset != 1 || e.Placeholder.Length != 3 || e.KwargsData != "dGVuc29y" {
		t.Errorf("unexpected entry: %+v", e)
	}
}

func TestRenderStep_GenerateFormat_MultipleImages(t *testing.T) {
	step, err := NewRenderStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.DefaultGeneratePath,
		Body: map[string]any{
			"model":     "test-model",
			"token_ids": []any{float64(1), float64(32000), float64(32000), float64(3), float64(41000), float64(41000), float64(2)},
			"features": map[string]any{
				"mm_hashes": map[string]any{"image": []any{"abc123", "def456"}},
				"mm_placeholders": map[string]any{"image": []any{
					map[string]any{"offset": float64(1), "length": float64(2)},
					map[string]any{"offset": float64(4), "length": float64(2)},
				}},
				"kwargs_data": map[string]any{"image": []any{"dGVuc29yMA==", "dGVuc29yMQ=="}},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 2 {
		t.Fatalf("expected 2 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
	want := []pipeline.MultimodalEntry{
		{Modality: ModalityImage, Hash: "abc123", KwargsData: "dGVuc29yMA==", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
		{Modality: ModalityImage, Hash: "def456", KwargsData: "dGVuc29yMQ==", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 2}},
	}
	for i, w := range want {
		if reqCtx.MultimodalEntries[i] != w {
			t.Errorf("entry %d: expected %+v, got %+v", i, w, reqCtx.MultimodalEntries[i])
		}
	}
}

// TestRenderStep_GenerateFormat_MultipleModalities exercises the
// generate path with mm features carrying image, audio, and video
// entries in one request. The render step walks modalities in
// alphabetical order (audio, image, video) so MultimodalEntries comes
// back tagged with the right modality per slot and each entry's Hash /
// KwargsData / Placeholder pair through cleanly.
func TestRenderStep_GenerateFormat_MultipleModalities(t *testing.T) {
	step, err := NewRenderStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.DefaultGeneratePath,
		Body: map[string]any{
			"model": "test-model",
			// 10 tokens, three non-overlapping placeholder spans below.
			"token_ids": []any{
				float64(1), float64(51000), float64(51000),
				float64(3), float64(32000), float64(32000), float64(32000),
				float64(4), float64(71000), float64(71000),
			},
			"features": map[string]any{
				"mm_hashes": map[string]any{
					"audio": []any{"aud-hash"},
					"image": []any{"img-hash"},
					"video": []any{"vid-hash"},
				},
				"mm_placeholders": map[string]any{
					"audio": []any{map[string]any{"offset": float64(1), "length": float64(2)}},
					"image": []any{map[string]any{"offset": float64(4), "length": float64(3)}},
					"video": []any{map[string]any{"offset": float64(8), "length": float64(2)}},
				},
				"kwargs_data": map[string]any{
					"audio": []any{"YXVkaW8="},
					"image": []any{"aW1hZ2U="},
					"video": []any{"dmlkZW8="},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 3 {
		t.Fatalf("expected 3 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
	want := []pipeline.MultimodalEntry{
		{Modality: ModalityAudio, Hash: "aud-hash", KwargsData: "YXVkaW8=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
		{Modality: ModalityImage, Hash: "img-hash", KwargsData: "aW1hZ2U=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
		{Modality: ModalityVideo, Hash: "vid-hash", KwargsData: "dmlkZW8=", Placeholder: pipeline.PlaceholderRange{Offset: 8, Length: 2}},
	}
	for i, w := range want {
		if reqCtx.MultimodalEntries[i] != w {
			t.Errorf("entry %d: expected %+v, got %+v", i, w, reqCtx.MultimodalEntries[i])
		}
	}
}

func TestRenderStep_GenerateFormat_MalformedFeatures(t *testing.T) {
	for _, tc := range []struct {
		name     string
		features any
	}{
		{"string", "not-an-object"},
		{"array", []any{"a", "b"}},
		{"number", float64(42)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step, _ := NewRenderStep(nil, map[string]any{})
			reqCtx := &pipeline.RequestContext{
				OriginalPath: gateway.DefaultGeneratePath,
				Body: map[string]any{
					"model":     "test-model",
					"token_ids": []any{float64(1), float64(2), float64(3)},
					"features":  tc.features,
				},
			}
			err := step.Execute(context.Background(), reqCtx)
			if err == nil {
				t.Fatal("expected error for malformed features")
			}
			if !errors.Is(err, pipeline.ErrBadRequest) {
				t.Errorf("expected ErrBadRequest, got %v", err)
			}
		})
	}
}

func TestRenderStep_GenerateFormat_PlaceholderOutOfBounds(t *testing.T) {
	for _, tc := range []struct {
		name   string
		offset float64
		length float64
	}{
		{"offset_out_of_range", 5, 1},
		{"length_exceeds_prompt", 1, 9007199254740992},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step, _ := NewRenderStep(nil, map[string]any{})
			reqCtx := &pipeline.RequestContext{
				OriginalPath: gateway.DefaultGeneratePath,
				Body: map[string]any{
					"model":     "test-model",
					"token_ids": []any{float64(1), float64(32000), float64(32000), float64(2)},
					"features": map[string]any{
						"mm_hashes": map[string]any{"image": []any{"abc123"}},
						"mm_placeholders": map[string]any{"image": []any{
							map[string]any{"offset": tc.offset, "length": tc.length},
						}},
						"kwargs_data": map[string]any{"image": []any{"dGVuc29y"}},
					},
				},
			}
			err := step.Execute(context.Background(), reqCtx)
			if err == nil {
				t.Fatal("expected error for out-of-bounds placeholder")
			}
			if !errors.Is(err, pipeline.ErrBadRequest) {
				t.Errorf("expected ErrBadRequest, got %v", err)
			}
		})
	}
}

func TestRenderStep_GenerateFormat_InvalidSamplingParams(t *testing.T) {
	for _, tc := range []struct {
		name           string
		samplingParams any
	}{
		{"array", []any{float64(1), float64(2)}},
		{"string", "greedy"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step, _ := NewRenderStep(nil, map[string]any{})
			reqCtx := &pipeline.RequestContext{
				OriginalPath: gateway.DefaultGeneratePath,
				Body: map[string]any{
					"model":           "test-model",
					"token_ids":       []any{float64(1), float64(2), float64(3)},
					"sampling_params": tc.samplingParams,
				},
			}
			err := step.Execute(context.Background(), reqCtx)
			if err == nil {
				t.Fatal("expected error for non-object sampling_params")
			}
			if !errors.Is(err, pipeline.ErrBadRequest) {
				t.Errorf("expected ErrBadRequest, got %v", err)
			}
		})
	}
}

func TestRenderStep_GenerateFormat_NullFeatures(t *testing.T) {
	step, _ := NewRenderStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.DefaultGeneratePath,
		Body: map[string]any{
			"model":     "test-model",
			"token_ids": []any{float64(1), float64(2), float64(3)},
			"features":  nil,
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error for null features: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestRenderStep_GenerateFormat_MissingTokenIDs(t *testing.T) {
	step, _ := NewRenderStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		OriginalPath: gateway.DefaultGeneratePath,
		Body:         map[string]any{"model": "test-model"},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for missing token_ids")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Errorf("expected ErrBadRequest, got %v", err)
	}
}
