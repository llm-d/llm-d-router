//go:build e2e

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

// Package e2e contains end-to-end tests that exercise the RenderStep against a
// real rendering service. These tests are skipped during normal `go test ./...`
// runs (build tag `e2e`).
//
// To run:
//
//	go test -tags=e2e ./test/real-vllm-e2e/...
//
// Configuration via environment variables:
//
//	RENDER_E2E_URL    base URL of the rendering service (default http://localhost:8000)
//	RENDER_E2E_MODEL  model name to send in the request body (default Qwen/Qwen3-VL-2B-Instruct)
package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

const (
	defaultRenderURL = "http://localhost:8000"
	defaultModel     = "Qwen/Qwen3-VL-2B-Instruct"

	// 1x1 transparent PNG, used as a minimal valid image payload.
	pixelPNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

func renderURL() string {
	if v := os.Getenv("RENDER_E2E_URL"); v != "" {
		return v
	}
	return defaultRenderURL
}

func modelName() string {
	if v := os.Getenv("RENDER_E2E_MODEL"); v != "" {
		return v
	}
	return defaultModel
}

func newRenderStep(t *testing.T, address string) *steps.RenderStep {
	t.Helper()
	step, err := steps.NewRenderStep(nil, map[string]any{"timeout": "30s"})
	if err != nil {
		t.Fatalf("NewRenderStep: %v", err)
	}
	rs := step.(*steps.RenderStep)
	rs.SetServiceAddress(address)
	return rs
}

func TestE2E_ChatCompletions_SimpleMessage(t *testing.T) {
	rs := newRenderStep(t, renderURL())

	reqCtx := &pipeline.RequestContext{
		RequestID:    "e2e-chat-simple",
		OriginalPath: gateway.PathChatCompletions,
		Model:        modelName(),
		Body: map[string]any{
			"model": modelName(),
			"messages": []any{
				map[string]any{"role": "user", "content": "Say hello."},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	if err := rs.Execute(ctx, reqCtx); err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if len(reqCtx.TokenIDs) == 0 {
		t.Fatal("expected non-empty TokenIDs")
	}
}

func TestE2E_ChatCompletions_TwoImages(t *testing.T) {
	rs := newRenderStep(t, renderURL())

	reqCtx := &pipeline.RequestContext{
		RequestID:    "e2e-chat-two-images",
		OriginalPath: gateway.PathChatCompletions,
		Model:        modelName(),
		Body: map[string]any{
			"model": modelName(),
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "What's in these images?"},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": pixelPNG}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": pixelPNG}},
					},
				},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Index: 0},
			{Index: 1},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	if err := rs.Execute(ctx, reqCtx); err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if len(reqCtx.TokenIDs) == 0 {
		t.Fatal("expected non-empty TokenIDs")
	}
	for i, entry := range reqCtx.MultimodalEntries {
		if entry.Hash == "" {
			t.Errorf("entry %d: Hash not populated", i)
		}
		if entry.Placeholder.Length == 0 {
			t.Errorf("entry %d: Placeholder.Length is 0", i)
		}
		if entry.KwargsData == "" {
			t.Errorf("entry %d: KwargsData not populated", i)
		}
	}
}

func TestE2E_Completions_TextPrompt(t *testing.T) {
	rs := newRenderStep(t, renderURL())

	reqCtx := &pipeline.RequestContext{
		RequestID:    "e2e-completions-text",
		OriginalPath: gateway.PathCompletions,
		Model:        modelName(),
		Body: map[string]any{
			"model":  modelName(),
			"prompt": "hello world",
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	if err := rs.Execute(ctx, reqCtx); err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if len(reqCtx.TokenIDs) == 0 {
		t.Fatal("expected non-empty TokenIDs")
	}
	if _, ok := reqCtx.Body["prompt"].([]int); !ok {
		t.Fatalf("expected Body[\"prompt\"] to be []int after render, got %T", reqCtx.Body["prompt"])
	}
}

func TestE2E_Completions_TokenArray(t *testing.T) {
	// This case short-circuits inside RenderStep without calling the upstream
	// service, so it works even if the rendering service is unreachable.
	rs := newRenderStep(t, "http://127.0.0.1:1") // deliberately unreachable

	tokens := []any{float64(1), float64(2345), float64(6789)}
	reqCtx := &pipeline.RequestContext{
		RequestID:    "e2e-completions-token-array",
		OriginalPath: gateway.PathCompletions,
		Model:        modelName(),
		Body: map[string]any{
			"model":  modelName(),
			"prompt": tokens,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := rs.Execute(ctx, reqCtx); err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if got, want := reqCtx.TokenIDs, []int{1, 2345, 6789}; !equalInts(got, want) {
		t.Fatalf("TokenIDs = %v, want %v", got, want)
	}
}

// TestE2E_DumpRenderChatCompletions POSTs a chat-completions body containing
// the real test image to <renderURL>/v1/chat/completions/render and prints
// the request, the response, the response size, len(token_ids), and the
// per-modality feature counts (mm_hashes / mm_placeholders / kwargs_data).
func TestE2E_DumpRenderChatCompletions(t *testing.T) {
	imageURL := loadTestImageDataURL(t)

	body := map[string]any{
		"model": modelName(),
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "Describe this image."},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}},
				},
			},
		},
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	pretty, err := json.MarshalIndent(body, "", "  ")
	if err != nil {
		t.Fatalf("pretty body: %v", err)
	}

	url := renderURL() + gateway.PathChatCompletions + "/render"

	t.Log("=== REQUEST ===")
	t.Logf("POST %s", url)
	t.Logf("body (%d bytes, image data: URL redacted; real length %d):\n%s",
		len(bodyBytes), len(imageURL),
		strings.ReplaceAll(string(pretty), imageURL, fmt.Sprintf("<data:image/jpeg;base64,... %d bytes total>", len(imageURL))))

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response: %v", err)
	}

	t.Log("=== RESPONSE ===")
	t.Logf("HTTP %d", resp.StatusCode)
	for k, v := range resp.Header {
		t.Logf("  %s: %s", k, strings.Join(v, ", "))
	}
	t.Logf("response size: %d bytes", len(respBytes))

	if resp.StatusCode/100 != 2 {
		t.Fatalf("non-2xx response: %s", string(respBytes))
	}

	// Parse to extract sizes/lengths.
	var parsed struct {
		TokenIDs []int `json:"token_ids"`
		Features struct {
			MMHashes       map[string][]string `json:"mm_hashes"`
			MMPlaceholders map[string][]any    `json:"mm_placeholders"`
			KwargsData     map[string][]string `json:"kwargs_data"`
		} `json:"features"`
	}
	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	t.Logf("len(token_ids): %d", len(parsed.TokenIDs))
	for modality, hashes := range parsed.Features.MMHashes {
		t.Logf("features.mm_hashes[%q]: count=%d, values=%v", modality, len(hashes), hashes)
	}
	for modality, placeholders := range parsed.Features.MMPlaceholders {
		t.Logf("features.mm_placeholders[%q]: count=%d, values=%v", modality, len(placeholders), placeholders)
	}
	for modality, kwargs := range parsed.Features.KwargsData {
		totalBytes := 0
		for _, k := range kwargs {
			totalBytes += len(k)
		}
		t.Logf("features.kwargs_data[%q]: count=%d, total bytes (sum of base64 lengths)=%d", modality, len(kwargs), totalBytes)
	}

	// Pretty-print the response body, redacting kwargs_data which dominates the size.
	for modality, kwargs := range parsed.Features.KwargsData {
		for i, k := range kwargs {
			placeholder := fmt.Sprintf("<base64 kwargs_data %s[%d], %d bytes>", modality, i, len(k))
			respBytes = bytes.ReplaceAll(respBytes, []byte(k), []byte(placeholder))
		}
	}
	var redactedAny any
	if err := json.Unmarshal(respBytes, &redactedAny); err == nil {
		if redactedPretty, err := json.MarshalIndent(redactedAny, "", "  "); err == nil {
			t.Logf("body (kwargs_data redacted):\n%s", string(redactedPretty))
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
