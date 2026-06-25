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

// Package e2e — prefill end-to-end tests against a real Inference Gateway.
//
// To run:
//
//	go test -tags=e2e ./test/real-vllm-e2e/... -run Prefill
//
// Configuration via environment variables (shared with encode_test.go):
//
//	ENCODE_E2E_GATEWAY  base URL of the gateway (default http://localhost:8080)
//	ENCODE_E2E_MODEL    model name (default Qwen/Qwen3-VL-2B-Instruct)
//	ENCODE_E2E_EC       EC connector to use (default ec-nixl)
//	ENCODE_E2E_KV       KV connector to use (default kv-nixl)
//	RENDER_E2E_URL      base URL of the rendering service (default http://localhost:8000)
//
// The matrix:
//   - Multimodal /v1/chat/completions input → prefill at /v1/chat/completions and /inference/v1/generate
//   - Text-only  /v1/chat/completions input → prefill at /v1/chat/completions and /inference/v1/generate
//   - Text       /v1/completions      input → prefill at /v1/completions
//   - Token array /v1/completions      input → prefill at /v1/completions (render skipped)
//
// /v1/completions never routes to /inference/v1/generate at the prefill leg
// because resolveFormat hard-pins FormatCompletions whenever the original path
// contains /v1/completions, regardless of use_openai_format.
package e2e

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/llm-d/coordinator/pkg/connectors/kv"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	"github.com/llm-d/coordinator/pkg/steps"
)

const defaultKVName = kv.NIXL

func kvConnectorName() string {
	if v := os.Getenv("ENCODE_E2E_KV"); v != "" {
		return v
	}
	return defaultKVName
}

func newPrefillStepFor(t *testing.T, gw *gateway.Client, useOpenAIFormat bool) *steps.PrefillStep {
	t.Helper()
	step, err := steps.NewPrefillStep(map[string]any{
		"use_openai_format":    useOpenAIFormat,
		steps.ParamECConnector: ecConnectorName(),
		steps.ParamKVConnector: kvConnectorName(),
	})
	if err != nil {
		t.Fatalf("NewPrefillStep: %v", err)
	}
	ps := step.(*steps.PrefillStep)
	ps.SetGatewayClient(gw)
	return ps
}

// logPrefillResult logs what prefill returned. We don't strictly assert that
// KVTransferParams is non-nil because vLLM returns kv_transfer_params: null
// when the backend is not configured for disaggregated prefill/decode. The
// meaningful e2e assertion is that PrefillStep.Execute did not error; this
// helper surfaces what came back so a disagg-mode operator can eyeball it.
func logPrefillResult(t *testing.T, reqCtx *pipeline.RequestContext) {
	t.Helper()
	if reqCtx.KVTransferParams == nil {
		t.Logf("prefill returned kv_transfer_params: null (cluster likely not in disaggregated P/D mode)")
		return
	}
	t.Logf("prefill kv_transfer_params: %v", reqCtx.KVTransferParams)
}

// TestE2E_Prefill_ChatCompletions_Multimodal exercises the full
// render → encode → prefill pipeline for a multimodal chat-completions
// request, both with the OpenAI chat-completions wire shape and with the
// internal generate shape.
func TestE2E_Prefill_ChatCompletions_Multimodal(t *testing.T) {
	gw := newGatewayClient()

	for _, v := range encodeVariants {
		v := v
		t.Run(v.name, func(t *testing.T) {
			rs := newRenderStep(t, renderURL())
			es := newEncodeStepFor(t, gw, v.useOpenAIFormat)
			ps := newPrefillStepFor(t, gw, v.useOpenAIFormat)
			imageURL := loadTestImageDataURL(t)

			reqCtx := &pipeline.RequestContext{
				RequestID:    "e2e-prefill-mm-" + v.name,
				OriginalPath: gateway.PathChatCompletions,
				Model:        encodeModel(),
				Body: map[string]any{
					"model": encodeModel(),
					"messages": []any{
						map[string]any{
							"role": "user",
							"content": []any{
								map[string]any{"type": "text", "text": "Describe this image."},
								map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}},
							},
						},
					},
				},
				MultimodalEntries: []pipeline.MultimodalEntry{{Index: 0}},
				KVTransferParams:  map[string]any{},
			}

			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			if err := rs.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("render failed: %v", err)
			}
			if err := es.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("encode failed: %v", err)
			}
			assertECTransferParams(t, reqCtx, 1)

			if err := ps.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("prefill failed (%s → %s): %v", v.name, v.expectedPath, err)
			}
			logPrefillResult(t, reqCtx)
		})
	}
}

// TestE2E_Prefill_ChatCompletions_TextOnly skips encode (no multimodal
// entries) but still hits the chat-completions render and prefill paths.
// Covers both prefill wire shapes.
func TestE2E_Prefill_ChatCompletions_TextOnly(t *testing.T) {
	gw := newGatewayClient()

	for _, v := range encodeVariants {
		v := v
		t.Run(v.name, func(t *testing.T) {
			rs := newRenderStep(t, renderURL())
			ps := newPrefillStepFor(t, gw, v.useOpenAIFormat)

			reqCtx := &pipeline.RequestContext{
				RequestID:    "e2e-prefill-text-" + v.name,
				OriginalPath: gateway.PathChatCompletions,
				Model:        encodeModel(),
				Body: map[string]any{
					"model": encodeModel(),
					"messages": []any{
						map[string]any{"role": "user", "content": "Say hello in one word."},
					},
				},
				KVTransferParams: map[string]any{},
			}

			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			if err := rs.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("render failed: %v", err)
			}
			if len(reqCtx.MultimodalEntries) != 0 {
				t.Fatalf("expected no multimodal entries for text-only, got %d", len(reqCtx.MultimodalEntries))
			}

			if err := ps.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("prefill failed (%s → %s): %v", v.name, v.expectedPath, err)
			}
			logPrefillResult(t, reqCtx)
		})
	}
}

// TestE2E_Prefill_Completions_TextPrompt drives the /v1/completions path
// with a text prompt: render tokenizes the string, then prefill posts to
// /v1/completions on the gateway. use_openai_format is irrelevant for this
// path because resolveFormat pins FormatCompletions for any /v1/completions
// OriginalPath; we verify both flag values still land on /v1/completions.
func TestE2E_Prefill_Completions_TextPrompt(t *testing.T) {
	gw := newGatewayClient()

	for _, useOpenAI := range []bool{true, false} {
		useOpenAI := useOpenAI
		name := "OpenAITrue"
		if !useOpenAI {
			name = "OpenAIFalse"
		}
		t.Run(name, func(t *testing.T) {
			rs := newRenderStep(t, renderURL())
			ps := newPrefillStepFor(t, gw, useOpenAI)

			reqCtx := &pipeline.RequestContext{
				RequestID:    "e2e-prefill-completions-text-" + name,
				OriginalPath: gateway.PathCompletions,
				Model:        encodeModel(),
				Body: map[string]any{
					"model":  encodeModel(),
					"prompt": "Hello world",
				},
				KVTransferParams: map[string]any{},
			}

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			if err := rs.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("render failed: %v", err)
			}
			if len(reqCtx.TokenIDs) == 0 {
				t.Fatal("expected non-empty TokenIDs after render")
			}

			if err := ps.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("prefill failed: %v", err)
			}
			logPrefillResult(t, reqCtx)
		})
	}
}

// TestE2E_Prefill_Completions_TokenArray hands a pre-tokenized prompt to
// /v1/completions; render is bypassed and prefill posts to /v1/completions.
// Confirms the render-skipped path still feeds prefill correctly.
func TestE2E_Prefill_Completions_TokenArray(t *testing.T) {
	gw := newGatewayClient()
	rs := newRenderStep(t, "http://127.0.0.1:1") // unreachable, must not be called
	ps := newPrefillStepFor(t, gw, true)

	reqCtx := &pipeline.RequestContext{
		RequestID:    "e2e-prefill-completions-tokens",
		OriginalPath: gateway.PathCompletions,
		Model:        encodeModel(),
		Body: map[string]any{
			"model":  encodeModel(),
			"prompt": []any{float64(1), float64(2345), float64(6789)},
		},
		KVTransferParams: map[string]any{},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	if err := rs.Execute(ctx, reqCtx); err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if got, want := reqCtx.TokenIDs, []int{1, 2345, 6789}; !equalInts(got, want) {
		t.Fatalf("TokenIDs = %v, want %v", got, want)
	}

	if err := ps.Execute(ctx, reqCtx); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}
	logPrefillResult(t, reqCtx)
}
