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

// Package e2e — encoder end-to-end tests against a real Inference Gateway.
//
// To run:
//
//	go test -tags=e2e ./test/real-vllm-e2e/... -run Encode
//
// Configuration via environment variables:
//
//	ENCODE_E2E_GATEWAY  base URL of the gateway (default http://localhost:8080)
//	ENCODE_E2E_MODEL    model name to send in the request body (default Qwen/Qwen3-VL-2B-Instruct)
//	ENCODE_E2E_EC       EC connector to use (default ec-nixl)
//	RENDER_E2E_URL      base URL of the rendering service for chained tests (default http://localhost:8000)
//
// Each test exercises both encoder endpoints:
//
//   - /v1/chat/completions  (use_openai_format=true, chat-completions OriginalPath)
//   - /inference/v1/generate  (use_openai_format=false)
//
// The synthetic test runs without the renderer; the chained test renders
// first to obtain realistic Hash / KwargsData / Placeholder values and then
// drives the encoder against the same gateway.
package e2e

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

const (
	defaultGatewayURL = "http://localhost:8080"
	defaultECName     = ec.NIXL
	testImagePath     = "test-data/200.jpg"
)

// loadTestImageDataURL reads testImagePath and returns it as a
// "data:image/jpeg;base64,..." URL suitable for an OpenAI image_url message.
func loadTestImageDataURL(t *testing.T) string {
	t.Helper()
	path, err := filepath.Abs(testImagePath)
	if err != nil {
		t.Fatalf("resolve %s: %v", testImagePath, err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(raw)
}

type encodeVariant struct {
	name            string
	useOpenAIFormat bool
	originalPath    string
	expectedPath    string
	// needsRealKwargs is true when the variant's request body sends
	// features.kwargs_data straight to the backend (Generate path). The
	// backend deserializes it as the real preprocessor output, so synthetic
	// payloads fail with HTTP 500. ChatCompletions sends image_url instead
	// and the backend re-preprocesses, so synthetic data is fine there.
	needsRealKwargs bool
}

var encodeVariants = []encodeVariant{
	{
		name:            "ChatCompletions",
		useOpenAIFormat: true,
		originalPath:    gateway.PathChatCompletions,
		expectedPath:    gateway.PathChatCompletions,
		needsRealKwargs: false,
	},
	{
		name:            "Generate",
		useOpenAIFormat: false,
		originalPath:    gateway.PathChatCompletions,
		expectedPath:    gateway.DefaultGeneratePath,
		needsRealKwargs: true,
	},
}

func gatewayURL() string {
	if v := os.Getenv("ENCODE_E2E_GATEWAY"); v != "" {
		return v
	}
	return defaultGatewayURL
}

func encodeModel() string {
	if v := os.Getenv("ENCODE_E2E_MODEL"); v != "" {
		return v
	}
	return defaultModel
}

func ecConnectorName() string {
	if v := os.Getenv("ENCODE_E2E_EC"); v != "" {
		return v
	}
	return defaultECName
}

func newGatewayClient() *gateway.Client {
	return gateway.New(config.GatewayConfig{
		Address:             gatewayURL(),
		MaxIdleConnsPerHost: 32,
		IdleConnTimeout:     30 * time.Second,
		Timeout:             60 * time.Second,
	})
}

func newEncodeStepFor(t *testing.T, gw *gateway.Client, useOpenAIFormat bool) *steps.EncodeStep {
	t.Helper()
	step, err := steps.NewEncodeStep(gw, map[string]any{
		"use_openai_format":    useOpenAIFormat,
		"max_parallel":         4,
		steps.ParamECConnector: ecConnectorName(),
	})
	if err != nil {
		t.Fatalf("NewEncodeStep: %v", err)
	}
	return step.(*steps.EncodeStep)
}

// TestE2E_Encode_Synthetic drives the encoder with hand-crafted multimodal
// entries. This avoids depending on the rendering service. The encoder
// backend must accept the supplied KwargsData; if it validates the tensor
// payload against a real preprocessor output, prefer the chained test below.
func TestE2E_Encode_Synthetic(t *testing.T) {
	gw := newGatewayClient()

	for _, v := range encodeVariants {
		v := v
		t.Run(v.name, func(t *testing.T) {
			if v.needsRealKwargs {
				t.Skipf("variant %s sends features.kwargs_data to the backend; synthetic payloads cannot be deserialized — covered by TestE2E_Encode_RenderThenEncode/%s", v.name, v.name)
			}
			// Sanity: confirm the EncodeStep would route to the expected endpoint.
			if got := gateway.PathForFormat(resolveFormatForVariant(v)); got != v.expectedPath {
				t.Fatalf("variant %s targets %q, want %q", v.name, got, v.expectedPath)
			}

			es := newEncodeStepFor(t, gw, v.useOpenAIFormat)
			imageURL := loadTestImageDataURL(t)

			reqCtx := &pipeline.RequestContext{
				RequestID:    "e2e-encode-synthetic-" + v.name,
				OriginalPath: v.originalPath,
				Model:        encodeModel(),
				TokenIDs:     []int{1, 32000, 32000, 32000, 32000, 2345},
				MultimodalEntries: []pipeline.MultimodalEntry{
					{
						Index:       0,
						Hash:        "e2e-synthetic-hash-0",
						KwargsData:  "AAAA",
						Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 4},
					},
				},
				Body: map[string]any{
					"model": encodeModel(),
					"messages": []any{
						map[string]any{
							"role": "user",
							"content": []any{
								map[string]any{"type": "text", "text": "What's in this image?"},
								map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}},
							},
						},
					},
				},
			}

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			if err := es.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("encode failed (%s → %s): %v", v.name, v.expectedPath, err)
			}
			assertECTransferParams(t, reqCtx, 1)
		})
	}
}

// TestE2E_Encode_RenderThenEncode renders first to populate Hash / KwargsData
// / Placeholder from the real preprocessor, then runs the encoder. This is
// the closest approximation of the production pipeline.
func TestE2E_Encode_RenderThenEncode(t *testing.T) {
	gw := newGatewayClient()

	for _, v := range encodeVariants {
		v := v
		t.Run(v.name, func(t *testing.T) {
			rs := newRenderStep(t, renderURL())
			es := newEncodeStepFor(t, gw, v.useOpenAIFormat)
			imageURL := loadTestImageDataURL(t)

			reqCtx := &pipeline.RequestContext{
				RequestID:    "e2e-encode-render-chain-" + v.name,
				OriginalPath: v.originalPath,
				Model:        encodeModel(),
				Body: map[string]any{
					"model": encodeModel(),
					"messages": []any{
						map[string]any{
							"role": "user",
							"content": []any{
								map[string]any{"type": "text", "text": "Describe both images."},
								map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}},
								map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}},
							},
						},
					},
				},
				MultimodalEntries: []pipeline.MultimodalEntry{
					{Index: 0},
					{Index: 1},
				},
			}

			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()

			if err := rs.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("render failed: %v", err)
			}
			for i, entry := range reqCtx.MultimodalEntries {
				if entry.Hash == "" || entry.KwargsData == "" || entry.Placeholder.Length == 0 {
					t.Fatalf("entry %d not populated by render: %+v", i, entry)
				}
			}

			if err := es.Execute(ctx, reqCtx); err != nil {
				t.Fatalf("encode failed (%s → %s): %v", v.name, v.expectedPath, err)
			}
			assertECTransferParams(t, reqCtx, len(reqCtx.MultimodalEntries))
		})
	}
}

// resolveFormatForVariant mirrors EncodeStep.resolveFormat so the test can
// assert which gateway path the variant will hit before running it.
func resolveFormatForVariant(v encodeVariant) gateway.RequestFormat {
	detected := gateway.DetectFormat(v.originalPath)
	if detected == gateway.FormatCompletions {
		return gateway.FormatCompletions
	}
	if !v.useOpenAIFormat {
		return gateway.FormatGenerate
	}
	return detected
}

// TestE2E_Encode_DumpChatCompletionsRequest builds and POSTs the same
// chat-completions encode body that EncodeStep would send for a single
// image, then prints the request payload (with the base64 image redacted)
// and the gateway's response. Useful for inspecting wire shape.
func TestE2E_Encode_DumpChatCompletionsRequest(t *testing.T) {
	imageURL := loadTestImageDataURL(t)

	body := map[string]any{
		"model": encodeModel(),
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}},
				},
			},
		},
		"tokens": map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 32000, 2345},
			"features": map[string]any{
				"mm_hashes":       map[string][]string{"image": {"e2e-dump-hash-0"}},
				"mm_placeholders": map[string][]any{"image": {map[string]any{"offset": 1, "length": 4}}},
			},
		},
		"max_tokens": 1,
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	pretty, err := json.MarshalIndent(body, "", "  ")
	if err != nil {
		t.Fatalf("pretty body: %v", err)
	}
	redacted := redactDataURL(string(pretty), imageURL, len(imageURL))

	url := gatewayURL() + gateway.PathChatCompletions
	headers := map[string]string{
		"Content-Type":         "application/json",
		"x-request-id":         "e2e-dump-chat-completions",
		gateway.EPPPhaseHeader: gateway.PhaseEncode,
	}

	t.Log("=== REQUEST ===")
	t.Logf("POST %s", url)
	for k, v := range headers {
		t.Logf("  %s: %s", k, v)
	}
	t.Logf("body (%d bytes, image data: URL redacted):\n%s", len(bodyBytes), redacted)

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	req = req.WithContext(ctx)

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
	t.Logf("body (%d bytes):\n%s", len(respBytes), formatJSONOrRaw(respBytes))
}

// TestE2E_Encode_DumpGenerateRequest renders to obtain a real KwargsData
// blob, then builds and POSTs the same body EncodeStep would send to
// /inference/v1/generate for a single image. Prints request + response.
func TestE2E_Encode_DumpGenerateRequest(t *testing.T) {
	imageURL := loadTestImageDataURL(t)

	rs := newRenderStep(t, renderURL())
	reqCtx := &pipeline.RequestContext{
		RequestID:    "e2e-dump-generate-render",
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
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := rs.Execute(ctx, reqCtx); err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(reqCtx.MultimodalEntries))
	}
	entry := reqCtx.MultimodalEntries[0]
	if entry.Hash == "" || entry.KwargsData == "" || entry.Placeholder.Length == 0 {
		t.Fatalf("entry not populated: %+v", entry)
	}

	// Build the same body EncodeStep emits for FormatGenerate.
	bos := 1
	placeholderTokenID := 0
	if len(reqCtx.TokenIDs) > 0 {
		bos = reqCtx.TokenIDs[0]
		if entry.Placeholder.Offset < len(reqCtx.TokenIDs) {
			placeholderTokenID = reqCtx.TokenIDs[entry.Placeholder.Offset]
		}
	}
	tokenIDs := make([]int, 1+entry.Placeholder.Length)
	tokenIDs[0] = bos
	for j := 1; j <= entry.Placeholder.Length; j++ {
		tokenIDs[j] = placeholderTokenID
	}

	body := map[string]any{
		"model":     encodeModel(),
		"token_ids": tokenIDs,
		"features": map[string]any{
			"mm_hashes":       map[string][]string{"image": {entry.Hash}},
			"mm_placeholders": map[string][]any{"image": {map[string]any{"offset": 1, "length": entry.Placeholder.Length}}},
			"kwargs_data":     map[string][]string{"image": {entry.KwargsData}},
		},
		"sampling_params": map[string]any{"max_tokens": 1},
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	pretty, err := json.MarshalIndent(body, "", "  ")
	if err != nil {
		t.Fatalf("pretty body: %v", err)
	}
	redacted := redactKwargsData(string(pretty), entry.KwargsData)

	url := gatewayURL() + gateway.DefaultGeneratePath
	headers := map[string]string{
		"Content-Type":         "application/json",
		"x-request-id":         "e2e-dump-generate",
		gateway.EPPPhaseHeader: gateway.PhaseEncode,
	}

	t.Log("=== REQUEST ===")
	t.Logf("POST %s", url)
	for k, v := range headers {
		t.Logf("  %s: %s", k, v)
	}
	t.Logf("body (%d bytes, kwargs_data redacted; real length %d):\n%s", len(bodyBytes), len(entry.KwargsData), redacted)

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	req = req.WithContext(ctx)

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
	t.Logf("body (%d bytes):\n%s", len(respBytes), formatJSONOrRaw(respBytes))
}

func redactKwargsData(s, kwargs string) string {
	if kwargs == "" {
		return s
	}
	placeholder := fmt.Sprintf("<base64 kwargs_data, %d bytes>", len(kwargs))
	return strings.ReplaceAll(s, kwargs, placeholder)
}

func redactDataURL(s, url string, urlLen int) string {
	placeholder := fmt.Sprintf("<data:image/jpeg;base64,... %d bytes total>", urlLen)
	return strings.ReplaceAll(s, url, placeholder)
}

// TestE2E_Encode_DiagnoseGenerateFailure runs the real EncodeStep with the
// Generate variant against a request-capturing transport, then prints the
// exact wire bytes (request line, headers, body) and the response. Useful
// for diffing against TestE2E_Encode_DumpGenerateRequest, which sends a
// nominally-identical body via http.DefaultClient and currently succeeds
// even though the EncodeStep path returns HTTP 400.
func TestE2E_Encode_DiagnoseGenerateFailure(t *testing.T) {
	imageURL := loadTestImageDataURL(t)

	rs := newRenderStep(t, renderURL())
	reqCtx := &pipeline.RequestContext{
		RequestID:    "e2e-diagnose-generate",
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
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := rs.Execute(ctx, reqCtx); err != nil {
		t.Fatalf("render failed: %v", err)
	}

	// Build an EncodeStep that sends through a capturing transport.
	captured := &capturingTransport{wrapped: &http.Transport{
		MaxIdleConnsPerHost:   32,
		IdleConnTimeout:       30 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
	}}
	gw := gateway.NewWithTransport(captured, gatewayURL())
	es, err := steps.NewEncodeStep(gw, map[string]any{
		"use_openai_format":    false, // Generate variant
		"max_parallel":         1,
		steps.ParamECConnector: ecConnectorName(),
	})
	if err != nil {
		t.Fatalf("NewEncodeStep: %v", err)
	}

	execErr := es.Execute(ctx, reqCtx)

	if len(captured.requests) == 0 {
		t.Fatal("no request captured")
	}
	for i, r := range captured.requests {
		t.Logf("=== captured request %d ===", i)
		t.Logf("%s %s", r.method, r.url)
		for k, v := range r.headers {
			t.Logf("  %s: %s", k, strings.Join(v, ", "))
		}
		t.Logf("body (%d bytes, kwargs_data redacted):\n%s", len(r.body), redactKwargsInJSON(r.body))
		t.Logf("--- response %d ---", i)
		t.Logf("HTTP %d", r.respStatus)
		for k, v := range r.respHeaders {
			t.Logf("  %s: %s", k, strings.Join(v, ", "))
		}
		t.Logf("body (%d bytes):\n%s", len(r.respBody), formatJSONOrRaw(r.respBody))
	}

	if execErr != nil {
		t.Logf("EncodeStep.Execute returned: %v", execErr)
	}
}

type capturedRequest struct {
	method      string
	url         string
	headers     http.Header
	body        []byte
	respStatus  int
	respHeaders http.Header
	respBody    []byte
}

type capturingTransport struct {
	wrapped  http.RoundTripper
	requests []capturedRequest
}

func (c *capturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	cap := capturedRequest{
		method:  req.Method,
		url:     req.URL.String(),
		headers: req.Header.Clone(),
	}
	if req.Body != nil {
		buf, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		req.Body.Close()
		cap.body = buf
		req.Body = io.NopCloser(bytes.NewReader(buf))
	}
	resp, err := c.wrapped.RoundTrip(req)
	if resp != nil {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		cap.respStatus = resp.StatusCode
		cap.respHeaders = resp.Header.Clone()
		cap.respBody = respBody
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
	}
	c.requests = append(c.requests, cap)
	return resp, err
}

func redactKwargsInJSON(body []byte) string {
	var v any
	if err := json.Unmarshal(body, &v); err != nil {
		return string(body)
	}
	redactKwargsAny(v)
	pretty, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return string(body)
	}
	return string(pretty)
}

func redactKwargsAny(v any) {
	switch val := v.(type) {
	case map[string]any:
		if kd, ok := val["kwargs_data"].(map[string]any); ok {
			for modality, list := range kd {
				if arr, ok := list.([]any); ok {
					for i, s := range arr {
						if str, ok := s.(string); ok {
							arr[i] = fmt.Sprintf("<base64 kwargs_data %s[%d], %d bytes>", modality, i, len(str))
						}
					}
				}
			}
		}
		for _, child := range val {
			redactKwargsAny(child)
		}
	case []any:
		for _, child := range val {
			redactKwargsAny(child)
		}
	}
}

func formatJSONOrRaw(data []byte) string {
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		return string(data)
	}
	pretty, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return string(data)
	}
	return string(pretty)
}

func assertECTransferParams(t *testing.T, reqCtx *pipeline.RequestContext, want int) {
	t.Helper()
	if got := len(reqCtx.ECTransferParams); got != want {
		t.Fatalf("ECTransferParams: got %d entries, want %d (%v)", got, want, reqCtx.ECTransferParams)
	}
	for i, entry := range reqCtx.ECTransferParams {
		if len(entry) == 0 {
			t.Errorf("ECTransferParams[%d] is empty", i)
		}
	}
}
