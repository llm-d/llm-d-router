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

package tokenizer

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"strings"
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"

	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
)

const (
	defaultHTTPRenderURL       = "http://localhost:8000"
	defaultHTTPRenderTimeout   = 5 * time.Second
	defaultHTTPRenderMMTimeout = 30 * time.Second

	completionsRenderPath = "/v1/completions/render"
	chatRenderPath        = "/v1/chat/completions/render"

	// maxErrorBodySnippetBytes truncates non-2xx response bodies before
	// embedding them in the returned error, so a misconfigured upstream that
	// returns a large HTML error page can't blow up log size.
	maxErrorBodySnippetBytes = 1024
)

// vllmConfig configures the vLLM /render backend. Future protocol fields
// (e.g., grpc) can be added alongside http.
type vllmConfig struct {
	// HTTP is the base URL of the vLLM render endpoint (no trailing slash).
	// Can be a loopback sidecar or a dedicated Service.
	// Defaults to http://localhost:8000.
	HTTP string `json:"http,omitempty"`
	// Timeout is the per-request timeout for text-only requests
	// (Go duration string, e.g. "5s"). Defaults to 5s.
	Timeout string `json:"timeout,omitempty"`
	// MMTimeout is the per-request timeout for multimodal requests
	// (image download/processing). Defaults to 30s.
	MMTimeout string `json:"mmTimeout,omitempty"`
}

// vllmHTTPRenderer implements the tokenizer interface by calling vLLM's
// /v1/completions/render and /v1/chat/completions/render endpoints.
type vllmHTTPRenderer struct {
	client    *http.Client
	baseURL   string
	modelName string
	timeout   time.Duration
	mmTimeout time.Duration
}

func newVLLMHTTPRenderer(cfg *vllmConfig, modelName string) (*vllmHTTPRenderer, error) {
	url := strings.TrimRight(cfg.HTTP, "/")
	if url == "" {
		url = defaultHTTPRenderURL
	}
	timeout, err := parseHTTPDuration(cfg.Timeout, defaultHTTPRenderTimeout)
	if err != nil {
		return nil, fmt.Errorf("invalid 'timeout': %w", err)
	}
	mmTimeout, err := parseHTTPDuration(cfg.MMTimeout, defaultHTTPRenderMMTimeout)
	if err != nil {
		return nil, fmt.Errorf("invalid 'mmTimeout': %w", err)
	}
	return &vllmHTTPRenderer{
		client:    &http.Client{Transport: newRenderTransport()},
		baseURL:   url,
		modelName: modelName,
		timeout:   timeout,
		mmTimeout: mmTimeout,
	}, nil
}

// newRenderTransport returns an http.Transport tuned for the render endpoint:
// HTTP/2 is disabled (vLLM doesn't support it) and the idle-connection pool
// is sized for the in-pod sidecar case while still being reasonable for a
// dedicated render Service.
func newRenderTransport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.MaxIdleConns = 0
	t.MaxIdleConnsPerHost = 16
	t.IdleConnTimeout = 90 * time.Second
	// Disable HTTP/2: vLLM doesn't support it. ForceAttemptHTTP2 alone is
	// not enough — clearing TLSNextProto prevents ALPN-negotiated h2 too.
	t.ForceAttemptHTTP2 = false
	t.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}
	return t
}

func parseHTTPDuration(s string, def time.Duration) (time.Duration, error) {
	if s == "" {
		return def, nil
	}
	return time.ParseDuration(s)
}


// Render calls /v1/completions/render. The PayloadMap is forwarded verbatim
// (preserving backend-specific fields such as reasoning) with the configured
// model name stamped in. Char offsets are not returned by vLLM's render endpoint.
func (r *vllmHTTPRenderer) Render(ctx context.Context, payload fwkrh.RequestPayload) ([]uint32, []tokenizerTypes.Offset, error) {
	pm, ok := payload.AsMap()
	if !ok {
		return nil, nil, errors.New("vLLM HTTP tokenizer requires a parsed PayloadMap")
	}
	body := maps.Clone(pm)
	body["model"] = r.modelName
	return r.postCompletionsRender(ctx, body)
}

func (r *vllmHTTPRenderer) postCompletionsRender(ctx context.Context, body any) ([]uint32, []tokenizerTypes.Offset, error) {
	var resp []renderResponse
	if err := r.postJSON(ctx, completionsRenderPath, body, r.timeout, &resp); err != nil {
		return nil, nil, err
	}
	if len(resp) == 0 {
		return nil, nil, errors.New("vLLM render returned empty response")
	}
	return resp[0].TokenIDs, nil, nil
}

// RenderChat calls /v1/chat/completions/render. The PayloadMap is forwarded
// verbatim with the configured model name stamped in.
func (r *vllmHTTPRenderer) RenderChat(ctx context.Context, payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	pm, ok := payload.AsMap()
	if !ok {
		return nil, nil, errors.New("vLLM HTTP tokenizer requires a parsed PayloadMap")
	}
	body := maps.Clone(pm)
	body["model"] = r.modelName
	return r.postChatRender(ctx, body, r.chatTimeout(pm))
}

func (r *vllmHTTPRenderer) postChatRender(ctx context.Context, body any, timeout time.Duration) ([]uint32, *tokenization.MultiModalFeatures, error) {
	var resp renderResponse
	if err := r.postJSON(ctx, chatRenderPath, body, timeout, &resp); err != nil {
		return nil, nil, err
	}
	return resp.TokenIDs, toKVCacheMM(resp.Features), nil
}

func (r *vllmHTTPRenderer) chatTimeout(payload fwkrh.PayloadMap) time.Duration {
	var chat fwkrh.ChatCompletionsRequest
	if data, err := json.Marshal(payload); err == nil {
		_ = json.Unmarshal(data, &chat)
	}
	for _, msg := range chat.Messages {
		if len(msg.Content.Structured) > 0 {
			return r.mmTimeout
		}
	}
	return r.timeout
}

// renderResponse is the subset of vLLM's GenerateRequest we consume.
type renderResponse struct {
	TokenIDs []uint32          `json:"token_ids"`
	Features *renderMMFeatures `json:"features,omitempty"`
}

type renderMMFeatures struct {
	MMHashes       map[string][]string            `json:"mm_hashes"`
	MMPlaceholders map[string][]renderPlaceholder `json:"mm_placeholders"`
}

type renderPlaceholder struct {
	Offset int `json:"offset"`
	Length int `json:"length"`
}

// toKVCacheMM converts vLLM's wire-format multimodal features into the kvcache
// map shape used by the rest of the tokenization pipeline.
func toKVCacheMM(f *renderMMFeatures) *tokenization.MultiModalFeatures {
	if f == nil || (len(f.MMHashes) == 0 && len(f.MMPlaceholders) == 0) {
		return nil
	}
	out := &tokenization.MultiModalFeatures{
		MMHashes:       f.MMHashes,
		MMPlaceholders: make(map[string][]kvblock.PlaceholderRange, len(f.MMPlaceholders)),
	}
	for k, prs := range f.MMPlaceholders {
		ranges := make([]kvblock.PlaceholderRange, len(prs))
		for i, pr := range prs {
			ranges[i] = kvblock.PlaceholderRange{Offset: pr.Offset, Length: pr.Length}
		}
		out.MMPlaceholders[k] = ranges
	}
	return out
}

func (r *vllmHTTPRenderer) postJSON(ctx context.Context, path string, body any, timeout time.Duration, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, r.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := r.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("post %s: %w", path, err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(httpResp.Body, maxErrorBodySnippetBytes))
		return fmt.Errorf("vLLM render returned status %d: %s", httpResp.StatusCode, string(snippet))
	}
	if err := json.NewDecoder(httpResp.Body).Decode(out); err != nil {
		return fmt.Errorf("unmarshal response: %w", err)
	}
	return nil
}
