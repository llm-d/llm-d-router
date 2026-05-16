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

// Render calls /v1/completions/render for prompt-only callers. The Produce path
// uses renderCompletionsPayload so the original request payload is preserved.
// Char offsets are not provided by vLLM's render endpoint and the upstream call
// site discards them, so we return nil.
func (r *vllmHTTPRenderer) Render(ctx context.Context, prompt string) ([]uint32, []tokenizerTypes.Offset, error) {
	return r.postCompletionsRender(ctx, map[string]any{
		"model":  r.modelName,
		"prompt": prompt,
	})
}

func (r *vllmHTTPRenderer) renderCompletionsPayload(ctx context.Context, payload fwkrh.PayloadMap) ([]uint32, []tokenizerTypes.Offset, error) {
	body := maps.Clone(payload)
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

// RenderChat calls /v1/chat/completions/render with an OpenAI-shaped request
// body, then converts the response's wire-format multimodal features into the
// kvcache map shape expected by the upstream interface.
func (r *vllmHTTPRenderer) RenderChat(ctx context.Context, req *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error) {
	body := buildChatRenderRequest(r.modelName, req)

	return r.postChatRender(ctx, body, r.chatTimeout(req))
}

func (r *vllmHTTPRenderer) renderChatPayload(ctx context.Context, payload fwkrh.PayloadMap, req *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error) {
	body := maps.Clone(payload)
	body["model"] = r.modelName

	return r.postChatRender(ctx, body, r.chatTimeout(req))
}

func (r *vllmHTTPRenderer) postChatRender(ctx context.Context, body any, timeout time.Duration) ([]uint32, *tokenization.MultiModalFeatures, error) {
	var resp renderResponse
	if err := r.postJSON(ctx, chatRenderPath, body, timeout, &resp); err != nil {
		return nil, nil, err
	}
	return resp.TokenIDs, toKVCacheMM(resp.Features), nil
}

func (r *vllmHTTPRenderer) chatTimeout(req *tokenizerTypes.RenderChatRequest) time.Duration {
	for _, msg := range req.Conversation {
		if len(msg.Content.Structured) > 0 {
			return r.mmTimeout
		}
	}
	return r.timeout
}

// chatRenderRequest is the wire body for POST /v1/chat/completions/render.
type chatRenderRequest struct {
	Model                string         `json:"model"`
	Messages             []chatMessage  `json:"messages"`
	Tools                []any          `json:"tools,omitempty"`
	Documents            []any          `json:"documents,omitempty"`
	ChatTemplate         string         `json:"chat_template,omitempty"`
	AddGenerationPrompt  bool           `json:"add_generation_prompt,omitempty"`
	ContinueFinalMessage bool           `json:"continue_final_message,omitempty"`
	ChatTemplateKWArgs   map[string]any `json:"chat_template_kwargs,omitempty"`
}

// chatMessage is one OpenAI-shaped message. Content is either a plain string
// or an array of parts; chatContent's MarshalJSON picks the right wire form.
type chatMessage struct {
	Role    string      `json:"role"`
	Content chatContent `json:"content"`
}

// chatContent serializes either Raw (string) or Parts (array of typed parts).
// When both are empty it serializes as "" (an empty user message).
type chatContent struct {
	Raw   string
	Parts []chatPart
}

func (c chatContent) MarshalJSON() ([]byte, error) {
	if len(c.Parts) > 0 {
		return json.Marshal(c.Parts)
	}
	return json.Marshal(c.Raw)
}

// chatPart is one OpenAI content part. Only the field matching Type is set.
type chatPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *chatImageURL `json:"image_url,omitempty"`
}

type chatImageURL struct {
	URL string `json:"url"`
}

// buildChatRenderRequest projects the kvcache RenderChatRequest into the
// OpenAI-shaped wire body expected by vLLM's /v1/chat/completions/render.
// Unknown content-block types are skipped (mirrors the UDS path's behavior).
func buildChatRenderRequest(model string, req *tokenizerTypes.RenderChatRequest) chatRenderRequest {
	msgs := make([]chatMessage, len(req.Conversation))
	for idx, c := range req.Conversation {
		msgs[idx] = chatMessage{Role: c.Role, Content: toChatContent(c.Content)}
	}
	return chatRenderRequest{
		Model:                model,
		Messages:             msgs,
		Tools:                req.Tools,
		Documents:            req.Documents,
		ChatTemplate:         req.ChatTemplate,
		AddGenerationPrompt:  req.AddGenerationPrompt,
		ContinueFinalMessage: req.ContinueFinalMessage,
		ChatTemplateKWArgs:   req.ChatTemplateKWArgs,
	}
}

func toChatContent(c tokenizerTypes.Content) chatContent {
	if len(c.Structured) == 0 {
		return chatContent{Raw: c.Raw}
	}
	parts := make([]chatPart, len(c.Structured))
	for idx, b := range c.Structured {
		switch b.Type {
		case "text":
			parts[idx] = chatPart{Type: "text", Text: b.Text}
		case "image_url":
			parts[idx] = chatPart{Type: "image_url", ImageURL: &chatImageURL{URL: b.ImageURL.URL}}
		default:
			// Unsupported by the kvcache ContentBlock schema; skip.
		}
	}
	return chatContent{Parts: parts}
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
