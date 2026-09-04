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
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	defaultHTTPRenderURL       = "http://localhost:8000"
	defaultHTTPRenderTimeout   = 5 * time.Second
	defaultHTTPRenderMMTimeout = 30 * time.Second

	completionsRenderPath = "/v1/completions/render"
	chatRenderPath        = "/v1/chat/completions/render"
	messagesRenderPath    = "/v1/messages/render"

	// maxErrorBodySnippetBytes truncates non-2xx response bodies before
	// embedding them in the returned error, so a misconfigured upstream that
	// returns a large HTML error page can't blow up log size.
	maxErrorBodySnippetBytes = 1024
)

// authHeaderCtxKey carries the inbound request's Authorization header from
// Plugin.Produce to the render call without widening tokenInputProducer.produce.
type authHeaderCtxKey struct{}

// withAuthHeader returns ctx carrying the Authorization header value verbatim.
func withAuthHeader(ctx context.Context, value string) context.Context {
	return context.WithValue(ctx, authHeaderCtxKey{}, value)
}

// authHeaderFromContext returns the Authorization header value on ctx, or "".
func authHeaderFromContext(ctx context.Context) string {
	value, _ := ctx.Value(authHeaderCtxKey{}).(string)
	return value
}

// renderStatusError is a non-2xx response from the render endpoint.
type renderStatusError struct {
	StatusCode int
	Body       string
}

func (e *renderStatusError) Error() string {
	return fmt.Sprintf("vLLM render returned status %d: %s", e.StatusCode, e.Body)
}

// isRenderAuthError reports whether err carries a 401 or 403 render response.
func isRenderAuthError(err error) bool {
	var se *renderStatusError
	return errors.As(err, &se) && (se.StatusCode == http.StatusUnauthorized || se.StatusCode == http.StatusForbidden)
}

// vllmConfig configures the vLLM /render backend. Future protocol fields
// (e.g., grpc) can be added alongside url.
type vllmConfig struct {
	// URL is the base URL of the vLLM render endpoint (no trailing slash).
	// Can be a loopback sidecar or a dedicated Service.
	// Defaults to http://localhost:8000.
	URL string `json:"url,omitempty"`
	// Timeout is the per-request timeout for completions
	// (Go duration string, e.g. "5s"). Defaults to 5s.
	Timeout string `json:"timeout,omitempty"`
	// MMTimeout allows image download/processing for Chat and Messages requests.
	// These endpoints use max(Timeout, MMTimeout) without inspecting content.
	// Defaults to 30s.
	MMTimeout string `json:"mmTimeout,omitempty"`
	// CACertPath is a PEM CA bundle used to verify the render endpoint's
	// server certificate when the URL scheme is https. When empty, the
	// system CA pool is used.
	CACertPath string `json:"caCertPath,omitempty"`
	// ClientCertPath and ClientKeyPath present a client certificate for
	// mTLS with the render endpoint. Both must be set together.
	ClientCertPath string `json:"clientCertPath,omitempty"`
	ClientKeyPath  string `json:"clientKeyPath,omitempty"`
	// InsecureSkipVerify disables verification of the render endpoint's
	// server certificate. Use for self-signed certificates in development
	// or when the cluster manages its own PKI.
	InsecureSkipVerify bool `json:"insecureSkipVerify,omitempty"`
}

// vllmHTTPRenderer implements the tokenizer interface by calling vLLM's
// native render endpoints.
type vllmHTTPRenderer struct {
	client    *http.Client
	baseURL   string
	timeout   time.Duration
	mmTimeout time.Duration
}

func newVLLMHTTPRenderer(cfg *vllmConfig) (*vllmHTTPRenderer, error) {
	url := strings.TrimRight(cfg.URL, "/")
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
	transport, err := newRenderTransport(cfg)
	if err != nil {
		return nil, err
	}
	return &vllmHTTPRenderer{
		client: &http.Client{Transport: otelhttp.NewTransport(transport, otelhttp.WithSpanNameFormatter(
			// Name the outbound span after the render route instead of the
			// transport's default "HTTP POST" so traces identify render calls.
			func(_ string, r *http.Request) string { return "tokenize_render " + r.URL.Path },
		))},
		baseURL:   url,
		timeout:   timeout,
		mmTimeout: mmTimeout,
	}, nil
}

// newRenderTransport returns an http.Transport tuned for the render endpoint:
// HTTP/2 is disabled (vLLM doesn't support it) and the idle-connection pool
// is sized for the in-pod sidecar case while still being reasonable for a
// dedicated render Service. When the config carries TLS fields (caCertPath,
// clientCertPath/clientKeyPath, insecureSkipVerify), the transport is
// configured for https.
func newRenderTransport(cfg *vllmConfig) (*http.Transport, error) {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.MaxIdleConns = 0
	t.MaxIdleConnsPerHost = 16
	t.IdleConnTimeout = 90 * time.Second
	// Disable HTTP/2: vLLM doesn't support it. ForceAttemptHTTP2 alone is
	// not enough — clearing TLSNextProto prevents ALPN-negotiated h2 too.
	t.ForceAttemptHTTP2 = false
	t.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}

	if cfg.hasTLS() {
		tlsCfg, err := renderTLSConfig(cfg)
		if err != nil {
			return nil, err
		}
		t.TLSClientConfig = tlsCfg
	}
	return t, nil
}

func (c *vllmConfig) hasTLS() bool {
	return c.InsecureSkipVerify || c.CACertPath != "" || c.ClientCertPath != "" || c.ClientKeyPath != ""
}

func renderTLSConfig(cfg *vllmConfig) (*tls.Config, error) {
	tc := &tls.Config{InsecureSkipVerify: cfg.InsecureSkipVerify} //nolint:gosec

	if !cfg.InsecureSkipVerify && cfg.CACertPath != "" {
		pem, err := os.ReadFile(cfg.CACertPath)
		if err != nil {
			return nil, fmt.Errorf("reading render CA cert %s: %w", cfg.CACertPath, err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("no valid CA certs in %s", cfg.CACertPath)
		}
		tc.RootCAs = pool
	}

	if cfg.ClientCertPath != "" || cfg.ClientKeyPath != "" {
		cert, err := tls.LoadX509KeyPair(cfg.ClientCertPath, cfg.ClientKeyPath)
		if err != nil {
			return nil, fmt.Errorf("loading render client cert: %w", err)
		}
		tc.Certificates = []tls.Certificate{cert}
	}
	return tc, nil
}

func parseHTTPDuration(s string, def time.Duration) (time.Duration, error) {
	if s == "" {
		return def, nil
	}
	return time.ParseDuration(s)
}

// Render forwards a completions request, including token arrays, to vLLM.
func (r *vllmHTTPRenderer) Render(ctx context.Context, payload fwkrh.RequestPayload) ([][]uint32, [][]tokenizerTypes.Offset, error) {
	var resp []renderResponse
	if err := r.postJSON(ctx, completionsRenderPath, payload, r.timeout, &resp); err != nil {
		return nil, nil, err
	}
	if len(resp) == 0 {
		return nil, nil, errors.New("vLLM render returned empty response")
	}
	allTokenIDs := make([][]uint32, len(resp))
	for i, r := range resp {
		allTokenIDs[i] = r.TokenIDs
	}
	return allTokenIDs, nil, nil
}

// RenderChat forwards the native Chat Completions body without model substitution.
func (r *vllmHTTPRenderer) RenderChat(ctx context.Context, payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	return r.renderConversation(ctx, chatRenderPath, payload)
}

// RenderMessages leaves Anthropic conversion to vLLM.
func (r *vllmHTTPRenderer) RenderMessages(ctx context.Context, payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	return r.renderConversation(ctx, messagesRenderPath, payload)
}

func (r *vllmHTTPRenderer) renderConversation(ctx context.Context, path string, payload fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
	var resp renderResponse
	if err := r.postJSON(ctx, path, payload, r.produceTimeout(), &resp); err != nil {
		return nil, nil, err
	}
	return resp.TokenIDs, toKVCacheMM(resp.Features), nil
}

// produceTimeout permits multimodal rendering without inspecting content.
func (r *vllmHTTPRenderer) produceTimeout() time.Duration {
	return max(r.timeout, r.mmTimeout)
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

func (r *vllmHTTPRenderer) postJSON(ctx context.Context, path string, body fwkrh.RequestPayload, timeout time.Duration, out any) error {
	var payload []byte
	switch body := body.(type) {
	case fwkrh.RawPayload:
		payload = body
	case fwkrh.Marshaler:
		var err error
		payload, err = body.Marshal()
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}
	default:
		return errors.New("native vLLM rendering requires an HTTP JSON payload")
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, r.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	// The render endpoint may require the same credential as inference;
	// forward the inbound Authorization when present.
	if auth := authHeaderFromContext(ctx); auth != "" {
		httpReq.Header.Set("Authorization", auth)
	}

	httpResp, err := r.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("post %s: %w", path, err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(httpResp.Body, maxErrorBodySnippetBytes))
		return &renderStatusError{StatusCode: httpResp.StatusCode, Body: string(snippet)}
	}
	if err := json.NewDecoder(httpResp.Body).Decode(out); err != nil {
		return fmt.Errorf("unmarshal response: %w", err)
	}
	return nil
}
