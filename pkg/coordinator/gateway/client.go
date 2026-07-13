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

package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/llm-d-router/pkg/coordinator/common/httplog"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
)

// Client is an HTTP client configured for persistent connections to the Inference Gateway.
type Client struct {
	httpClient *http.Client
	baseURL    string
}

// New creates a Client from a GatewayConfig, constructing an http.Transport
// with the configured connection pool and timeout settings.
func New(cfg config.GatewayConfig) *Client {
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConnsPerHost:   cfg.MaxIdleConnsPerHost,
		IdleConnTimeout:       cfg.IdleConnTimeout,
		ResponseHeaderTimeout: cfg.Timeout,
		ForceAttemptHTTP2:     true,
	}

	return NewWithTransport(transport, cfg.Address)
}

// NewWithTransport creates a Client using the provided transport and base URL.
// A nil transport is valid; http.Client will fall back to http.DefaultTransport.
func NewWithTransport(transport *http.Transport, baseURL string) *Client {
	return &Client{
		httpClient: &http.Client{Transport: transport},
		baseURL:    baseURL,
	}
}

// Request sends an HTTP request to the gateway at the given path. The returned
// response body is always readable; it is buffered into memory only at TRACE,
// where it is logged and re-wrapped, and otherwise handed back as the unread
// stream. The caller owns closing the body.
func (c *Client) Request(ctx context.Context, method, path string, body []byte, headers map[string]string) (*http.Response, error) {
	url := c.baseURL + path

	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	req.Header.Set(ContentTypeHeader, ContentTypeJSON)
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	logger := log.FromContext(ctx).WithName("gateway")
	if body != nil {
		if v := logger.V(logutil.TRACE); v.Enabled() {
			v.Info("request body", "method", method, "path", path, "headers", httplog.RedactedHeaders(req.Header), "body", redactBody(body))
		}
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request to gateway: %w", err)
	}

	// Buffer the body only to log it at TRACE; otherwise hand the caller the
	// unread stream so large prefill/encode responses are not held in memory.
	if v := logger.V(logutil.TRACE); v.Enabled() {
		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("reading response from gateway: %w", err)
		}
		v.Info("response body", "status", resp.StatusCode, "body", redactBody(respBody))
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
	}

	return resp, nil
}

// Post is a convenience method for POST requests.
func (c *Client) Post(ctx context.Context, path string, body []byte, headers map[string]string) (*http.Response, error) {
	return c.Request(ctx, http.MethodPost, path, body, headers)
}

// BaseURL returns the gateway base URL (e.g. "http://inference-gateway:80").
func (c *Client) BaseURL() string {
	return c.baseURL
}

// Transport returns the underlying HTTP transport for reuse by reverse proxies.
func (c *Client) Transport() http.RoundTripper {
	return c.httpClient.Transport
}

// Log-redaction limits, applied when a request/response body is logged at TRACE.
const (
	// maxRedactStringLen is the longest JSON string value kept verbatim; longer
	// values are replaced with a marker so tensor blobs, base64, and URLs do not
	// drown out the structural fields.
	maxRedactStringLen = 50
	// maxRedactRawBodyLen bounds a non-JSON body logged verbatim.
	maxRedactRawBodyLen = 200
	// maxRedactElems caps how many array elements are kept before a count marker.
	maxRedactElems = 10
	// maxRedactDepth caps redactStrings recursion so a deeply nested adversarial
	// body cannot drive it toward the 500-level encoding/json limit on a trace log.
	maxRedactDepth = 32
)

// redactBody parses JSON and redacts oversized string values (see
// maxRedactStringLen); a body that is not valid JSON is returned verbatim, or
// truncated at maxRedactRawBodyLen.
func redactBody(data []byte) any {
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		if len(data) > maxRedactRawBodyLen {
			return string(data[:maxRedactRawBodyLen]) + "..."
		}
		return string(data)
	}
	return redactStrings(v)
}

func redactStrings(v any) any {
	return redactStringsDepth(v, 0)
}

func redactStringsDepth(v any, depth int) any {
	if depth >= maxRedactDepth {
		return "[truncated]"
	}
	switch val := v.(type) {
	case string:
		if len(val) > maxRedactStringLen {
			if strings.HasPrefix(val, "data:") {
				return "[base64]"
			}
			if strings.HasPrefix(val, "http://") || strings.HasPrefix(val, "https://") {
				return "[url]"
			}
			return "..."
		}
		return val
	case map[string]any:
		out := make(map[string]any, len(val))
		for k, item := range val {
			out[k] = redactStringsDepth(item, depth+1)
		}
		return out
	case []any:
		if len(val) > maxRedactElems {
			out := make([]any, maxRedactElems+1)
			for i := 0; i < maxRedactElems; i++ {
				out[i] = redactStringsDepth(val[i], depth+1)
			}
			out[maxRedactElems] = fmt.Sprintf("... +%d more", len(val)-maxRedactElems)
			return out
		}
		out := make([]any, len(val))
		for i, item := range val {
			out[i] = redactStringsDepth(item, depth+1)
		}
		return out
	default:
		return v
	}
}
