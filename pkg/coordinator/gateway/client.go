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

	"github.com/llm-d/coordinator/pkg/common/httplog"
	"github.com/llm-d/coordinator/pkg/config"
)

// Client is an HTTP client configured for persistent connections to the Envoy Gateway.
type Client struct {
	httpClient *http.Client
	baseURL    string
}

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

	return &Client{
		httpClient: &http.Client{
			Transport: transport,
		},
		baseURL: cfg.Address,
	}
}

// Request sends an HTTP request to the gateway at the given path.
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

	respBody, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("reading response from gateway: %w", err)
	}
	if v := logger.V(logutil.TRACE); v.Enabled() {
		v.Info("response body", "status", resp.StatusCode, "body", redactBody(respBody))
	}
	resp.Body = io.NopCloser(bytes.NewReader(respBody))

	return resp, nil
}

// Post is a convenience method for POST requests.
func (c *Client) Post(ctx context.Context, path string, body []byte, headers map[string]string) (*http.Response, error) {
	return c.Request(ctx, http.MethodPost, path, body, headers)
}

// BaseURL returns the gateway base URL (e.g. "http://envoy-gateway:80").
func (c *Client) BaseURL() string {
	return c.baseURL
}

// Transport returns the underlying HTTP transport for reuse by reverse proxies.
func (c *Client) Transport() http.RoundTripper {
	return c.httpClient.Transport
}

// redactBody parses JSON and replaces string values longer than 50 chars with
// "..." so tensor blobs don't drown out the structural fields.
func redactBody(data []byte) any {
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		if len(data) > 200 {
			return string(data[:200]) + "..."
		}
		return string(data)
	}
	return redactStrings(v)
}

func redactStrings(v any) any {
	switch val := v.(type) {
	case string:
		if len(val) > 50 {
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
			out[k] = redactStrings(item)
		}
		return out
	case []any:
		const maxElems = 10
		if len(val) > maxElems {
			out := make([]any, maxElems+1)
			for i := 0; i < maxElems; i++ {
				out[i] = redactStrings(val[i])
			}
			out[maxElems] = fmt.Sprintf("... +%d more", len(val)-maxElems)
			return out
		}
		out := make([]any, len(val))
		for i, item := range val {
			out[i] = redactStrings(item)
		}
		return out
	default:
		return v
	}
}
