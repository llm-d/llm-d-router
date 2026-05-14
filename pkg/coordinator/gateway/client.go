package gateway

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

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

	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request to gateway: %w", err)
	}
	return resp, nil
}

// Post is a convenience method for POST requests.
func (c *Client) Post(ctx context.Context, path string, body []byte, headers map[string]string) (*http.Response, error) {
	return c.Request(ctx, http.MethodPost, path, body, headers)
}
