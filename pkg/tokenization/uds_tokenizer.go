/*
Copyright 2025 The llm-d Authors.

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

package tokenization

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/daulet/tokenizers"
	"golang.org/x/net/http2"
)

// UdsTokenizerConfig represents the configuration for the UDS-based tokenizer,
// including the socket file path.
type UdsTokenizerConfig struct {
	SocketFile string `json:"socketFile"`
}

func (cfg *UdsTokenizerConfig) IsEnabled() bool {
	return cfg != nil && cfg.SocketFile != ""
}

// UdsTokenizer communicates with a Unix Domain Socket server for tokenization.
type UdsTokenizer struct {
	httpClient *http.Client
	baseURL    string
}

// TokenizedInput represents the response from the tokenize endpoint.
type TokenizedInput struct {
	InputIDs      []uint32            `json:"input_ids"`
	OffsetMapping []tokenizers.Offset `json:"offset_mapping"`
}

const (
	defaultSocketFile = "/tmp/tokenizer/tokenizer-uds.socket"
	baseURL           = "http://tokenizer"

	// Default timeout for requests.
	defaultTimeout    = 5 * time.Second
	defaultMaxRetries = 2

	// Initial delay for exponential backoff.
	initialRetryDelay = 100 * time.Millisecond
)

// NewUdsTokenizer creates a new UDS-based tokenizer client with connection pooling.
func NewUdsTokenizer(config *UdsTokenizerConfig) (Tokenizer, error) {
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}

	socketFile := config.SocketFile
	if socketFile == "" {
		socketFile = defaultSocketFile
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return dialer.DialContext(ctx, "unix", socketFile)
		},
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}

	if err := http2.ConfigureTransport(transport); err != nil {
		return nil, fmt.Errorf("failed to configure HTTP/2 transport: %w", err)
	}

	client := &http.Client{
		Transport: transport,
	}

	return &UdsTokenizer{
		httpClient: client,
		baseURL:    baseURL,
	}, nil
}

// Encode tokenizes the input string and returns the token IDs and offsets.
func (u *UdsTokenizer) Encode(input, modelName string) ([]uint32, []tokenizers.Offset, error) {
	req, err := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		u.baseURL+"/tokenize",
		strings.NewReader(input),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := u.executeRequest(req, defaultTimeout, defaultMaxRetries)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	var tokenized TokenizedInput
	if err := json.Unmarshal(body, &tokenized); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return tokenized.InputIDs, tokenized.OffsetMapping, nil
}

// RenderChatTemplate renders a chat template using the UDS tokenizer service.
func (u *UdsTokenizer) RenderChatTemplate(messages interface{}) (string, error) {
	messagesBytes, err := json.Marshal(messages)
	if err != nil {
		return "", fmt.Errorf("failed to marshal messages: %w", err)
	}

	req, err := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		u.baseURL+"/chat-template",
		bytes.NewBuffer(messagesBytes),
	)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := u.executeRequest(req, defaultTimeout, defaultMaxRetries)
	if err != nil {
		return "", fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	return string(body), nil
}

// executeRequest executes an HTTP request with timeout and retry logic.
func (u *UdsTokenizer) executeRequest(req *http.Request,
	timeout time.Duration, maxRetries int,
) (*http.Response, error) {
	if timeout == 0 {
		timeout = defaultTimeout
	}
	if maxRetries < 0 {
		maxRetries = defaultMaxRetries
	}

	// Try the request up to maxRetries+1 times
	var lastErr error
	delay := initialRetryDelay

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Create a context with timeout
		ctx, cancel := context.WithTimeout(req.Context(), timeout)
		req = req.WithContext(ctx)

		// Execute the request
		resp, err := u.httpClient.Do(req)
		lastErr = err

		cancel()

		// If no error, check status code
		if err == nil {
			// For non-5xx status codes, don't retry
			if resp.StatusCode < 500 {
				return resp, nil
			}
			// Close the response body before retrying
			resp.Body.Close()
		}

		// If this was the last attempt, break
		if attempt == maxRetries {
			break
		}

		// Wait before retrying with exponential backoff
		time.Sleep(delay)
		delay *= 2 // Exponential backoff

		// Add some jitter to prevent thundering herd
		jitter, err := rand.Int(rand.Reader, big.NewInt(int64(delay/2)))
		if err != nil {
			// Fallback to using the full delay without jitter
			jitter = big.NewInt(int64(delay / 2))
		}
		delay += time.Duration(jitter.Int64())
	}

	return nil, fmt.Errorf("request failed after %d retries: %w", maxRetries, lastErr)
}
