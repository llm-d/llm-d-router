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

package steps

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// TestNewDecodeProxyRequest_ForwardsPeerTopology verifies that a non-empty
// reqCtx.PeerTopology, captured by PrefillStep from the prefill response, is
// set on the outgoing decode request's x-peer-topology header.
func TestNewDecodeProxyRequest_ForwardsPeerTopology(t *testing.T) {
	gwClient := gateway.New(config.GatewayConfig{Address: "http://example.invalid"})

	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-1",
		OriginalPath: "/inference/v1/generate",
		PeerTopology: "host=node12,zone=us-east1-a",
	}

	req, err := newDecodeProxyRequest(context.Background(), logr.Discard(), "decode", reqCtx, gwClient, map[string]any{}, nil)
	if err != nil {
		t.Fatalf("newDecodeProxyRequest: %v", err)
	}
	if got, want := req.Header.Get(gateway.PeerTopologyHeader), "host=node12,zone=us-east1-a"; got != want {
		t.Fatalf("x-peer-topology header = %q, want %q", got, want)
	}
}

// TestNewDecodeProxyRequest_OmitsPeerTopologyWhenEmpty verifies that no
// x-peer-topology header is set on the decode request when the prefill
// response carried none (single-EPP-in-coordinator or no topology-stamp-handler).
func TestNewDecodeProxyRequest_OmitsPeerTopologyWhenEmpty(t *testing.T) {
	gwClient := gateway.New(config.GatewayConfig{Address: "http://example.invalid"})

	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-1",
		OriginalPath: "/inference/v1/generate",
	}

	req, err := newDecodeProxyRequest(context.Background(), logr.Discard(), "decode", reqCtx, gwClient, map[string]any{}, nil)
	if err != nil {
		t.Fatalf("newDecodeProxyRequest: %v", err)
	}
	if got := req.Header.Get(gateway.PeerTopologyHeader); got != "" {
		t.Fatalf("expected no x-peer-topology header, got %q", got)
	}
}

// TestNewDecodeProxy_MidStreamTruncationLogged drives the proxy against an
// upstream that promises a large Content-Length, writes a few bytes, then drops
// the connection. The copy fails after the 200 has been sent, so the only
// signal is the proxy's ErrorLog, which must reach the request logger with the
// partial-response marker.
func TestNewDecodeProxy_MidStreamTruncationLogged(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("ResponseWriter is not a Hijacker")
		}
		conn, buf, err := hj.Hijack()
		if err != nil {
			t.Fatalf("hijack: %v", err)
		}
		// Promise 1000 bytes, send 5, then close: the copy hits an
		// unexpected EOF mid-body.
		_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nhello")
		_ = buf.Flush()
		_ = conn.Close()
	}))
	defer upstream.Close()

	var mu sync.Mutex
	var msgs []string
	logger := funcr.New(func(_, args string) {
		mu.Lock()
		msgs = append(msgs, args)
		mu.Unlock()
	}, funcr.Options{})

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, upstream.URL, nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}

	proxy := newDecodeProxy(logger, http.DefaultTransport, nil)
	proxy.ServeHTTP(httptest.NewRecorder(), req)

	mu.Lock()
	defer mu.Unlock()
	for _, m := range msgs {
		if strings.Contains(m, "partial response") {
			return
		}
	}
	t.Fatalf("expected a partial-response error log, got %v", msgs)
}
