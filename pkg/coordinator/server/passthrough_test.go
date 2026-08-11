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

package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
)

// captured records what an upstream test gateway saw so tests can assert on
// method, path, query, headers, and body.
type captured struct {
	mu      sync.Mutex
	method  string
	path    string
	query   string
	headers http.Header
	body    []byte
}

func (c *captured) get() (string, string, string, http.Header, []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.method, c.path, c.query, c.headers.Clone(), append([]byte(nil), c.body...)
}

// newCapturingUpstream returns an httptest.Server that records the incoming
// request and replies with the given status and body.
func newCapturingUpstream(t *testing.T, status int, respBody string) (*httptest.Server, *captured) {
	t.Helper()
	c := &captured{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("upstream read body: %v", err)
		}
		c.mu.Lock()
		c.method = r.Method
		c.path = r.URL.Path
		c.query = r.URL.RawQuery
		c.headers = r.Header.Clone()
		c.body = body
		c.mu.Unlock()
		w.WriteHeader(status)
		_, _ = w.Write([]byte(respBody))
	}))
	t.Cleanup(srv.Close)
	return srv, c
}

func doPassthrough(t *testing.T, srv *Server, req *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.httpServer.Handler.ServeHTTP(rec, req)
	return rec
}

func TestPassthrough_ForwardsUnregisteredGET(t *testing.T) {
	// /v1/models is a GET that the coordinator does not register; the passthrough
	// must forward it verbatim to the gateway with EPP-Profile: decode.
	upstream, cap := newCapturingUpstream(t, http.StatusOK, `{"data":[]}`)
	srv := newTestServerWithGateway(nil, upstream.URL)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := doPassthrough(t, srv, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%q", rec.Code, rec.Body.String())
	}
	if rec.Body.String() != `{"data":[]}` {
		t.Fatalf("unexpected response body: %q", rec.Body.String())
	}
	method, path, _, headers, _ := cap.get()
	if method != http.MethodGet {
		t.Fatalf("upstream method: got %q want GET", method)
	}
	if path != "/v1/models" {
		t.Fatalf("upstream path: got %q want /v1/models", path)
	}
	if got := headers.Get(gateway.EPPProfileHeader); got != gateway.PhaseDecode {
		t.Fatalf("upstream %s: got %q want %q", gateway.EPPProfileHeader, got, gateway.PhaseDecode)
	}
}

func TestPassthrough_PreservesMethodBodyAndQuery(t *testing.T) {
	// POST with a body and a query string must reach the gateway unchanged.
	upstream, cap := newCapturingUpstream(t, http.StatusAccepted, "ok")
	srv := newTestServerWithGateway(nil, upstream.URL)

	body := `{"prompt":"hi"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages?stream=true", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := doPassthrough(t, srv, req)

	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected 202, got %d", rec.Code)
	}
	method, path, query, headers, gotBody := cap.get()
	if method != http.MethodPost {
		t.Fatalf("method: got %q want POST", method)
	}
	if path != "/v1/messages" {
		t.Fatalf("path: got %q want /v1/messages", path)
	}
	if query != "stream=true" {
		t.Fatalf("query: got %q want %q", query, "stream=true")
	}
	if string(gotBody) != body {
		t.Fatalf("body: got %q want %q", string(gotBody), body)
	}
	if got := headers.Get("Content-Type"); got != "application/json" {
		t.Fatalf("content-type: got %q want application/json", got)
	}
}

func TestPassthrough_MaliciousRequestIDReplaced(t *testing.T) {
	// A request id with disallowed characters must be replaced before it reaches
	// the gateway, matching handleInference's sanitization.
	upstream, cap := newCapturingUpstream(t, http.StatusOK, "")
	srv := newTestServerWithGateway(nil, upstream.URL)

	malicious := "evil\r\nInjected: value"
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set(reqcommon.RequestIDHeaderKey, malicious)
	doPassthrough(t, srv, req)

	_, _, _, headers, _ := cap.get()
	upstreamID := headers.Get(reqcommon.RequestIDHeaderKey)
	if upstreamID == "" || upstreamID == malicious {
		t.Fatalf("malicious request_id must not reach the gateway: got %q", upstreamID)
	}
	if strings.ContainsAny(upstreamID, "\r\n ") {
		t.Fatalf("replacement request_id must not contain CR/LF/space: got %q", upstreamID)
	}
}

func TestPassthrough_ValidRequestIDPreserved(t *testing.T) {
	// A well-formed client request id must survive intact so upstream logs can
	// correlate with the client trace.
	upstream, cap := newCapturingUpstream(t, http.StatusOK, "")
	srv := newTestServerWithGateway(nil, upstream.URL)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set(reqcommon.RequestIDHeaderKey, "req-abc-123")
	doPassthrough(t, srv, req)

	_, _, _, headers, _ := cap.get()
	if got := headers.Get(reqcommon.RequestIDHeaderKey); got != "req-abc-123" {
		t.Fatalf("request_id: got %q want %q", got, "req-abc-123")
	}
}

func TestPassthrough_TransportErrorReturns502(t *testing.T) {
	// Point at a closed listener: the transport fails and the passthrough must
	// answer 502 rather than surface a raw error to the client.
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	closedURL := upstream.URL
	upstream.Close() // close before serving so subsequent dials get ECONNREFUSED

	srv := newTestServerWithGateway(nil, closedURL)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := doPassthrough(t, srv, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 on transport error, got %d", rec.Code)
	}
}

func TestPassthrough_RegisteredPathsBypass(t *testing.T) {
	// A registered path must not reach the passthrough. Point the gateway at a
	// handler that fails the test if hit, then POST a known route.
	upstream := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		t.Errorf("registered path %s reached the passthrough gateway", r.URL.Path)
	}))
	defer upstream.Close()

	srv := newTestServerWithGateway(nil, upstream.URL)
	req := httptest.NewRequest(http.MethodPost, gateway.PathChatCompletions, strings.NewReader(`{"model":"m"}`))
	rec := doPassthrough(t, srv, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("registered path expected 200, got %d", rec.Code)
	}
}
