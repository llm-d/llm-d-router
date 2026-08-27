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
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
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

func TestPassthrough_StreamsChunksAsTheyArrive(t *testing.T) {
	// Upstream writes two chunks with an explicit Flush() between them. The
	// upstream sets an explicit Content-Length and a non-SSE Content-Type so
	// httputil.ReverseProxy's auto-flush for text/event-stream and unknown
	// (chunked) content lengths does not paper over a missing FlushInterval: -1
	// on the coordinator's proxy.
	const body = "chunk1chunk2"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("upstream ResponseWriter is not a Flusher")
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("chunk1"))
		flusher.Flush()
		_, _ = w.Write([]byte("chunk2"))
		flusher.Flush()
	}))
	defer upstream.Close()

	srv := newTestServerWithGateway(nil, upstream.URL)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := doPassthrough(t, srv, req)

	if !rec.Flushed {
		t.Fatal("expected passthrough to flush downstream; FlushInterval: -1 regression?")
	}
	if got := rec.Body.String(); got != body {
		t.Fatalf("body: got %q want %q", got, body)
	}
}

// logCaptureSink is a logr.LogSink that records every Info and Error call.
// Tests use it to assert which log level a code path takes.
type logCaptureSink struct {
	infos  []capturedLog
	errors []capturedLog
}

type capturedLog struct {
	level int
	msg   string
	err   error
}

func (s *logCaptureSink) Init(_ logr.RuntimeInfo) {}
func (s *logCaptureSink) Enabled(_ int) bool      { return true }
func (s *logCaptureSink) Info(level int, msg string, _ ...any) {
	s.infos = append(s.infos, capturedLog{level: level, msg: msg})
}
func (s *logCaptureSink) Error(err error, msg string, _ ...any) {
	s.errors = append(s.errors, capturedLog{msg: msg, err: err})
}
func (s *logCaptureSink) WithValues(_ ...any) logr.LogSink { return s }
func (s *logCaptureSink) WithName(_ string) logr.LogSink   { return s }

// backendTimeoutErr fakes net/http's errTimeout: a transport-level timeout
// whose Is method matches context.DeadlineExceeded even though the client's
// request context is not done. The ErrorHandler must not misclassify it as
// a client cancellation.
type backendTimeoutErr struct{}

func (backendTimeoutErr) Error() string     { return "net/http: timeout awaiting response headers" }
func (backendTimeoutErr) Is(err error) bool { return err == context.DeadlineExceeded }

func TestPassthrough_ClientCancelNotLoggedAsError(t *testing.T) {
	// Classification is driven by the request context's state, not by the
	// error argument: a request whose context is done is a client-side
	// lifecycle event (disconnect or client deadline), anything else is a
	// backend fault. A transport-level timeout whose Is method matches
	// context.DeadlineExceeded (net/http's errTimeout, fired by
	// ResponseHeaderTimeout) must stay in the error log.
	gwURL, err := url.Parse("http://gw:80")
	if err != nil {
		t.Fatalf("parse gateway URL: %v", err)
	}

	pastDeadline := time.Now().Add(-time.Second)

	cases := []struct {
		name          string
		reqCtx        func() (context.Context, context.CancelFunc)
		err           error
		wantErrorLogs int
	}{
		{
			name: "client cancelled",
			reqCtx: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx, func() {}
			},
			err:           errors.New("connection reset by peer"),
			wantErrorLogs: 0,
		},
		{
			name: "client deadline exceeded",
			reqCtx: func() (context.Context, context.CancelFunc) {
				return context.WithDeadline(context.Background(), pastDeadline)
			},
			err:           errors.New("connection reset by peer"),
			wantErrorLogs: 0,
		},
		{
			name: "backend response header timeout stays an error",
			reqCtx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			err:           backendTimeoutErr{},
			wantErrorLogs: 1,
		},
		{
			name: "transport failure",
			reqCtx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			err:           errors.New("boom"),
			wantErrorLogs: 1,
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			sink := &logCaptureSink{}
			proxy := newPassthroughProxy(logr.New(sink), gwURL, http.DefaultTransport, "req-1")

			ctx, cancel := tt.reqCtx()
			defer cancel()
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, "/v1/models", nil).WithContext(ctx)
			proxy.ErrorHandler(rec, req, tt.err)

			if rec.Code != http.StatusBadGateway {
				t.Fatalf("status: got %d want 502", rec.Code)
			}
			if got := len(sink.errors); got != tt.wantErrorLogs {
				t.Fatalf("Error-level logs: got %d (%v), want %d", got, sink.errors, tt.wantErrorLogs)
			}
		})
	}
}

func TestPassthrough_BodyOverConfiguredCapMapsTo413(t *testing.T) {
	// A body larger than server.max_request_body_size (in MB) must be rejected
	// on the passthrough path with 413, matching the cap the inference path
	// enforces. Use a 1 MB cap and send 1 MB + 1 byte to trigger the limit.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The passthrough will tear the outbound body mid-stream when the cap
		// trips; discard whatever fragment arrives without asserting on the
		// read error.
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	p := pipeline.New([]pipeline.Step{stubStep{name: "stub"}})
	gw := gateway.NewWithTransport(&http.Transport{}, upstream.URL)
	srv, err := New(config.ServerConfig{MaxRequestBodySize: 1}, p, gw)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	oversize := strings.Repeat("x", config.BytesPerMB+1)
	req := httptest.NewRequest(http.MethodPost, "/v1/models", strings.NewReader(oversize))
	rec := doPassthrough(t, srv, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413 for oversize passthrough body, got %d", rec.Code)
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
