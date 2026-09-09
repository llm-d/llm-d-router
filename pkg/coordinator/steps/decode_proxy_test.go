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
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
)

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

	proxy, _ := newDecodeProxy(logger, http.DefaultTransport, nil)
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

// TestTimedRoundTripper_ExcludesStreamingBody drives the full decode proxy
// against an upstream that writes headers, flushes, then delays before the
// body. ServeHTTP wall-clock covers the body delay, but the wrapper's recorded
// duration must not: that is what keeps upstream_request_duration_seconds
// comparable to the non-streaming steps, which time only up to headers.
func TestTimedRoundTripper_ExcludesStreamingBody(t *testing.T) {
	const bodyDelay = 200 * time.Millisecond
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		time.Sleep(bodyDelay)
		_, _ = w.Write([]byte("done"))
	}))
	defer upstream.Close()

	var recorded time.Duration
	rt := &timedRoundTripper{
		inner:  http.DefaultTransport,
		record: func(d time.Duration) { recorded = d },
	}
	logger := funcr.New(func(_, _ string) {}, funcr.Options{})
	proxy, _ := newDecodeProxy(logger, rt, nil)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, upstream.URL, nil)
	require.NoError(t, err)
	rec := httptest.NewRecorder()

	start := time.Now()
	proxy.ServeHTTP(rec, req)
	total := time.Since(start)

	require.GreaterOrEqual(t, total, bodyDelay, "ServeHTTP must cover the full body write")
	require.Less(t, recorded, bodyDelay, "recorded must exclude body streaming; got %v", recorded)
}

func TestTimedRoundTripper_RecordsOnTransportError(t *testing.T) {
	called := false
	rt := &timedRoundTripper{
		inner:  roundTripperFunc(func(_ *http.Request) (*http.Response, error) { return nil, http.ErrHandlerTimeout }),
		record: func(_ time.Duration) { called = true },
	}
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "http://example.invalid/", nil)
	require.NoError(t, err)
	resp, err := rt.RoundTrip(req)
	if resp != nil {
		_ = resp.Body.Close()
	}
	require.Error(t, err)
	require.True(t, called, "record must fire on transport error to match gwClient.Post semantics")
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

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

func TestNewDecodeProxy_ClientCancelNotLoggedAsError(t *testing.T) {
	// Classification is driven by the request context's state, not by the
	// error argument: a request whose context is done is a client-side
	// lifecycle event (disconnect or client deadline), anything else is a
	// backend fault. A transport-level timeout whose Is method matches
	// context.DeadlineExceeded (net/http's errTimeout, fired by
	// ResponseHeaderTimeout) must stay in the error log. errCacheMiss stays
	// swallowed (no log, no response) so the miss can fall through to the
	// rest of the pipeline.
	pastDeadline := time.Now().Add(-time.Second)

	cases := []struct {
		name          string
		reqCtx        func() (context.Context, context.CancelFunc)
		err           error
		wantStatus    int
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
			wantStatus:    http.StatusBadGateway,
			wantErrorLogs: 0,
		},
		{
			name: "client deadline exceeded",
			reqCtx: func() (context.Context, context.CancelFunc) {
				return context.WithDeadline(context.Background(), pastDeadline)
			},
			err:           errors.New("connection reset by peer"),
			wantStatus:    http.StatusBadGateway,
			wantErrorLogs: 0,
		},
		{
			name: "backend response header timeout stays an error",
			reqCtx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			err:           backendTimeoutErr{},
			wantStatus:    http.StatusBadGateway,
			wantErrorLogs: 1,
		},
		{
			name: "transport failure",
			reqCtx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			err:           errors.New("boom"),
			wantStatus:    http.StatusBadGateway,
			wantErrorLogs: 1,
		},
		{
			name: "cache miss falls through",
			reqCtx: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			err:           errCacheMiss,
			wantStatus:    http.StatusOK,
			wantErrorLogs: 0,
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			sink := &logCaptureSink{}
			proxy, _ := newDecodeProxy(logr.New(sink), http.DefaultTransport, nil)

			ctx, cancel := tt.reqCtx()
			defer cancel()
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
			proxy.ErrorHandler(rec, req, tt.err)

			if rec.Code != tt.wantStatus {
				t.Fatalf("status: got %d want %d", rec.Code, tt.wantStatus)
			}
			if got := len(sink.errors); got != tt.wantErrorLogs {
				t.Fatalf("Error-level logs: got %d (%v), want %d", got, sink.errors, tt.wantErrorLogs)
			}
		})
	}
}
