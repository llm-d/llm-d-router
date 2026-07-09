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

	"github.com/go-logr/logr/funcr"
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
