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

package proxy

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// A prefill target carrying an http:// scheme must resolve to the same cached
// handler as its bare host:port form. A cache miss on the scheme-prefixed value
// would build a new handler and dial a new connection on every request.
func TestPrefillerProxyHandler_CachesSchemePrefixedTarget(t *testing.T) {
	for _, tc := range []struct {
		name   string
		scheme bool
	}{
		{"bare host:port", false},
		{"http:// prefixed", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var opened atomic.Int32
			backend := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.WriteHeader(http.StatusOK)
			}))
			backend.Config.ConnState = func(_ net.Conn, st http.ConnState) {
				if st == http.StateNew {
					opened.Add(1)
				}
			}
			backend.Start()
			defer backend.Close()

			target := strings.TrimPrefix(backend.URL, "http://")
			if tc.scheme {
				target = backend.URL
			}

			s := NewProxy(Config{})
			s.logger = log.Log

			h1, err := s.prefillerProxyHandler(target)
			if err != nil {
				t.Fatal(err)
			}
			h2, err := s.prefillerProxyHandler(target)
			if err != nil {
				t.Fatal(err)
			}
			if h1 != h2 {
				t.Errorf("prefillerProxyHandler(%q) built a second handler for the same target", target)
			}

			const n = 3
			for i := 0; i < n; i++ {
				h, err := s.prefillerProxyHandler(target)
				if err != nil {
					t.Fatal(err)
				}
				rec := httptest.NewRecorder()
				req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(`{}`))
				h.ServeHTTP(rec, req)
				if rec.Code != http.StatusOK {
					t.Fatalf("request %d: status %d", i, rec.Code)
				}
			}
			if got := opened.Load(); got != 1 {
				t.Fatalf("%d sequential requests to %q opened %d connections; want 1 keep-alive connection", n, target, got)
			}
		})
	}
}
