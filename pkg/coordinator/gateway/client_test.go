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
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/go-logr/logr/funcr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
)

// The gateway reaches in-cluster destinations (Inference Gateway, EPP, model-serving pods).
// Its client must NOT route through HTTP_PROXY/HTTPS_PROXY: it builds an explicit
// http.Transport and leaves the Proxy field nil ("never proxy"). This is the
// opposite of the multimedia downloader, which relies on http.DefaultTransport to
// honor the proxy env. This test fails if someone adds a Proxy to the transport,
// which would send in-cluster traffic through an external forward proxy.
func TestClient_IgnoresProxyEnv(t *testing.T) {
	c := New(config.GatewayConfig{Address: "http://gw"})

	tr, ok := c.httpClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected *http.Transport, got %T", c.httpClient.Transport)
	}
	if tr.Proxy != nil {
		t.Fatal("gateway transport must not set Proxy: in-cluster traffic must not route through HTTP(S)_PROXY")
	}
}

// redactStrings is the log-redaction safety net: it keeps base64 blobs,
// credential-bearing URLs, and oversized arrays out of trace logs. These cases
// pin each branch.
func TestRedactStrings(t *testing.T) {
	const shortStr = "keep me"
	longBase64 := "data:image/png;base64," + strings.Repeat("A", 60)
	longURL := "https://example.com/" + strings.Repeat("a", 60)
	longHTTP := "http://example.com/" + strings.Repeat("a", 60)
	longPlain := strings.Repeat("x", 60)

	tests := []struct {
		name string
		in   any
		want any
	}{
		{"short string unchanged", shortStr, shortStr},
		{"exactly max length unchanged", strings.Repeat("a", maxRedactStringLen), strings.Repeat("a", maxRedactStringLen)},
		{"long data URI redacted", longBase64, "[base64]"},
		{"long https URL redacted", longURL, "[url]"},
		{"long http URL redacted", longHTTP, "[url]"},
		{"long plain string truncated", longPlain, "..."},
		{"non-string scalar unchanged", 42.0, 42.0},
		{
			"map recurses into values",
			map[string]any{"keep": shortStr, "blob": longBase64},
			map[string]any{"keep": shortStr, "blob": "[base64]"},
		},
		{
			"small array recurses elementwise",
			[]any{shortStr, longURL},
			[]any{shortStr, "[url]"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := redactStrings(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("redactStrings(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

// An array longer than the cap keeps the first 10 elements (redacted) and a
// trailing count marker, so a long prompt list cannot flood the log.
func TestRedactStrings_TruncatesLongArray(t *testing.T) {
	const extra = 5
	in := make([]any, maxRedactElems+extra)
	for i := range in {
		in[i] = "elem"
	}

	got, ok := redactStrings(in).([]any)
	if !ok {
		t.Fatalf("expected []any, got %T", redactStrings(in))
	}
	if len(got) != maxRedactElems+1 {
		t.Fatalf("expected %d elements + marker, got %d", maxRedactElems, len(got))
	}
	if marker, want := got[maxRedactElems], fmt.Sprintf("... +%d more", extra); marker != want {
		t.Errorf("truncation marker = %v, want %q", marker, want)
	}
}

// Deeply nested input is cut off at maxRedactDepth with a sentinel, so an
// adversarial body cannot drive recursion toward the encoding/json limit.
func TestRedactStrings_CapsDepth(t *testing.T) {
	var v any = "leaf"
	for i := 0; i < maxRedactDepth+10; i++ {
		v = map[string]any{"k": v}
	}

	got := redactStrings(v)
	for i := 0; i < maxRedactDepth; i++ {
		m, ok := got.(map[string]any)
		if !ok {
			t.Fatalf("level %d: expected map, got %T (%v)", i, got, got)
		}
		got = m["k"]
	}
	if got != "[truncated]" {
		t.Errorf("at depth %d = %v, want %q", maxRedactDepth, got, "[truncated]")
	}
}

func TestRedactBody(t *testing.T) {
	t.Run("valid JSON redacts string values", func(t *testing.T) {
		blob := strings.Repeat("Z", 60)
		body, err := json.Marshal(map[string]any{"model": "m", "image": blob})
		if err != nil {
			t.Fatal(err)
		}

		got, ok := redactBody(body).(map[string]any)
		if !ok {
			t.Fatalf("expected map[string]any, got %T", redactBody(body))
		}
		if got["model"] != "m" {
			t.Errorf("short value changed: %v", got["model"])
		}
		if got["image"] != "..." {
			t.Errorf("long value not redacted: %v", got["image"])
		}
	})

	t.Run("non-JSON under max length returned verbatim", func(t *testing.T) {
		if got := redactBody([]byte("not json")); got != "not json" {
			t.Errorf("redactBody = %v, want verbatim string", got)
		}
	})

	t.Run("non-JSON exactly max length returned verbatim", func(t *testing.T) {
		raw := strings.Repeat("q", maxRedactRawBodyLen)
		if got := redactBody([]byte(raw)); got != raw {
			t.Errorf("redactBody = %v, want verbatim (boundary is > not >=)", got)
		}
	})

	t.Run("non-JSON over max length truncated", func(t *testing.T) {
		raw := strings.Repeat("q", maxRedactRawBodyLen+50)
		got, ok := redactBody([]byte(raw)).(string)
		if !ok {
			t.Fatalf("expected string, got %T", redactBody([]byte(raw)))
		}
		if want := raw[:maxRedactRawBodyLen] + "..."; got != want {
			t.Errorf("redactBody truncation = %q, want %q", got, want)
		}
	})
}

// Request returns a readable body regardless of log level: buffered only at
// TRACE, otherwise the unread stream. This round-trip pins that contract: the
// returned body is fully readable and carries the upstream status.
func TestClient_RequestReturnsReadableBody(t *testing.T) {
	const want = `{"ok":true}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(ContentTypeHeader) != ContentTypeJSON {
			t.Errorf("content-type = %q, want %q", r.Header.Get(ContentTypeHeader), ContentTypeJSON)
		}
		if got := r.Header.Get("X-Custom"); got != "v" {
			t.Errorf("custom header not forwarded, got %q", got)
		}
		w.WriteHeader(http.StatusCreated)
		_, _ = io.WriteString(w, want)
	}))
	defer srv.Close()

	c := New(config.GatewayConfig{Address: srv.URL})
	resp, err := c.Post(context.Background(), "/v1/chat/completions", []byte(`{"q":1}`), map[string]string{"X-Custom": "v"})
	if err != nil {
		t.Fatalf("Post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusCreated)
	}
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading body: %v", err)
	}
	if string(got) != want {
		t.Errorf("body = %q, want %q", got, want)
	}
}

// At TRACE the body is buffered into an in-memory reader so it can be logged
// and still handed to the caller; with TRACE off it is left unread to avoid
// holding large prefill/encode responses in memory.
func TestClient_RequestBuffersBodyOnlyAtTrace(t *testing.T) {
	const want = `{"ok":true}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, want)
	}))
	defer srv.Close()

	c := New(config.GatewayConfig{Address: srv.URL})
	bufferedType := reflect.TypeOf(io.NopCloser(bytes.NewReader(nil)))

	t.Run("trace off leaves body unread", func(t *testing.T) {
		resp, err := c.Post(context.Background(), "/v1/chat/completions", []byte(`{"q":1}`), nil)
		if err != nil {
			t.Fatalf("Post: %v", err)
		}
		defer resp.Body.Close()
		if got := reflect.TypeOf(resp.Body); got == bufferedType {
			t.Errorf("resp.Body buffered with trace off, type = %v", got)
		}
	})

	t.Run("trace on buffers body", func(t *testing.T) {
		logger := funcr.NewJSON(func(string) {}, funcr.Options{Verbosity: logutil.TRACE})
		ctx := log.IntoContext(context.Background(), logger)
		resp, err := c.Post(ctx, "/v1/chat/completions", []byte(`{"q":1}`), nil)
		if err != nil {
			t.Fatalf("Post: %v", err)
		}
		defer resp.Body.Close()
		if got := reflect.TypeOf(resp.Body); got != bufferedType {
			t.Errorf("resp.Body type = %v, want buffered %v", got, bufferedType)
		}
		got, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("reading buffered body: %v", err)
		}
		if string(got) != want {
			t.Errorf("body = %q, want %q", got, want)
		}
	})
}

func TestClient_BaseURLAndTransport(t *testing.T) {
	c := New(config.GatewayConfig{Address: "http://gw:80"})
	if c.BaseURL() != "http://gw:80" {
		t.Errorf("BaseURL = %q, want %q", c.BaseURL(), "http://gw:80")
	}
	if c.Transport() == nil {
		t.Error("Transport must not be nil")
	}
}
