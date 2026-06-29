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
	"encoding/base64"
	"errors"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

// newLoopbackStep builds a step whose SSRF guard permits loopback. httptest
// servers bind to 127.0.0.1, which the guard blocks by default, so download
// tests that talk to a local server must opt loopback back in.
func newLoopbackStep(t *testing.T, params map[string]any) *ReplaceMediaURLsStep {
	t.Helper()
	step, err := NewReplaceMediaURLsStep(nil, params)
	if err != nil {
		t.Fatal(err)
	}
	rmu := step.(*ReplaceMediaURLsStep)
	rmu.guard.allowLoopback = true
	return rmu
}

func TestReplaceMediaURLsStep_DownloadsAndInlines(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "describe this"},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/photo.jpg"},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}
	if reqCtx.MultimodalEntries[0].ContentType != "image/jpeg" {
		t.Fatalf("expected content type image/jpeg, got %s", reqCtx.MultimodalEntries[0].ContentType)
	}
	if reqCtx.MultimodalEntries[0].Base64Data == "" {
		t.Fatal("expected Base64Data to be set")
	}

	msgs := reqCtx.Body["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	imgPart := content[1].(map[string]any)["image_url"].(map[string]any)
	url := imgPart["url"].(string)
	if url[:len("data:image/jpeg;base64,")] != "data:image/jpeg;base64," {
		t.Fatalf("expected data URI, got %s", url)
	}
}

func TestReplaceMediaURLsStep_NoImages(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": "just text"},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected 0 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_DownloadFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": server.URL + "/missing.png"},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for failed download")
	}
}

func TestReplaceMediaURLsStep_DataURIInput(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})

	const dataURI = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "describe this"},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": dataURI},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}
	got := reqCtx.MultimodalEntries[0]
	if got.ContentType != "image/jpeg" {
		t.Fatalf("expected content type image/jpeg, got %s", got.ContentType)
	}
	if got.Base64Data != "/9j/4AAQSkZJRg==" {
		t.Fatalf("expected base64 payload preserved, got %q", got.Base64Data)
	}

	msgs := reqCtx.Body["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	imgPart := content[1].(map[string]any)["image_url"].(map[string]any)
	if imgPart["url"].(string) != dataURI {
		t.Fatalf("expected url unchanged, got %s", imgPart["url"])
	}
}

// MultimodalEntry.Index must reflect the position of each image in the
// request, regardless of whether it came from a download or an inline
// data: URI. EncodeStep.buildSingleImageContent indexes by entry.Index so
// drift would associate hashes/placeholders with the wrong image. Asserted
// in both source orderings.
func TestReplaceMediaURLsStep_MixedHTTPAndDataURIOrdering(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("downloaded-image-bytes"))
	}))
	defer imageServer.Close()

	const dataURI = "data:image/jpeg;base64,SU5MSU5F"
	httpURL := imageServer.URL + "/img.png"

	httpPart := map[string]any{"type": "image_url", "image_url": map[string]any{"url": httpURL}}
	dataPart := map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}}

	type want struct {
		contentType string
		base64Data  string
	}
	tests := []struct {
		name  string
		parts []any
		want  []want
	}{
		{
			name:  "http then data",
			parts: []any{httpPart, dataPart},
			want: []want{
				{contentType: "image/png", base64Data: base64.StdEncoding.EncodeToString([]byte("downloaded-image-bytes"))},
				{contentType: "image/jpeg", base64Data: "SU5MSU5F"},
			},
		},
		{
			name:  "data then http",
			parts: []any{dataPart, httpPart},
			want: []want{
				{contentType: "image/jpeg", base64Data: "SU5MSU5F"},
				{contentType: "image/png", base64Data: base64.StdEncoding.EncodeToString([]byte("downloaded-image-bytes"))},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			step := newLoopbackStep(t, map[string]any{})
			reqCtx := &pipeline.RequestContext{
				Body: map[string]any{
					"messages": []any{
						map[string]any{"role": "user", "content": tt.parts},
					},
				},
			}

			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(reqCtx.MultimodalEntries) != len(tt.want) {
				t.Fatalf("expected %d multimodal entries, got %d", len(tt.want), len(reqCtx.MultimodalEntries))
			}
			for i, w := range tt.want {
				got := reqCtx.MultimodalEntries[i]
				if got.Index != i {
					t.Errorf("entry[%d].Index = %d, want %d", i, got.Index, i)
				}
				if got.ContentType != w.contentType {
					t.Errorf("entry[%d].ContentType = %q, want %q", i, got.ContentType, w.contentType)
				}
				if got.Base64Data != w.base64Data {
					t.Errorf("entry[%d].Base64Data = %q, want %q", i, got.Base64Data, w.base64Data)
				}
			}
		})
	}
}

func TestParseDataURI(t *testing.T) {
	tests := []struct {
		name        string
		uri         string
		wantType    string
		wantPayload string
		wantErr     bool
	}{
		{
			name:        "jpeg base64",
			uri:         "data:image/jpeg;base64,/9j/4AAQ",
			wantType:    "image/jpeg",
			wantPayload: "/9j/4AAQ",
		},
		{
			name:        "png base64",
			uri:         "data:image/png;base64,iVBORw0K",
			wantType:    "image/png",
			wantPayload: "iVBORw0K",
		},
		{
			name:    "missing media type",
			uri:     "data:;base64,YWJj",
			wantErr: true,
		},
		{
			name:        "content type normalized to lowercase and trimmed",
			uri:         "data:IMAGE/PNG ;base64,iVBORw0K",
			wantType:    "image/png",
			wantPayload: "iVBORw0K",
		},
		{
			name:    "missing comma",
			uri:     "data:image/jpeg;base64",
			wantErr: true,
		},
		{
			name:    "missing base64 marker",
			uri:     "data:image/jpeg,raw",
			wantErr: true,
		},
		{
			name:    "no semicolon before comma",
			uri:     "data:image/jpeg,abc",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ct, b64, err := parseDataURI(tt.uri)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got contentType=%q payload=%q", ct, b64)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ct != tt.wantType {
				t.Fatalf("contentType: want %q, got %q", tt.wantType, ct)
			}
			if b64 != tt.wantPayload {
				t.Fatalf("payload: want %q, got %q", tt.wantPayload, b64)
			}
		})
	}
}

func TestReplaceMediaURLsStep_RejectsTooManyEntries(t *testing.T) {
	var hits atomic.Int32
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("png-data"))
	}))
	defer imageServer.Close()

	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_multimodal_entries": 2})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/a.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/b.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/c.png"}},
					},
				},
			},
		},
	}

	err = step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for exceeding max_multimodal_entries")
	}
	if !strings.Contains(err.Error(), "too many multimodal entries") {
		t.Fatalf("unexpected error message: %v", err)
	}
	if !strings.Contains(err.Error(), "got 3") || !strings.Contains(err.Error(), "max 2") {
		t.Fatalf("error should include counts: %v", err)
	}
	if hits.Load() != 0 {
		t.Fatalf("expected no downloads on rejection, got %d hits", hits.Load())
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no entries populated on rejection, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_RejectsNegativeMaxEntries(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_multimodal_entries": -1})
	if err == nil {
		t.Fatal("expected error for negative max_multimodal_entries")
	}
}

func TestReplaceMediaURLsStep_AllowsAtLimit(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("png-data"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"max_multimodal_entries": 2})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/a.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/b.png"}},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error at limit: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_MultipleImages(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("png-data"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/a.png"},
						},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/b.png"},
						},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/c.png"},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(reqCtx.MultimodalEntries))
	}
	for i, entry := range reqCtx.MultimodalEntries {
		if entry.Base64Data == "" {
			t.Fatalf("entry %d: expected Base64Data to be set", i)
		}
	}
}

func TestReplaceMediaURLsStep_RejectsNonPositiveMaxConcurrent(t *testing.T) {
	for _, v := range []int{0, -1} {
		if _, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_concurrent_downloads": v}); err == nil {
			t.Fatalf("expected error for max_concurrent_downloads=%d", v)
		}
	}
	if _, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_concurrent_downloads": 5}); err != nil {
		t.Fatalf("unexpected error for positive max_concurrent_downloads: %v", err)
	}
}

func TestReplaceMediaURLsStep_Name(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if step.Name() != ReplaceMediaURLsStepName {
		t.Fatalf("Name() = %q, want %q", step.Name(), ReplaceMediaURLsStepName)
	}
}

func TestReplaceMediaURLsStep_MalformedBody(t *testing.T) {
	tests := []struct {
		name string
		body map[string]any
	}{
		{
			name: "no messages key",
			body: map[string]any{"model": "x"},
		},
		{
			name: "message not a map",
			body: map[string]any{"messages": []any{"not-a-map"}},
		},
		{
			name: "content part not a map",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user", "content": []any{"not-a-map"}},
				},
			},
		},
		{
			name: "image_url field not a map",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user", "content": []any{
						map[string]any{"type": "image_url", "image_url": "not-a-map"},
					}},
				},
			},
		},
		{
			name: "url field not a string",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user", "content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": 123}},
					}},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
			reqCtx := &pipeline.RequestContext{Body: tt.body}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(reqCtx.MultimodalEntries) != 0 {
				t.Fatalf("expected 0 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
			}
		})
	}
}

func TestReplaceMediaURLsStep_InvalidDataURI(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:image/jpeg;base64"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for malformed data URI")
	}
	if !strings.Contains(err.Error(), "parsing data URI") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestReplaceMediaURLsStep_EmptyContentType(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header()["Content-Type"] = nil // suppress net/http content sniffing
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("raw-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/raw"},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}
	if reqCtx.MultimodalEntries[0].ContentType != defaultContentType {
		t.Fatalf("expected %s, got %q", defaultContentType, reqCtx.MultimodalEntries[0].ContentType)
	}
}

func TestReplaceMediaURLsStep_DownloadUnreachable(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	deadURL := imageServer.URL + "/gone.png"
	imageServer.Close() // nothing is listening on this address now

	step := newLoopbackStep(t, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": deadURL},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for unreachable download host")
	}
	if !strings.Contains(err.Error(), "downloading") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

// Structural guard, not a behavioral proxy test. The SSRF dial guard requires a
// custom transport, so the downloader clones http.DefaultTransport to retain
// its Proxy: http.ProxyFromEnvironment. That is the only reason image fetches
// honor HTTP_PROXY/HTTPS_PROXY. A custom transport without a Proxy field (as in
// pkg/gateway/client.go) would silently bypass the proxy; this test fails if
// that regression is introduced here.
func TestReplaceMediaURLsStep_ClientPreservesProxy(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	rmu, ok := step.(*ReplaceMediaURLsStep)
	if !ok {
		t.Fatalf("expected *ReplaceMediaURLsStep, got %T", step)
	}
	transport, ok := rmu.client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected *http.Transport, got %T", rmu.client.Transport)
	}
	if transport.Proxy == nil {
		t.Fatal("downloader transport must keep Proxy (http.ProxyFromEnvironment) so HTTP(S)_PROXY is honored")
	}
}

func TestReplaceMediaURLsStep_DownloadInvalidURL(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	rmu := step.(*ReplaceMediaURLsStep)

	// 0x7f (DEL) is an invalid control character in a URL; NewRequestWithContext
	// fails before any network call.
	_, _, err := rmu.download(context.Background(), "http://\x7f/control-char")
	if err == nil {
		t.Fatal("expected error building request for URL with control character")
	}
}

func TestReplaceMediaURLsStep_RejectsOversizedBody(t *testing.T) {
	var hits atomic.Int32
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "image/png")
		// No Content-Length set: force the size check to happen during the read.
		w.(http.Flusher).Flush()
		_, _ = w.Write(make([]byte, 64))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"max_download_size": 16})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/big.png"}},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for oversized download")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no entries populated on rejection, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_RejectsOversizedContentLength(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Length", "1000")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(make([]byte, 1000))
	}))
	defer imageServer.Close()

	rmu := newLoopbackStep(t, map[string]any{"max_download_size": 16})

	_, _, err := rmu.download(context.Background(), imageServer.URL+"/big.png")
	if err == nil {
		t.Fatal("expected error for oversized Content-Length")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

func TestReplaceMediaURLsStep_AllowsBodyAtCap(t *testing.T) {
	const size = 16
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(make([]byte, size))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"max_download_size": size})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/atcap.png"}},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error for body exactly at cap: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(reqCtx.MultimodalEntries))
	}
	if want := base64.StdEncoding.EncodeToString(make([]byte, size)); reqCtx.MultimodalEntries[0].Base64Data != want {
		t.Fatalf("entry data mismatch: got %q want %q", reqCtx.MultimodalEntries[0].Base64Data, want)
	}
}

// A request may carry several image_url entries. The per-download cap must
// bound each one independently: a single oversized entry rejects the whole
// request even when the others are within the cap.
func TestReplaceMediaURLsStep_RejectsOneOversizedAmongMany(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		if strings.HasPrefix(r.URL.Path, "/big") {
			_, _ = w.Write(make([]byte, 64))
			return
		}
		_, _ = w.Write(make([]byte, 4))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"max_download_size": 16})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/small1.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/big.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/small2.png"}},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error when one of several entries is oversized")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

func TestReplaceMediaURLsStep_RejectsInvalidMaxDownloadSize(t *testing.T) {
	// math.MaxInt is rejected so the maxDownloadSize+1 sentinel read in
	// download() cannot overflow to a negative limit (which io.LimitReader
	// would treat as immediate EOF, accepting an oversized body as empty).
	for _, v := range []int{0, -1, math.MaxInt} {
		if _, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_download_size": v}); err == nil {
			t.Fatalf("expected error for max_download_size=%d", v)
		}
	}
}

func TestReplaceMediaURLsStep_DownloadTruncatedBody(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			return
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			return
		}
		// Promise 100 bytes, send 5, then close: the client's io.ReadAll sees an
		// unexpected EOF.
		_, _ = conn.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort"))
		_ = conn.Close()
	}))
	defer imageServer.Close()

	rmu := newLoopbackStep(t, map[string]any{})

	_, _, err := rmu.download(context.Background(), imageServer.URL+"/truncated")
	if err == nil {
		t.Fatal("expected error reading truncated response body")
	}
}

func TestAddressGuard_BlockedIP(t *testing.T) {
	tests := []struct {
		name string
		ip   string
		want bool
	}{
		{"metadata link-local", "169.254.169.254", true},
		{"loopback v4", "127.0.0.1", true},
		{"loopback v6", "::1", true},
		{"link-local v6", "fe80::1", true},
		{"unspecified v4", "0.0.0.0", true},
		{"unspecified v6", "::", true},
		{"cgnat", "100.64.1.1", true},
		{"private 10", "10.0.0.1", true},
		{"private 172", "172.16.0.1", true},
		{"private 192", "192.168.1.1", true},
		{"unique-local v6", "fc00::1", true},
		{"ipv4-mapped metadata", "::ffff:169.254.169.254", true},
		{"ipv4-mapped private", "::ffff:10.0.0.1", true},
		{"public v4", "8.8.8.8", false},
		{"public v6", "2001:4860:4860::8888", false},
	}
	guard := &addressGuard{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ip := net.ParseIP(tt.ip)
			if ip == nil {
				t.Fatalf("could not parse %q", tt.ip)
			}
			if got := guard.blockedIP(ip); got != tt.want {
				t.Fatalf("blockedIP(%s) = %v, want %v", tt.ip, got, tt.want)
			}
		})
	}
}

func TestAddressGuard_AllowPrivate(t *testing.T) {
	guard := &addressGuard{allowPrivate: true}
	// RFC1918 ranges are permitted when opted in...
	for _, ip := range []string{"10.0.0.1", "172.16.0.1", "192.168.1.1"} {
		if guard.blockedIP(net.ParseIP(ip)) {
			t.Errorf("blockedIP(%s) = true, want false with allowPrivate", ip)
		}
	}
	// ...but the metadata endpoint and other special ranges stay blocked.
	// allowPrivate is RFC1918-only: IPv6 unique-local (fc00::/7) must not leak
	// through, even though net.IP.IsPrivate treats it as private.
	for _, ip := range []string{"169.254.169.254", "127.0.0.1", "0.0.0.0", "100.64.1.1", "fc00::1"} {
		if !guard.blockedIP(net.ParseIP(ip)) {
			t.Errorf("blockedIP(%s) = false, want true even with allowPrivate", ip)
		}
	}
}

func TestAddressGuard_HostAllowed(t *testing.T) {
	open := &addressGuard{}
	if !open.hostAllowed("anything.example.com") {
		t.Fatal("empty allowlist must allow any host")
	}

	restricted := &addressGuard{allowedDomains: map[string]struct{}{"images.example.com": {}}}
	if !restricted.hostAllowed("images.example.com") {
		t.Fatal("listed host must be allowed")
	}
	if !restricted.hostAllowed("IMAGES.EXAMPLE.COM") {
		t.Fatal("host match must be case-insensitive")
	}
	if restricted.hostAllowed("evil.example.com") {
		t.Fatal("unlisted host must be rejected")
	}
}

// download rejects non-http(s) schemes before any network call.
func TestReplaceMediaURLsStep_RejectsScheme(t *testing.T) {
	rmu := newLoopbackStep(t, map[string]any{})
	for _, raw := range []string{"file:///etc/passwd", "gopher://host/1", "ftp://host/x"} {
		_, _, err := rmu.download(context.Background(), raw)
		if err == nil {
			t.Fatalf("expected scheme %q to be rejected", raw)
		}
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Fatalf("scheme rejection must be a bad request, got %v", err)
		}
	}
}

// A dial to a blocked range surfaces as a client error (ErrBadRequest), not a
// generic gateway fault, so the handler maps it to a 4xx.
func TestReplaceMediaURLsStep_BlocksMetadataIP(t *testing.T) {
	rmu := newLoopbackStep(t, map[string]any{"download_timeout": "2s"})
	_, _, err := rmu.download(context.Background(), "http://169.254.169.254/latest/meta-data/")
	if err == nil {
		t.Fatal("expected metadata IP fetch to be blocked")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("blocked address must classify as bad request, got %v", err)
	}
}

// An allowed public host that 302-redirects to a blocked address is rejected at
// the dial of the redirect hop.
func TestReplaceMediaURLsStep_BlocksRedirectToPrivate(t *testing.T) {
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://169.254.169.254/latest/meta-data/", http.StatusFound)
	}))
	defer redirector.Close()

	// Loopback allowed so the first hop (the httptest server) connects; the
	// metadata redirect target is link-local and stays blocked regardless.
	rmu := newLoopbackStep(t, map[string]any{"download_timeout": "2s"})
	_, _, err := rmu.download(context.Background(), redirector.URL+"/start")
	if err == nil {
		t.Fatal("expected redirect to metadata IP to be blocked")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("blocked redirect must classify as bad request, got %v", err)
	}
}

// A hostname that resolves to a blocked IP is caught at dial time, defeating a
// DNS-rebinding bypass. localhost resolves to loopback, blocked by default.
func TestReplaceMediaURLsStep_BlocksHostnameResolvingToPrivate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("data"))
	}))
	defer server.Close()

	_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}

	// Default guard: loopback blocked. "localhost" resolves to 127.0.0.1/::1.
	built, err := NewReplaceMediaURLsStep(nil, map[string]any{"download_timeout": "2s"})
	if err != nil {
		t.Fatal(err)
	}
	step := built.(*ReplaceMediaURLsStep)
	_, _, err = step.download(context.Background(), "http://localhost:"+port+"/x")
	if err == nil {
		t.Fatal("expected hostname resolving to loopback to be blocked")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("blocked resolved host must classify as bad request, got %v", err)
	}
}

// With a domain allowlist set, only listed hosts are fetched; others are
// rejected before any connection.
func TestReplaceMediaURLsStep_DomainAllowlist(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("img"))
	}))
	defer server.Close()

	host := strings.TrimPrefix(server.URL, "http://")
	hostname, _, err := net.SplitHostPort(host)
	if err != nil {
		t.Fatal(err)
	}

	allowed := newLoopbackStep(t, map[string]any{"allowed_domains": []any{hostname}})
	if _, _, err := allowed.download(context.Background(), server.URL+"/ok.png"); err != nil {
		t.Fatalf("listed host must be fetchable: %v", err)
	}

	denied := newLoopbackStep(t, map[string]any{"allowed_domains": []any{"images.example.com"}})
	_, _, err = denied.download(context.Background(), server.URL+"/ok.png")
	if err == nil {
		t.Fatal("unlisted host must be rejected")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("allowlist rejection must classify as bad request, got %v", err)
	}
}

// allowed_domains entries must be strings.
func TestReplaceMediaURLsStep_RejectsNonStringAllowedDomain(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{"allowed_domains": []any{123}})
	if err == nil {
		t.Fatal("expected error for non-string allowed_domains entry")
	}
}

// A list arriving as []string (a programmatic caller, not the YAML path) must
// build the allowlist, not silently fall back to allow-all.
func TestReplaceMediaURLsStep_AllowedDomainsStringSlice(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"allowed_domains": []string{"images.example.com"}})
	if err != nil {
		t.Fatal(err)
	}
	guard := step.(*ReplaceMediaURLsStep).guard
	if guard.hostAllowed("evil.example.com") {
		t.Fatal("allowlist must reject unlisted host")
	}
	if !guard.hostAllowed("images.example.com") {
		t.Fatal("allowlist must permit listed host")
	}
}

// An allowed_domains value of an unsupported type must error, not silently
// disable the allowlist.
func TestReplaceMediaURLsStep_RejectsNonListAllowedDomains(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{"allowed_domains": "images.example.com"})
	if err == nil {
		t.Fatal("expected error for non-list allowed_domains")
	}
}

func TestReplaceMediaURLsStep_RejectsNonImageDataURI(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:text/html;base64,PGgxPmhpPC9oMT4="},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for non-image data URI content type")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

func TestReplaceMediaURLsStep_RejectsMissingMediaType(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:;base64,AAAA"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for data URI missing media type")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

func TestReplaceMediaURLsStep_CancelledContextSkipsDataURIParse(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(map[string]any{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:image/jpeg,raw"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(ctx, reqCtx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}
