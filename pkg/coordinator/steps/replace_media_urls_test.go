package steps

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestReplaceMediaURLsStep_DownloadsAndInlines(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step, err := NewReplaceMediaURLsStep(map[string]any{"download_timeout": "5s"})
	if err != nil {
		t.Fatal(err)
	}

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

	err = step.Execute(context.Background(), reqCtx)
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
	step, _ := NewReplaceMediaURLsStep(map[string]any{})

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

	step, _ := NewReplaceMediaURLsStep(map[string]any{})

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
	step, _ := NewReplaceMediaURLsStep(map[string]any{})

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
			step, _ := NewReplaceMediaURLsStep(map[string]any{})
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
			name:        "missing media type defaults to octet-stream",
			uri:         "data:;base64,YWJj",
			wantType:    defaultContentType,
			wantPayload: "YWJj",
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

	step, err := NewReplaceMediaURLsStep(map[string]any{"max_multimodal_entries": 2})
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
	_, err := NewReplaceMediaURLsStep(map[string]any{"max_multimodal_entries": -1})
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

	step, _ := NewReplaceMediaURLsStep(map[string]any{"max_multimodal_entries": 2})

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

	step, _ := NewReplaceMediaURLsStep(map[string]any{})

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
		if _, err := NewReplaceMediaURLsStep(map[string]any{"max_concurrent_downloads": v}); err == nil {
			t.Fatalf("expected error for max_concurrent_downloads=%d", v)
		}
	}
	if _, err := NewReplaceMediaURLsStep(map[string]any{"max_concurrent_downloads": 5}); err != nil {
		t.Fatalf("unexpected error for positive max_concurrent_downloads: %v", err)
	}
}

func TestReplaceMediaURLsStep_Name(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(map[string]any{})
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
			step, _ := NewReplaceMediaURLsStep(map[string]any{})
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
	step, _ := NewReplaceMediaURLsStep(map[string]any{})
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

	step, _ := NewReplaceMediaURLsStep(map[string]any{})
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

	step, _ := NewReplaceMediaURLsStep(map[string]any{})
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

// Structural guard, not a behavioral proxy test. The downloader must leave
// http.Client.Transport nil so it uses http.DefaultTransport, whose Proxy is
// http.ProxyFromEnvironment. That is the only reason image fetches honor
// HTTP_PROXY/HTTPS_PROXY. A custom transport without a Proxy field (as in
// pkg/gateway/client.go) would silently bypass the proxy; this test fails if
// that regression is introduced here.
func TestReplaceMediaURLsStep_ClientUsesDefaultTransport(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	rmu, ok := step.(*ReplaceMediaURLsStep)
	if !ok {
		t.Fatalf("expected *ReplaceMediaURLsStep, got %T", step)
	}
	if rmu.client.Transport != nil {
		t.Fatal("downloader client.Transport must be nil so http.DefaultTransport (Proxy: ProxyFromEnvironment) is used")
	}
}

func TestReplaceMediaURLsStep_DownloadInvalidURL(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(map[string]any{})
	rmu := step.(*ReplaceMediaURLsStep)

	// 0x7f (DEL) is an invalid control character in a URL; NewRequestWithContext
	// fails before any network call.
	_, _, err := rmu.download(context.Background(), "http://\x7f/control-char")
	if err == nil {
		t.Fatal("expected error building request for URL with control character")
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

	step, _ := NewReplaceMediaURLsStep(map[string]any{})
	rmu := step.(*ReplaceMediaURLsStep)

	_, _, err := rmu.download(context.Background(), imageServer.URL+"/truncated")
	if err == nil {
		t.Fatal("expected error reading truncated response body")
	}
}
