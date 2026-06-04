package steps

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
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

	// Verify URL was replaced with data URI
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

	// Existing data URI must be left in place; nothing rewrites it.
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
			wantType:    "application/octet-stream",
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
