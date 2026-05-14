package steps

import (
	"context"
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
