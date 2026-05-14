package steps

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/llm-d/coordinator/pkg/logging"
	"github.com/llm-d/coordinator/pkg/pipeline"
	"golang.org/x/sync/errgroup"
)

func init() {
	pipeline.Register("replace-media-urls", NewReplaceMediaURLsStep)
}

type ReplaceMediaURLsStep struct {
	downloadTimeout        time.Duration
	maxConcurrentDownloads int
	client                 *http.Client
}

func NewReplaceMediaURLsStep(params map[string]any) (pipeline.Step, error) {
	timeout := 10 * time.Second
	if v, ok := params["download_timeout"].(string); ok {
		d, err := time.ParseDuration(v)
		if err == nil {
			timeout = d
		}
	}

	maxConcurrent := 10
	if v, ok := params["max_concurrent_downloads"].(int); ok {
		if v <= 0 {
			return nil, fmt.Errorf("max_concurrent_downloads must be positive, got %d", v)
		}
		maxConcurrent = v
	}

	return &ReplaceMediaURLsStep{
		downloadTimeout:        timeout,
		maxConcurrentDownloads: maxConcurrent,
		client:                 &http.Client{Timeout: timeout},
	}, nil
}

func (s *ReplaceMediaURLsStep) Name() string { return "replace-media-urls" }

func (s *ReplaceMediaURLsStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := logging.FromContext(ctx).WithName("replace-media-urls")

	messages, ok := reqCtx.Body["messages"].([]any)
	if !ok {
		return nil
	}

	var imageURLs []imageRef
	for msgIdx, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		content, ok := msgMap["content"].([]any)
		if !ok {
			continue
		}
		for partIdx, part := range content {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if partMap["type"] != "image_url" {
				continue
			}
			imageURL, ok := partMap["image_url"].(map[string]any)
			if !ok {
				continue
			}
			url, ok := imageURL["url"].(string)
			if !ok {
				continue
			}
			imageURLs = append(imageURLs, imageRef{
				msgIdx:  msgIdx,
				partIdx: partIdx,
				url:     url,
			})
		}
	}

	if len(imageURLs) == 0 {
		return nil
	}

	logger.V(logging.TRACE).Info("downloading images", "count", len(imageURLs))

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxConcurrentDownloads)

	results := make([]downloadResult, len(imageURLs))
	for i, ref := range imageURLs {
		g.Go(func() error {
			data, contentType, err := s.download(gCtx, ref.url)
			if err != nil {
				return fmt.Errorf("downloading %s: %w", ref.url, err)
			}
			results[i] = downloadResult{
				ref:         ref,
				base64Data:  base64.StdEncoding.EncodeToString(data),
				contentType: contentType,
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	for _, r := range results {
		dataURI := fmt.Sprintf("data:%s;base64,%s", r.contentType, r.base64Data)

		msg := messages[r.ref.msgIdx].(map[string]any)
		content := msg["content"].([]any)
		part := content[r.ref.partIdx].(map[string]any)
		imageURL := part["image_url"].(map[string]any)
		imageURL["url"] = dataURI

		reqCtx.MultimodalEntries = append(reqCtx.MultimodalEntries, pipeline.MultimodalEntry{
			Index:       len(reqCtx.MultimodalEntries),
			Base64Data:  r.base64Data,
			ContentType: r.contentType,
		})
	}

	return nil
}

func (s *ReplaceMediaURLsStep) download(ctx context.Context, url string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", err
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	return data, contentType, nil
}

type imageRef struct {
	msgIdx  int
	partIdx int
	url     string
}

type downloadResult struct {
	ref         imageRef
	base64Data  string
	contentType string
}
