package steps

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/coordinator/pkg/pipeline"
	"golang.org/x/sync/errgroup"
)

const ReplaceMediaURLsStepName = "replace-media-urls"

const imageURLPartType = "image_url"

func init() {
	pipeline.Register(ReplaceMediaURLsStepName, NewReplaceMediaURLsStep)
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

func (s *ReplaceMediaURLsStep) Name() string { return ReplaceMediaURLsStepName }

func (s *ReplaceMediaURLsStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(ReplaceMediaURLsStepName)

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
			if partMap["type"] != imageURLPartType {
				continue
			}
			imageURL, ok := partMap[imageURLPartType].(map[string]any)
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

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxConcurrentDownloads)

	results := make([]downloadResult, len(imageURLs))
	for i, ref := range imageURLs {
		if strings.HasPrefix(ref.url, "data:") {
			contentType, b64, err := parseDataURI(ref.url)
			if err != nil {
				return fmt.Errorf("parsing data URI at message %d part %d: %w", ref.msgIdx, ref.partIdx, err)
			}
			results[i] = downloadResult{ref: ref, base64Data: b64, contentType: contentType}
			continue
		}
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

	logger.V(logutil.TRACE).Info("downloading images", "count", len(imageURLs))

	if err := g.Wait(); err != nil {
		return err
	}

	for _, r := range results {
		if !strings.HasPrefix(r.ref.url, "data:") {
			dataURI := fmt.Sprintf("data:%s;base64,%s", r.contentType, r.base64Data)
			msg := messages[r.ref.msgIdx].(map[string]any)
			content := msg["content"].([]any)
			part := content[r.ref.partIdx].(map[string]any)
			imageURL := part[imageURLPartType].(map[string]any)
			imageURL["url"] = dataURI
		}

		appendMultimodalEntry(reqCtx, r.contentType, r.base64Data)
	}

	return nil
}

func appendMultimodalEntry(reqCtx *pipeline.RequestContext, contentType, b64 string) {
	reqCtx.MultimodalEntries = append(reqCtx.MultimodalEntries, pipeline.MultimodalEntry{
		Index:       len(reqCtx.MultimodalEntries),
		Base64Data:  b64,
		ContentType: contentType,
	})
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

func parseDataURI(uri string) (contentType, b64 string, err error) {
	rest := strings.TrimPrefix(uri, "data:")
	meta, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return "", "", errors.New("missing comma in data URI")
	}
	ct, params, _ := strings.Cut(meta, ";")
	hasBase64 := false
	for _, p := range strings.Split(params, ";") {
		if strings.EqualFold(strings.TrimSpace(p), "base64") {
			hasBase64 = true
			break
		}
	}
	if !hasBase64 {
		return "", "", errors.New("data URI must be base64-encoded")
	}
	if ct == "" {
		ct = "application/octet-stream"
	}
	return ct, payload, nil
}
