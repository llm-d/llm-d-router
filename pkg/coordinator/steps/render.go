package steps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

const RenderStepName = "render"

func init() {
	pipeline.Register(RenderStepName, NewRenderStep)
}

type RenderStep struct {
	serviceAddress string
	endpoint       string
	client         *http.Client
}

func NewRenderStep(params map[string]any) (pipeline.Step, error) {
	endpoint := "/v1/chat/completions/render"
	if v, ok := params["endpoint"].(string); ok {
		endpoint = v
	}

	timeout := 30 * time.Second
	if v, ok := params["timeout"].(string); ok {
		if d, err := time.ParseDuration(v); err == nil {
			timeout = d
		}
	}

	return &RenderStep{
		endpoint: endpoint,
		client:   &http.Client{Timeout: timeout},
	}, nil
}

func (s *RenderStep) SetServiceAddress(addr string) {
	s.serviceAddress = addr
}

func (s *RenderStep) Name() string { return RenderStepName }

func (s *RenderStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName("render")

	body, err := json.Marshal(reqCtx.Body)
	if err != nil {
		return fmt.Errorf("marshaling request for render: %w", err)
	}

	url := s.serviceAddress + s.endpoint
	logger.V(logutil.DEFAULT).Info("sending request", "url", url)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	if err != nil {
		return fmt.Errorf("creating render request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Body = io.NopCloser(jsonReader(body))
	req.ContentLength = int64(len(body))

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("render request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("render service returned HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var renderResp renderResponse
	if err := json.NewDecoder(resp.Body).Decode(&renderResp); err != nil {
		return fmt.Errorf("decoding render response: %w", err)
	}

	reqCtx.TokenIDs = renderResp.TokenIDs

	imageHashes := renderResp.Features.MMHashes["image"]
	imagePlaceholders := renderResp.Features.MMPlaceholders["image"]
	imageKwargs := renderResp.Features.KwargsData["image"]

	expected := len(reqCtx.MultimodalEntries)
	if len(imageHashes) != expected {
		return fmt.Errorf("render returned %d mm_hashes but expected %d", len(imageHashes), expected)
	}
	if len(imagePlaceholders) != expected {
		return fmt.Errorf("render returned %d mm_placeholders but expected %d", len(imagePlaceholders), expected)
	}
	if len(imageKwargs) != expected {
		return fmt.Errorf("render returned %d kwargs_data but expected %d", len(imageKwargs), expected)
	}

	for i := range reqCtx.MultimodalEntries {
		reqCtx.MultimodalEntries[i].Hash = imageHashes[i]
		reqCtx.MultimodalEntries[i].KwargsData = imageKwargs[i]
		reqCtx.MultimodalEntries[i].Placeholder = imagePlaceholders[i]
	}

	logger.V(logutil.DEBUG).Info("response", "mm_hashes", imageHashes, "mm_placeholders", imagePlaceholders, "kwargs_data_len", len(imageKwargs))
	logger.V(logutil.DEFAULT).Info("complete", "token_ids_len", len(renderResp.TokenIDs), "images", len(imageHashes))
	return nil
}

type renderResponse struct {
	TokenIDs []int          `json:"token_ids"`
	Features renderFeatures `json:"features"`
}

type renderFeatures struct {
	MMHashes       map[string][]string                    `json:"mm_hashes"`
	MMPlaceholders map[string][]pipeline.PlaceholderRange `json:"mm_placeholders"`
	KwargsData     map[string][]string                    `json:"kwargs_data"`
}
