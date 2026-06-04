package steps

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

const RenderStepName = "render"

func init() {
	pipeline.Register(RenderStepName, NewRenderStep)
}

type RenderStep struct {
	serviceAddress string
	client         *http.Client
}

func NewRenderStep(params map[string]any) (pipeline.Step, error) {
	timeout := 30 * time.Second
	if v, ok := params["timeout"].(string); ok {
		if d, err := time.ParseDuration(v); err == nil {
			timeout = d
		}
	}

	address, _ := params["address"].(string)

	return &RenderStep{
		serviceAddress: address,
		client:         &http.Client{Timeout: timeout},
	}, nil
}

func (s *RenderStep) SetServiceAddress(addr string) {
	s.serviceAddress = addr
}

func (s *RenderStep) Name() string { return RenderStepName }

func (s *RenderStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	if strings.Contains(reqCtx.OriginalPath, gateway.PathCompletions) {
		return s.executeCompletions(ctx, reqCtx)
	} else if strings.Contains(reqCtx.OriginalPath, gateway.PathChatCompletions) {
		return s.executeChatCompletions(ctx, reqCtx)
	}
	logger := log.FromContext(ctx).WithName(RenderStepName)
	logger.V(logutil.DEFAULT).Info("skipping render step", "path", reqCtx.OriginalPath)
	return nil
}

func (s *RenderStep) executeCompletions(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(RenderStepName)

	prompt := reqCtx.Body["prompt"]

	switch p := prompt.(type) {
	case string:
		// /v1/completions/render returns an array with one element per prompt.
		// We reject batched prompts above, so we always expect length 1. We
		// decode into a minimal struct so completions stays decoupled from the
		// chat-completions response shape.
		var renderResp []completionsRenderResponse
		if err := s.postRender(ctx, reqCtx, gateway.PathCompletions, &renderResp); err != nil {
			return err
		}
		if len(renderResp) != 1 {
			return fmt.Errorf("render: expected 1 response element, got %d", len(renderResp))
		}
		tokenIDs := renderResp[0].TokenIDs
		reqCtx.TokenIDs = tokenIDs
		reqCtx.Body["prompt"] = tokenIDs
		logger.V(logutil.DEFAULT).Info("complete", "token_ids_len", len(tokenIDs))
		return nil

	case []any:
		// OpenAI /v1/completions accepts four prompt shapes: string, []string,
		// []int (token IDs), [][]int (batched token IDs). The first three are
		// covered here; only single-sequence variants are supported downstream
		// because RequestContext.TokenIDs is []int.
		if len(p) == 0 {
			reqCtx.TokenIDs = []int{}
			logger.V(logutil.DEFAULT).Info("prompt is empty array, skipping render", "token_ids_len", 0)
			return nil
		}
		switch p[0].(type) {
		case float64, json.Number:
			// Convert to []int for downstream steps and validate every element
			// is numeric; a heterogeneous array fails fast here rather than
			// reaching vLLM as garbage tokens.
			tokenIDs, err := toIntSlice(p)
			if err != nil {
				return err
			}
			reqCtx.TokenIDs = tokenIDs
			logger.V(logutil.DEFAULT).Info("prompt is token array, skipping render", "token_ids_len", len(tokenIDs))
			return nil
		case string:
			return errors.New("render: batched string prompts ([]string) are not supported")
		case []any:
			return errors.New("render: batched token prompts ([][]int) are not supported")
		default:
			return fmt.Errorf("render: invalid prompt array element: %T", p[0])
		}

	default:
		return fmt.Errorf("render: prompt must be a string or token array, got %T", prompt)
	}
}

func (s *RenderStep) executeChatCompletions(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(RenderStepName)

	var renderResp renderResponse
	if err := s.postRender(ctx, reqCtx, gateway.PathChatCompletions, &renderResp); err != nil {
		return err
	}

	reqCtx.TokenIDs = renderResp.TokenIDs

	imageHashes := renderResp.Features.MMHashes[ModalityImage]
	imagePlaceholders := renderResp.Features.MMPlaceholders[ModalityImage]
	imageKwargs := renderResp.Features.KwargsData[ModalityImage]

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

// postRender marshals reqCtx.Body, POSTs it to <serviceAddress><basePath>/render,
// and decodes the JSON response into out.
func (s *RenderStep) postRender(ctx context.Context, reqCtx *pipeline.RequestContext, basePath string, out any) error {
	if s.serviceAddress == "" {
		return errors.New("render: service address not configured (set 'address' in render step params)")
	}

	logger := log.FromContext(ctx).WithName(RenderStepName)

	body, err := json.Marshal(reqCtx.Body)
	if err != nil {
		return fmt.Errorf("marshaling request for render: %w", err)
	}

	url := s.serviceAddress + basePath + "/render"
	logger.V(logutil.DEFAULT).Info("sending request", "url", url)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, io.NopCloser(jsonReader(body)))
	if err != nil {
		return fmt.Errorf("creating render request: %w", err)
	}
	req.ContentLength = int64(len(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range reqCtx.ForwardedHeaders() {
		req.Header.Set(k, v)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("render request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("render service returned HTTP %d: %s", resp.StatusCode, string(respBody))
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decoding render response: %w", err)
	}
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

// completionsRenderResponse is a minimal view of the per-prompt object returned
// by /v1/completions/render. Only token_ids is consumed; other fields
// (request_id, sampling_params, model, etc.) are ignored.
type completionsRenderResponse struct {
	TokenIDs []int `json:"token_ids"`
}

func toIntSlice(values []any) ([]int, error) {
	out := make([]int, 0, len(values))
	for _, v := range values {
		switch n := v.(type) {
		case float64:
			if n < 0 || n != math.Trunc(n) {
				return nil, fmt.Errorf("render: invalid token in prompt array: %v (must be a non-negative integer)", v)
			}
			out = append(out, int(n))
		case json.Number:
			i, err := n.Int64()
			if err != nil {
				return nil, fmt.Errorf("render: invalid token in prompt array: %v", v)
			}
			if i < 0 {
				return nil, fmt.Errorf("render: invalid token in prompt array: %v (must be a non-negative integer)", v)
			}
			out = append(out, int(i))
		default:
			return nil, fmt.Errorf("render: invalid token in prompt array: %T", v)
		}
	}
	return out, nil
}
