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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
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
	serviceAddress            string
	maxTotalTokens            int
	maxTotalPlaceholderTokens int
	client                    *http.Client
}

func NewRenderStep(_ *gateway.Client, params map[string]any) (pipeline.Step, error) {
	timeout := 30 * time.Second
	if v, ok, err := paramDuration(params, "timeout"); err != nil {
		return nil, err
	} else if ok {
		timeout = v
	}

	maxIdleConnsPerHost := 100
	if v, ok, err := paramInt(params, "max_idle_conns_per_host"); err != nil {
		return nil, err
	} else if ok {
		if v < 0 {
			return nil, fmt.Errorf("max_idle_conns_per_host must be non-negative, got %d", v)
		}
		maxIdleConnsPerHost = v
	}

	idleConnTimeout := 90 * time.Second
	if v, ok, err := paramDuration(params, "idle_conn_timeout"); err != nil {
		return nil, err
	} else if ok {
		idleConnTimeout = v
	}

	address, _ := params["address"].(string)

	maxTokens := 0
	if v, ok, err := paramInt(params, "max_total_tokens"); err != nil {
		return nil, err
	} else if ok {
		if v < 0 {
			return nil, fmt.Errorf("max_total_tokens must be non-negative, got %d", v)
		}
		maxTokens = v
	}

	maxPlaceholders := 0
	if v, ok, err := paramInt(params, "max_total_placeholder_tokens"); err != nil {
		return nil, err
	} else if ok {
		if v < 0 {
			return nil, fmt.Errorf("max_total_placeholder_tokens must be non-negative, got %d", v)
		}
		maxPlaceholders = v
	}

	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConnsPerHost: maxIdleConnsPerHost,
		IdleConnTimeout:     idleConnTimeout,
		ForceAttemptHTTP2:   true,
	}

	return &RenderStep{
		serviceAddress:            address,
		maxTotalTokens:            maxTokens,
		maxTotalPlaceholderTokens: maxPlaceholders,
		client:                    &http.Client{Timeout: timeout, Transport: transport},
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
		if err := s.checkTokenLimit(len(reqCtx.TokenIDs)); err != nil {
			return err
		}
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
			// Reject oversized arrays before iterating: toIntSlice does an
			// O(n) type-assert per element, so for a runaway prompt the
			// length check saves real work.
			if err := s.checkTokenLimit(len(p)); err != nil {
				return err
			}
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
			return fmt.Errorf("render: batched string prompts ([]string) are not supported: %w", pipeline.ErrBadRequest)
		case []any:
			return fmt.Errorf("render: batched token prompts ([][]int) are not supported: %w", pipeline.ErrBadRequest)
		default:
			return fmt.Errorf("render: invalid prompt array element: %T: %w", p[0], pipeline.ErrBadRequest)
		}

	default:
		return fmt.Errorf("render: prompt must be a string or token array, got %T: %w", prompt, pipeline.ErrBadRequest)
	}
}

func (s *RenderStep) executeChatCompletions(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(RenderStepName)

	var renderResp renderResponse
	if err := s.postRender(ctx, reqCtx, gateway.PathChatCompletions, &renderResp); err != nil {
		return err
	}

	reqCtx.TokenIDs = renderResp.TokenIDs
	if err := s.checkTokenLimit(len(reqCtx.TokenIDs)); err != nil {
		return err
	}

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

	if err := s.checkPlaceholderLimit(reqCtx.MultimodalEntries); err != nil {
		return err
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

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("creating render request: %w", err)
	}
	req.ContentLength = int64(len(body))
	req.Header.Set(gateway.ContentTypeHeader, gateway.ContentTypeJSON)
	for k, v := range reqCtx.ForwardedHeaders() {
		req.Header.Set(k, v)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("render request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody := readErrorBody(resp.Body)
		return upstreamError(RenderStepName, resp.StatusCode, respBody)
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

func (s *RenderStep) checkTokenLimit(tokenCount int) error {
	if s.maxTotalTokens > 0 && tokenCount > s.maxTotalTokens {
		return fmt.Errorf("too many total tokens: got %d, max %d: %w", tokenCount, s.maxTotalTokens, pipeline.ErrBadRequest)
	}
	return nil
}

func (s *RenderStep) checkPlaceholderLimit(entries []pipeline.MultimodalEntry) error {
	if s.maxTotalPlaceholderTokens <= 0 {
		return nil
	}
	total := 0
	for _, e := range entries {
		total += e.Placeholder.Length
	}
	if total > s.maxTotalPlaceholderTokens {
		return fmt.Errorf("too many placeholder tokens: got %d, max %d: %w", total, s.maxTotalPlaceholderTokens, pipeline.ErrBadRequest)
	}
	return nil
}

func toIntSlice(values []any) ([]int, error) {
	out := make([]int, 0, len(values))
	for _, v := range values {
		switch n := v.(type) {
		case float64:
			if n < 0 || n != math.Trunc(n) {
				return nil, fmt.Errorf("render: invalid token in prompt array: %v (must be a non-negative integer): %w", v, pipeline.ErrBadRequest)
			}
			out = append(out, int(n))
		case json.Number:
			i, err := n.Int64()
			if err != nil {
				return nil, fmt.Errorf("render: invalid token in prompt array: %v: %w", v, pipeline.ErrBadRequest)
			}
			if i < 0 {
				return nil, fmt.Errorf("render: invalid token in prompt array: %v (must be a non-negative integer): %w", v, pipeline.ErrBadRequest)
			}
			out = append(out, int(i))
		default:
			return nil, fmt.Errorf("render: invalid token in prompt array: %T: %w", v, pipeline.ErrBadRequest)
		}
	}
	return out, nil
}
