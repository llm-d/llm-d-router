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

package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/llm-d-router/pkg/coordinator/engine"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// PrepareRender normalizes supplied tokens or prepares a renderer request.
// A nil request means no renderer HTTP call is needed.
//
//nolint:nilnil // Local preprocessing needs no HTTP request.
func (e Engine) PrepareRender(ctx context.Context, reqCtx *pipeline.RequestContext) (*engine.Request, error) {
	if reqCtx.OriginalPath == gateway.DefaultGeneratePath {
		return nil, e.normalizeGenerate(ctx, reqCtx)
	}
	if strings.Contains(reqCtx.OriginalPath, gateway.PathCompletions) {
		return e.prepareCompletions(ctx, reqCtx)
	}
	if strings.Contains(reqCtx.OriginalPath, gateway.PathChatCompletions) {
		return &engine.Request{Path: gateway.PathChatCompletions + "/render", Body: reqCtx.Body}, nil
	}
	log.FromContext(ctx).WithName("render").V(logutil.DEFAULT).Info("skipping render step", "path", reqCtx.OriginalPath)
	return nil, nil
}

//nolint:nilnil // Token prompts need no renderer HTTP call.
func (e Engine) prepareCompletions(ctx context.Context, reqCtx *pipeline.RequestContext) (*engine.Request, error) {
	prompt := reqCtx.Body["prompt"]
	switch p := prompt.(type) {
	case string:
		return &engine.Request{Path: gateway.PathCompletions + "/render", Body: reqCtx.Body}, nil
	case []any:
		if len(p) == 0 {
			reqCtx.TokenIDs = []int{}
			log.FromContext(ctx).WithName("render").V(logutil.DEFAULT).Info("prompt is empty array, skipping render", "token_ids_len", 0)
			return nil, nil
		}
		switch p[0].(type) {
		case float64, json.Number:
			// Bound the array before converting every token.
			if err := e.limits.CheckTokenLimit(len(p)); err != nil {
				return nil, err
			}
			tokenIDs, err := toIntSlice(p)
			if err != nil {
				return nil, fmt.Errorf("render: %w", err)
			}
			reqCtx.TokenIDs = tokenIDs
			log.FromContext(ctx).WithName("render").V(logutil.DEFAULT).Info("prompt is token array, skipping render", "token_ids_len", len(tokenIDs))
			return nil, nil
		case string:
			return nil, fmt.Errorf("render: batched string prompts ([]string) are not supported: %w", pipeline.ErrBadRequest)
		case []any:
			return nil, fmt.Errorf("render: batched token prompts ([][]int) are not supported: %w", pipeline.ErrBadRequest)
		default:
			return nil, fmt.Errorf("render: invalid prompt array element: %T: %w", p[0], pipeline.ErrBadRequest)
		}
	default:
		return nil, fmt.Errorf("render: prompt must be a string or token array, got %T: %w", prompt, pipeline.ErrBadRequest)
	}
}

// ApplyRenderResponse records rendered tokens and multimodal metadata in reqCtx
// and checks the configured input limits.
func (e Engine) ApplyRenderResponse(ctx context.Context, reqCtx *pipeline.RequestContext, body io.Reader) error {
	if strings.Contains(reqCtx.OriginalPath, gateway.PathCompletions) {
		var response []completionsRenderResponse
		if err := json.NewDecoder(body).Decode(&response); err != nil {
			return fmt.Errorf("decoding render response: %w", err)
		}
		if len(response) != 1 {
			return fmt.Errorf("render: expected 1 response element, got %d", len(response))
		}
		reqCtx.TokenIDs = response[0].TokenIDs
		if err := e.limits.CheckTokenLimit(len(reqCtx.TokenIDs)); err != nil {
			return err
		}
		reqCtx.Body["prompt"] = reqCtx.TokenIDs
		log.FromContext(ctx).WithName("render").V(logutil.DEFAULT).Info("complete", "token_ids_len", len(reqCtx.TokenIDs))
		return nil
	}
	var renderResp renderResponse
	if err := json.NewDecoder(body).Decode(&renderResp); err != nil {
		return fmt.Errorf("decoding render response: %w", err)
	}
	reqCtx.TokenIDs = renderResp.TokenIDs
	if err := e.limits.CheckTokenLimit(len(reqCtx.TokenIDs)); err != nil {
		return err
	}

	imageHashes := renderResp.Features.MMHashes[modalityImage]
	imagePlaceholders := renderResp.Features.MMPlaceholders[modalityImage]
	imageKwargs := renderResp.Features.KwargsData[modalityImage]

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

	if err := e.limits.CheckPlaceholderLimit(reqCtx.MultimodalEntries); err != nil {
		return err
	}
	logger := log.FromContext(ctx).WithName("render")
	logger.V(logutil.DEBUG).Info("response", "mm_hashes", imageHashes, "mm_placeholders", imagePlaceholders, "kwargs_data_len", len(imageKwargs))
	logger.V(logutil.DEFAULT).Info("complete", "token_ids_len", len(renderResp.TokenIDs), "images", len(imageHashes))
	return nil
}

// normalizeGenerate handles the tokens-in generate path. It does not tokenize:
// the client supplies token_ids and features directly. It normalizes them into
// reqCtx.TokenIDs and reqCtx.MultimodalEntries, the typed fields every
// downstream step (encode/prefill/decode) reads instead of the raw body, and
// enforces the generate-only bounds vLLM does not (validatePlaceholderBounds,
// token/placeholder limits) before EncodeStep indexes token_ids[offset] and
// allocates from length.
func (e Engine) normalizeGenerate(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	tokenIDs, err := extractTokenIDs(reqCtx.Body["token_ids"])
	if err != nil {
		return fmt.Errorf("render: %w", err)
	}
	if err := e.limits.CheckTokenLimit(len(tokenIDs)); err != nil {
		return err
	}
	reqCtx.TokenIDs = tokenIDs

	if err := validateSamplingParams(reqCtx.Body); err != nil {
		return fmt.Errorf("render: %w", err)
	}

	rawFeatures := reqCtx.Body["features"]
	var features map[string]any
	if rawFeatures != nil {
		var ok bool
		features, ok = rawFeatures.(map[string]any)
		if !ok {
			return fmt.Errorf("render: features must be an object, got %T: %w", rawFeatures, pipeline.ErrBadRequest)
		}
	}
	entries, err := extractMultimodalEntries(features)
	if err != nil {
		return fmt.Errorf("render: %w", err)
	}
	// The client supplies placeholder geometry directly on this path, so bound
	// every span to the prompt before EncodeStep allocates from length. Unlike
	// CheckPlaceholderLimit, this guard is unconditional: it does not depend on
	// the optional max_total_placeholder_tokens knob.
	if err := validatePlaceholderBounds(entries, len(tokenIDs)); err != nil {
		return fmt.Errorf("render: %w", err)
	}
	if err := e.limits.CheckPlaceholderLimit(entries); err != nil {
		return err
	}
	reqCtx.MultimodalEntries = entries

	log.FromContext(ctx).WithName("render").V(logutil.DEFAULT).Info("complete", "token_ids_len", len(tokenIDs), "images", len(entries))
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
