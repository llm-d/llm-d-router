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
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/common/httplog"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"golang.org/x/sync/errgroup"
)

const EncodeStepName = "encode"

func init() {
	pipeline.Register(EncodeStepName, NewEncodeStep)
}

type EncodeStep struct {
	useOpenAIFormat bool
	maxParallel     int
	gwClient        *gateway.Client
	ec              ec.Connector
}

func NewEncodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("encode: gateway client is required")
	}
	useOpenAI, err := parseUseOpenAIFormat(params)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	maxParallel := 8
	if v, ok, err := paramInt(params, "max_parallel"); err != nil {
		return nil, err
	} else if ok {
		if v <= 0 {
			return nil, fmt.Errorf("max_parallel must be positive, got %d", v)
		}
		maxParallel = v
	}
	ecName, err := paramString(params, ParamECConnector)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	ecConn, err := ec.Build(ecName)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	return &EncodeStep{
		useOpenAIFormat: useOpenAI,
		maxParallel:     maxParallel,
		gwClient:        gwClient,
		ec:              ecConn,
	}, nil
}

func (s *EncodeStep) Name() string { return EncodeStepName }

func (s *EncodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	if len(reqCtx.MultimodalEntries) == 0 {
		return nil
	}

	logger := log.FromContext(ctx).WithName(EncodeStepName)

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxParallel)

	results := make([]map[string]any, len(reqCtx.MultimodalEntries))

	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	var imageParts []map[string]any
	if format == gateway.FormatChatCompletions {
		imageParts = collectImageParts(reqCtx.Body)
	}

	for i, entry := range reqCtx.MultimodalEntries {
		g.Go(func() error {
			tokenIDs := s.buildEncodeTokenIDs(reqCtx.TokenIDs, entry)

			body := s.buildEncodeBody(reqCtx, tokenIDs, entry, format, imageParts)

			bodyBytes, err := json.Marshal(body)
			if err != nil {
				err = fmt.Errorf("encode[%d]: marshal: %w", i, err)
				logger.Error(err, "encode fanout marshal", "index", i)
				return err
			}

			path := gateway.PathForFormat(format)
			logger.V(logutil.DEFAULT).Info("sending sub-request", "index", i, "path", path)

			headers := reqCtx.ForwardedHeaders()
			headers[reqcommon.RequestIDHeaderKey] = reqCtx.RequestID
			headers[gateway.EPPPhaseHeader] = gateway.PhaseEncode

			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("sub-request body", "index", i, "method", "POST", "path", path, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(headers))
			}

			resp, err := s.gwClient.Post(gCtx, path, bodyBytes, headers)
			if err != nil {
				err = fmt.Errorf("encode[%d]: request: %w", i, err)
				logger.Error(err, "encode fanout request", "index", i, "path", path)
				return err
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				respBody := readErrorBody(resp.Body)
				err := upstreamError(fmt.Sprintf("%s[%d]", EncodeStepName, i), resp.StatusCode, respBody)
				logger.Error(err, "encode fanout status", "index", i, "status", resp.StatusCode)
				return err
			}

			var encResp encodeResponse
			if err := json.NewDecoder(resp.Body).Decode(&encResp); err != nil {
				err = fmt.Errorf("encode[%d]: decode response: %w", i, err)
				logger.Error(err, "encode fanout decode", "index", i)
				return err
			}

			results[i] = coerceParamsMap(logger.WithValues("index", i), encResp.ECTransferParams, "ec_transfer_params")
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	for _, r := range results {
		s.ec.MergeEncodeResponse(ctx, reqCtx, r)
	}

	logger.V(logutil.DEFAULT).Info("all sub-requests complete", "count", len(results))
	return nil
}

func (s *EncodeStep) buildEncodeTokenIDs(fullTokenIDs []int, entry pipeline.MultimodalEntry) []int {
	bos := 1
	placeholderTokenID := 0
	if len(fullTokenIDs) > 0 {
		bos = fullTokenIDs[0]
		if entry.Placeholder.Offset < len(fullTokenIDs) {
			placeholderTokenID = fullTokenIDs[entry.Placeholder.Offset]
		}
	}

	tokenIDs := make([]int, 1+entry.Placeholder.Length)
	tokenIDs[0] = bos
	for j := 1; j <= entry.Placeholder.Length; j++ {
		tokenIDs[j] = placeholderTokenID
	}
	return tokenIDs
}

func (s *EncodeStep) buildEncodeBody(reqCtx *pipeline.RequestContext, tokenIDs []int, entry pipeline.MultimodalEntry, format gateway.RequestFormat, imageParts []map[string]any) map[string]any {
	switch format {
	case gateway.FormatChatCompletions:
		imageContent := buildSingleImageContent(imageParts, entry.Index)
		body := map[string]any{
			"model": reqCtx.Model,
			"messages": []any{
				map[string]any{
					"role":    "user",
					"content": []any{imageContent},
				},
			},
			"tokens": map[string]any{
				"token_ids": tokenIDs,
				"features": map[string]any{
					"mm_hashes":       map[string][]string{ModalityImage: {entry.Hash}},
					"mm_placeholders": map[string][]any{ModalityImage: {map[string]any{"offset": 1, "length": entry.Placeholder.Length}}},
				},
			},
		}
		reqcommon.PrimeSingleTokenRequest(body, reqCtx.Body)
		return body
	default:
		return map[string]any{
			"model":     reqCtx.Model,
			"token_ids": tokenIDs,
			"features": map[string]any{
				"mm_hashes":       map[string][]string{ModalityImage: {entry.Hash}},
				"mm_placeholders": map[string][]any{ModalityImage: {map[string]any{"offset": 1, "length": entry.Placeholder.Length}}},
				"kwargs_data":     map[string][]string{ModalityImage: {entry.KwargsData}},
			},
			"sampling_params": map[string]any{"max_tokens": 1},
		}
	}
}

// collectImageParts walks the request messages once and returns the image_url
// parts in order, so the fan-out loop can index by position instead of
// re-walking all parts per image (O(N*M) -> O(N+M)).
func collectImageParts(body map[string]any) []map[string]any {
	messages, _ := body["messages"].([]any)
	var parts []map[string]any
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		content, ok := msgMap["content"].([]any)
		if !ok {
			continue
		}
		for _, part := range content {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if partMap["type"] == imageURLPartType {
				parts = append(parts, partMap)
			}
		}
	}
	return parts
}

func buildSingleImageContent(imageParts []map[string]any, index int) map[string]any {
	if index >= 0 && index < len(imageParts) {
		return map[string]any{
			"type":      imageURLPartType,
			"image_url": imageParts[index][imageURLPartType],
		}
	}
	return map[string]any{
		"type":      imageURLPartType,
		"image_url": map[string]any{"url": ""},
	}
}

type encodeResponse struct {
	// ECTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	ECTransferParams any `json:"ec_transfer_params"`
}
