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

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
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
	ecConn, err := buildECConnector(params)
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

	// On the generate path the prefill worker runs the vision encoder inline from
	// kwargs_data, so the encode fan-out and EC handoff are redundant. Skipping it
	// avoids shipping the oversized preprocessed pixel tensor a second time
	// (see https://github.com/vllm-project/vllm/issues/46722).
	if reqcommon.DetectAPIType(reqCtx.OriginalPath) == reqcommon.APITypeVLLMGenerate {
		logger.V(logutil.DEFAULT).Info("skipping encode for generate request")
		return nil
	}

	results := make([]map[string]any, len(reqCtx.MultimodalEntries))
	responseHeaders := make([]http.Header, len(reqCtx.MultimodalEntries))

	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	var imageParts []imagePart
	if items, ok := promptItems(reqCtx.Body, format); ok {
		imageParts = collectImageParts(items, format)
	}

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxParallel)
	for i := range reqCtx.MultimodalEntries {
		g.Go(func() error {
			result, headers, err := s.executeOne(gCtx, logger, reqCtx, i, reqCtx.MultimodalEntries[i], format, imageParts)
			results[i] = result
			responseHeaders[i] = headers
			return err
		})
	}

	if err := g.Wait(); err != nil {
		// Headers from successful siblings are discarded so a failed encode
		// step cannot publish a partial aggregate.
		return err
	}

	for _, r := range results {
		s.ec.MergeEncodeResponse(ctx, reqCtx, r)
	}
	reqCtx.CaptureResponseHeaders(responseHeaders...)

	logger.V(logutil.DEFAULT).Info("all sub-requests complete", "count", len(results))
	return nil
}

func (s *EncodeStep) executeOne(
	ctx context.Context,
	logger logr.Logger,
	reqCtx *pipeline.RequestContext,
	index int,
	entry pipeline.MultimodalEntry,
	format reqcommon.APIType,
	imageParts []imagePart,
) (map[string]any, http.Header, error) {
	logger = logger.WithValues("index", index)

	body, err := s.buildEncodeBody(reqCtx, entry, format, imageParts)
	if err != nil {
		err = fmt.Errorf("encode[%d]: %w", index, err)
		logger.Error(err, "encode fanout build body")
		return nil, nil, err
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		err = fmt.Errorf("encode[%d]: marshal: %w", index, err)
		logger.Error(err, "encode fanout marshal")
		return nil, nil, err
	}

	path := format.Path()
	headers := gatewayHeaders(reqCtx, gateway.PhaseEncode)
	headers[reqcommon.RequestIDHeaderKey] = fmt.Sprintf("%s-enc-%d", reqCtx.RequestID, index)
	logger.V(logutil.DEFAULT).Info("sending sub-request", "path", path)
	resp, err := postToGateway(ctx, logger, s.gwClient, gatewayRequest{
		logMsg:   "sub-request body",
		step:     fmt.Sprintf("%s[%d]", EncodeStepName, index),
		upstream: coordmetrics.UpstreamEncode,
		path:     path,
		body:     bodyBytes,
		headers:  headers,
	})
	if err != nil {
		var upstream *pipeline.UpstreamError
		if errors.As(err, &upstream) {
			logger.Error(err, "encode fanout status", "status", upstream.StatusCode)
		} else {
			logger.Error(err, "encode fanout request", "path", path)
		}
		return nil, nil, err
	}
	defer resp.Body.Close()

	var encResp encodeResponse
	if err := json.NewDecoder(resp.Body).Decode(&encResp); err != nil {
		err = fmt.Errorf("encode[%d]: decode response: %w", index, err)
		logger.Error(err, "encode fanout decode")
		return nil, nil, err
	}
	return coerceParamsMap(logger, encResp.ECTransferParams, "ec_transfer_params"), resp.Header, nil
}

func (s *EncodeStep) buildEncodeTokenIDs(fullTokenIDs []int, entry pipeline.MultimodalEntry) []int {
	bos := 1
	placeholderTokenID := 0
	if len(fullTokenIDs) > 0 {
		bos = fullTokenIDs[0]
		// Only the upper bound is checked here; offset >= 0 is guaranteed for all
		// paths, either by extractMultimodalEntries (generate) or by the trusted
		// render-service response (chat/completions). A negative offset would
		// index out of range.
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

func (s *EncodeStep) buildEncodeBody(reqCtx *pipeline.RequestContext, entry pipeline.MultimodalEntry, format reqcommon.APIType, imageParts []imagePart) (map[string]any, error) {
	switch format {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses:
		if entry.Index < 0 || entry.Index >= len(imageParts) {
			return nil, fmt.Errorf("no image part at index %d of %d: %w", entry.Index, len(imageParts), pipeline.ErrBadRequest)
		}
		part := imageParts[entry.Index].part
		// Without a URL the sub-request primes the encoder against a part it
		// cannot fetch, under a hash the prefiller later looks up.
		// replace-media-urls rejects this shape as it builds the entry this
		// index came from, so the guard is defensive.
		if reqcommon.MediaPartURL(part) == "" {
			return nil, fmt.Errorf("image part %d carries no fetchable URL: %w", entry.Index, pipeline.ErrBadRequest)
		}
		// The part goes out unreshaped, so the options each API keeps beside
		// the URL (Responses' detail sibling, chat's nested image_url fields)
		// come along without per-format copying.
		return reqcommon.NewEncoderPrimingBody(reqCtx.Body, part, format), nil
	case reqcommon.APITypeVLLMGenerate:
		// Unlike the OpenAI formats, this body carries no image: the encoder
		// preprocesses nothing, so the client's mm_processor_kwargs and
		// media_io_kwargs have no effect here. Render already applied them and
		// returned the result as entry.Hash and entry.KwargsData.
		body := map[string]any{
			"model":     reqCtx.Model,
			"token_ids": s.buildEncodeTokenIDs(reqCtx.TokenIDs, entry),
			"features": map[string]any{
				"mm_hashes":       map[string][]string{ModalityImage: {entry.Hash}},
				"mm_placeholders": map[string][]any{ModalityImage: {map[string]any{"offset": 1, "length": entry.Placeholder.Length}}},
				"kwargs_data":     mmKwargsField([]string{entry.KwargsData}),
			},
		}
		reqcommon.CapSingleToken(body, format)
		return body, nil
	default:
		// resolveFormat can also return APITypeCompletions, but a completions
		// request never carries images: render's executeCompletions never
		// populates MultimodalEntries, so this fan-out never runs for one. That
		// leaves APITypeCompletions and any future format value as cases that
		// should not reach here; treat them as a programming error instead of
		// silently sending a generate-shaped body to the wrong endpoint.
		return nil, fmt.Errorf("unsupported request format %v", format)
	}
}

type encodeResponse struct {
	// ECTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	ECTransferParams any `json:"ec_transfer_params"`
}
