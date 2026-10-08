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
	if err := validateEntryModalities(reqCtx.MultimodalEntries); err != nil {
		return fmt.Errorf("encode: %w", err)
	}

	logger := log.FromContext(ctx).WithName(EncodeStepName)

	// On the generate path the prefill worker runs the encoder inline from
	// kwargs_data (image tensors, audio spectrograms, and video frames all take
	// this path), so the encode fanout and EC handoff would be redundant and
	// would ship the preprocessed tensor a second time
	// (see https://github.com/vllm-project/vllm/issues/46722).
	if reqcommon.DetectAPIType(reqCtx.OriginalPath) == reqcommon.APITypeVLLMGenerate {
		logger.V(logutil.DEFAULT).Info("skipping encode for generate request")
		return nil
	}

	results := make([]map[string]any, len(reqCtx.MultimodalEntries))
	responseHeaders := make([]http.Header, len(reqCtx.MultimodalEntries))

	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	var partsByMod map[reqcommon.Modality][]mediaPart
	if items, ok := promptItems(reqCtx.Body, format); ok {
		partsByMod = groupMediaPartsByModality(collectMediaParts(items, format))
	}

	// Entry i's local index is its position among the entries sharing its
	// modality. Resolved up front so each sub-request carries its own
	// coordinate; see collectMediaParts for how entries and parts stay lined up.
	localIdx := modalityLocalIndexes(reqCtx.MultimodalEntries)

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxParallel)
	for i := range reqCtx.MultimodalEntries {
		g.Go(func() error {
			result, headers, err := s.executeOne(gCtx, logger, reqCtx, i, reqCtx.MultimodalEntries[i], localIdx[i], format, partsByMod)
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
	localIdx int,
	format reqcommon.APIType,
	partsByMod map[reqcommon.Modality][]mediaPart,
) (map[string]any, http.Header, error) {
	logger = logger.WithValues("index", index)

	body, err := s.buildEncodeBody(reqCtx, entry, localIdx, format, partsByMod)
	if err != nil {
		// Every failure here is a coordinator bug rather than a bad request:
		// either entries and parts got out of line upstream, though
		// collectMediaParts builds both from one walk by one rule, or a format
		// reached this fan-out that should never carry media. Fail rather than
		// send the encoder a request known to be wrong. Which one it was is in
		// the wrapped err, so the message names the stage and asserts nothing;
		// the part count the pairing failures care about is in there too.
		err = fmt.Errorf("encode[%d]: %w", index, err)
		logger.Error(err, "encode fanout build body",
			"modality", entry.Modality,
			"local_index", localIdx)
		return nil, nil, err
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		err = fmt.Errorf("encode[%d]: marshal: %w", index, err)
		logger.Error(err, "encode fanout marshal")
		return nil, nil, err
	}

	path := format.Path()
	logger.V(logutil.DEFAULT).Info("sending sub-request", "path", path)
	resp, err := postToGateway(ctx, logger, s.gwClient, gatewayRequest{
		logMsg:   "sub-request body",
		step:     fmt.Sprintf("%s[%d]", EncodeStepName, index),
		upstream: coordmetrics.UpstreamEncode,
		path:     path,
		body:     bodyBytes,
		headers:  gatewayHeaders(reqCtx, gateway.PhaseEncode),
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

// buildEncodeBody builds one fanout sub-request. localIdx is the entry's
// position among the entries sharing its modality, resolved by Execute; the
// modality itself comes off the entry.
func (s *EncodeStep) buildEncodeBody(reqCtx *pipeline.RequestContext, entry pipeline.MultimodalEntry, localIdx int, format reqcommon.APIType, partsByMod map[reqcommon.Modality][]mediaPart) (map[string]any, error) {
	mod := entry.Modality
	switch format {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses:
		// Neither failure below is ErrBadRequest. replace-media-urls builds the
		// entries from this same walk and rejects a part with no payload as it
		// goes (see collectMediaRefs), so a client request cannot reach either
		// guard: both mean entries and parts got out of line inside the
		// coordinator, which should surface as a 5xx rather than blame the
		// caller. validateEntryModalities states the same rule for its field.
		parts := partsByMod[mod]
		if localIdx < 0 || localIdx >= len(parts) {
			return nil, fmt.Errorf("no %s media part at index %d, request has %d", mod, localIdx, len(parts))
		}
		part := parts[localIdx].part
		if !mediaPartCarriesPayload(part) {
			return nil, fmt.Errorf("%s media part at index %d carries no media", mod, localIdx)
		}
		// The part goes out unreshaped, so the options each API keeps beside
		// the URL (Responses' detail sibling, chat's nested image_url fields,
		// an input_audio format) come along without per-format copying.
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
				"mm_hashes":       map[reqcommon.Modality][]string{mod: {entry.Hash}},
				"mm_placeholders": map[reqcommon.Modality][]any{mod: {map[string]any{"offset": 1, "length": entry.Placeholder.Length}}},
				"kwargs_data":     singleEntryKwargs(mod, entry.KwargsData),
			},
		}
		reqcommon.CapSingleToken(body, format)
		return body, nil
	default:
		// resolveFormat can also return APITypeCompletions, but a completions
		// request never carries media: render's executeCompletions never
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
