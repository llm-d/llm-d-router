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

	// On the generate path the prefill worker runs the vision encoder inline from
	// kwargs_data, so the encode fan-out and EC handoff are redundant. Skipping it
	// avoids shipping the oversized preprocessed pixel tensor a second time
	// (see https://github.com/vllm-project/vllm/issues/46722).
	if reqCtx.OriginalPath == gateway.DefaultGeneratePath {
		logger.V(logutil.DEFAULT).Info("skipping encode for generate request")
		return nil
	}

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxParallel)

	results := make([]map[string]any, len(reqCtx.MultimodalEntries))

	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	var partsByMod map[string][]map[string]any
	if format == gateway.FormatChatCompletions {
		partsByMod = collectMediaParts(reqCtx.Body)
	}

	// Per-modality running counter: entry i's local index is the number of
	// earlier entries sharing its modality. One pass, O(n) total.
	modCounter := make(map[string]int)
	for i, entry := range reqCtx.MultimodalEntries {
		mod := entryModality(entry)
		localIdx := modCounter[mod]
		modCounter[mod]++
		g.Go(func() error {
			tokenIDs := s.buildEncodeTokenIDs(reqCtx.TokenIDs, entry)

			body := s.buildEncodeBody(reqCtx, tokenIDs, entry, localIdx, format, partsByMod)

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
			headers[gateway.EPPProfileHeader] = gateway.PhaseEncode

			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("sub-request body", "index", i, "method", "POST", "path", path, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(headers))
			}

			call := coordmetrics.StartUpstreamCall(coordmetrics.UpstreamEncode)
			resp, err := s.gwClient.Post(gCtx, path, bodyBytes, headers)
			call.Done()
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

func (s *EncodeStep) buildEncodeBody(reqCtx *pipeline.RequestContext, tokenIDs []int, entry pipeline.MultimodalEntry, localIdx int, format gateway.RequestFormat, partsByMod map[string][]map[string]any) map[string]any {
	mod := entryModality(entry)
	placeholder := map[string]any{"offset": 1, "length": entry.Placeholder.Length}
	switch format {
	case gateway.FormatChatCompletions:
		mediaContent := buildSingleMediaContent(partsByMod, mod, localIdx)
		body := map[string]any{
			"model": reqCtx.Model,
			"messages": []any{
				map[string]any{
					"role":    "user",
					"content": []any{mediaContent},
				},
			},
			"tokens": map[string]any{
				"token_ids": tokenIDs,
				"features": map[string]any{
					"mm_hashes":       map[string][]string{mod: {entry.Hash}},
					"mm_placeholders": map[string][]any{mod: {placeholder}},
				},
			},
		}
		capSingleTokenOutput(body, format)
		return body
	default:
		body := map[string]any{
			"model":     reqCtx.Model,
			"token_ids": tokenIDs,
			"features": map[string]any{
				"mm_hashes":       map[string][]string{mod: {entry.Hash}},
				"mm_placeholders": map[string][]any{mod: {placeholder}},
				"kwargs_data":     singleEntryKwargs(mod, entry.KwargsData),
			},
		}
		capSingleTokenOutput(body, format)
		return body
	}
}

// collectMediaParts walks the request messages once and returns the media
// parts grouped by modality, each per-modality list in walker (request)
// order. The encode fanout looks up the content part for an entry by
// (entry.Modality, per-modality position), reading from the per-modality
// list at the per-modality position. Non-media parts (text, tool_use, etc.)
// are skipped. Parts that fail mediaPartIsWellFormed are also skipped so
// this walker's per-modality indexing stays in lock-step with the
// MultimodalEntries replace_media_urls produced: any part it silently
// dropped must not shift the pairing here. Uses partTypeModality from
// replace_media_urls.go as the authoritative list of recognized media part
// types.
func collectMediaParts(body map[string]any) map[string][]map[string]any {
	messages, _ := body["messages"].([]any)
	partsByMod := make(map[string][]map[string]any)
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
			partType, _ := partMap["type"].(string)
			modality, isMedia := partTypeModality[partType]
			if !isMedia {
				continue
			}
			if !mediaPartIsWellFormed(partMap, partType) {
				continue
			}
			partsByMod[modality] = append(partsByMod[modality], partMap)
		}
	}
	return partsByMod
}

// modalityFallbackPartType names the URL-shaped content-part type used
// for the buildSingleMediaContent fallback when localIdx is out of range.
// Keying by modality keeps the emitted sub-request self-consistent, an
// audio fanout entry does not get shipped inside an image_url part.
var modalityFallbackPartType = map[string]string{
	ModalityImage: imageURLPartType,
	ModalityAudio: audioURLPartType,
	ModalityVideo: videoURLPartType,
}

// buildSingleMediaContent returns the OpenAI content-part representing the
// entry at (modality, localIdx) in the per-modality parts map. It emits
// the part verbatim in its native shape, {type: <partType>, <partType>:
// <innerMap>}, so a caller can drop the returned map directly into an
// encode sub-request's messages[0].content slice.
//
// If localIdx is out of range for the given modality, an empty-shaped
// URL part of the matching modality is returned as a safe fallback.
// Reaching that path means the coordinator's entry<->part pairing is
// broken; the fallback prevents a panic but does not hide the bug, the
// encoder receives an empty URL of the right modality and rejects the
// sub-request loudly.
func buildSingleMediaContent(partsByMod map[string][]map[string]any, modality string, localIdx int) map[string]any {
	parts := partsByMod[modality]
	if localIdx < 0 || localIdx >= len(parts) {
		partType, ok := modalityFallbackPartType[modality]
		if !ok {
			partType = imageURLPartType
		}
		return map[string]any{
			"type":   partType,
			partType: map[string]any{"url": ""},
		}
	}
	p := parts[localIdx]
	partType, _ := p["type"].(string)
	return map[string]any{
		"type":   partType,
		partType: p[partType],
	}
}

type encodeResponse struct {
	// ECTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	ECTransferParams any `json:"ec_transfer_params"`
}
