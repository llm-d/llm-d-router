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
	"errors"
	"fmt"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const DecodeStepName = "decode"

func init() {
	pipeline.Register(DecodeStepName, NewDecodeStep)
}

type DecodeStep struct {
	gwClient    *gateway.Client
	kv          kv.Connector
	forceStream bool
	budget      *forceStreamBudget
}

func NewDecodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("decode: gateway client is required")
	}
	if err := rejectUseOpenAIFormatOverride(DecodeStepName, params); err != nil {
		return nil, err
	}
	kvConn, err := buildKVConnector(params)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	forceStream, _, err := paramBool(params, ParamForceStream)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	step := &DecodeStep{gwClient: gwClient, kv: kvConn, forceStream: forceStream}
	if forceStream {
		budget, err := parseForceStreamBudget(params)
		if err != nil {
			return nil, fmt.Errorf("decode: %w", err)
		}
		step.budget = budget
	}
	return step, nil
}

func (s *DecodeStep) Name() string { return DecodeStepName }

func (s *DecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	if err := s.prepareDecodeBody(ctx, reqCtx); err != nil {
		return err
	}

	// Force-streaming applies only to a non-streaming request. A chat or text
	// request whose reply may carry fields the reassembler drops (tool calls,
	// logprobs) takes the pass-through to keep them. A chat or text single-choice
	// reply is written incrementally, with no reservation. The buffered shapes
	// (responses, generate, n>1) reserve budget and reassemble the whole reply;
	// one with no output token limit to reserve, or one that finds the budget
	// full, falls through to the pass-through below, which buffers nothing.
	if s.forceStream && !reqCtx.Stream {
		if shape, shapeOK := shapeForAPIType(reqcommon.DetectAPIType(reqCtx.OriginalPath)); shapeOK {
			switch {
			case forceStreamLossy(shape, reqCtx.Body):
				coordmetrics.IncForceStreamTotal(reqCtx.Model, coordmetrics.ForceStreamResultFallbackUnsupported)
			case canStreamIncrementally(shape, reqCtx):
				if err := s.executeForceStreamIncremental(ctx, logger, reqCtx, shape); !recordForceStreamFallback(reqCtx.Model, err) {
					return err
				}
				// Fell back with nothing written; continue to the pass-through below.
			default:
				reserved, ok := s.estimateReservation(reqCtx)
				switch {
				case !ok:
					coordmetrics.IncForceStreamTotal(reqCtx.Model, coordmetrics.ForceStreamResultFallbackUnbounded)
				case s.budget.tryReserve(reserved):
					if err := s.executeForceStream(ctx, logger, reqCtx, shape, reserved); !recordForceStreamFallback(reqCtx.Model, err) {
						return err
					}
					// Fell back with nothing written; continue to the pass-through below.
				default:
					coordmetrics.IncForceStreamTotal(reqCtx.Model, coordmetrics.ForceStreamResultFallbackBudget)
				}
			}
		}
	}

	logger.V(logutil.DEFAULT).Info("sending request", "path", reqCtx.OriginalPath, "stream", reqCtx.Stream)

	proxyReq, err := newDecodeProxyRequest(ctx, logger, DecodeStepName, reqCtx, s.gwClient, reqCtx.Body, nil)
	if err != nil {
		return err
	}

	out := serveDecode(logger, s.gwClient.Transport(), reqCtx.ResponseWriter, proxyReq, coordmetrics.UpstreamDecode, nil)
	return out.streamedError(DecodeStepName)
}

// recordForceStreamFallback reports whether a force-stream attempt fell back to
// the pass-through, recording the matching outcome when it did. A fallback leaves
// the client uncommitted, so the caller continues to the pass-through; nil (a
// forced success) and every other error are not fallbacks, so the caller returns
// them directly.
func recordForceStreamFallback(modelName string, err error) bool {
	switch {
	case errors.Is(err, errForceStreamOversize):
		coordmetrics.IncForceStreamTotal(modelName, coordmetrics.ForceStreamResultFallbackOversize)
	case errors.Is(err, errForceStreamUnreadable):
		coordmetrics.IncForceStreamTotal(modelName, coordmetrics.ForceStreamResultFallbackUnreadable)
	default:
		return false
	}
	return true
}

// prepareDecodeBody mutates reqCtx.Body in place rather than on a clone (unlike
// prefill and conditional-decode). decode is the terminal pipeline step: its body
// is streamed straight to the client and no later step reads reqCtx.Body. A clone
// would also be insufficient, since injectUUIDs mutates nested values that a shallow
// maps.Clone would still share. This is sound only while the pipeline runs steps
// sequentially; if it ever goes concurrent, decode must copy like the others.
func (s *DecodeStep) prepareDecodeBody(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)
	format := reqcommon.DetectAPIType(reqCtx.OriginalPath)

	kvParams := s.kv.PrepareDecodeKVParams(ctx, reqCtx)
	s.injectUUIDs(reqCtx, logger)

	switch format {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses, reqcommon.APITypeMessages, reqcommon.APITypeVLLMGenerate:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
	case reqcommon.APITypeCompletions:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
		if len(reqCtx.TokenIDs) > 0 {
			reqCtx.Body["prompt"] = reqCtx.TokenIDs
		}
	default:
		// kvParams and injectUUIDs above already ran; both are harmless here
		// since the request fails on this return and reqCtx.Body is never sent.
		return unreachableFormatError(format)
	}
	return nil
}

// injectUUIDs tags each media content part with the uuid the decode backend
// uses for prefix-cache keying.
//
// It keys on DetectAPIType(reqCtx.OriginalPath): decode proxies reqCtx.Body to
// reqCtx.OriginalPath, so the wire shape to walk is whatever the client sent.
// resolveFormat's answer instead reflects the encode/prefill wire-format
// setting, which can differ from the client's own shape.
//
// A Messages image block has no uuid field: vLLM's Anthropic conversion drops
// one, so the worker hashes the image itself and decode stamps nothing.
func (s *DecodeStep) injectUUIDs(reqCtx *pipeline.RequestContext, logger logr.Logger) {
	// Without entries there is nothing to stamp, and the prompt items stay raw.
	if len(reqCtx.MultimodalEntries) == 0 {
		return
	}
	apiType := reqcommon.DetectAPIType(reqCtx.OriginalPath)
	if apiType == reqcommon.APITypeMessages {
		return
	}
	if items, ok := promptItems(reqCtx.Body, apiType); ok {
		injectMediaPartUUIDs(items, apiType, reqCtx.MultimodalEntries, logger)
		setPromptItems(reqCtx.Body, apiType, items)
	}
}

// injectMediaPartUUIDs stamps each media content part with the hash of its
// corresponding multimodal entry, pairing the two by position within a modality.
// A surplus part is left unstamped, so the worker hashes the media itself rather
// than reading an entry primed under another part's hash; a surplus entry has no
// part to stamp. Neither is fatal; the branches below record what each costs.
func injectMediaPartUUIDs(items []any, apiType reqcommon.APIType, entries []pipeline.MultimodalEntry, logger logr.Logger) {
	// Grouped in entry order, so the walk below indexes by the per-modality
	// position: O(1) per part after an O(n) build.
	hashesByMod := make(map[reqcommon.Modality][]string)
	for _, entry := range entries {
		hashesByMod[entry.Modality] = append(hashesByMod[entry.Modality], entry.Hash)
	}

	modCounter := make(map[reqcommon.Modality]int)
	for _, media := range collectMediaParts(items, apiType) {
		localIdx := modCounter[media.modality]
		modCounter[media.modality]++
		hashes := hashesByMod[media.modality]
		if localIdx < len(hashes) {
			media.part["uuid"] = hashes[localIdx]
			continue
		}
		// A miss means entries and parts got out of line upstream (see
		// collectMediaParts). The request is still answered, but the worker
		// hashes and re-processes that media itself -- at the caps
		// coordinator.yaml suggests, up to 200 MB of video, redoing the encode
		// work the EPD split exists to do once elsewhere.
		logger.V(logutil.DEBUG).Info("no MultimodalEntry for media part",
			"location", media.location,
			"modality", media.modality,
			"local_index", localIdx,
			"modality_entry_count", len(hashes))
	}

	// The walk above iterates parts, so it can only see a surplus part; the
	// other direction needs its own pass, and is the worse of the two. A
	// surplus entry's hash still reaches the prefiller, since
	// PreparePrefillECParams flattens every encode response into
	// ec_transfer_params, so the prefill body describes an EC buffer no part of
	// the decode body names by uuid.
	for mod, hashes := range hashesByMod {
		if partCount := modCounter[mod]; partCount < len(hashes) {
			logger.V(logutil.DEBUG).Info("MultimodalEntry with no media part",
				"modality", mod,
				"part_count", partCount,
				"modality_entry_count", len(hashes))
		}
	}
}
