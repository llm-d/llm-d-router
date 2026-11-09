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
	gwClient          *gateway.Client
	kv                kv.Connector
	promptTokenParity string
	// budget is set when the pipeline enables force-streaming.
	budget *forceStreamBudget
}

func NewDecodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("decode: gateway client is required")
	}
	if err := rejectUseOpenAIFormatOverride(DecodeStepName, params); err != nil {
		return nil, err
	}
	if err := rejectForceStreamParams(DecodeStepName, params); err != nil {
		return nil, err
	}
	kvConn, err := buildKVConnector(params)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	parity, err := parsePromptTokenParity(params)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &DecodeStep{gwClient: gwClient, kv: kvConn, promptTokenParity: parity}, nil
}

func (s *DecodeStep) Name() string { return DecodeStepName }

func (s *DecodeStep) enableForceStream(budget *forceStreamBudget) { s.budget = budget }

func (s *DecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	if reqCtx.IsSGLangGRPC() {
		return s.executeSGLangGRPC(ctx, reqCtx)
	}
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	if err := s.prepareDecodeBody(ctx, reqCtx); err != nil {
		return err
	}
	parity, injectedUsage := setupParity(DecodeStepName, s.promptTokenParity, logger, reqCtx, reqCtx.Body)

	// Force-streaming applies only to a non-streaming request. A request whose
	// reply may carry fields the reassembler drops (tool calls, logprobs,
	// reasoning), or that the engine would not serve as a stream, takes the
	// pass-through; forceStreamLossy catches what the request declares up front,
	// and a server-side default is caught on the frame mid-stream (both fall back
	// here). Every other candidate reserves budget and reassembles the whole reply;
	// one whose estimate exceeds the per-request cap, or that finds the budget
	// full, falls through to the pass-through below, which buffers nothing.
	if s.budget != nil && !reqCtx.Stream {
		if shape, shapeOK := shapeForAPIType(reqcommon.DetectAPIType(reqCtx.OriginalPath)); shapeOK {
			result := coordmetrics.ForceStreamResultFallbackUnsupported
			// An SGLang decode server has no local-prefill path, so a fallback could
			// not recompute a prompt whose handoff the forced attempt used.
			if s.kv.Name() != kv.SGLang {
				leg := forceStreamLeg{step: DecodeStepName, upstream: coordmetrics.UpstreamDecode}
				var sent bool
				var err error
				result, sent, err = tryForceStream(ctx, logger, s.gwClient, s.budget, leg, reqCtx, reqCtx.Body, shape, parity)
				if result == "" {
					return err
				}
				// The forced attempt failed with nothing written. Unless the connector
				// is replay-safe, it already used the one-shot handoff the body names,
				// so the fallback drops it and the decode pod recomputes the prompt; the
				// reply is slower but correct.
				if sent && !kvHandoffReplaySafe(s.kv) {
					logger.Info("force-stream failed and its KV block handoff is spent; decode will recompute the prompt", "result", result)
					delete(reqCtx.Body, reqcommon.FieldKVTransferParams)
					// The prefill cache-hit count describes the handoff just dropped.
					reqCtx.PrefillCachedTokens = nil
				}
			}
			coordmetrics.IncForceStreamTotal(reqCtx.Model, DecodeStepName, result)
		}
	}

	logger.V(logutil.DEFAULT).Info("sending request", "path", reqCtx.OriginalPath, "stream", reqCtx.Stream)

	proxyReq, err := newDecodeProxyRequest(ctx, logger, DecodeStepName, reqCtx, s.gwClient, reqCtx.Body, nil)
	if err != nil {
		return err
	}

	writer, finalize := parityResponseWriter(reqCtx, parity, injectedUsage)
	out := serveDecode(logger, s.gwClient.Transport(), writer, proxyReq, coordmetrics.UpstreamDecode, nil)
	if err := finalize(); err != nil {
		// The buffered tail could not be forwarded, so the client holds a
		// truncated response. Nothing can be written to say so: the status and
		// the body before the tail are already on the wire.
		logger.Error(err, "flushing the rewritten usage block failed: client received a partial response")
	}
	return out.streamedError(DecodeStepName)
}

// kvHandoffReplaySafe reports whether a decode body's kv_transfer_params may be
// sent a second time. Only kv-shared-storage qualifies: its decode pod re-reads
// the shared store, so a replay finds the KV again. The other connectors name a
// one-shot handoff: the NIXL blocks the prefiller frees once pulled, the
// kv-offloading P2P batch the prefiller releases once fetched, or the SGLang
// bootstrap room. A connector not listed here is treated as one-shot too.
func kvHandoffReplaySafe(conn kv.Connector) bool {
	return conn.Name() == kv.SharedStorage
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
			reqCtx.Body[reqcommon.FieldPrompt] = reqCtx.TokenIDs
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
			media.part[reqcommon.FieldUUID] = hashes[localIdx]
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
