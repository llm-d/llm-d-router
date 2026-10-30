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
	"github.com/llm-d/llm-d-router/pkg/common/usage"

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const DecodeStepName = "decode"

// ParamPromptTokenParity selects what the decode step does when the worker's
// usage reports a prompt token count different from the rendered one:
// "off" skips the check, "warn" (default) counts and logs it, "error" also
// records a step error. The response to the client is never changed.
const ParamPromptTokenParity = "prompt_token_parity"

const (
	promptTokenParityOff   = "off"
	promptTokenParityWarn  = "warn"
	promptTokenParityError = "error"
)

func init() {
	pipeline.Register(DecodeStepName, NewDecodeStep)
}

type DecodeStep struct {
	gwClient          *gateway.Client
	kv                kv.Connector
	promptTokenParity string
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
	parity, err := parsePromptTokenParity(params)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &DecodeStep{gwClient: gwClient, kv: kvConn, promptTokenParity: parity}, nil
}

func parsePromptTokenParity(params map[string]any) (string, error) {
	mode, err := paramString(params, ParamPromptTokenParity)
	if err != nil {
		return "", err
	}
	switch mode {
	case "":
		return promptTokenParityWarn, nil
	case promptTokenParityOff, promptTokenParityWarn, promptTokenParityError:
		return mode, nil
	default:
		return "", fmt.Errorf("%s must be %q, %q or %q, got %q",
			ParamPromptTokenParity, promptTokenParityOff, promptTokenParityWarn, promptTokenParityError, mode)
	}
}

func (s *DecodeStep) Name() string { return DecodeStepName }

func (s *DecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	if err := s.prepareDecodeBody(ctx, reqCtx); err != nil {
		return err
	}
	parity := s.newParityCheck(logger, reqCtx)
	injectedUsage := parity != nil && injectIncludeUsage(reqCtx)

	logger.V(logutil.DEFAULT).Info("sending request", "path", reqCtx.OriginalPath, "stream", reqCtx.Stream)

	proxyReq, err := newDecodeProxyRequest(ctx, logger, DecodeStepName, reqCtx, s.gwClient, reqCtx.Body, nil)
	if err != nil {
		return err
	}

	writer, finalize := s.responseWriter(reqCtx, parity, injectedUsage)
	out := serveDecode(logger, s.gwClient.Transport(), writer, proxyReq, coordmetrics.UpstreamDecode, nil)
	if err := finalize(); err != nil {
		// The buffered tail could not be forwarded, so the client holds a
		// truncated response. Nothing can be written to say so: the status and
		// the body before the tail are already on the wire.
		logger.Error(err, "flushing the rewritten usage block failed: client received a partial response")
	}
	return out.streamedError(DecodeStepName)
}

// responseWriter returns the writer the decode response is streamed through,
// and the function that flushes whatever it buffered. Without a prefill count
// and without a parity check the client gets the decode worker's bytes
// untouched.
func (s *DecodeStep) responseWriter(reqCtx *pipeline.RequestContext, parity *parityCheck, injectedUsage bool) (http.ResponseWriter, func() error) {
	if reqCtx.PrefillCachedTokens == nil && parity == nil {
		return reqCtx.ResponseWriter, func() error { return nil }
	}
	opts := usage.Options{
		CachedTokens:       reqCtx.PrefillCachedTokens,
		DropUsageOnlyFrame: injectedUsage,
	}
	if parity != nil {
		opts.OnUsage = parity.onUsage
	}
	return usage.NewResponseWriter(reqCtx.ResponseWriter, opts, reqCtx.Stream)
}

// parityCheck compares the prompt token count the decode worker reports with
// the number of token IDs the render step produced, which is what prefill ran
// on. A difference means decode rendered a different prompt than prefill did:
// the worker ignored forwarded IDs (vLLM does for messages with non-text
// content or with echo set) or the two renderers disagree (tokenizer, chat
// template, processor version, chat_template_kwargs). Either way prefill's KV
// cache does not match and decode silently recomputes the prompt, so the only
// visible symptom without this check is a slower request.
type parityCheck struct {
	logger       logr.Logger
	mode         string
	requestID    string
	model        string
	apiType      reqcommon.APIType
	idsForwarded bool
	expected     int
	reported     bool
}

// newParityCheck returns nil when the check is off or no tokens were rendered.
func (s *DecodeStep) newParityCheck(logger logr.Logger, reqCtx *pipeline.RequestContext) *parityCheck {
	if s.promptTokenParity == promptTokenParityOff || len(reqCtx.TokenIDs) == 0 {
		return nil
	}
	apiType := reqcommon.DetectAPIType(reqCtx.OriginalPath)
	return &parityCheck{
		logger:    logger,
		mode:      s.promptTokenParity,
		requestID: reqCtx.RequestID,
		model:     reqCtx.Model,
		apiType:   apiType,
		// Completions sends the rendered IDs as the prompt (prepareDecodeBody);
		// the generate route carries them by definition. The other routes
		// forward the client's text today.
		idsForwarded: apiType == reqcommon.APITypeCompletions || apiType == reqcommon.APITypeVLLMGenerate,
		expected:     len(reqCtx.TokenIDs),
	}
}

// onUsage is called for each usage block in the decode response. Only the
// first block that carries a prompt count is judged: a streamed Messages
// response reports input_tokens on message_start and again on the final
// usage, and both describe the same prompt. usage.PromptTokens folds the
// Messages cache split back in, so a P/D request whose prompt was served
// from prefill's KV compares as a match, not a short count.
func (p *parityCheck) onUsage(block map[string]any) {
	if p.reported {
		return
	}
	got, ok := usage.PromptTokens(block)
	if !ok {
		return
	}
	p.reported = true
	if got == p.expected {
		return
	}
	coordmetrics.IncPromptTokenMismatch(p.model, p.apiType.String(), p.idsForwarded)
	keysAndValues := []any{
		"request_id", p.requestID,
		"rendered_tokens", p.expected,
		"decode_prompt_tokens", got,
		"api_type", p.apiType,
		"ids_forwarded", p.idsForwarded,
	}
	if p.mode == promptTokenParityError {
		coordmetrics.IncStepErrorTotal(DecodeStepName, coordmetrics.ErrorCodePromptTokenMismatch)
		p.logger.Error(nil, "prompt token mismatch between prefill and decode", keysAndValues...)
		return
	}
	p.logger.Info("prompt token mismatch between prefill and decode", keysAndValues...)
}

// injectIncludeUsage asks a streamed chat or completions leg for the trailing
// usage frame when the client did not, so the parity check can read it. It
// reports whether it added the option; the response writer then drops that
// frame so the client's stream stays as requested. Responses and Messages
// streams carry usage on their own terminal events.
func injectIncludeUsage(reqCtx *pipeline.RequestContext) bool {
	if !reqCtx.Stream {
		return false
	}
	switch reqcommon.DetectAPIType(reqCtx.OriginalPath) {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeCompletions:
	default:
		return false
	}
	options := streamOptions(reqCtx.Body)
	if include, ok := options[fieldIncludeUsage].(bool); ok && include {
		return false
	}
	// vLLM reports usage on every chunk when include_usage and
	// continuous_usage_stats are both set. A client that sent only the latter
	// gets no usage today; adding include_usage would put a usage block in
	// each of its content chunks, which the writer cannot drop, so the check
	// is skipped rather than reshaping the stream.
	if continuous, ok := options[fieldContinuousUsageStats].(bool); ok && continuous {
		return false
	}
	merged := make(map[string]any, len(options)+1)
	for k, v := range options {
		merged[k] = v
	}
	merged[fieldIncludeUsage] = true
	reqCtx.Body[reqcommon.FieldStreamOptions] = merged
	return true
}

// streamOptions returns the request's stream_options object. The server
// decodes only the fields the pipeline inspects and keeps the rest as raw
// JSON (decodeRequestBody), so the value is usually json.RawMessage here.
func streamOptions(body map[string]any) map[string]any {
	switch v := body[reqcommon.FieldStreamOptions].(type) {
	case map[string]any:
		return v
	case json.RawMessage:
		var options map[string]any
		if err := json.Unmarshal(v, &options); err != nil {
			return nil
		}
		return options
	default:
		return nil
	}
}

const (
	fieldIncludeUsage         = "include_usage"
	fieldContinuousUsageStats = "continuous_usage_stats"
)

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
