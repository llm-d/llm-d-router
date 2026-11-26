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
	"fmt"
	"net/http"

	"github.com/go-logr/logr"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/usage"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// Values of the prompt_token_parity step parameter (ParamPromptTokenParity).
const (
	promptTokenParityOff   = "off"
	promptTokenParityWarn  = "warn"
	promptTokenParityError = "error"
)

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

// parityCheck compares the prompt token count a worker reports with the
// number of token IDs the render step produced, which is what the EPP scored
// and, when prefill ran, what it ran on. A difference means the worker rendered a different
// prompt: it ignored forwarded IDs (vLLM does for messages with non-text
// content or with echo set) or the two renderers disagree (tokenizer, chat
// template, processor version, chat_template_kwargs). Either way a KV cache
// built from the rendered sequence does not match and the worker silently
// recomputes the prompt, so the only visible symptom without this check is a
// slower request. The decode step and the conditional-decode fast path both
// run it; a served cache hit is a KV-reuse case like any other.
type parityCheck struct {
	logger       logr.Logger
	step         string
	mode         string
	requestID    string
	model        string
	apiType      reqcommon.APIType
	idsForwarded bool
	expected     int
	reported     bool
}

// setupParity prepares the parity check for one leg whose request body is
// about to be sent. It returns nil when the mode is off, no tokens were
// rendered, or the client's stream_options is present but not an object: the
// body is then forwarded byte for byte and the worker's own answer to it is
// not something to compare. Otherwise it may add stream_options.include_usage
// to body so a streamed chat or completions reply carries a usage frame to
// compare, and reports that through injected so the response writer drops
// that frame again.
func setupParity(step, mode string, logger logr.Logger, reqCtx *pipeline.RequestContext, body map[string]any) (check *parityCheck, injected bool) {
	if mode == promptTokenParityOff || len(reqCtx.TokenIDs) == 0 {
		return nil, false
	}
	apiType := reqcommon.DetectAPIType(reqCtx.OriginalPath)
	injected, invalid := reqcommon.InjectIncludeUsage(body, reqCtx.Stream, apiType)
	if invalid {
		return nil, false
	}
	return &parityCheck{
		logger:       logger,
		step:         step,
		mode:         mode,
		requestID:    reqCtx.RequestID,
		model:        reqCtx.Model,
		apiType:      apiType,
		idsForwarded: idsForwardedIn(body, apiType),
		expected:     len(reqCtx.TokenIDs),
	}, injected
}

// idsForwardedIn reports whether the leg's body carries the rendered token IDs,
// so the worker should not render at all: the generate route by definition,
// completions when the step put the IDs in as the prompt, and the chat-shaped
// routes when kv_transfer_params.prompt_token_ids is set (vLLM reuses it on
// text requests). It reads the body rather than the route so the label follows
// whatever a step sends.
func idsForwardedIn(body map[string]any, apiType reqcommon.APIType) bool {
	switch apiType {
	case reqcommon.APITypeVLLMGenerate:
		return true
	case reqcommon.APITypeCompletions:
		_, ok := body[reqcommon.FieldPrompt].([]uint32)
		return ok
	}
	kv, ok := body[reqcommon.FieldKVTransferParams].(map[string]any)
	return ok && kv[reqcommon.FieldPromptTokenIDs] != nil
}

// parityResponseWriter returns the writer a leg's response is streamed
// through, and the function that flushes whatever it buffered. Without a
// prefill count and without a parity check the client gets the worker's bytes
// untouched; with only a check the writer is a passthrough that reads usage on
// the side.
func parityResponseWriter(reqCtx *pipeline.RequestContext, check *parityCheck, injected bool) (http.ResponseWriter, func() error) {
	if reqCtx.PrefillCachedTokens == nil && check == nil {
		return reqCtx.ResponseWriter, func() error { return nil }
	}
	opts := usage.Options{
		CachedTokens:       reqCtx.PrefillCachedTokens,
		DropUsageOnlyFrame: injected,
	}
	if check != nil {
		opts.OnUsage = check.onUsage
	}
	return usage.NewResponseWriter(reqCtx.ResponseWriter, opts, reqCtx.Stream)
}

// onUsage is called for each usage block in the response. Only the first
// block that carries a prompt count is judged: a streamed Messages response
// reports input_tokens on message_start and again on the final usage, and both
// describe the same prompt. usage.PromptTokens folds the Messages cache split
// back in, so a request whose prompt was served from prefill's KV compares as
// a match, not a short count.
func (p *parityCheck) onUsage(block map[string]any) {
	if p.reported {
		return
	}
	got, ok := usage.PromptTokens(block)
	if !ok {
		return
	}
	p.reported = true
	recordPromptTokenMismatch(p.logger, p.mode, p.step, p.requestID, p.model, p.apiType, p.idsForwarded, p.expected, got)
}

// recordPromptTokenMismatch compares got against expected and records the
// prompt_token_mismatch_total metric and log entry when they differ. On HTTP
// paths expected is the render step's token count; on gRPC P/D paths expected
// is the prefill worker's reported prompt token count.
func recordPromptTokenMismatch(
	logger logr.Logger,
	mode, step, requestID, model string,
	apiType reqcommon.APIType,
	idsForwarded bool,
	expected, got int,
) {
	if mode == promptTokenParityOff || got == expected {
		return
	}
	coordmetrics.IncPromptTokenMismatch(model, apiType.String(), idsForwarded)
	keysAndValues := []any{
		"request_id", requestID,
		"step", step,
		"rendered_tokens", expected,
		"worker_prompt_tokens", got,
		"api_type", apiType,
		"ids_forwarded", idsForwarded,
	}
	if mode == promptTokenParityError {
		coordmetrics.IncStepErrorTotal(step, coordmetrics.ErrorCodePromptTokenMismatch)
		logger.Error(nil, "prompt token mismatch between render and worker", keysAndValues...)
		return
	}
	logger.Info("prompt token mismatch between render and worker", keysAndValues...)
}
