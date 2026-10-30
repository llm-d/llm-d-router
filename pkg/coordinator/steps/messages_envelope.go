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
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/usage"
)

// wrapMessages rewrites a vLLM generate response in output_mode text into an
// Anthropic Messages API response, the Messages counterpart of
// wrapChatCompletion.
//
// The message holds one text block with the single choice's text. A
// finish_reason of stop becomes end_turn and length becomes max_tokens; any
// other reason is an error, since the Messages API has no equivalent to report.
//
// Anthropic's input_tokens excludes tokens read from or written to the prompt
// cache, so prompt_tokens minus cached and creation tokens is reported as
// input_tokens, and cached_tokens as cache_read_input_tokens.
//
// requestID, model, and wantMode have the same meaning as in wrapChatCompletion.
func wrapMessages(generate []byte, requestID, model, wantMode string) ([]byte, error) {
	var resp map[string]any
	if err := json.Unmarshal(generate, &resp); err != nil {
		return nil, fmt.Errorf("decoding generate response: %w", err)
	}

	gotMode, _ := resp[reqcommon.FieldOutputMode].(string)
	if gotMode != wantMode {
		return nil, fmt.Errorf("worker answered output_mode %q, want %q: it carries no text to answer with",
			gotMode, wantMode)
	}

	id, _ := resp[reqcommon.FieldRequestID].(string)
	if id == "" {
		id = requestID
	}
	if !strings.HasPrefix(id, "msg_") {
		id = "msg_" + id
	}

	m, _ := resp[reqcommon.FieldModel].(string)
	if m == "" {
		m = model
	}
	if m == "" {
		return nil, errors.New("messages response carries no model")
	}

	choices, _ := resp[reqcommon.FieldChoices].([]any)
	if len(choices) != 1 {
		return nil, fmt.Errorf("generate response carries %d choices, want exactly 1", len(choices))
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return nil, errors.New("choice 0 is not an object")
	}

	var stopReason string
	switch reason, _ := choice[reqcommon.FieldFinishReason].(string); reason {
	case reqcommon.FinishReasonStop:
		stopReason = reqcommon.StopReasonEndTurn
	case reqcommon.FinishReasonLength:
		stopReason = reqcommon.StopReasonMaxTokens
	default:
		return nil, fmt.Errorf("choice 0 finish_reason %q has no Messages API equivalent", reason)
	}

	text, _ := choice["text"].(string)

	out := map[string]any{
		reqcommon.FieldID:   id,
		reqcommon.FieldType: reqcommon.TypeMessage,
		reqcommon.FieldRole: reqcommon.RoleAssistant,
		reqcommon.FieldContent: []any{map[string]any{
			reqcommon.FieldType: reqcommon.BlockTypeText,
			"text":              text,
		}},
		reqcommon.FieldModel:        m,
		reqcommon.FieldStopReason:   stopReason,
		reqcommon.FieldStopSequence: nil,
	}

	if block, ok := resp[reqcommon.FieldUsage].(map[string]any); ok {
		mapped := map[string]any{}
		var prompt float64
		if p, ok := block[reqcommon.FieldPromptTokens]; ok && p != nil {
			prompt, _ = p.(float64)
			if prompt < 0 {
				return nil, fmt.Errorf("usage %s is negative: %v", reqcommon.FieldPromptTokens, prompt)
			}
		}
		if cached, ok := usage.CachedTokens(block); ok {
			if cached < 0 {
				return nil, fmt.Errorf("usage cached tokens is negative: %d", cached)
			}
			if float64(cached) > prompt {
				return nil, fmt.Errorf("usage cached tokens %d exceeds prompt tokens %v", cached, prompt)
			}
			mapped[reqcommon.FieldCacheReadInputTokens] = cached
			prompt -= float64(cached)
		}
		if creationVal, ok := block[reqcommon.FieldCacheCreationInputTokens]; ok && creationVal != nil {
			creation, _ := creationVal.(float64)
			if creation < 0 {
				return nil, fmt.Errorf("usage %s is negative: %v", reqcommon.FieldCacheCreationInputTokens, creation)
			}
			if creation > prompt {
				return nil, fmt.Errorf("usage %s %v exceeds remaining prompt tokens %v", reqcommon.FieldCacheCreationInputTokens, creation, prompt)
			}
			mapped[reqcommon.FieldCacheCreationInputTokens] = creation
			prompt -= creation
		}
		mapped[reqcommon.FieldInputTokens] = prompt
		var comp float64
		if compVal, ok := block[reqcommon.FieldCompletionTokens]; ok && compVal != nil {
			comp, _ = compVal.(float64)
			if comp < 0 {
				return nil, fmt.Errorf("usage %s is negative: %v", reqcommon.FieldCompletionTokens, comp)
			}
		}
		mapped[reqcommon.FieldOutputTokens] = comp
		out[reqcommon.FieldUsage] = mapped
	}

	wrapped, err := json.Marshal(out)
	if err != nil {
		return nil, fmt.Errorf("encoding messages response: %w", err)
	}
	return wrapped, nil
}
