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

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// wrapResponses rewrites a vLLM generate response in output_mode text into a
// Responses API response, the Responses counterpart of wrapChatCompletion.
//
// The single choice becomes an assistant message output item holding the choice
// text and empty annotations. A finish_reason of length marks both the response
// and its output item incomplete; any reason other than stop or length is an
// error, since the Responses API has no equivalent to report. Usage counts are
// renamed to input_tokens and output_tokens. The result is built from the
// fields the Responses API declares, so no generate-only field reaches the
// client.
//
// requestID, model, and wantMode have the same meaning as in wrapChatCompletion.
func wrapResponses(generate []byte, requestID, model, wantMode string) ([]byte, error) {
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

	m, _ := resp[reqcommon.FieldModel].(string)
	if m == "" {
		m = model
	}
	if m == "" {
		return nil, errors.New("responses response carries no model")
	}

	choices, _ := resp[reqcommon.FieldChoices].([]any)
	if len(choices) != 1 {
		return nil, fmt.Errorf("generate response carries %d choices, want exactly 1", len(choices))
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return nil, errors.New("choice 0 is not an object")
	}

	status := reqcommon.StatusCompleted
	switch reason, _ := choice[reqcommon.FieldFinishReason].(string); reason {
	case reqcommon.FinishReasonStop:
	case reqcommon.FinishReasonLength:
		status = reqcommon.StatusIncomplete
	default:
		return nil, fmt.Errorf("choice 0 finish_reason %q has no Responses API equivalent", reason)
	}

	text, _ := choice["text"].(string)
	outputItem := map[string]any{
		reqcommon.FieldID:     fmt.Sprintf("msg_%s_0", id),
		reqcommon.FieldType:   reqcommon.TypeMessage,
		reqcommon.FieldRole:   reqcommon.RoleAssistant,
		reqcommon.FieldStatus: status,
		reqcommon.FieldContent: []any{map[string]any{
			reqcommon.FieldType: reqcommon.PartTypeOutputText,
			"text":              text,
			"annotations":       []any{},
		}},
	}

	out := map[string]any{
		reqcommon.FieldID:     id,
		reqcommon.FieldObject: reqcommon.ObjectResponse,
		reqcommon.FieldStatus: status,
		reqcommon.FieldOutput: []any{outputItem},
		reqcommon.FieldModel:  m,
	}
	if status == reqcommon.StatusIncomplete {
		out["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	}
	if created, ok := resp[reqcommon.FieldCreated]; ok {
		out[reqcommon.FieldCreatedAt] = created
	}
	if usage, ok := resp[reqcommon.FieldUsage].(map[string]any); ok {
		for _, k := range []string{reqcommon.FieldPromptTokens, reqcommon.FieldCompletionTokens, reqcommon.FieldTotalTokens} {
			if v, ok := usage[k].(float64); ok && v < 0 {
				return nil, fmt.Errorf("usage %s is negative: %v", k, v)
			}
		}
		mapped := map[string]any{}
		copyField(mapped, reqcommon.FieldInputTokens, usage, reqcommon.FieldPromptTokens)
		copyField(mapped, reqcommon.FieldOutputTokens, usage, reqcommon.FieldCompletionTokens)
		copyField(mapped, reqcommon.FieldTotalTokens, usage, reqcommon.FieldTotalTokens)
		copyField(mapped, reqcommon.FieldInputTokensDetails, usage, reqcommon.FieldPromptTokensDetails)
		out[reqcommon.FieldUsage] = mapped
	}

	wrapped, err := json.Marshal(out)
	if err != nil {
		return nil, fmt.Errorf("encoding responses response: %w", err)
	}
	return wrapped, nil
}

// copyField sets dst[to] to src[from] when src carries a non-null value.
func copyField(dst map[string]any, to string, src map[string]any, from string) {
	if v, ok := src[from]; ok && v != nil {
		dst[to] = v
	}
}
