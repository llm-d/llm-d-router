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

package usage

import (
	"encoding/json"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// Count reads a JSON-decoded token count. A count arrives as float64 from a
// plain decode, json.Number from a UseNumber decode, a Go integer set
// in-process, or the string form some non-conforming providers emit; each
// form is read by NonNegativeInt, so a value that is negative, fractional,
// out of range, or not a number reports false and reads as absent rather
// than failing the response.
func Count(value any) (int, bool) {
	if s, ok := value.(string); ok {
		value = json.Number(s)
	}
	n, err := reqcommon.NonNegativeInt(value)
	return n, err == nil
}

// UsageBlock returns the usage object a decoded response or stream event
// carries. It is the top-level "usage" for chat, completions, generate, and
// the final Responses and Messages objects; "response.usage" for Responses
// stream events (response.completed); and "message.usage" for the Messages
// message_start event. A null usage, which Responses sends on its early
// events, reports false.
func UsageBlock(response map[string]any) (map[string]any, bool) {
	if block, ok := response[reqcommon.FieldUsage].(map[string]any); ok {
		return block, true
	}
	for _, wrapper := range []string{reqcommon.FieldResponse, reqcommon.FieldMessage} {
		inner, ok := response[wrapper].(map[string]any)
		if !ok {
			continue
		}
		if block, ok := inner[reqcommon.FieldUsage].(map[string]any); ok {
			return block, true
		}
	}
	return nil, false
}

// PromptTokens reads the whole prompt's token count out of a usage block:
// prompt_tokens on the chat, completions and generate routes, which already
// includes cached tokens; input_tokens on Responses; and on Messages
// input_tokens + cache_read_input_tokens + cache_creation_input_tokens, since
// vLLM subtracts the cached and created counts from input_tokens there (the
// EPP's Anthropic parser sums them the same way). It reports false when no
// prompt count is present.
func PromptTokens(block any) (int, bool) {
	fields, ok := block.(map[string]any)
	if !ok {
		return 0, false
	}
	if n, ok := Count(fields[reqcommon.FieldPromptTokens]); ok {
		return n, true
	}
	n, ok := Count(fields[reqcommon.FieldInputTokens])
	if !ok {
		return 0, false
	}
	read, _ := Count(fields[reqcommon.FieldCacheReadInputTokens])
	created, _ := Count(fields[reqcommon.FieldCacheCreationInputTokens])
	return n + read + created, true
}
