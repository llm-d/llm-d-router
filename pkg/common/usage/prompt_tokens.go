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

import reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

// Keys under which a streamed event nests the object that carries usage:
// Responses stream events wrap the response object, and the Messages
// message_start event wraps the message object.
const (
	fieldResponse = "response"
	fieldMessage  = "message"
	fieldChoices  = "choices"
)

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
	for _, wrapper := range []string{fieldResponse, fieldMessage} {
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

// Messages usage splits the prompt into the part served from cache and the
// rest; input_tokens alone excludes the cached part.
const (
	fieldCacheReadInputTokens     = "cache_read_input_tokens"
	fieldCacheCreationInputTokens = "cache_creation_input_tokens"
)

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
	if n, ok := intValue(fields[reqcommon.FieldPromptTokens]); ok {
		return n, true
	}
	n, ok := intValue(fields[reqcommon.FieldInputTokens])
	if !ok {
		return 0, false
	}
	read, _ := intValue(fields[fieldCacheReadInputTokens])
	created, _ := intValue(fields[fieldCacheCreationInputTokens])
	return n + read + created, true
}
