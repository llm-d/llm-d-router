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

package request

import (
	"fmt"
	"strings"
)

// Inference API paths served by the sidecar and the coordinator.
const (
	PathChatCompletions = "/v1/chat/completions"
	PathCompletions     = "/v1/completions"
	PathResponses       = "/v1/responses"
	PathMessages        = "/v1/messages"
	PathGenerate        = "/inference/v1/generate"
)

// APIType is the inference API a request speaks. It selects the JSON field
// names a request carries and the path a synthesized request is sent to.
type APIType int

const (
	// APITypeChatCompletions is the Chat Completions API (/v1/chat/completions)
	// and the Anthropic Messages API (/v1/messages), which share its field names.
	APITypeChatCompletions APIType = iota
	// APITypeCompletions is the legacy Completions API (/v1/completions).
	APITypeCompletions
	// APITypeResponses is the Responses API (/v1/responses).
	APITypeResponses
	// APITypeGenerate is vLLM's token-in generate API (/inference/v1/generate).
	APITypeGenerate
)

// String implements fmt.Stringer so structured logs show readable API names.
func (a APIType) String() string {
	switch a {
	case APITypeChatCompletions:
		return "chat_completions"
	case APITypeCompletions:
		return "completions"
	case APITypeResponses:
		return "responses"
	case APITypeGenerate:
		return "generate"
	default:
		return fmt.Sprintf("APIType(%d)", int(a))
	}
}

// Path returns the canonical request path for the API. APITypeChatCompletions
// maps to PathChatCompletions; PathMessages shares its field names but is not
// a synthesis target.
func (a APIType) Path() string {
	switch a {
	case APITypeChatCompletions:
		return PathChatCompletions
	case APITypeCompletions:
		return PathCompletions
	case APITypeResponses:
		return PathResponses
	default:
		return PathGenerate
	}
}

// DetectAPIType classifies a request path. An unrecognized path maps to
// APITypeGenerate: callers that route only known paths never reach the
// fallback, and a path the router does not register is not a client fault.
func DetectAPIType(path string) APIType {
	switch {
	case strings.Contains(path, PathChatCompletions):
		return APITypeChatCompletions
	case strings.Contains(path, PathCompletions):
		return APITypeCompletions
	case strings.Contains(path, PathResponses):
		return APITypeResponses
	case strings.Contains(path, PathMessages):
		return APITypeChatCompletions
	default:
		return APITypeGenerate
	}
}

// JSON request field names that carry output token limits, by API.
// Do not mutate these slices.
var (
	chatCompletionTokenLimitFields = []string{FieldMaxTokens, FieldMaxCompletionTokens, FieldMinTokens}
	responsesTokenLimitFields      = []string{FieldMaxOutputTokens}
	generateTokenLimitFields       = []string{FieldMaxTokens, FieldMinTokens}
)

// TokenLimitFields returns the token limit field names the API uses.
// The returned slices are shared package-level vars; callers must not mutate them.
func (a APIType) TokenLimitFields() []string {
	switch a {
	case APITypeResponses:
		return responsesTokenLimitFields
	case APITypeGenerate:
		return generateTokenLimitFields
	default:
		return chatCompletionTokenLimitFields
	}
}

// TokenLimitMap returns the map inside body that holds the token limit fields:
// sampling_params for the generate API (created if absent), body itself
// otherwise. The second return value reports whether an empty sampling_params
// map was synthesized; callers must drop it before dispatching downstream if it
// stays empty.
func (a APIType) TokenLimitMap(body map[string]any) (map[string]any, bool) {
	if a != APITypeGenerate {
		return body, false
	}
	if sp, ok := body[FieldSamplingParams].(map[string]any); ok {
		return sp, false
	}
	sp := map[string]any{}
	body[FieldSamplingParams] = sp
	return sp, true
}
