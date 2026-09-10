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
	"maps"
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
// names a request carries and the path a synthesized request is sent to. A
// value outside the constants below degrades to APITypeChatCompletions.
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

// Path returns the canonical request path for the API. PathMessages shares the
// chat completions field names but is not a synthesis target.
func (a APIType) Path() string {
	switch a {
	case APITypeCompletions:
		return PathCompletions
	case APITypeResponses:
		return PathResponses
	case APITypeGenerate:
		return PathGenerate
	default:
		return PathChatCompletions
	}
}

// LookupAPIType classifies a request path and reports whether the path matched
// a known API. A step that must not process a path the router does not register
// reads the second value; the first is APITypeChatCompletions when it is false.
func LookupAPIType(path string) (APIType, bool) {
	switch {
	case strings.Contains(path, PathChatCompletions):
		return APITypeChatCompletions, true
	case strings.Contains(path, PathCompletions):
		return APITypeCompletions, true
	case strings.Contains(path, PathResponses):
		return APITypeResponses, true
	case strings.Contains(path, PathMessages):
		return APITypeChatCompletions, true
	case strings.Contains(path, PathGenerate):
		return APITypeGenerate, true
	default:
		return APITypeChatCompletions, false
	}
}

// DetectAPIType is LookupAPIType for callers that pass only known paths.
func DetectAPIType(path string) APIType {
	apiType, _ := LookupAPIType(path)
	return apiType
}

// JSON request field names that cap output tokens, by API. Chat completions caps
// both max_tokens and max_completion_tokens: vLLM and SGLang accept the two
// together and prefer max_completion_tokens, so capping both bounds the request
// regardless of which field the engine consults. The Completions and generate
// APIs share a list: neither defines max_completion_tokens, so capping it would
// put a field on the wire that a strict server is free to reject.
var (
	chatCompletionTokenLimitFields = []string{FieldMaxTokens, FieldMaxCompletionTokens}
	maxTokensOnlyTokenLimitFields  = []string{FieldMaxTokens}
	responsesTokenLimitFields      = []string{FieldMaxOutputTokens}
)

// tokenLimitFields returns the output token cap field names the API uses.
// The returned slices are shared package-level vars; callers must not mutate them.
func (a APIType) tokenLimitFields() []string {
	switch a {
	case APITypeResponses:
		return responsesTokenLimitFields
	case APITypeCompletions, APITypeGenerate:
		return maxTokensOnlyTokenLimitFields
	default:
		return chatCompletionTokenLimitFields
	}
}

// tokenLimitMap returns the map inside body that holds the token limit fields:
// sampling_params for the generate API, body itself otherwise. The generate map
// is always replaced with one body owns, so a caller that writes into the result
// never reaches a nested map the body was cloned from.
func (a APIType) tokenLimitMap(body map[string]any) map[string]any {
	switch a {
	case APITypeGenerate:
		sp, _ := body[FieldSamplingParams].(map[string]any)
		owned := make(map[string]any, len(sp)+1)
		maps.Copy(owned, sp)
		body[FieldSamplingParams] = owned
		return owned
	default:
		return body
	}
}
