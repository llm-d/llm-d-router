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

// APIType is the inference API a request was sent to. A value outside the
// constants below degrades to APITypeChatCompletions.
type APIType int

const (
	// APITypeChatCompletions is the Chat Completions API (/v1/chat/completions).
	APITypeChatCompletions APIType = iota
	// APITypeCompletions is the legacy Completions API (/v1/completions).
	APITypeCompletions
	// APITypeResponses is the Responses API (/v1/responses).
	APITypeResponses
	// APITypeGenerate is vLLM's token-in generate API (/inference/v1/generate).
	APITypeGenerate
	// APITypeMessages is the Anthropic Messages API (/v1/messages).
	APITypeMessages
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
	case APITypeMessages:
		return "messages"
	default:
		return fmt.Sprintf("APIType(%d)", int(a))
	}
}

// Path returns the canonical request path for the API.
func (a APIType) Path() string {
	switch a {
	case APITypeCompletions:
		return PathCompletions
	case APITypeResponses:
		return PathResponses
	case APITypeGenerate:
		return PathGenerate
	case APITypeMessages:
		return PathMessages
	default:
		return PathChatCompletions
	}
}

// DetectAPIType classifies a request path. An unrecognized path maps to
// APITypeChatCompletions: callers that route only known paths never reach the
// fallback.
func DetectAPIType(path string) APIType {
	switch {
	case strings.Contains(path, PathChatCompletions):
		return APITypeChatCompletions
	case strings.Contains(path, PathCompletions):
		return APITypeCompletions
	case strings.Contains(path, PathResponses):
		return APITypeResponses
	case strings.Contains(path, PathMessages):
		return APITypeMessages
	case strings.Contains(path, PathGenerate):
		return APITypeGenerate
	default:
		return APITypeChatCompletions
	}
}

// JSON request field names that cap output tokens, by API. Chat completions caps
// both max_tokens and max_completion_tokens: vLLM and SGLang accept the two
// together and prefer max_completion_tokens, so capping both bounds the request
// regardless of which field the engine consults. The Completions, Messages, and
// generate APIs share a list: none of them defines max_completion_tokens, so
// capping it would put a field on the wire that a strict server is free to
// reject.
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
	case APITypeCompletions, APITypeGenerate, APITypeMessages:
		return maxTokensOnlyTokenLimitFields
	default:
		return chatCompletionTokenLimitFields
	}
}
