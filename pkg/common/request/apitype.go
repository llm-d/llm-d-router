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
	PathVLLMGenerate    = "/inference/v1/generate"
	PathSGLangGenerate  = "/generate"
)

// output_mode values of the vLLM generate API, naming what a response carries
// beyond its token IDs. The field defaults to tokens, and a server
// predating it ignores the value and answers with token IDs either way.
//
// OutputModeText adds the detokenized text to each choice. vLLM rejects it
// with 400 on a worker that loads no tokenizer (--tokens-only,
// --skip-tokenizer-init) or whose request sets sampling_params.detokenize to
// false.
//
// OutputModeDerender adds the parsed assistant message (content, reasoning,
// tool_calls) instead, applying the engine's own parsers in the decode call.
// vLLM rejects it with 400 alongside stream: true, and on a server with a tool
// or reasoning parser configured it also requires a parse_context built by
// /render. It postdates OutputModeText, so a worker that knows only the
// earlier modes answers 400 for an unknown mode.
//
// The default, tokens, has no constant here: the coordinator never asks for it.
// Decode streams its response to the client, so a token-IDs-only body would
// reach the caller with no text in it.
const (
	OutputModeText     = "text"
	OutputModeDerender = "derender"
)

// APIType is the inference API a request was sent to. Path and the output
// token cap treat a value outside the constants below as
// APITypeChatCompletions; String reports it as APIType(N).
type APIType int

const (
	// APITypeChatCompletions is the Chat Completions API (/v1/chat/completions).
	APITypeChatCompletions APIType = iota
	// APITypeCompletions is the legacy Completions API (/v1/completions).
	APITypeCompletions
	// APITypeResponses is the Responses API (/v1/responses).
	APITypeResponses
	// APITypeVLLMGenerate is vLLM's token-in generate API (/inference/v1/generate).
	APITypeVLLMGenerate
	// APITypeSGLangGenerate is SGLang's native generation API (/generate).
	APITypeSGLangGenerate
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
	case APITypeVLLMGenerate:
		return "vllm_generate"
	case APITypeSGLangGenerate:
		return "sglang_generate"
	case APITypeMessages:
		return "messages"
	default:
		return fmt.Sprintf("APIType(%d)", int(a))
	}
}

func (a APIType) Path() string {
	switch a {
	case APITypeCompletions:
		return PathCompletions
	case APITypeResponses:
		return PathResponses
	case APITypeVLLMGenerate:
		return PathVLLMGenerate
	case APITypeSGLangGenerate:
		return PathSGLangGenerate
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
	case strings.Contains(path, PathVLLMGenerate):
		return APITypeVLLMGenerate
	case strings.Contains(path, PathSGLangGenerate):
		return APITypeSGLangGenerate
	default:
		return APITypeChatCompletions
	}
}

// JSON request field names that cap output tokens, by API. Chat completions caps
// both max_tokens and max_completion_tokens: vLLM and SGLang accept the two
// together and prefer max_completion_tokens, so capping both bounds the request
// regardless of which field the engine consults. The Completions, Messages, and
// vLLM generate APIs share a list: none of them defines max_completion_tokens, so
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
	case APITypeCompletions, APITypeVLLMGenerate, APITypeMessages:
		return maxTokensOnlyTokenLimitFields
	default:
		return chatCompletionTokenLimitFields
	}
}
