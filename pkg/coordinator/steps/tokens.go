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
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
)

// capSingleTokenOutput rewrites body into a single-output-token, non-streaming
// request for the synthetic prefill and encode legs. max_tokens is pinned to 1
// and min_tokens is stripped: min_tokens defaults to 0 in vLLM, so removing it
// keeps min_tokens <= max_tokens satisfied without raising the floor above
// max_tokens=1. These limits live in the sampling_params sub-map for the generate
// schema (synthesized if absent) and at the top level for the OpenAI schemas.
//
// stream is forced false and stream_options stripped for every format, so no leg
// returns a streamed response the coordinator cannot decode. max_completion_tokens
// is capped to 1 only when the body already carries it (in practice only the chat
// completions schema does), never added when absent.
func capSingleTokenOutput(body map[string]any, format gateway.RequestFormat) {
	target := body
	if format == gateway.FormatGenerate {
		sp, ok := body[reqcommon.FieldSamplingParams].(map[string]any)
		if !ok {
			sp = map[string]any{}
			body[reqcommon.FieldSamplingParams] = sp
		}
		target = sp
	}

	target[reqcommon.FieldMaxTokens] = 1
	delete(target, reqcommon.FieldMinTokens)

	if _, ok := body[reqcommon.FieldMaxCompletionTokens]; ok {
		body[reqcommon.FieldMaxCompletionTokens] = 1
	}

	body[reqcommon.FieldStream] = false
	delete(body, reqcommon.FieldStreamOptions)
}
