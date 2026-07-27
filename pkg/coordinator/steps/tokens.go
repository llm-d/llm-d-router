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
// request for the synthetic prefill and encode legs.
//
// Intentionally distinct from the sidecar's reqcommon.PrimeSingleTokenRequest
// for now.
// TODO: unify the two into one shared single-token helper in a future refactor.
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
	// Strip rather than clamp min_tokens: it defaults to 0 in vLLM, so removing it
	// keeps min_tokens <= max_tokens=1 without raising the floor above the cap.
	delete(target, reqcommon.FieldMinTokens)

	if _, ok := body[reqcommon.FieldMaxCompletionTokens]; ok {
		body[reqcommon.FieldMaxCompletionTokens] = 1
	}

	// TODO: max_output_tokens is another client-supplied output cap (Responses
	// API) that a client can send instead of max_tokens/max_completion_tokens; it
	// should be capped to 1 here as well so the synthetic legs stay single-token.

	body[reqcommon.FieldStream] = false
	delete(body, reqcommon.FieldStreamOptions)
}
