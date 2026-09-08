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

import "maps"

// CapSingleToken rewrites body into a synthetic, non-streaming,
// single-output-token request for a prefill or encode leg.
//
// max_completion_tokens is capped alongside max_tokens where the API has it:
// vLLM and SGLang accept the two together and prefer max_completion_tokens, so
// setting both caps the leg regardless of which field the engine consults.
// min_tokens is stripped rather than clamped: it defaults to 0 in vLLM, so
// removing it keeps min_tokens <= max_tokens=1 without raising the floor above
// the cap (vLLM's SamplingParams rejects min_tokens > max_tokens).
//
// TODO: the Responses API caps output with max_output_tokens (see
// APIType.TokenLimitFields), which this function does not touch, so a
// /v1/responses prefill leg carrying only that field is not single-token.
func CapSingleToken(body map[string]any, apiType APIType) {
	limits, created := apiType.TokenLimitMap(body)
	if !created && apiType == APITypeGenerate {
		// TokenLimitMap handed back the caller's own sampling_params. Callers
		// copy the request body one level deep and marshal the original for the
		// decode leg, so writing through this map would cap decode as well.
		limits = maps.Clone(limits)
		body[FieldSamplingParams] = limits
	}

	limits[FieldMaxTokens] = 1
	delete(limits, FieldMinTokens)
	if apiType != APITypeGenerate {
		limits[FieldMaxCompletionTokens] = 1
	}

	body[FieldStream] = false
	delete(body, FieldStreamOptions)
}
