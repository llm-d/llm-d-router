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
// The caps to rewrite come from APIType.TokenLimitFields, so each API's output
// caps are named in one place. min_tokens is a floor rather than a cap, so it
// is stripped unconditionally instead: it defaults to 0 in vLLM, so removing it
// keeps min_tokens <= max_tokens=1 without raising the floor above the cap
// (vLLM's SamplingParams rejects min_tokens > max_tokens).
//
// Chat completions lists both max_tokens and max_completion_tokens: vLLM and
// SGLang accept the two together and prefer max_completion_tokens, so capping
// both bounds the leg regardless of which field the engine consults.
//
// body is rewritten in place, so the caller passes its own copy. A one-level
// copy is enough: the generate API caps inside sampling_params, and that nested
// map is replaced rather than written through, so a body that still shares it
// with the decode leg keeps the client's limits.
func CapSingleToken(body map[string]any, apiType APIType) {
	limits, created := apiType.TokenLimitMap(body)
	if !created && apiType == APITypeGenerate {
		limits = maps.Clone(limits)
		body[FieldSamplingParams] = limits
	}

	for _, field := range apiType.TokenLimitFields() {
		if field != FieldMinTokens {
			limits[field] = 1
		}
	}
	delete(limits, FieldMinTokens)

	body[FieldStream] = false
	delete(body, FieldStreamOptions)
}
