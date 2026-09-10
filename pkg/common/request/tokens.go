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

// CapSingleToken rewrites body into a synthetic, non-streaming,
// single-output-token request for a prefill or encode leg. It returns the map
// the caps were written into, which is where the generate API also expects
// transfer params, so a caller adding them needs no second lookup.
//
// The caps to rewrite come from APIType.tokenLimitFields, so each API's output
// caps are named in one place. min_tokens is a floor rather than a cap, so it is
// stripped instead of capped: it defaults to 0 in vLLM, so removing it keeps
// min_tokens <= max_tokens=1 without raising the floor above the cap (vLLM's
// SamplingParams rejects min_tokens > max_tokens).
//
// body is rewritten in place, so the caller passes its own copy. A one-level
// copy is enough: APIType.tokenLimitMap never writes through a nested
// sampling_params the copy still shares with the decode leg.
func CapSingleToken(body map[string]any, apiType APIType) map[string]any {
	limits := apiType.tokenLimitMap(body)
	for _, field := range apiType.tokenLimitFields() {
		limits[field] = 1
	}
	delete(limits, FieldMinTokens)

	body[FieldStream] = false
	delete(body, FieldStreamOptions)
	return limits
}
