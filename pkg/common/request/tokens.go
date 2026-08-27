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

// CapMaxTokensField caps target's max_tokens to 1 and strips min_tokens.
// min_tokens is stripped rather than clamped: it defaults to 0 in vLLM, so
// removing it keeps min_tokens <= max_tokens=1 without raising the floor
// above the cap (vLLM's SamplingParams rejects min_tokens > max_tokens).
func CapMaxTokensField(target map[string]any) {
	target[FieldMaxTokens] = 1
	delete(target, FieldMinTokens)
}

// PrimeSingleTokenRequest mutates target in place into a synthetic,
// non-streaming, single-output-token chat-completions or completions
// request. max_completion_tokens is unconditionally capped to 1 alongside
// max_tokens: vLLM and SGLang both accept the two fields together
// (max_completion_tokens takes precedence over max_tokens when present),
// so setting both guarantees the cap regardless of which field the serving
// engine consults.
func PrimeSingleTokenRequest(target map[string]any) {
	CapMaxTokensField(target)
	target[FieldMaxCompletionTokens] = 1

	target[FieldStream] = false
	delete(target, FieldStreamOptions)
}
