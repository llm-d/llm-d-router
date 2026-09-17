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
// single-output-token prefill or encode request. It returns the map
// the caps were written into: sampling_params for the vLLM generate API, body itself
// otherwise. The vLLM generate API also expects transfer params in that map, so a
// caller adding them needs no second lookup.
//
// The caps to rewrite come from APIType.tokenLimitFields, so each API's output
// caps are named in one place. min_tokens is a floor rather than a cap, so it is
// stripped instead of capped: it defaults to 0 in vLLM, so removing it keeps
// min_tokens <= max_tokens=1 without raising the floor above the cap (vLLM's
// SamplingParams rejects min_tokens > max_tokens).
//
// body is rewritten in place, so the caller passes its own copy. A one-level
// copy is enough: the generate sampling_params is always replaced with a map
// body owns, so the rewrite never reaches a nested map the body was cloned from.
func CapSingleToken(body map[string]any, apiType APIType) map[string]any {
	limits := body
	if apiType == APITypeVLLMGenerate {
		sp, _ := body[FieldSamplingParams].(map[string]any)
		limits = make(map[string]any, len(sp)+1)
		maps.Copy(limits, sp)
		body[FieldSamplingParams] = limits
	}
	for _, field := range apiType.tokenLimitFields() {
		limits[field] = 1
	}
	delete(limits, FieldMinTokens)

	body[FieldStream] = false
	delete(body, FieldStreamOptions)
	return limits
}

// DropStatefulResponsesFields removes stateful Responses fields that neither
// the coordinator's pipeline nor the sidecar can
// honor. "store" is forced to false rather
// than removed: vLLM defaults store to true when the field is absent, so
// deleting it would leave storage enabled instead of disabling it.
//
// Callers call this once on the request body before any per-request
// cloning, so every downstream body built from it, or from a clone of it,
// inherits the same stripped fields. It returns the names of the fields it
// changed, in the order above, so callers can log only when something
// actually changed; a caller whose decoded value for store is never a Go
// bool (e.g. the sidecar, which leaves uninspected fields as raw JSON)
// would otherwise treat every request as changed.
func DropStatefulResponsesFields(body map[string]any) []string {
	var changed []string
	if _, ok := body[FieldPreviousResponseID]; ok {
		delete(body, FieldPreviousResponseID)
		changed = append(changed, FieldPreviousResponseID)
	}
	if _, ok := body[FieldConversation]; ok {
		delete(body, FieldConversation)
		changed = append(changed, FieldConversation)
	}
	if store, ok := body[FieldStore].(bool); !ok || store {
		body[FieldStore] = false
		changed = append(changed, FieldStore)
	}
	if _, ok := body[FieldBackground]; ok {
		delete(body, FieldBackground)
		changed = append(changed, FieldBackground)
	}
	return changed
}
