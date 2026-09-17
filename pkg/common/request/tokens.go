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
	"maps"

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

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
// the coordinator's disaggregated pipeline nor the sidecar's
// connectors can honor. Pure vllm-d supports stateless /responses requests.
// The "store" field is forced to false rather than
// removed: vLLM defaults "store" to true when the field is absent, so deleting
// it would leave storage enabled instead of disabling it.
//
// Callers pass the request body before any per-request cloning, so every request body
// is built from it (or from a clone of it) inherits the same stripped fields.
func DropStatefulResponsesFields(logger logr.Logger, body map[string]any) {
	var changed []string
	if _, ok := body[FieldPreviousResponseID]; ok {
		delete(body, FieldPreviousResponseID)
		changed = append(changed, FieldPreviousResponseID)
	}
	if store, ok := body[FieldStore].(bool); !ok || store {
		body[FieldStore] = false
		changed = append(changed, FieldStore)
	}
	if _, ok := body[FieldBackground]; ok {
		delete(body, FieldBackground)
		changed = append(changed, FieldBackground)
	}
	if len(changed) > 0 {
		logger.V(logutil.DEFAULT).Info("clearing unsupported responses fields", "fields", changed)
	}
}
