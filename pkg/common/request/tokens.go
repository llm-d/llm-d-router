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
	"encoding/json"
	"fmt"
	"maps"
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

// RejectStatefulResponsesFields reports an error naming the first field it
// finds that depends on state the router does not keep: previous_response_id
// and conversation reference a prior turn, background asks for an async job
// the router cannot poll, and file_id is part of the Responses file
// hydration API, referring to a file the router never stored.
//
// store is left unchecked: it is handled upstream by the stateful proxy
// (the agentic-api layer strips it before the request reaches the router),
// and forwarding it is harmless regardless since it defaults to true.
//
// body may hold its values decoded or as json.RawMessage, so a caller that
// decodes only the fields it reads passes its body as-is.
func RejectStatefulResponsesFields(body map[string]any) error {
	for _, field := range []string{FieldPreviousResponseID, FieldConversation} {
		if _, set := fieldValue(body, field); set {
			return fmt.Errorf("field %q is not supported by the router", field)
		}
	}
	if backgroundRequested(body) {
		return fmt.Errorf("field %q is not supported by the router", FieldBackground)
	}
	if inputReferencesFile(arrayFromAny(body[FieldInput])) {
		return fmt.Errorf("field %q is not supported by the router", FieldFileID)
	}
	return nil
}

// fieldValue resolves body[field], reporting whether the client set it. A
// field the caller left as raw JSON bytes is decoded here. An explicit null is
// not set, since SDKs serialize an unset optional that way and refusing it
// would name a field the client believes it omitted. Bytes that do not decode
// report set with a nil value, so a field that cannot be read is refused
// rather than passed on.
func fieldValue(body map[string]any, field string) (any, bool) {
	v, ok := body[field]
	if !ok {
		return nil, false
	}
	if raw, isRaw := v.(json.RawMessage); isRaw {
		if err := json.Unmarshal(raw, &v); err != nil {
			return nil, true
		}
	}
	return v, v != nil
}

// backgroundRequested reports whether body asks for a background response.
// Only an unset field or a value reading as false counts as not asking: vLLM
// validates background with pydantic, which coerces 1, "true" and "yes" to
// true, so a value this package cannot read as false is refused.
func backgroundRequested(body map[string]any) bool {
	v, set := fieldValue(body, FieldBackground)
	if !set {
		return false
	}
	background, isBool := v.(bool)
	return !isBool || background
}

// arrayFromAny coerces a JSON-decoded value into a []any, accepting the raw-bytes
// form the sidecar proxy leaves behind for the fields it does not decode. A value
// that is absent, not an array (a Responses input is a string or an array), or
// bytes that do not decode all yield nil, which a caller walks as empty.
func arrayFromAny(v any) []any {
	switch t := v.(type) {
	case []any:
		return t
	case json.RawMessage:
		var decoded []any
		if json.Unmarshal(t, &decoded) == nil {
			return decoded
		}
	}
	return nil
}

// inputReferencesFile reports whether a Responses input array contains a
// content part with a file_id field. file_id is not a top-level field:
// OpenAI's Responses API nests it inside an input_image, input_file, or
// input_audio content part, so finding it takes a walk of the input array
// rather than a map lookup.
func inputReferencesFile(input []any) bool {
	for _, item := range input {
		itemMap, ok := item.(map[string]any)
		if !ok {
			continue
		}
		content, ok := itemMap[FieldContent].([]any)
		if !ok {
			continue
		}
		for _, part := range content {
			partMap, ok := part.(map[string]any)
			if ok {
				if _, ok := partMap[FieldFileID]; ok {
					return true
				}
			}
		}
	}
	return false
}
