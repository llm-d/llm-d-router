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
// decodes only the fields it reads passes its body as-is. An input that reads
// as neither an array nor a string is refused too: the file_id walk cannot run
// on it, and answering "no file_id" for an input that was never inspected would
// forward the request the walk exists to catch.
func RejectStatefulResponsesFields(body map[string]any) error {
	for _, field := range []string{FieldPreviousResponseID, FieldConversation} {
		if _, set := fieldValue(body, field); set {
			return fmt.Errorf("field %q is not supported by the router", field)
		}
	}
	if backgroundRequested(body) {
		return fmt.Errorf("field %q is not supported by the router", FieldBackground)
	}
	items, readable := arrayFromAny(body[FieldInput])
	if !readable {
		return fmt.Errorf("field %q could not be read as a JSON array or string", FieldInput)
	}
	if inputReferencesFile(items) {
		return fmt.Errorf("field %q is not supported by the router", FieldFileID)
	}
	return nil
}

// fieldValue resolves body[field], reporting whether the client set it. An
// explicit null is not set: SDKs serialize an unset optional that way, and
// refusing it would name a field the client believes it omitted. Bytes that do
// not decode report set with a nil value, so a field that cannot be read is
// refused rather than passed on.
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
// Only an unset field or a JSON bool false counts as not asking. vLLM coerces
// background through pydantic, so any other value could arrive there as true;
// refusing all of them avoids reimplementing that coercion.
func backgroundRequested(body map[string]any) bool {
	v, set := fieldValue(body, FieldBackground)
	if !set {
		return false
	}
	background, isBool := v.(bool)
	return !isBool || background
}

// arrayFromAny coerces a JSON-decoded value into a []any, accepting the
// json.RawMessage form a caller that decodes selectively leaves behind. An
// absent value, and a Responses input sent as a string, yield a nil slice a
// caller walks as empty.
//
// Raw bytes that decode as neither report false, so a caller refuses an input it
// could not inspect rather than reading the failure as "nothing found": every
// number in an array decodes through float64, so one value out of that range
// (1e999) fails the whole array while leaving the enclosing document valid.
func arrayFromAny(v any) ([]any, bool) {
	switch t := v.(type) {
	case []any:
		return t, true
	case json.RawMessage:
		var decoded []any
		if json.Unmarshal(t, &decoded) == nil {
			return decoded, true
		}
		var text string
		return nil, json.Unmarshal(t, &text) == nil
	}
	return nil, true
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
