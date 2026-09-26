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
	"bytes"
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
// A Responses body also pins store to false: vLLM defaults it to true, so a
// synthetic leg would leave behind a stored response object nothing reaps. The
// other APIs define no store field, so setting it there would put an unknown
// field on the wire.
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
	if apiType == APITypeResponses {
		body[FieldStore] = false
	}
	return limits
}

// RejectStatefulResponsesFields reports an error naming the first field it
// finds that depends on state the router does not keep: previous_response_id
// and conversation reference a prior turn, background asks for an async job
// the router cannot poll, and file_id is part of the Responses file
// hydration API, referring to a file the router never stored.
//
// body may hold its values decoded or as json.RawMessage, so a caller that
// decodes only the fields it reads passes its body as-is. An input that reads
// as neither an array nor a string is refused: the file_id walk cannot run on
// it, so reporting no file_id would pass on a request whose input was never
// inspected.
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
	if referencesFile(items) {
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

// backgroundRequested reports whether body asks for a background response. An
// absent field, an explicit null and a JSON bool false do not ask; any other
// value does. vLLM coerces background through pydantic, so a value this package
// cannot read as false could arrive there as true, and refusing all of them
// avoids reimplementing that coercion.
func backgroundRequested(body map[string]any) bool {
	v, set := fieldValue(body, FieldBackground)
	if !set {
		return false
	}
	background, isBool := v.(bool)
	return !isBool || background
}

// arrayFromAny coerces a JSON-decoded value, or the json.RawMessage a caller
// that decodes selectively leaves behind, into a []any. An absent value and a
// string both yield a nil slice the caller walks as empty; anything else reports
// false so the caller refuses an input it could not inspect. UseNumber keeps that
// refusal off bodies the model server parses: a number outside float64 range
// fails a default []any decode while leaving the document valid.
func arrayFromAny(v any) ([]any, bool) {
	switch t := v.(type) {
	case nil, string:
		return nil, true
	case []any:
		return t, true
	case json.RawMessage:
		dec := json.NewDecoder(bytes.NewReader(t))
		dec.UseNumber()
		var decoded []any
		if dec.Decode(&decoded) == nil {
			return decoded, true
		}
		var text string
		return nil, json.Unmarshal(t, &text) == nil
	}
	return nil, false
}

// fileHydrationPartTypes are the content part types whose FieldFileID names a
// Files API upload. A file_id on any other object describes a file the serving
// engine never fetches: a file_citation annotation on a replayed assistant turn
// carries one, and vLLM reads only the text of such an item.
var fileHydrationPartTypes = map[string]bool{
	PartTypeInputImage:         true,
	PartTypeInputFile:          true,
	PartTypeComputerScreenshot: true,
}

// referencesFile reports whether v nests a content part naming a Files API
// upload. The depth varies: a message content part, a computer_call_output's
// output object, and a function_call_output's output array all carry one.
//
// An annotations array is skipped: every part type it holds reports a file the
// completed turn cited, so descending would refuse a turn replayed verbatim
// from a prior response.
//
// A file_id of null names no upload. The input schema declares the field
// nullable on an input_image and an input_file, and a computer_screenshot
// replayed from a prior response carries one, so refusing it would name a field
// the client believes it omitted. Values reach this walk fully decoded, so a
// JSON null arrives as a nil.
func referencesFile(v any) bool {
	switch t := v.(type) {
	case map[string]any:
		if partType, _ := t[FieldType].(string); fileHydrationPartTypes[partType] {
			if fileID, ok := t[FieldFileID]; ok && fileID != nil {
				return true
			}
		}
		for key, nested := range t {
			if key == FieldAnnotations {
				continue
			}
			if referencesFile(nested) {
				return true
			}
		}
	case []any:
		for _, nested := range t {
			if referencesFile(nested) {
				return true
			}
		}
	}
	return false
}
