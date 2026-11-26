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
)

// Top-level request fields that only a trusted caller may set on an external
// text route. FieldPromptTokenIDs and FieldKVTransferParams are declared in
// constants.go.
const (
	FieldPromptRenderFingerprint = "prompt_render_fingerprint"
	FieldRenderFingerprint       = "render_fingerprint"
	FieldInputIDs                = "input_ids"
)

// ExternalTextRouteSanitizedFields lists the top-level JSON fields that
// untrusted clients must not supply on external text routes
// (/v1/chat/completions, string /v1/completions, /v1/messages, /v1/responses).
//
// Removing kv_transfer_params at the top level also removes nested fields such
// as kv_transfer_params.prompt_token_ids and do_remote_prefill/do_remote_decode.
// FindExternalTextFields and StripExternalTextFields also cover the copies of
// kv_transfer_params in nestedKVTransferParams.
var ExternalTextRouteSanitizedFields = []string{
	FieldPromptTokenIDs,
	FieldPromptRenderFingerprint,
	FieldRenderFingerprint,
	FieldKVTransferParams,
	FieldInputIDs,
}

// nestedKVTransferParams lists the objects, besides the top level, from which a
// vLLM worker reads kv_transfer_params. The engine reads the field from
// SamplingParams.extra_args. A client writes that map as
// sampling_params.extra_args on /inference/v1/generate. vllm-rs also accepts
// the name vllm_xargs there, and it accepts any JSON value in the top-level
// vllm_xargs of the OpenAI routes.
var nestedKVTransferParams = []struct {
	parents []string
	name    string
}{
	{parents: []string{FieldVLLMXArgs}, name: "vllm_xargs.kv_transfer_params"},
	{parents: []string{FieldSamplingParams, FieldExtraArgs}, name: "sampling_params.extra_args.kv_transfer_params"},
	{parents: []string{FieldSamplingParams, FieldVLLMXArgs}, name: "sampling_params.vllm_xargs.kv_transfer_params"},
}

// nestedKey reports whether the object at path parents inside value holds key.
// With strip it also returns value without key. An object is a decoded
// map[string]any, which is edited in place, or a json.RawMessage, which is
// encoded again. Any other value holds no key.
func nestedKey(value any, parents []string, key string, strip bool) (any, bool) {
	switch obj := value.(type) {
	case map[string]any:
		if len(parents) == 0 {
			if _, ok := obj[key]; !ok {
				return value, false
			}
			if strip {
				delete(obj, key)
			}
			return value, true
		}
		child, found := nestedKey(obj[parents[0]], parents[1:], key, strip)
		if found && strip {
			obj[parents[0]] = child
		}
		return value, found
	case json.RawMessage:
		// A key that is not written literally has an escape, and an escape
		// has a backslash. An object without both cannot hold key, so most
		// requests skip the decoding.
		if !bytes.Contains(obj, []byte(key)) && bytes.IndexByte(obj, '\\') < 0 {
			return value, false
		}
		var fields map[string]json.RawMessage
		if json.Unmarshal(obj, &fields) != nil || fields == nil {
			return value, false
		}
		if len(parents) == 0 {
			if _, ok := fields[key]; !ok {
				return value, false
			}
			delete(fields, key)
		} else {
			child, found := nestedKey(fields[parents[0]], parents[1:], key, strip)
			if !found {
				return value, false
			}
			fields[parents[0]], _ = child.(json.RawMessage)
		}
		// The fields come from valid JSON, so the encoding cannot fail. A nil
		// result would encode as null, which also removes the key. Without
		// strip, the caller does not use the encoded object.
		encoded, _ := json.Marshal(fields)
		return json.RawMessage(encoded), true
	default:
		return value, false
	}
}

// nestedKVTransferParamsIn returns the names of the nestedKVTransferParams
// copies in body, and deletes them when strip is true.
func nestedKVTransferParamsIn(body map[string]any, strip bool) []string {
	var found []string
	for _, field := range nestedKVTransferParams {
		if _, ok := nestedKey(body, field.parents, FieldKVTransferParams, strip); ok {
			found = append(found, field.name)
		}
	}
	return found
}

// FindKVTransferParams returns the names of the client copies of
// kv_transfer_params in body: the top-level field and the
// nestedKVTransferParams copies. Returns nil when none are present.
func FindKVTransferParams(body map[string]any) []string {
	var found []string
	if _, ok := body[FieldKVTransferParams]; ok {
		found = append(found, FieldKVTransferParams)
	}
	return append(found, nestedKVTransferParamsIn(body, false)...)
}

// StripKVTransferParams deletes the fields that FindKVTransferParams reports
// and returns their names. A dedicated token route keeps the client token IDs
// and drops only these fields.
func StripKVTransferParams(body map[string]any) []string {
	var stripped []string
	if _, ok := body[FieldKVTransferParams]; ok {
		delete(body, FieldKVTransferParams)
		stripped = append(stripped, FieldKVTransferParams)
	}
	return append(stripped, nestedKVTransferParamsIn(body, true)...)
}

// isTokenArrayCompletionsPrompt reports whether a decoded /v1/completions
// prompt value is a token-ID array ([int, ...]) or batch of token-ID arrays
// ([[int, ...], ...]). String prompts, string arrays, empty arrays, and
// malformed elements return false.
func isTokenArrayCompletionsPrompt(prompt any) bool {
	arr, ok := prompt.([]any)
	if !ok || len(arr) == 0 {
		return false
	}
	if _, isSubArray := arr[0].([]any); isSubArray {
		for _, elem := range arr {
			sub, ok := elem.([]any)
			if !ok || !isNonEmptyIntArray(sub) {
				return false
			}
		}
		return true
	}
	return isNonEmptyIntArray(arr)
}

func isNonEmptyIntArray(arr []any) bool {
	if len(arr) == 0 {
		return false
	}
	for _, elem := range arr {
		if _, err := NonNegativeInt(elem); err != nil {
			return false
		}
	}
	return true
}

// IsExternalTextRoute reports whether apiType with the given decoded
// completions prompt value is an external text route subject to prompt-token
// and KV-transfer field sanitization. Dedicated token routes
// (/inference/v1/generate, /generate, and token-array /v1/completions) return
// false.
func IsExternalTextRoute(apiType APIType, completionsPrompt any) bool {
	switch apiType {
	case APITypeChatCompletions, APITypeMessages, APITypeResponses:
		return true
	case APITypeCompletions:
		return !isTokenArrayCompletionsPrompt(completionsPrompt)
	default:
		return false
	}
}

// FindExternalTextFields returns the names of any forbidden external text-route
// fields present in body: the ExternalTextRouteSanitizedFields in their order,
// then the nestedKVTransferParams copies. Returns nil when none are present.
func FindExternalTextFields(body map[string]any) []string {
	var found []string
	for _, field := range ExternalTextRouteSanitizedFields {
		if _, ok := body[field]; ok {
			found = append(found, field)
		}
	}
	return append(found, nestedKVTransferParamsIn(body, false)...)
}

// StripExternalTextFields deletes the fields that FindExternalTextFields
// reports from body and returns their names in the same order. Returns nil
// when none are present.
func StripExternalTextFields(body map[string]any) []string {
	var stripped []string
	for _, field := range ExternalTextRouteSanitizedFields {
		if _, ok := body[field]; ok {
			delete(body, field)
			stripped = append(stripped, field)
		}
	}
	return append(stripped, nestedKVTransferParamsIn(body, true)...)
}
