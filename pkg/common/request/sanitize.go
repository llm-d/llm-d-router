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
var ExternalTextRouteSanitizedFields = []string{
	FieldPromptTokenIDs,
	FieldPromptRenderFingerprint,
	FieldRenderFingerprint,
	FieldKVTransferParams,
	FieldInputIDs,
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
// fields present in body, in ExternalTextRouteSanitizedFields order. Returns
// nil without allocating when none are present.
func FindExternalTextFields(body map[string]any) []string {
	var found []string
	for _, field := range ExternalTextRouteSanitizedFields {
		if _, ok := body[field]; ok {
			found = append(found, field)
		}
	}
	return found
}

// StripExternalTextFields deletes any forbidden external text-route fields from
// body and returns the deleted field names in ExternalTextRouteSanitizedFields
// order. Returns nil without allocating when none are present.
func StripExternalTextFields(body map[string]any) []string {
	var stripped []string
	for _, field := range ExternalTextRouteSanitizedFields {
		if _, ok := body[field]; ok {
			delete(body, field)
			stripped = append(stripped, field)
		}
	}
	return stripped
}
