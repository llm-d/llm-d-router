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
	"maps"
)

// InjectIncludeUsage asks a streamed chat or completions request for the
// trailing usage frame when the client did not. It reports whether it added the
// option; the caller then drops that frame from the response so the client's
// stream stays as requested. invalid is true when the client sent a
// stream_options that is not an object: the body is left as it is for the
// worker to judge, and nothing is injected. Responses and Messages streams carry
// usage on their own terminal events.
func InjectIncludeUsage(body map[string]any, stream bool, apiType APIType) (injected bool, invalid bool) {
	if !stream {
		return false, false
	}
	switch apiType {
	case APITypeChatCompletions, APITypeCompletions:
	default:
		return false, false
	}
	options, present, ok := streamOptions(body)
	if present && !ok {
		return false, true
	}
	if include, ok := options[FieldIncludeUsage].(bool); ok && include {
		return false, false
	}
	// vLLM reports usage on every chunk when include_usage and
	// continuous_usage_stats are both set. A client that sent only the latter
	// gets no usage; adding include_usage would put a usage block in each of
	// its content chunks, which cannot be dropped, so nothing is injected
	// rather than reshaping the stream.
	if continuous, ok := options[FieldContinuousUsageStats].(bool); ok && continuous {
		return false, false
	}
	// A copy, since the client's stream_options object can be shared with
	// another view of the same request.
	merged := maps.Clone(options)
	if merged == nil {
		merged = make(map[string]any, 1)
	}
	merged[FieldIncludeUsage] = true
	body[FieldStreamOptions] = merged
	return true, false
}

// streamOptions returns the request's stream_options object. present says
// whether the field is in the body at all; ok says whether it decoded to an
// object (a JSON null counts as absent). Callers that decode only the fields
// they inspect keep the rest as raw JSON, so the value can be a
// json.RawMessage as well as a decoded map.
func streamOptions(body map[string]any) (options map[string]any, present bool, ok bool) {
	raw, present := body[FieldStreamOptions]
	if !present || raw == nil {
		return nil, false, true
	}
	switch v := raw.(type) {
	case map[string]any:
		return v, true, true
	case json.RawMessage:
		var decoded map[string]any
		if err := json.Unmarshal(v, &decoded); err != nil {
			return nil, true, false
		}
		if decoded == nil {
			return nil, false, true
		}
		return decoded, true, true
	default:
		return nil, true, false
	}
}
