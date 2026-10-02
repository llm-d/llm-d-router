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

// MediaPartURL returns the URL a media content part references, or "" when
// there is none to fetch: an inline input_audio part, or a part whose URL field
// is absent or not a string.
func MediaPartURL(part map[string]any) string {
	switch partType, _ := part[FieldType].(string); partType {
	case PartTypeImageURL, PartTypeAudioURL, PartTypeVideoURL:
		nested, ok := part[partType].(map[string]any)
		if !ok {
			return ""
		}
		url, _ := nested[FieldURL].(string)
		return url
	case PartTypeInputImage:
		url, _ := part[FieldImageURL].(string)
		return url
	}
	return ""
}

// encoderPassthroughFields are the client fields NewEncoderPrimingBody
// forwards: both kwargs fields change preprocessing and feed vLLM's multimodal
// hash, so an encoder primed at the deployment default stores its entry under a
// hash the prefiller never looks up.
var encoderPassthroughFields = []string{
	FieldModel,
	FieldMMProcessorKwargs,
	FieldMediaIOKwargs,
}

// NewEncoderPrimingBody builds a single-part encoder request: the
// encoderPassthroughFields off clientBody plus one synthetic user turn wrapping
// part, capped to a single output token. It builds from scratch because copying
// clientBody would hand the encoder fields its own API does not define.
//
// Anything but APITypeResponses is treated as chat completions, matching the
// path fanoutEncoder posts to. part is forwarded unreshaped: it already has the
// shape of the API it is posted under.
func NewEncoderPrimingBody(clientBody map[string]any, part map[string]any, apiType APIType) map[string]any {
	if apiType != APITypeResponses {
		apiType = APITypeChatCompletions
	}

	body := make(map[string]any, len(encoderPassthroughFields)+3)
	for _, field := range encoderPassthroughFields {
		if v, ok := clientBody[field]; ok {
			body[field] = v
		}
	}

	turn := map[string]any{FieldRole: "user", FieldContent: []map[string]any{part}}
	if apiType == APITypeResponses {
		body[FieldInput] = []map[string]any{turn}
	} else {
		body[FieldMessages] = []map[string]any{turn}
	}

	CapSingleToken(body, apiType)

	return body
}
