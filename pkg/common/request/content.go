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

// PartArray is one content part array of a message or input item, named by the
// body field it came from so a caller can report which array it walked.
type PartArray struct {
	Field string
	Parts []any
}

// ItemPartArrays returns the content part arrays an item carries, in the order
// a walk visits them.
//
// Every API holds its parts under content. A Responses function_call_output
// instead holds them under output, and vLLM forwards that array as a tool
// message's content, so media in it reaches the model like any other part. A
// computer_call_output's output is an object rather than an array and names no
// part type a media walk collects. A chat-completions message defines no
// output, so walking one there would collect a part the client never sent.
//
// Every walk over a request's media parts takes its arrays from here: the
// coordinator steps that index multimodal entries by position and the sidecar's
// encoder fan-out all have to agree on the set of parts a request carries.
func ItemPartArrays(item map[string]any, apiType APIType) []PartArray {
	var arrays []PartArray
	if content, ok := item[FieldContent].([]any); ok {
		arrays = append(arrays, PartArray{Field: FieldContent, Parts: content})
	}
	if apiType == APITypeResponses {
		if output, ok := item[FieldOutput].([]any); ok {
			arrays = append(arrays, PartArray{Field: FieldOutput, Parts: output})
		}
	}
	return arrays
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
