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

// MediaPartURLRef returns the URL a media content part references, a setter
// that writes a replacement back to the field it came from, and whether the
// part carries a readable URL at all. ok is false for a part type that holds no
// URL, such as an inline input_audio part, and for one whose URL field is
// absent or not a string; set is nil in that case.
//
// Where the URL lives differs by part type, and this is the only place that
// knows: chat-completions nests it at part[type]["url"], a Responses
// input_image holds it as a bare string at part["image_url"]. A caller that
// rewrites a URL in place goes through set so it cannot write the wrong shape.
func MediaPartURLRef(part map[string]any) (url string, set func(string), ok bool) {
	switch partType, _ := part[FieldType].(string); partType {
	case PartTypeImageURL, PartTypeAudioURL, PartTypeVideoURL:
		nested, isMap := part[partType].(map[string]any)
		if !isMap {
			return "", nil, false
		}
		url, ok = nested[FieldURL].(string)
		if !ok {
			return "", nil, false
		}
		return url, func(v string) { nested[FieldURL] = v }, true
	case PartTypeInputImage:
		url, ok = part[FieldImageURL].(string)
		if !ok {
			return "", nil, false
		}
		return url, func(v string) { part[FieldImageURL] = v }, true
	}
	return "", nil, false
}

// MediaPartURL returns the URL a media content part references, or "" when
// there is none to fetch.
func MediaPartURL(part map[string]any) string {
	url, _, _ := MediaPartURLRef(part)
	return url
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
// What callers share is this array-selection rule, not the parts they keep from
// it: the sidecar's encoder fan-out primes every modality and drops a part with
// no fetchable URL, while the coordinator steps keep one image type and drop
// nothing, since they pair parts with multimodal entries by position. A caller
// that selected arrays for itself could disagree about which parts exist at
// all, which is the one thing none of them may do.
//
// Parts aliases the item it came from. A coordinator step writes a uuid or a
// rewritten URL through it; the sidecar decodes its own copy, where a write
// would reach nothing.
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
