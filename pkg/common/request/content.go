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

import "fmt"

// MediaPartURLRef returns the URL a media content part references and a setter
// that writes a replacement back to the field it came from. set is nil when
// there is no readable URL: a part type that holds none, such as an inline
// input_audio part, or one whose URL field is absent or not a string.
//
// Where the URL lives differs by part type, and this is the only place that
// knows: chat-completions nests it at part[type]["url"], a Responses
// input_image holds it as a bare string at part["image_url"], and an Anthropic
// Messages image block holds a source object. A caller that rewrites a URL in
// place goes through set so it cannot write the wrong shape.
func MediaPartURLRef(part map[string]any) (url string, set func(string)) {
	switch partType, _ := part[FieldType].(string); partType {
	case PartTypeImageURL, PartTypeAudioURL, PartTypeVideoURL:
		nested, isMap := part[partType].(map[string]any)
		if !isMap {
			return "", nil
		}
		url, isString := nested[FieldURL].(string)
		if !isString {
			return "", nil
		}
		return url, func(v string) { nested[FieldURL] = v }
	case PartTypeInputImage:
		url, isString := part[FieldImageURL].(string)
		if !isString {
			return "", nil
		}
		return url, func(v string) { part[FieldImageURL] = v }
	case PartTypeImage:
		source, isMap := part[FieldSource].(map[string]any)
		if !isMap {
			return "", nil
		}
		return messagesImageSourceRef(source)
	}
	return "", nil
}

// messagesImageSourceRef reads an Anthropic Messages image source the way
// vLLM's Anthropic conversion does: a url source names its URL, and any other
// source is base64 data, read here as a data URL. A source with no data, such
// as a Files API reference, has nothing to fetch. set replaces either kind with
// a url source, which vLLM loads the same way when the URL is a data URL.
func messagesImageSourceRef(source map[string]any) (url string, set func(string)) {
	set = func(v string) {
		clear(source)
		source[FieldType] = ImageSourceTypeURL
		source[FieldURL] = v
	}
	if source[FieldType] == ImageSourceTypeURL {
		url, isString := source[FieldURL].(string)
		if !isString {
			return "", nil
		}
		return url, set
	}
	data, isString := source[FieldData].(string)
	if !isString || data == "" {
		return "", nil
	}
	mediaType, _ := source[FieldMediaType].(string)
	if mediaType == "" {
		mediaType = DefaultImageMediaType
	}
	return "data:" + mediaType + ";base64," + data, set
}

// MediaPartURL returns the URL a media content part references, or "" when
// there is none to fetch.
func MediaPartURL(part map[string]any) string {
	url, _ := MediaPartURLRef(part)
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
// A Messages turn follows vLLM's Anthropic conversion. A system turn keeps only
// its text, so it carries no array. A user turn's tool_result blocks become
// messages placed ahead of the turn's own content, so the content array of each
// comes first, in order. An assistant turn's tool_result becomes text.
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
	if apiType == APITypeMessages {
		return messagesPartArrays(item)
	}
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

func messagesPartArrays(item map[string]any) []PartArray {
	role, _ := item[FieldRole].(string)
	content, ok := item[FieldContent].([]any)
	if !ok || role == RoleSystem {
		return nil
	}
	var arrays []PartArray
	if role == RoleUser {
		for i, block := range content {
			blockMap, isMap := block.(map[string]any)
			if !isMap || blockMap[FieldType] != PartTypeToolResult {
				continue
			}
			if nested, isArray := blockMap[FieldContent].([]any); isArray {
				arrays = append(arrays, PartArray{Field: fmt.Sprintf("%s[%d].%s", FieldContent, i, FieldContent), Parts: nested})
			}
		}
	}
	return append(arrays, PartArray{Field: FieldContent, Parts: content})
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

// messagesEncoderPassthroughFields are the fields a Messages body forwards.
// vLLM's /v1/messages accepts neither kwargs field, so the prefiller hashes a
// Messages image at the deployment defaults and the encoder has to as well.
var messagesEncoderPassthroughFields = []string{FieldModel}

// NewEncoderPrimingBody builds a single-part encoder request: the passthrough
// fields off clientBody plus one synthetic user turn wrapping part, capped to a
// single output token. It builds from scratch because copying clientBody would
// hand the encoder fields its own API does not define.
//
// Responses and Messages bodies are built for their own API; anything else is
// treated as chat completions. part is forwarded unreshaped: it already has the
// shape of the API the caller posts the body under.
func NewEncoderPrimingBody(clientBody map[string]any, part map[string]any, apiType APIType) map[string]any {
	if apiType != APITypeResponses && apiType != APITypeMessages {
		apiType = APITypeChatCompletions
	}
	fields := encoderPassthroughFields
	if apiType == APITypeMessages {
		fields = messagesEncoderPassthroughFields
	}

	body := make(map[string]any, len(fields)+3)
	for _, field := range fields {
		if v, ok := clientBody[field]; ok {
			body[field] = v
		}
	}

	turn := map[string]any{FieldRole: RoleUser, FieldContent: []map[string]any{part}}
	if apiType == APITypeResponses {
		body[FieldInput] = []map[string]any{turn}
	} else {
		body[FieldMessages] = []map[string]any{turn}
	}

	CapSingleToken(body, apiType)

	return body
}
