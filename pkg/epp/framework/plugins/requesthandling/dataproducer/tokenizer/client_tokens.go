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

package tokenizer

import (
	"encoding/json"
	"math"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

// clientPromptTokenIDs returns the prompt token IDs a request already carries,
// so the backend does not render the prompt again: a token-array completions
// prompt, or kv_transfer_params.prompt_token_ids on the APIs vLLM's chat path
// serves, which consumes the field in place of templating and tokenizing the
// messages. Returns nil when the request carries none, and when it carries
// media, since vLLM then ignores the field and renders the messages.
func clientPromptTokenIDs(body *fwkrequest.InferenceRequestBody) [][]uint32 {
	if body.Completions != nil {
		// The completions path reads the prompt; it does not read
		// kv_transfer_params.prompt_token_ids, so a text prompt must be rendered.
		if len(body.Completions.Prompt.TokenIDs) > 0 {
			return body.Completions.Prompt.TokenIDs
		}
		return nil
	}
	if !chatPathTextOnly(body) {
		return nil
	}
	payload, ok := body.Payload.(fwkrequest.PayloadMap)
	if !ok {
		return nil
	}
	var rawIDs any
	switch v := payload[reqcommon.FieldKVTransferParams].(type) {
	case map[string]any:
		rawIDs = v[reqcommon.FieldPromptTokenIDs]
	case json.RawMessage:
		// Decoding into raw values keeps the id array as bytes for
		// ParseCanonicalTokenIDs, rather than one json.Number per id.
		var params map[string]json.RawMessage
		if err := parserutil.Unmarshal(v, &params); err != nil {
			return nil
		}
		if ids, ok := params[reqcommon.FieldPromptTokenIDs]; ok {
			rawIDs = ids
		}
	default:
		return nil
	}
	if ids := parseTokenIDList(rawIDs); ids != nil {
		return [][]uint32{ids}
	}
	return nil
}

// chatPathTextOnly reports whether the request is one vLLM's chat path serves
// (chat completions, Messages, Responses) and names no media. vLLM renders the
// messages instead of using prompt_token_ids when any content part is not text,
// and only a render yields the multimodal features downstream plugins read.
func chatPathTextOnly(body *fwkrequest.InferenceRequestBody) bool {
	switch {
	case body.ChatCompletions != nil:
		for _, msg := range body.ChatCompletions.Messages {
			for _, block := range msg.Content.Structured {
				if !isTextPartType(block.Type) {
					return false
				}
			}
		}
		return true
	case body.Messages != nil:
		for _, msg := range body.Messages.Messages {
			for _, block := range msg.Content.Structured {
				if _, ok := reqcommon.MessagesBlockModality(block.Type); ok {
					return false
				}
				if block.Type == reqcommon.BlockTypeToolResult {
					for _, nested := range block.ToolResultContent().Structured {
						if _, ok := reqcommon.MessagesBlockModality(nested.Type); ok {
							return false
						}
					}
				}
			}
		}
		return true
	case body.Responses != nil:
		items, _ := body.Responses.Input.([]any)
		for _, item := range items {
			itemMap, ok := item.(map[string]any)
			if !ok {
				continue
			}
			for _, array := range reqcommon.ItemPartArrays(itemMap, reqcommon.APITypeResponses) {
				for _, part := range array.Parts {
					partMap, ok := part.(map[string]any)
					if !ok {
						continue
					}
					if partType, _ := partMap[reqcommon.FieldType].(string); !isTextPartType(partType) {
						return false
					}
				}
			}
		}
		return true
	}
	return false
}

// isTextPartType reports whether a chat or Responses content part of this type
// is parsed as text. The ContentBlock decoder types a part without a type by its
// media field, so an untyped media part does not read as text here.
func isTextPartType(partType string) bool {
	switch partType {
	case reqcommon.PartTypeText, reqcommon.PartTypeInputText, reqcommon.PartTypeOutputText, reqcommon.PartTypeRefusal, blockTypeThinking:
		return true
	}
	return false
}

// parseTokenIDList converts a prompt_token_ids value to token IDs. The value is
// raw JSON when the envelope kept kv_transfer_params opaque, a decoded list
// whose numbers are json.Number or float64, or a []uint32 written by a
// mutation. Returns nil for an absent, empty, or malformed value, so
// tokenization proceeds normally.
func parseTokenIDList(value any) []uint32 {
	switch v := value.(type) {
	case []uint32:
		if len(v) > 0 {
			return v
		}
	case json.RawMessage:
		if ids, ok := fwkrequest.ParseCanonicalTokenIDs(v); ok {
			return ids
		}
		var decoded []any
		if err := parserutil.Unmarshal(v, &decoded); err != nil {
			return nil
		}
		return parseTokenIDList(decoded)
	case []any:
		ids := make([]uint32, 0, len(v))
		for _, item := range v {
			id, ok := tokenID(item)
			if !ok {
				return nil
			}
			ids = append(ids, id)
		}
		if len(ids) > 0 {
			return ids
		}
	}
	return nil
}

// tokenID converts one decoded JSON number to a token ID.
func tokenID(v any) (uint32, bool) {
	var f float64
	switch n := v.(type) {
	case json.Number:
		parsed, err := n.Float64()
		if err != nil {
			return 0, false
		}
		f = parsed
	case float64:
		f = n
	default:
		return 0, false
	}
	if f < 0 || f != math.Trunc(f) || f > math.MaxUint32 {
		return 0, false
	}
	return uint32(f), true
}
