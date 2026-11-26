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

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

// clientPromptTokenIDs returns the prompt token IDs a request already carries:
// a token-array completions prompt, a token-array embeddings input, or
// kv_transfer_params.prompt_token_ids on a text-only chat-path request (see
// chatPathTextOnly). Returns nil otherwise, when truncate_prompt_tokens is set
// on completions or embeddings, or when echo is set, since vLLM then transforms
// or renders the input instead of using the raw IDs. When the request carries
// IDs that are not used, ignored names the reason.
func clientPromptTokenIDs(body *fwkrequest.InferenceRequestBody) (ids [][]uint32, ignored string) {
	if body.Completions != nil {
		// The completions path reads the prompt; it does not read
		// kv_transfer_params.prompt_token_ids, so a text prompt must be rendered.
		if len(body.Completions.Prompt.TokenIDs) == 0 {
			return nil, ""
		}
		// vLLM truncates the prompt when truncate_prompt_tokens is set, so only a
		// render yields the tokens the engine serves.
		if hasTruncatePromptTokens(body) {
			return nil, "truncate_prompt_tokens is set"
		}
		return body.Completions.Prompt.TokenIDs, ""
	}
	if body.Embeddings != nil {
		if len(body.Embeddings.Input.TokenIDs) == 0 {
			return nil, ""
		}
		if hasTruncatePromptTokens(body) {
			return nil, "truncate_prompt_tokens is set"
		}
		return body.Embeddings.Input.TokenIDs, ""
	}
	payload, ok := body.Payload.(fwkrequest.PayloadMap)
	if !ok {
		return nil, ""
	}
	rawIDs, present := kvPromptTokenIDs(payload)
	switch {
	case !present:
		return nil, ""
	case !chatPathTextOnly(body):
		return nil, "request is outside the chat path or carries media"
	}
	if echo, _ := payload[reqcommon.FieldEcho].(bool); echo {
		return nil, "echo is set"
	}
	if parsed := parseTokenIDList(rawIDs); parsed != nil {
		return [][]uint32{parsed}, ""
	}
	return nil, "malformed or empty"
}

// hasTruncatePromptTokens reports whether the request payload carries a
// non-null truncate_prompt_tokens parameter.
func hasTruncatePromptTokens(body *fwkrequest.InferenceRequestBody) bool {
	payload, ok := body.Payload.(fwkrequest.PayloadMap)
	return ok && payload[reqcommon.FieldTruncatePromptTokens] != nil
}

// kvPromptTokenIDs returns the undecoded kv_transfer_params.prompt_token_ids
// value and whether the field is present.
func kvPromptTokenIDs(payload fwkrequest.PayloadMap) (any, bool) {
	switch v := payload[reqcommon.FieldKVTransferParams].(type) {
	case map[string]any:
		ids, ok := v[reqcommon.FieldPromptTokenIDs]
		return ids, ok
	case json.RawMessage:
		// Decode to raw values so the id array stays as bytes for
		// ParseCanonicalTokenIDs.
		var params map[string]json.RawMessage
		if err := parserutil.Unmarshal(v, &params); err != nil {
			return nil, false
		}
		ids, ok := params[reqcommon.FieldPromptTokenIDs]
		return ids, ok
	}
	return nil, false
}

// chatPathTextOnly reports whether the request is one vLLM's chat path serves
// (chat completions, Messages, Responses) and carries no media part. vLLM's chat
// path consumes prompt_token_ids in place of templating and tokenizing the
// messages, but renders them when a part carries media, and only a render
// yields the multimodal features downstream plugins read.
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
// counts as text for the prompt_token_ids check. The set matches vLLM's
// TEXT_PART_TYPES. The ContentBlock decoder types an untyped part by its media
// field, so an untyped media part is not text here.
func isTextPartType(partType string) bool {
	switch partType {
	case reqcommon.PartTypeText, reqcommon.PartTypeInputText, reqcommon.PartTypeOutputText, reqcommon.PartTypeRefusal, blockTypeThinking:
		return true
	}
	return false
}

// parseTokenIDList converts a prompt_token_ids value to token IDs. The value is
// raw JSON when the envelope kept kv_transfer_params opaque, a decoded list
// whose numbers are json.Number or float64, or a []uint32 a plugin set
// in-process on the payload map. Returns nil for an absent, empty, or malformed value, so
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
		ids, err := reqcommon.ParseTokenIDs(v)
		if err != nil {
			return nil
		}
		if len(ids) > 0 {
			return ids
		}
	}
	return nil
}
