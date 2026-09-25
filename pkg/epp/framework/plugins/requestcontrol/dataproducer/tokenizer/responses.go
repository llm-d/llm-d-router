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
	"context"
	"errors"
	"fmt"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

// responsesItemTypeMessage is the Input item "type" discriminator this code
// converts; other discriminators (function_call, function_call_output,
// reasoning, and so on) are left for a follow-up.
const responsesItemTypeMessage = "message"

// renderResponses reshapes a /v1/responses body into a chat-completions render
// call, since vLLM has no /v1/responses/render endpoint yet. The reshape covers
// string Input, Input items that are simple {role, content} messages, and
// Instructions as a leading system message; other Input item kinds (for
// example function_call, function_call_output, reasoning) are left for a
// follow-up rather than tokenized incorrectly here.
func (b renderBackend) renderResponses(ctx context.Context, r *fwkrh.ResponsesRequest) (*fwkrh.TokenizedRequest, error) {
	payload, err := responsesPayload(r)
	if err != nil {
		return nil, err
	}
	payload["model"] = b.modelName
	tokenIDs, mmFeatures, err := b.tk.RenderChat(ctx, payload)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}
	return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{
		TokenIDs:           tokenIDs,
		MultiModalFeatures: convertMMFeaturesToUpstream(mmFeatures),
	}}}, nil
}

func responsesPayload(r *fwkrh.ResponsesRequest) (fwkrh.PayloadMap, error) {
	conversation := responsesToConversation(r)
	if len(conversation) == 0 {
		return nil, errors.New("responses request has no renderable input")
	}
	rr := buildChatRenderRequest(&tokenizerTypes.RenderChatRequest{
		Conversation: conversation,
		Tools:        convertResponsesTools(r.Tools),
	})
	pm := fwkrh.PayloadMap{"messages": rr.Messages}
	if len(rr.Tools) > 0 {
		pm["tools"] = rr.Tools
	}
	return pm, nil
}

func responsesToConversation(r *fwkrh.ResponsesRequest) []tokenizerTypes.Conversation {
	var conversation []tokenizerTypes.Conversation
	if sys, ok := r.Instructions.(string); ok && sys != "" {
		conversation = append(conversation, tokenizerTypes.Conversation{
			Role:    "system",
			Content: &tokenizerTypes.Content{Raw: sys},
		})
	}
	return append(conversation, responsesInputToConversation(r.Input)...)
}

// responsesInputToConversation converts the Input field. Input is a plain
// string, or an array of items; only items with string or text-part-array
// content pass through, matching the render endpoint's chat message shape.
func responsesInputToConversation(input any) []tokenizerTypes.Conversation {
	switch v := input.(type) {
	case string:
		if v == "" {
			return nil
		}
		return []tokenizerTypes.Conversation{{Role: "user", Content: &tokenizerTypes.Content{Raw: v}}}
	case []any:
		var out []tokenizerTypes.Conversation
		for _, item := range v {
			if conv, ok := simpleResponsesMessage(item); ok {
				out = append(out, conv)
			}
		}
		return out
	default:
		return nil
	}
}

// simpleResponsesMessage recognizes a Responses input item shaped like a plain
// chat message: {"role": ..., "content": ...}, with an optional
// "type": "message". Content is either a string or an array of content
// parts; text and image parts are converted (see responsesContentPart).
// Items carrying any other "type" (function_call, function_call_output,
// reasoning, and so on), or whose content yields no recognized parts, are
// skipped rather than guessed at.
func simpleResponsesMessage(item any) (tokenizerTypes.Conversation, bool) {
	m, ok := item.(map[string]any)
	if !ok {
		return tokenizerTypes.Conversation{}, false
	}
	if t, ok := m["type"].(string); ok && t != "" && t != responsesItemTypeMessage {
		return tokenizerTypes.Conversation{}, false
	}
	role, ok := m["role"].(string)
	if !ok || role == "" {
		return tokenizerTypes.Conversation{}, false
	}
	content, ok := responsesContent(m["content"])
	if !ok {
		return tokenizerTypes.Conversation{}, false
	}
	return tokenizerTypes.Conversation{Role: role, Content: content}, true
}

// responsesContentTextTypes are the Responses content-part "type" values
// this code converts to a chat-completions text block. Parts of any other
// recognized type convert differently (see responsesContentPart); an
// unrecognized type is skipped rather than guessed at.
var responsesContentTextTypes = map[string]bool{
	"input_text":  true,
	"output_text": true,
	blockTypeText: true,
}

// responsesContent converts a Responses input item's "content" field into
// chat-completions Content. A plain string passes through as Raw. An array
// of parts converts each recognized part (see responsesContentPart); a
// single resulting text part collapses to Raw, matching the plain-string
// case, and multiple parts (or a single image part) use Structured. Returns
// false when content is neither shape, or an array yields no recognized
// parts.
func responsesContent(raw any) (*tokenizerTypes.Content, bool) {
	switch v := raw.(type) {
	case string:
		return &tokenizerTypes.Content{Raw: v}, true
	case []any:
		var blocks []tokenizerTypes.ContentBlock
		for _, part := range v {
			if block, ok := responsesContentPart(part); ok {
				blocks = append(blocks, block)
			}
		}
		if len(blocks) == 0 {
			return nil, false
		}
		if len(blocks) == 1 && blocks[0].Type == blockTypeText {
			return &tokenizerTypes.Content{Raw: blocks[0].Text}, true
		}
		return &tokenizerTypes.Content{Structured: blocks}, true
	default:
		return nil, false
	}
}

// responsesContentPart converts one content part. input_text/output_text
// parts convert to a text block using their "text" field. input_image parts
// convert to an image block using their "image_url" field, which carries the
// URL as a bare string, unlike chat completions' nested
// {"image_url": {"url": ...}} shape. Any other type is skipped rather than
// guessed at.
func responsesContentPart(part any) (tokenizerTypes.ContentBlock, bool) {
	p, ok := part.(map[string]any)
	if !ok {
		return tokenizerTypes.ContentBlock{}, false
	}
	partType, _ := p["type"].(string)
	switch {
	case responsesContentTextTypes[partType]:
		text, ok := p["text"].(string)
		if !ok {
			return tokenizerTypes.ContentBlock{}, false
		}
		return tokenizerTypes.ContentBlock{Type: blockTypeText, Text: text}, true
	case partType == "input_image":
		url, ok := p["image_url"].(string)
		if !ok || url == "" {
			return tokenizerTypes.ContentBlock{}, false
		}
		return tokenizerTypes.ContentBlock{Type: blockTypeImageURL, ImageURL: tokenizerTypes.ImageBlock{URL: url}}, true
	default:
		return tokenizerTypes.ContentBlock{}, false
	}
}

// convertResponsesTools reshapes the Responses API's flat tool shape
// ({"type": "function", "name": ..., "description": ..., "parameters": ...})
// into the chat-completions shape ({"type": "function", "function": {...}})
// that the render endpoint's chat template expects. An entry that already
// carries a "function" object, or whose type is not "function", passes
// through unchanged.
func convertResponsesTools(tools any) []any {
	list, ok := tools.([]any)
	if !ok || len(list) == 0 {
		return nil
	}
	out := make([]any, 0, len(list))
	for _, t := range list {
		out = append(out, convertResponsesTool(t))
	}
	return out
}

func convertResponsesTool(t any) any {
	m, ok := t.(map[string]any)
	if !ok {
		return t
	}
	if _, alreadyNested := m["function"]; alreadyNested {
		return t
	}
	if kind, _ := m["type"].(string); kind != "function" {
		return t
	}
	fn := map[string]any{}
	for _, k := range []string{"name", "description", "parameters"} {
		if v, ok := m[k]; ok {
			fn[k] = v
		}
	}
	return map[string]any{"type": "function", "function": fn}
}
