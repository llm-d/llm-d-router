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

package requesthandling

import (
	"io"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// PrefixWriter receives the conversation prefix, typically a hash digest.
type PrefixWriter interface {
	io.Writer
	io.StringWriter
}

// prefixFieldSeparator ends each written field so adjacent fields cannot run together.
var prefixFieldSeparator = []byte{0}

// WriteConversationPrefix writes the cache salt and the text conversation up to
// and including its first user message. Every later turn of a session repeats
// that prefix, while sessions that share only a system prompt differ. It
// returns false when the request type has no turn boundary, when no text user
// message is present, or when the prefix contains non-text content; w then
// holds partial output and must be discarded.
func (b *InferenceRequestBody) WriteConversationPrefix(w PrefixWriter) bool {
	switch {
	case b == nil:
		return false
	case b.ChatCompletions != nil:
		return b.ChatCompletions.WriteConversationPrefix(w)
	case b.Messages != nil:
		return b.Messages.WriteConversationPrefix(w)
	case b.Responses != nil:
		return b.Responses.WriteConversationPrefix(w)
	default:
		// Completions and generate prompts carry no turn boundary, so a
		// fixed-length prefix would collapse every session behind a long
		// shared preamble onto one key. The remaining types are single-shot.
		return false
	}
}

// WriteConversationPrefix implements InferenceRequestBody.WriteConversationPrefix
// for chat completions. A message carrying tool calls ends the walk.
func (r *ChatCompletionsRequest) WriteConversationPrefix(w PrefixWriter) bool {
	writePrefixField(w, r.CacheSalt)
	for _, message := range r.Messages {
		if len(message.ToolCalls) > 0 {
			return false
		}
		writePrefixField(w, message.Role)
		hasText, ok := writeChatPrefixContent(w, message.Content)
		if !ok {
			return false
		}
		if message.Role == reqcommon.RoleUser {
			return hasText
		}
	}
	return false
}

// WriteConversationPrefix implements InferenceRequestBody.WriteConversationPrefix
// for Anthropic messages. The top-level system prompt precedes the messages.
func (r *MessagesRequest) WriteConversationPrefix(w PrefixWriter) bool {
	writePrefixField(w, r.CacheSalt)
	if _, ok := writeAnthropicPrefixContent(w, r.System); !ok {
		return false
	}
	for _, message := range r.Messages {
		writePrefixField(w, message.Role)
		hasText, ok := writeAnthropicPrefixContent(w, message.Content)
		if !ok {
			return false
		}
		if message.Role == reqcommon.RoleUser {
			return hasText
		}
	}
	return false
}

// WriteConversationPrefix implements InferenceRequestBody.WriteConversationPrefix
// for Responses. String instructions precede the input, and a string input is
// one user message, so a first turn sent as a string and a later turn sent as
// items write the same prefix. A request referencing server-side state omits
// earlier turns, and any non-message item ends the walk.
func (r *ResponsesRequest) WriteConversationPrefix(w PrefixWriter) bool {
	if r.PreviousResponseID != "" || r.Conversation != nil {
		return false
	}
	writePrefixField(w, r.CacheSalt)
	switch instructions := r.Instructions.(type) {
	case nil:
		writePrefixField(w, "")
	case string:
		writePrefixField(w, instructions)
	default:
		return false
	}
	switch input := r.Input.(type) {
	case string:
		writePrefixField(w, reqcommon.RoleUser)
		writePrefixField(w, input)
		return input != ""
	case []any:
		for _, item := range input {
			fields, ok := item.(map[string]any)
			if !ok {
				return false
			}
			if itemType, _ := fields[reqcommon.FieldType].(string); itemType != "" && itemType != reqcommon.ResponsesItemTypeMessage {
				return false
			}
			role, _ := fields[reqcommon.FieldRole].(string)
			if role == "" {
				return false
			}
			writePrefixField(w, role)
			hasText, ok := writeResponsesPrefixContent(w, fields[reqcommon.FieldContent])
			if !ok {
				return false
			}
			if role == reqcommon.RoleUser {
				return hasText
			}
		}
		return false
	default:
		return false
	}
}

func writePrefixField(w PrefixWriter, s string) {
	_, _ = w.WriteString(s)
	_, _ = w.Write(prefixFieldSeparator)
}

func writeChatPrefixContent(w PrefixWriter, content Content) (hasText, ok bool) {
	hasText = content.Raw != ""
	_, _ = w.WriteString(content.Raw)
	for _, block := range content.Structured {
		if block.Type != reqcommon.PartTypeText {
			return false, false
		}
		_, _ = w.WriteString(block.Text)
		hasText = hasText || block.Text != ""
	}
	_, _ = w.Write(prefixFieldSeparator)
	return hasText, true
}

func writeAnthropicPrefixContent(w PrefixWriter, content AnthropicContent) (hasText, ok bool) {
	hasText = content.Raw != ""
	_, _ = w.WriteString(content.Raw)
	for _, block := range content.Structured {
		if block.Type != reqcommon.PartTypeText {
			return false, false
		}
		_, _ = w.WriteString(block.Text)
		hasText = hasText || block.Text != ""
	}
	_, _ = w.Write(prefixFieldSeparator)
	return hasText, true
}

func writeResponsesPrefixContent(w PrefixWriter, content any) (hasText, ok bool) {
	switch content := content.(type) {
	case string:
		writePrefixField(w, content)
		return content != "", true
	case []any:
		for _, part := range content {
			fields, ok := part.(map[string]any)
			if !ok {
				return false, false
			}
			switch partType, _ := fields[reqcommon.FieldType].(string); partType {
			case reqcommon.PartTypeText, reqcommon.PartTypeInputText, reqcommon.PartTypeOutputText:
			default:
				return false, false
			}
			text, ok := fields[reqcommon.FieldText].(string)
			if !ok {
				return false, false
			}
			_, _ = w.WriteString(text)
			hasText = hasText || text != ""
		}
		_, _ = w.Write(prefixFieldSeparator)
		return hasText, true
	default:
		return false, false
	}
}
