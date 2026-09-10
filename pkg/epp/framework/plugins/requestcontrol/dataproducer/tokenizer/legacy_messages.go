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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

const (
	messagesRenderModeAuto   = "auto"
	messagesRenderModeLegacy = "legacy"
	messagesRenderModeNative = "native"
)

type legacyMessagesMode struct {
	name      string
	mode      string
	discovery chan struct{}
}

func configureLegacyMessages(ctx context.Context, name, mode string) (*legacyMessagesMode, error) {
	switch mode {
	case "", messagesRenderModeAuto:
		return &legacyMessagesMode{name: name, discovery: make(chan struct{}, 1)}, nil
	case messagesRenderModeLegacy:
		warnLegacyMessages(ctx, name)
		return &legacyMessagesMode{mode: mode}, nil
	case messagesRenderModeNative:
		return nil, nil //nolint:nilnil // Native rendering needs no compatibility state.
	default:
		return nil, fmt.Errorf("invalid vllm.messagesRenderMode %q: must be %q, %q or %q", mode, messagesRenderModeAuto, messagesRenderModeLegacy, messagesRenderModeNative)
	}
}

func warnLegacyMessages(ctx context.Context, name string) {
	log.FromContext(ctx).Info(
		"vllm.messagesRenderMode=legacy is deprecated and does not guarantee token parity; use native with a renderer supporting /v1/messages/render",
		"pluginName", name,
	)
}

func (m *legacyMessagesMode) useLegacy(ctx context.Context, tk tokenizer, model string) (bool, error) {
	if m == nil {
		return false, nil
	}
	if m.discovery == nil {
		return m.mode == messagesRenderModeLegacy, nil
	}
	// A waiting request must be able to cancel while another caller probes.
	select {
	case m.discovery <- struct{}{}:
		defer func() { <-m.discovery }()
	case <-ctx.Done():
		return false, ctx.Err()
	}
	if m.mode == "" {
		probe := fwkrh.PayloadMap{
			"model": model, "max_tokens": 1,
			"messages": []any{map[string]any{"role": "user", "content": "warmup"}},
		}
		mode := messagesRenderModeNative
		tokens, _, err := tk.RenderMessages(ctx, probe)
		if err != nil {
			var status *renderStatusError
			if !errors.As(err, &status) || (status.StatusCode != http.StatusNotFound && status.StatusCode != http.StatusMethodNotAllowed) {
				return false, fmt.Errorf("discover Messages rendering: %w", err)
			}
			// Confirm that the same model can render before selecting conversion.
			mode = messagesRenderModeLegacy
			tokens, _, err = tk.RenderChat(ctx, probe)
			if err != nil {
				return false, fmt.Errorf("discover legacy Messages rendering: %w", err)
			}
		}
		if len(tokens) == 0 {
			return false, errors.New("messages render discovery returned no tokens")
		}
		m.mode = mode
		if mode == messagesRenderModeLegacy {
			warnLegacyMessages(ctx, m.name)
		}
	}
	return m.mode == messagesRenderModeLegacy, nil
}

func (b renderBackend) renderLegacyMessages(ctx context.Context, msg *fwkrh.MessagesRequest) (*fwkrh.TokenizedRequest, error) {
	payload := legacyMessagesPayload(msg)
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

func legacyMessagesPayload(msg *fwkrh.MessagesRequest) fwkrh.PayloadMap {
	rr := buildChatRenderRequest(messagesToRenderChatRequest(msg))
	pm := fwkrh.PayloadMap{"messages": rr.Messages}
	if len(rr.Tools) > 0 {
		pm["tools"] = rr.Tools
	}
	return pm
}

func messagesToRenderChatRequest(msg *fwkrh.MessagesRequest) *tokenizerTypes.RenderChatRequest {
	conversation := make([]tokenizerTypes.Conversation, 0, 1+len(msg.Messages))

	if sys := anthropicSystemText(msg.System); sys != "" {
		conversation = append(conversation, tokenizerTypes.Conversation{
			Role:    "system",
			Content: &tokenizerTypes.Content{Raw: sys},
		})
	}

	for _, m := range msg.Messages {
		if m.Role == "system" {
			if text := anthropicSystemText(m.Content); text != "" {
				conversation = append(conversation, tokenizerTypes.Conversation{
					Role:    "system",
					Content: &tokenizerTypes.Content{Raw: text},
				})
			}
			continue
		}
		conversation = appendAnthropicMessage(conversation, m)
	}

	return &tokenizerTypes.RenderChatRequest{
		Conversation: conversation,
		Tools:        convertAnthropicTools(msg.Tools),
	}
}

// Tool replies must follow the assistant's tool calls in the Chat history.
func appendAnthropicMessage(conversation []tokenizerTypes.Conversation, m fwkrh.AnthropicMessage) []tokenizerTypes.Conversation {
	if m.Content.Raw != "" {
		return append(conversation, tokenizerTypes.Conversation{
			Role:    m.Role,
			Content: &tokenizerTypes.Content{Raw: m.Content.Raw},
		})
	}

	var contentBlocks []tokenizerTypes.ContentBlock
	var toolCalls []any
	var reasoning strings.Builder
	for _, b := range m.Content.Structured {
		switch b.Type {
		case blockTypeText:
			if b.Text != "" {
				contentBlocks = append(contentBlocks, tokenizerTypes.ContentBlock{Type: blockTypeText, Text: b.Text})
			}
		case blockTypeImage:
			contentBlocks = appendImageBlock(contentBlocks, b.Source)
		case blockTypeThinking:
			reasoning.WriteString(b.Thinking)
		case "redacted_thinking":
		case blockTypeToolUse:
			toolCalls = append(toolCalls, anthropicToolCall(b))
		case blockTypeToolResult:
			if m.Role == "user" {
				conversation = appendAnthropicToolResult(conversation, b)
			} else {
				text, _ := anthropicToolResultContent(b)
				contentBlocks = append(contentBlocks, tokenizerTypes.ContentBlock{
					Type: blockTypeText,
					Text: "Tool result: " + text,
				})
			}
		}
	}

	conv := tokenizerTypes.Conversation{Role: m.Role}
	if reasoning.Len() > 0 {
		conv.Reasoning = reasoning.String()
	}
	conv.ToolCalls = toolCalls
	switch {
	case len(contentBlocks) == 1 && contentBlocks[0].Type == blockTypeText:
		conv.Content = &tokenizerTypes.Content{Raw: contentBlocks[0].Text}
	case len(contentBlocks) > 0:
		conv.Content = &tokenizerTypes.Content{Structured: contentBlocks}
	}
	// Tool-result-only user messages are represented by their tool messages.
	if m.Role == "user" && conv.Content == nil {
		return conversation
	}
	return append(conversation, conv)
}

func anthropicToolCall(b fwkrh.AnthropicContentBlock) map[string]any {
	id := b.ID
	if id == "" {
		// A fixed stand-in preserves rendered length when an ID is absent.
		id = "call_0000000000"
	}
	return map[string]any{
		"id":   id,
		"type": "function",
		"function": map[string]any{
			"name":      b.Name,
			"arguments": pythonArguments(b.Input),
		},
	}
}

// CPython-formatted tool arguments become text in the rendered prompt.
func pythonArguments(raw json.RawMessage) string {
	switch string(bytes.TrimSpace(raw)) {
	case "", "null", "{}":
		return "{}"
	}
	if out, err := pythonDumps(raw); err == nil {
		return out
	}
	return "{}"
}

// Preserve legacy Chat role ordering for images in tool results.
func appendAnthropicToolResult(conversation []tokenizerTypes.Conversation, b fwkrh.AnthropicContentBlock) []tokenizerTypes.Conversation {
	text, imageBlocks := anthropicToolResultContent(b)
	conversation = append(conversation, tokenizerTypes.Conversation{
		Role:       "tool",
		ToolCallID: b.ToolUseID,
		Content:    &tokenizerTypes.Content{Raw: text},
	})
	if len(imageBlocks) > 0 {
		conversation = append(conversation, tokenizerTypes.Conversation{
			Role:    "user",
			Content: &tokenizerTypes.Content{Structured: imageBlocks},
		})
	}
	return conversation
}

// input_schema stays raw so serialization preserves the wire key order.
func convertAnthropicTools(tools []fwkrh.AnthropicTool) []any {
	if len(tools) == 0 {
		return nil
	}
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		var schema json.RawMessage = bytes.TrimSpace(t.InputSchema)
		if len(schema) == 0 || bytes.Equal(schema, []byte("null")) {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		fn := map[string]any{
			"name":       t.Name,
			"parameters": schema,
		}
		if t.Description != "" {
			fn["description"] = t.Description
		}
		if t.Strict != nil {
			fn["strict"] = *t.Strict
		}
		if t.DeferLoading != nil {
			fn["defer_loading"] = *t.DeferLoading
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

type chatRenderRequest struct {
	Messages []chatMessage `json:"messages"`
	Tools    []any         `json:"tools,omitempty"`
}

type chatMessage struct {
	Role       string       `json:"role"`
	Content    *chatContent `json:"content,omitempty"`
	ToolCalls  []any        `json:"tool_calls,omitempty"`
	Reasoning  string       `json:"reasoning,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
}

type chatContent struct {
	Raw   string
	Parts []chatPart
}

func (c chatContent) MarshalJSON() ([]byte, error) {
	if len(c.Parts) > 0 {
		return json.Marshal(c.Parts)
	}
	return json.Marshal(c.Raw)
}

type chatPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *chatImageURL `json:"image_url,omitempty"`
}

type chatImageURL struct {
	URL string `json:"url"`
}

func buildChatRenderRequest(req *tokenizerTypes.RenderChatRequest) chatRenderRequest {
	msgs := make([]chatMessage, len(req.Conversation))
	for idx, c := range req.Conversation {
		msgs[idx] = chatMessage{
			Role:       c.Role,
			Content:    toChatContent(c.Content),
			ToolCalls:  c.ToolCalls,
			Reasoning:  c.Reasoning,
			ToolCallID: c.ToolCallID,
		}
	}
	return chatRenderRequest{
		Messages: msgs,
		Tools:    req.Tools,
	}
}

func toChatContent(c *tokenizerTypes.Content) *chatContent {
	if c == nil {
		return nil
	}
	if len(c.Structured) == 0 {
		return &chatContent{Raw: c.Raw}
	}
	parts := make([]chatPart, 0, len(c.Structured))
	for _, b := range c.Structured {
		switch b.Type {
		case blockTypeText:
			parts = append(parts, chatPart{Type: blockTypeText, Text: b.Text})
		case blockTypeImageURL:
			parts = append(parts, chatPart{Type: blockTypeImageURL, ImageURL: &chatImageURL{URL: b.ImageURL.URL}})
		default:
		}
	}
	return &chatContent{Parts: parts}
}

func pythonDumps(raw json.RawMessage) (string, error) {
	var sb strings.Builder
	if err := dumpValue(&sb, bytes.TrimSpace(raw)); err != nil {
		return "", err
	}
	return sb.String(), nil
}

func dumpValue(sb *strings.Builder, raw []byte) error {
	switch {
	case len(raw) == 0:
		return errors.New("pythonDumps: empty JSON value")
	case raw[0] == '{':
		return dumpDelimited(sb, raw, '{')
	case raw[0] == '[':
		return dumpDelimited(sb, raw, '[')
	case raw[0] == '"':
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return fmt.Errorf("pythonDumps: decode string: %w", err)
		}
		writeJSONString(sb, s)
		return nil
	default:
		var n json.Number
		if string(raw) != "null" && string(raw) != "true" && string(raw) != "false" && json.Unmarshal(raw, &n) != nil {
			return fmt.Errorf("pythonDumps: invalid value %s", raw)
		}
		sb.Write(raw)
		return nil
	}
}

func dumpDelimited(sb *strings.Builder, raw []byte, open byte) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	if _, err := dec.Token(); err != nil {
		return fmt.Errorf("pythonDumps: decode start: %w", err)
	}
	closer, sep := '}', ": "
	if open == '[' {
		closer, sep = ']', ", "
	}
	sb.WriteByte(open)
	first := true
	for dec.More() {
		if !first {
			sb.WriteString(", ")
		}
		first = false
		if open == '{' {
			tok, err := dec.Token()
			if err != nil {
				return fmt.Errorf("pythonDumps: decode object key: %w", err)
			}
			key, ok := tok.(string)
			if !ok {
				return fmt.Errorf("pythonDumps: unexpected object key %v", tok)
			}
			writeJSONString(sb, key)
			sb.WriteString(sep)
		}
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return fmt.Errorf("pythonDumps: decode value: %w", err)
		}
		if err := dumpValue(sb, bytes.TrimSpace(val)); err != nil {
			return err
		}
	}
	sb.WriteRune(closer)
	return nil
}

func writeJSONString(sb *strings.Builder, s string) {
	sb.WriteByte('"')
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		switch {
		case r == '"':
			sb.WriteString(`\"`)
		case r == '\\':
			sb.WriteString(`\\`)
		case r == '\b':
			sb.WriteString(`\b`)
		case r == '\f':
			sb.WriteString(`\f`)
		case r == '\n':
			sb.WriteString(`\n`)
		case r == '\r':
			sb.WriteString(`\r`)
		case r == '\t':
			sb.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(sb, `\u%04x`, r)
		case r < utf8.RuneSelf:
			sb.WriteByte(byte(r))
		case r > 0xFFFF:
			r1, r2 := utf16.EncodeRune(r)
			fmt.Fprintf(sb, `\u%04x\u%04x`, r1, r2)
		default:
			fmt.Fprintf(sb, `\u%04x`, r)
		}
		i += size
	}
	sb.WriteByte('"')
}
