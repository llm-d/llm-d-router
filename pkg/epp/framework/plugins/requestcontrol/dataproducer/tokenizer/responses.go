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
	"net/http"

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	tokenizerTypes "github.com/llm-d/llm-d-router/pkg/kvcache/tokenization/types"
)

// responsesItemTypeMessage is the Input item "type" discriminator this code
// converts; other discriminators (function_call, function_call_output,
// reasoning, and so on) are left for a follow-up.
const responsesItemTypeMessage = "message"

const (
	responsesRenderModeAuto   = "auto"
	responsesRenderModeLegacy = "legacy"
	responsesRenderModeNative = "native"
)

// legacyResponsesMode tracks whether the render endpoint speaks
// /v1/responses/render natively, discovering it once (auto) or honoring an
// explicit override (native/legacy). A nil *legacyResponsesMode means native
// rendering with no compatibility state, matching legacyMessagesMode.
type legacyResponsesMode struct {
	name      string
	mode      string
	discovery chan struct{}
}

// configureLegacyResponses returns the Responses rendering mode tracker for
// mode ("", "auto", "native", or "legacy").
func configureLegacyResponses(ctx context.Context, name, mode string) (*legacyResponsesMode, error) {
	switch mode {
	case "", responsesRenderModeAuto:
		return &legacyResponsesMode{name: name, discovery: make(chan struct{}, 1)}, nil
	case responsesRenderModeLegacy:
		warnLegacyResponses(ctx, name)
		return &legacyResponsesMode{mode: mode}, nil
	case responsesRenderModeNative:
		return nil, nil //nolint:nilnil // Native rendering needs no compatibility state.
	default:
		return nil, fmt.Errorf("invalid vllm.responsesRenderMode %q: must be %q, %q or %q",
			mode, responsesRenderModeAuto, responsesRenderModeLegacy, responsesRenderModeNative)
	}
}

func warnLegacyResponses(ctx context.Context, name string) {
	log.FromContext(ctx).Info(
		"vllm.responsesRenderMode=legacy is deprecated and does not guarantee token parity; use native with a renderer supporting /v1/responses/render",
		"pluginName", name,
	)
}

// useLegacy reports whether Responses rendering should use the legacy
// chat-completions translation instead of the native /v1/responses/render
// endpoint, discovering and caching the answer on first use. Mirrors
// legacyMessagesMode.useLegacy.
func (m *legacyResponsesMode) useLegacy(ctx context.Context, tk tokenizer, model string) (bool, error) {
	if m == nil {
		return false, nil
	}
	if m.discovery == nil {
		return m.mode == responsesRenderModeLegacy, nil
	}
	// A waiting request must be able to cancel while another caller probes.
	select {
	case m.discovery <- struct{}{}:
		defer func() { <-m.discovery }()
	case <-ctx.Done():
		return false, ctx.Err()
	}
	if m.mode == "" {
		mode := responsesRenderModeNative
		tokens, _, err := tk.RenderResponses(ctx, fwkrh.PayloadMap{
			"model": model, "max_tokens": 1, "input": "warmup",
		})
		if err != nil {
			var status *renderStatusError
			if !errors.As(err, &status) || (status.StatusCode != http.StatusNotFound && status.StatusCode != http.StatusMethodNotAllowed) {
				return false, fmt.Errorf("discover Responses rendering: %w", err)
			}
			// The legacy path renders through the chat-completions endpoint, whose
			// payload shape differs from the native probe above (messages, not input).
			mode = responsesRenderModeLegacy
			legacyProbe, perr := responsesPayload(&fwkrh.ResponsesRequest{Input: "warmup"})
			if perr != nil {
				return false, fmt.Errorf("discover legacy Responses rendering: %w", perr)
			}
			legacyProbe["model"] = model
			legacyProbe["max_tokens"] = 1
			tokens, _, err = tk.RenderChat(ctx, legacyProbe)
			if err != nil {
				return false, fmt.Errorf("discover legacy Responses rendering: %w", err)
			}
		}
		if len(tokens) == 0 {
			return false, errors.New("responses render discovery returned no tokens")
		}
		m.mode = mode
		if mode == responsesRenderModeLegacy {
			warnLegacyResponses(ctx, m.name)
		}
	}
	return m.mode == responsesRenderModeLegacy, nil
}

// renderLegacyResponses reshapes a /v1/responses body into a chat-completions
// render call, for a render endpoint without /v1/responses/render. The
// reshape covers string Input, Input items that are simple {role, content}
// messages, and Instructions as a leading system message; other Input item
// kinds (for example function_call, function_call_output, reasoning) are
// left for a follow-up rather than tokenized incorrectly here.
func (b renderBackend) renderLegacyResponses(ctx context.Context, r *fwkrh.ResponsesRequest) (*fwkrh.TokenizedRequest, error) {
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
// parts; only text parts are converted, using their "text" field. Items
// carrying any other "type" (function_call, function_call_output, reasoning,
// and so on), or whose content yields no text, are skipped rather than
// guessed at.
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
// type (input_image, refusal, and so on) are skipped rather than guessed at.
var responsesContentTextTypes = map[string]bool{
	"input_text":  true,
	"output_text": true,
	blockTypeText: true,
}

// responsesContent converts a Responses input item's "content" field into
// chat-completions Content. A plain string passes through as Raw. An array
// of parts keeps only text parts, converting their "type" to blockTypeText;
// a single resulting text part collapses to Raw, matching the plain-string
// case. Returns false when content is neither shape, or an array yields no
// text parts.
func responsesContent(raw any) (*tokenizerTypes.Content, bool) {
	switch v := raw.(type) {
	case string:
		return &tokenizerTypes.Content{Raw: v}, true
	case []any:
		var blocks []tokenizerTypes.ContentBlock
		for _, part := range v {
			p, ok := part.(map[string]any)
			if !ok {
				continue
			}
			partType, _ := p["type"].(string)
			if !responsesContentTextTypes[partType] {
				continue
			}
			text, ok := p["text"].(string)
			if !ok {
				continue
			}
			blocks = append(blocks, tokenizerTypes.ContentBlock{Type: blockTypeText, Text: text})
		}
		if len(blocks) == 0 {
			return nil, false
		}
		if len(blocks) == 1 {
			return &tokenizerTypes.Content{Raw: blocks[0].Text}, true
		}
		return &tokenizerTypes.Content{Structured: blocks}, true
	default:
		return nil, false
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
