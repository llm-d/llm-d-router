/*
Copyright 2025 The Kubernetes Authors.

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

package payload

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// Attribute and event names. The gen_ai.* names are the upstream GenAI
// semantic-convention attributes (Opt-In, Development stability); the
// llm_d.payload.* names are llm-d extensions defined in the proposal, kept
// outside the reserved gen_ai.* namespace. The GenAI conventions live in
// the dedicated semantic-conventions-genai repo:
// https://github.com/open-telemetry/semantic-conventions-genai .
const (
	// EventInferenceDetails is the upstream GenAI event that carries payload
	// attributes: gen_ai.client.inference.operation.details (still at
	// opt_in / development stability as of 2026-08).
	EventInferenceDetails = "gen_ai.client.inference.operation.details"

	// AttrInputMessages is gen_ai.input.messages: the structured request
	// messages (role/parts), serialised to JSON when recorded on a span
	// event. Value follows the ChatMessage schema (model/gen-ai/
	// gen-ai-input-messages.json in the semconv-genai repo).
	AttrInputMessages = "gen_ai.input.messages"
	// AttrSystemInstructions is gen_ai.system_instructions: the current
	// authoritative schema (model/gen-ai/gen-ai-system-instructions.json)
	// defines this as an array of TextPart / GenericPart, NOT a plain
	// instruction string. Same JSON-serialised part shape as
	// gen_ai.input.messages, with a narrower part-type vocabulary.
	AttrSystemInstructions = "gen_ai.system_instructions"

	// AttrTruncated is llm_d.payload.truncated (llm-d extension): true when
	// content was dropped or truncated during capture.
	AttrTruncated = "llm_d.payload.truncated"
)

// Part types and modalities from the upstream input-messages schema:
// https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/gen-ai-input-messages.json
// Only the part types llm-d emits are named here; other schema types
// (BlobPart, FilePart, ServerToolCallPart, CompactionPart, GenericPart)
// are recognised implicitly via the Part struct's optional fields.
const (
	partTypeText             = "text"
	partTypeURI              = "uri"
	partTypeToolCall         = "tool_call"
	partTypeToolCallResponse = "tool_call_response"
	partTypeReasoning        = "reasoning"

	modalityImage = "image"
	modalityVideo = "video"
)

// blockTypeText is the "text" content-block type shared by the OpenAI and
// Anthropic request schemas.
const blockTypeText = "text"

// chatMessage mirrors one entry of the upstream gen_ai.input.messages schema.
type chatMessage struct {
	Role  string `json:"role"`
	Parts []part `json:"parts"`
}

// part mirrors the upstream message-part schema. Field sets are disjoint
// by Type:
//   - text / reasoning: Content
//   - uri:              URI, MimeType, Modality
//   - tool_call:        ID, Name, Arguments
//   - tool_call_response: ID, Response
//
// Arguments and Response are raw JSON so the model's own key order survives
// downstream re-serialization (matches the upstream ToolCallRequestPart /
// ToolCallResponsePart definitions).
type part struct {
	Type      string          `json:"type"`
	Content   string          `json:"content,omitempty"`
	URI       string          `json:"uri,omitempty"`
	MimeType  string          `json:"mime_type,omitempty"`
	Modality  string          `json:"modality,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	Response  json.RawMessage `json:"response,omitempty"`
}

// extraction is the intermediate result of converting a parsed request body
// into semantic-convention form. Both messages and system carry the same
// TextPart / GenericPart structured shape that the semconv schema requires;
// system is emitted as gen_ai.system_instructions, messages as
// gen_ai.input.messages.
type extraction struct {
	messages []chatMessage
	system   []part
	// truncated is set when content was dropped: inline blob parts (no offload
	// backend exists in Phase 1), non-textual prompts (token IDs), or opaque
	// tool-call payloads (OpenAI assistant tool_calls, which the parser
	// currently exposes as []any and we can't semantic-extract without a
	// bespoke re-parse).
	truncated bool
}

func (e extraction) empty() bool {
	return len(e.messages) == 0 && len(e.system) == 0 && !e.truncated
}

// appendSystemParts adds parts to ext.system, filtering to the part-type
// vocabulary the SystemInstructions schema allows (TextPart today; the
// schema also permits GenericPart for forward-compat, but llm-d emits only
// text at that carrier). Non-text parts mark the event truncated so
// consumers can tell a media / tool part was dropped rather than never sent.
func appendSystemParts(ext *extraction, parts []part) {
	for _, p := range parts {
		if p.Type == partTypeText {
			ext.system = append(ext.system, p)
			continue
		}
		ext.truncated = true
	}
}

// Capturer records request payloads as GenAI span events on the active
// gateway span. A nil *Capturer is valid and captures nothing.
type Capturer struct {
	store  PayloadStore
	logger logr.Logger
}

// NewCapturer returns a Capturer for the given configuration, or nil when the
// configuration produces no events (capture disabled, or the noop backend,
// which is the proposal's secondary kill switch).
func NewCapturer(cfg Config, logger logr.Logger) *Capturer {
	if !cfg.Enabled || cfg.Backend != BackendInline {
		return nil
	}
	return &Capturer{
		store:  InlineStore{MaxBytes: cfg.InlineSizeThresholdBytes},
		logger: logger.WithName("payload-capture"),
	}
}

// NewCapturerFromEnv builds a Capturer from the LLMD_PAYLOAD_* environment
// variables; nil when capture is disabled.
func NewCapturerFromEnv(logger logr.Logger) *Capturer {
	cfg := LoadConfigFromEnv(logger)
	c := NewCapturer(cfg, logger)
	if c != nil {
		logger.Info("GenAI payload capture enabled",
			"backend", cfg.Backend, "inlineSizeThresholdBytes", cfg.InlineSizeThresholdBytes)
	}
	return c
}

// CaptureRequest records the request payload as a GenAI span event on the
// span carried by ctx. It never fails the request: on any capture problem the
// event is degraded (llm_d.payload.truncated=true) or skipped entirely.
func (c *Capturer) CaptureRequest(ctx context.Context, body *fwkrh.InferenceRequestBody) {
	if c == nil || body == nil {
		return
	}
	span := trace.SpanFromContext(ctx)
	if !span.SpanContext().IsValid() || !span.IsRecording() {
		return
	}

	ext := extract(body)
	if ext.empty() {
		return
	}

	ref := PayloadRef{
		TraceID:   span.SpanContext().TraceID().String(),
		SpanID:    span.SpanContext().SpanID().String(),
		Kind:      KindPrompt,
		MediaType: "application/json",
	}

	attrs := make([]attribute.KeyValue, 0, 3)
	if kv, ok := c.inlineJSON(ctx, ref, AttrInputMessages, ext.messages, len(ext.messages) > 0, &ext.truncated); ok {
		attrs = append(attrs, kv)
	}
	if kv, ok := c.inlineJSON(ctx, ref, AttrSystemInstructions, ext.system, len(ext.system) > 0, &ext.truncated); ok {
		attrs = append(attrs, kv)
	}
	if ext.truncated {
		attrs = append(attrs, attribute.Bool(AttrTruncated, true))
	}
	if len(attrs) == 0 {
		return
	}
	span.AddEvent(EventInferenceDetails, trace.WithAttributes(attrs...))
}

// inlineJSON serialises v and offers it to the backend. It returns the
// attribute to attach when the backend accepts the payload inline; on
// ErrPayloadTooLarge the attribute is dropped and *truncated is set (Phase 1
// has no offload backend to fall back to). Any other Store error is logged
// so unexpected backend failures aren't lost silently — the attribute is
// still dropped and *truncated set to keep the request path infallible.
func (c *Capturer) inlineJSON(ctx context.Context, ref PayloadRef, key string, v any, present bool, truncated *bool) (attribute.KeyValue, bool) {
	if !present {
		return attribute.KeyValue{}, false
	}
	data, err := json.Marshal(v)
	if err != nil {
		c.logger.Error(err, "failed to serialise payload attribute", "attribute", key)
		*truncated = true
		return attribute.KeyValue{}, false
	}
	if _, err := c.store.Store(ctx, ref, data); err != nil {
		if !errors.Is(err, ErrPayloadTooLarge) {
			c.logger.Error(err, "payload store rejected attribute", "attribute", key)
		}
		*truncated = true
		return attribute.KeyValue{}, false
	}
	return attribute.String(key, string(data)), true
}

// extract converts the parsed request body into semantic-convention messages.
// Phase 1 covers the chat-completions, completions and Anthropic-messages
// shapes; other request types produce no event.
func extract(body *fwkrh.InferenceRequestBody) extraction {
	switch {
	case body.ChatCompletions != nil:
		return extractChatCompletions(body.ChatCompletions)
	case body.Completions != nil:
		return extractCompletions(body.Completions)
	case body.Messages != nil:
		return extractAnthropicMessages(body.Messages)
	default:
		return extraction{}
	}
}

func extractChatCompletions(req *fwkrh.ChatCompletionsRequest) extraction {
	var ext extraction
	for _, msg := range req.Messages {
		parts := contentToParts(msg.Content, &ext.truncated)
		// OpenAI models assistant tool-call requests as a separate ToolCalls
		// field (typed []any at the parser layer) rather than as content
		// parts. We don't semantic-extract those into tool_call parts yet —
		// mark the event truncated so consumers know assistant-side tool
		// history was elided.
		if len(msg.ToolCalls) > 0 {
			ext.truncated = true
		}
		// System-level guidance is recorded as gen_ai.system_instructions,
		// which follows the SystemInstructions schema (array of TextPart /
		// GenericPart) — the same structured shape as input.messages, with a
		// narrower part-type vocabulary. appendSystemParts enforces that
		// narrower vocabulary.
		if msg.Role == "system" || msg.Role == "developer" {
			appendSystemParts(&ext, parts)
			continue
		}
		if len(parts) == 0 {
			continue
		}
		ext.messages = append(ext.messages, chatMessage{Role: msg.Role, Parts: parts})
	}
	return ext
}

func contentToParts(content fwkrh.Content, truncated *bool) []part {
	if content.Raw != "" {
		return []part{{Type: partTypeText, Content: content.Raw}}
	}
	var parts []part
	for _, block := range content.Structured {
		switch block.Type {
		case blockTypeText:
			parts = append(parts, part{Type: partTypeText, Content: block.Text})
		case "image_url":
			parts = appendMediaURI(parts, block.ImageURL.URL, modalityImage, truncated)
		case "video_url":
			parts = appendMediaURI(parts, block.VideoURL.URL, modalityVideo, truncated)
		case "input_audio":
			// Audio arrives as raw base64 bytes (a blob part). Phase 1 has no
			// object-store backend to offload blobs to, so the part is dropped.
			*truncated = true
		default:
			*truncated = true
		}
	}
	return parts
}

// appendMediaURI records an external media reference as a uri part. Data URLs
// carry raw bytes inline (blob parts under the upstream schema); Phase 1 has
// no object-store backend to offload them to, so they are dropped and the
// event is marked truncated.
func appendMediaURI(parts []part, url, modality string, truncated *bool) []part {
	if url == "" {
		return parts
	}
	if strings.HasPrefix(url, "data:") {
		*truncated = true
		return parts
	}
	return append(parts, part{Type: partTypeURI, URI: url, Modality: modality})
}

func extractCompletions(req *fwkrh.CompletionsRequest) extraction {
	var ext extraction
	var parts []part
	if req.Prompt.Raw != "" {
		parts = []part{{Type: partTypeText, Content: req.Prompt.Raw}}
	} else {
		for _, s := range req.Prompt.Strings {
			parts = append(parts, part{Type: partTypeText, Content: s})
		}
	}
	if len(parts) == 0 {
		// Pre-tokenised prompts (token IDs) have no capturable text.
		if len(req.Prompt.TokenIDs) > 0 {
			ext.truncated = true
		}
		return ext
	}
	ext.messages = []chatMessage{{Role: "user", Parts: parts}}
	return ext
}

func extractAnthropicMessages(req *fwkrh.MessagesRequest) extraction {
	var ext extraction
	// Same schema restriction as ChatCompletions: system content is limited
	// to TextPart / GenericPart.
	appendSystemParts(&ext, anthropicContentToParts(req.System, &ext.truncated))
	for _, msg := range req.Messages {
		parts := anthropicContentToParts(msg.Content, &ext.truncated)
		if len(parts) == 0 {
			continue
		}
		ext.messages = append(ext.messages, chatMessage{Role: msg.Role, Parts: parts})
	}
	return ext
}

func anthropicContentToParts(content fwkrh.AnthropicContent, truncated *bool) []part {
	if content.Raw != "" {
		return []part{{Type: partTypeText, Content: content.Raw}}
	}
	var parts []part
	for _, block := range content.Structured {
		switch {
		case block.Type == blockTypeText:
			parts = append(parts, part{Type: partTypeText, Content: block.Text})
		case block.Type == "image" && block.Source != nil && block.Source.URL != "":
			parts = append(parts, part{
				Type:     partTypeURI,
				URI:      block.Source.URL,
				MimeType: block.Source.MediaType,
				Modality: modalityImage,
			})
		default:
			// base64 image sources are blob parts (no offload backend in
			// Phase 1).
			//
			// tool_use / tool_result / thinking blocks land with
			// llm-d/llm-d-router#2389 (which extends AnthropicContentBlock
			// with ID / Name / Input / ToolUseID / Content / Thinking
			// fields). Once that PR lands, wire them here as semconv
			// ToolCallRequestPart / ToolCallResponsePart / ReasoningPart —
			// the part-type constants and Part struct fields are already
			// in place. Until then unrecognised block types mark truncated
			// so consumers can tell the event isn't the whole request.
			*truncated = true
		}
	}
	return parts
}
