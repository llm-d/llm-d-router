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
// outside the reserved gen_ai.* namespace.
// https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
const (
	// EventInferenceDetails is the upstream GenAI event that carries payload
	// attributes: gen_ai.client.inference.operation.details.
	EventInferenceDetails = "gen_ai.client.inference.operation.details"

	// AttrInputMessages is gen_ai.input.messages: the structured request
	// messages (role/parts), serialised to JSON when recorded on a span event.
	AttrInputMessages = "gen_ai.input.messages"
	// AttrSystemInstructions is gen_ai.system_instructions: system-level
	// instruction *text*, per the upstream semantic convention. Non-text
	// system content (e.g. images in a system message) is dropped and marked
	// on llm_d.payload.truncated; multiple system-role messages and multiple
	// text parts within one message are joined with "\n\n".
	AttrSystemInstructions = "gen_ai.system_instructions"

	// AttrTruncated is llm_d.payload.truncated (llm-d extension): true when
	// content was dropped or truncated during capture.
	AttrTruncated = "llm_d.payload.truncated"
)

// Part types and modalities from the upstream input-messages schema.
// https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json
const (
	partTypeText = "text"
	partTypeURI  = "uri"

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

// part mirrors the upstream message-part schema. Exactly one of Content or
// URI is set, per the part Type.
type part struct {
	Type     string `json:"type"`
	Content  string `json:"content,omitempty"`
	URI      string `json:"uri,omitempty"`
	MimeType string `json:"mime_type,omitempty"`
	Modality string `json:"modality,omitempty"`
}

// extraction is the intermediate result of converting a parsed request body
// into semantic-convention form. system is a plain string (concatenation of
// text-only system content) per the semconv definition of
// gen_ai.system_instructions.
type extraction struct {
	messages []chatMessage
	system   string
	// truncated is set when content was dropped: inline blob parts (no offload
	// backend exists in Phase 1), non-textual prompts (token IDs), or non-text
	// content on a system-role message (system instructions are text-only).
	truncated bool
}

func (e extraction) empty() bool {
	return len(e.messages) == 0 && e.system == "" && !e.truncated
}

// partsToTextString extracts the concatenated text of parts, joining multiple
// text parts with a blank line. Non-text parts (uri, blob, etc.) are dropped
// and mark *truncated. This is how system-role content is normalised into the
// plain-string value required by gen_ai.system_instructions.
func partsToTextString(parts []part, truncated *bool) string {
	var texts []string
	for _, p := range parts {
		if p.Type == partTypeText {
			if p.Content != "" {
				texts = append(texts, p.Content)
			}
			continue
		}
		*truncated = true
	}
	return strings.Join(texts, "\n\n")
}

// appendSystemText joins additional system-instruction text with any already
// captured, using a blank-line separator so multiple system messages remain
// legible.
func appendSystemText(existing, next string) string {
	switch {
	case next == "":
		return existing
	case existing == "":
		return next
	default:
		return existing + "\n\n" + next
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
	if kv, ok := c.inlineString(ctx, ref, AttrSystemInstructions, ext.system, &ext.truncated); ok {
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

// inlineString attaches v as a plain string attribute after offering its bytes
// to the backend for the size check. Used for attributes whose semantic
// convention is a plain string (gen_ai.system_instructions). Follows the same
// error contract as inlineJSON.
func (c *Capturer) inlineString(ctx context.Context, ref PayloadRef, key, v string, truncated *bool) (attribute.KeyValue, bool) {
	if v == "" {
		return attribute.KeyValue{}, false
	}
	if _, err := c.store.Store(ctx, ref, []byte(v)); err != nil {
		if !errors.Is(err, ErrPayloadTooLarge) {
			c.logger.Error(err, "payload store rejected attribute", "attribute", key)
		}
		*truncated = true
		return attribute.KeyValue{}, false
	}
	return attribute.String(key, v), true
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
		// System-level guidance is recorded as gen_ai.system_instructions, a
		// plain string per the upstream convention; non-text parts (media,
		// blobs) are dropped and flagged via llm_d.payload.truncated.
		if msg.Role == "system" || msg.Role == "developer" {
			ext.system = appendSystemText(ext.system, partsToTextString(parts, &ext.truncated))
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
	// Anthropic's `system` field is text-oriented (either a bare string or a
	// list of content blocks); non-text blocks are dropped as truncated to
	// match the semconv definition of gen_ai.system_instructions.
	ext.system = partsToTextString(anthropicContentToParts(req.System, &ext.truncated), &ext.truncated)
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
			// Phase 1) and unrecognised block types are not representable.
			*truncated = true
		}
	}
	return parts
}
