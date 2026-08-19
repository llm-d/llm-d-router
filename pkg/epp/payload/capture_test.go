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
	"strings"
	"testing"

	"github.com/go-logr/logr"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// truncatedTrue is the emitted string form of llm_d.payload.truncated=true.
const truncatedTrue = "true"

// captureOnSpan runs CaptureRequest against a recording span and returns the
// finished span for inspection.
func captureOnSpan(t *testing.T, c *Capturer, body *fwkrh.InferenceRequestBody) tracetest.SpanStub {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })

	ctx, span := tp.Tracer("test").Start(context.Background(), "gateway.request", trace.WithSpanKind(trace.SpanKindServer))
	c.CaptureRequest(ctx, body)
	span.End()

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("recorded %d spans, want 1", len(spans))
	}
	return tracetest.SpanStubsFromReadOnlySpans(spans)[0]
}

func eventAttr(t *testing.T, stub tracetest.SpanStub, key string) (string, bool) {
	t.Helper()
	for _, ev := range stub.Events {
		if ev.Name != EventInferenceDetails {
			continue
		}
		for _, kv := range ev.Attributes {
			if string(kv.Key) == key {
				return kv.Value.String(), true
			}
		}
	}
	return "", false
}

func hasDetailsEvent(stub tracetest.SpanStub) bool {
	for _, ev := range stub.Events {
		if ev.Name == EventInferenceDetails {
			return true
		}
	}
	return false
}

func inlineCapturer(threshold int) *Capturer {
	return NewCapturer(Config{Enabled: true, Backend: BackendInline, InlineSizeThresholdBytes: threshold}, logr.Discard())
}

func TestCaptureChatCompletions(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{
				{Role: "system", Content: fwkrh.Content{Raw: "You are terse."}},
				{Role: "user", Content: fwkrh.Content{Raw: "What is the capital of France?"}},
				{Role: "assistant", Content: fwkrh.Content{Raw: "Paris."}},
			},
		},
	}

	stub := captureOnSpan(t, inlineCapturer(4096), body)

	msgJSON, ok := eventAttr(t, stub, AttrInputMessages)
	if !ok {
		t.Fatal("event missing gen_ai.input.messages")
	}
	var msgs []chatMessage
	if err := json.Unmarshal([]byte(msgJSON), &msgs); err != nil {
		t.Fatalf("gen_ai.input.messages is not valid JSON: %v", err)
	}
	if len(msgs) != 2 || msgs[0].Role != "user" || msgs[1].Role != "assistant" {
		t.Fatalf("unexpected input messages: %s", msgJSON)
	}
	if msgs[0].Parts[0].Type != "text" || msgs[0].Parts[0].Content != "What is the capital of France?" {
		t.Fatalf("unexpected user parts: %+v", msgs[0].Parts)
	}

	sysJSON, ok := eventAttr(t, stub, AttrSystemInstructions)
	if !ok {
		t.Fatal("event missing gen_ai.system_instructions")
	}
	// gen_ai.system_instructions is an array of TextPart / GenericPart per
	// the current semantic-conventions-genai schema
	// (model/gen-ai/gen-ai-system-instructions.json). Guard against
	// regressing to the plain-string emission an earlier Copilot review
	// asked for based on an outdated snapshot of the semconv.
	var sysParts []part
	if err := json.Unmarshal([]byte(sysJSON), &sysParts); err != nil {
		t.Fatalf("gen_ai.system_instructions is not valid JSON: %v", err)
	}
	if len(sysParts) != 1 || sysParts[0].Type != "text" || sysParts[0].Content != "You are terse." {
		t.Fatalf("unexpected system instructions: %s", sysJSON)
	}

	if _, ok := eventAttr(t, stub, AttrTruncated); ok {
		t.Error("unexpected llm_d.payload.truncated on fully-inline capture")
	}
}

// TestCaptureSystemInstructionsStructured covers the current semconv path:
// gen_ai.system_instructions is emitted as a JSON array of parts (same
// TextPart shape as gen_ai.input.messages entries), not a joined plain
// string. Multiple system-role messages append into one attribute; non-text
// system content is dropped and marks truncated.
func TestCaptureSystemInstructionsStructured(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{
				{Role: "system", Content: fwkrh.Content{Raw: "You are terse."}},
				{Role: "developer", Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
					{Type: "text", Text: "Answer in French."},
					{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/style-guide.png"}},
				}}},
				{Role: "user", Content: fwkrh.Content{Raw: "Bonjour."}},
			},
		},
	}

	stub := captureOnSpan(t, inlineCapturer(4096), body)

	sysJSON, ok := eventAttr(t, stub, AttrSystemInstructions)
	if !ok {
		t.Fatal("event missing gen_ai.system_instructions")
	}
	var sysParts []part
	if err := json.Unmarshal([]byte(sysJSON), &sysParts); err != nil {
		t.Fatalf("gen_ai.system_instructions is not valid JSON: %v", err)
	}
	// System / developer text lines each become one TextPart, in message
	// order; the image on the developer message is dropped and marks
	// truncated. Also asserts the URI attached to the earlier user
	// message never leaks into system_instructions.
	if len(sysParts) != 2 {
		t.Fatalf("want 2 system TextParts (system+developer text), got %d: %s", len(sysParts), sysJSON)
	}
	if sysParts[0].Type != "text" || sysParts[0].Content != "You are terse." {
		t.Fatalf("unexpected first system part: %+v", sysParts[0])
	}
	if sysParts[1].Type != "text" || sysParts[1].Content != "Answer in French." {
		t.Fatalf("unexpected second system part: %+v", sysParts[1])
	}
	if v, ok := eventAttr(t, stub, AttrTruncated); !ok || v != truncatedTrue {
		t.Error("expected llm_d.payload.truncated=true when non-text system content is dropped")
	}
}

// TestCaptureOpenAIAssistantToolCallsMarksTruncated documents the current
// gap: OpenAI's Message.ToolCalls field is typed []any at the parser layer,
// so we can't semantic-extract it into semconv tool_call parts without a
// bespoke re-parse. Until the parser exposes typed tool-call structs, an
// assistant message that carries tool_calls records the visible content but
// flags truncation so consumers know the assistant turn isn't complete.
func TestCaptureOpenAIAssistantToolCallsMarksTruncated(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{
				{Role: "user", Content: fwkrh.Content{Raw: "What's the weather in NYC?"}},
				{
					Role:      "assistant",
					Content:   fwkrh.Content{Raw: ""},
					ToolCalls: []any{map[string]any{"id": "call_1", "type": "function"}},
				},
			},
		},
	}

	stub := captureOnSpan(t, inlineCapturer(4096), body)

	if v, ok := eventAttr(t, stub, AttrTruncated); !ok || v != truncatedTrue {
		t.Error("expected llm_d.payload.truncated=true when assistant.tool_calls is present but not semantic-extracted")
	}
}

func TestCaptureMultimodalParts(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{
				{Role: "user", Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
					{Type: "text", Text: "Describe this image."},
					{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/cat.png"}},
					{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "data:image/png;base64,iVBORw0KGgo="}},
				}}},
			},
		},
	}

	stub := captureOnSpan(t, inlineCapturer(4096), body)

	msgJSON, ok := eventAttr(t, stub, AttrInputMessages)
	if !ok {
		t.Fatal("event missing gen_ai.input.messages")
	}
	var msgs []chatMessage
	if err := json.Unmarshal([]byte(msgJSON), &msgs); err != nil {
		t.Fatalf("gen_ai.input.messages is not valid JSON: %v", err)
	}
	parts := msgs[0].Parts
	if len(parts) != 2 {
		t.Fatalf("got %d parts, want 2 (text + external uri; data URL dropped): %s", len(parts), msgJSON)
	}
	if parts[1].Type != "uri" || parts[1].URI != "https://example.com/cat.png" || parts[1].Modality != "image" {
		t.Fatalf("unexpected uri part: %+v", parts[1])
	}

	// The data-URL image is a blob part with no offload backend in Phase 1.
	if v, ok := eventAttr(t, stub, AttrTruncated); !ok || v != truncatedTrue {
		t.Error("expected llm_d.payload.truncated=true when a blob part is dropped")
	}
}

func TestCaptureCompletionsPrompt(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "Once upon a time"}},
	}

	stub := captureOnSpan(t, inlineCapturer(4096), body)

	msgJSON, ok := eventAttr(t, stub, AttrInputMessages)
	if !ok {
		t.Fatal("event missing gen_ai.input.messages")
	}
	var msgs []chatMessage
	if err := json.Unmarshal([]byte(msgJSON), &msgs); err != nil {
		t.Fatalf("gen_ai.input.messages is not valid JSON: %v", err)
	}
	if len(msgs) != 1 || msgs[0].Role != "user" || msgs[0].Parts[0].Content != "Once upon a time" {
		t.Fatalf("unexpected messages: %s", msgJSON)
	}
}

func TestCaptureAnthropicMessages(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		Messages: &fwkrh.MessagesRequest{
			System: fwkrh.AnthropicContent{Raw: "Answer in French."},
			Messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
					{Type: "text", Text: "Hello"},
					{Type: "image", Source: &fwkrh.AnthropicImageSource{Type: "url", MediaType: "image/jpeg", URL: "https://example.com/dog.jpg"}},
				}}},
			},
		},
	}

	stub := captureOnSpan(t, inlineCapturer(4096), body)

	msgJSON, ok := eventAttr(t, stub, AttrInputMessages)
	if !ok {
		t.Fatal("event missing gen_ai.input.messages")
	}
	var msgs []chatMessage
	if err := json.Unmarshal([]byte(msgJSON), &msgs); err != nil {
		t.Fatalf("gen_ai.input.messages is not valid JSON: %v", err)
	}
	parts := msgs[0].Parts
	if len(parts) != 2 || parts[1].Type != "uri" || parts[1].MimeType != "image/jpeg" {
		t.Fatalf("unexpected parts: %s", msgJSON)
	}

	if sysJSON, ok := eventAttr(t, stub, AttrSystemInstructions); !ok || !strings.Contains(sysJSON, "Answer in French.") {
		t.Fatalf("missing or wrong system instructions: %q", sysJSON)
	}
}

func TestCaptureOverThresholdTruncates(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: strings.Repeat("x", 4096)}},
	}

	stub := captureOnSpan(t, inlineCapturer(64), body)

	if _, ok := eventAttr(t, stub, AttrInputMessages); ok {
		t.Error("gen_ai.input.messages should be dropped when over the inline threshold")
	}
	if v, ok := eventAttr(t, stub, AttrTruncated); !ok || v != truncatedTrue {
		t.Error("expected llm_d.payload.truncated=true when payload exceeds inline threshold")
	}
}

func TestCaptureTokenIDPromptMarksTruncated(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: []uint32{1, 2, 3}}},
	}

	stub := captureOnSpan(t, inlineCapturer(4096), body)

	if _, ok := eventAttr(t, stub, AttrInputMessages); ok {
		t.Error("token-ID prompts have no capturable text")
	}
	if v, ok := eventAttr(t, stub, AttrTruncated); !ok || v != truncatedTrue {
		t.Error("expected llm_d.payload.truncated=true for token-ID prompt")
	}
}

func TestCaptureSkipsUnsupportedAndNil(t *testing.T) {
	c := inlineCapturer(4096)

	// Unsupported request type: no event.
	stub := captureOnSpan(t, c, &fwkrh.InferenceRequestBody{Embeddings: &fwkrh.EmbeddingsRequest{}})
	if hasDetailsEvent(stub) {
		t.Error("unsupported request types should not emit a details event")
	}

	// Nil body: no event.
	stub = captureOnSpan(t, c, nil)
	if hasDetailsEvent(stub) {
		t.Error("nil body should not emit a details event")
	}

	// Nil capturer: must be a safe no-op.
	var nilCapturer *Capturer
	nilCapturer.CaptureRequest(context.Background(), &fwkrh.InferenceRequestBody{})
}

func TestCaptureWithoutSpanIsNoop(t *testing.T) {
	// No span in context: nothing to attach to, and no panic.
	inlineCapturer(4096).CaptureRequest(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
	})
}
