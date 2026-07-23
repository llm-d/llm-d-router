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

	sys, ok := eventAttr(t, stub, AttrSystemInstructions)
	if !ok {
		t.Fatal("event missing gen_ai.system_instructions")
	}
	// gen_ai.system_instructions is a plain string per the upstream semconv,
	// not a JSON-serialised []part. Guard against regressing to the old shape.
	if sys != "You are terse." {
		t.Fatalf("system instructions want plain string %q, got %q", "You are terse.", sys)
	}

	if _, ok := eventAttr(t, stub, AttrTruncated); ok {
		t.Error("unexpected llm_d.payload.truncated on fully-inline capture")
	}
}

// TestCaptureSystemInstructionsPlainString covers the semconv-compliance path:
// multiple system messages are joined into one plain-string attribute, and
// non-text system content is dropped and flagged as truncated.
func TestCaptureSystemInstructionsPlainString(t *testing.T) {
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

	sys, ok := eventAttr(t, stub, AttrSystemInstructions)
	if !ok {
		t.Fatal("event missing gen_ai.system_instructions")
	}
	// Multiple system-role messages are joined with a blank-line separator.
	want := "You are terse.\n\nAnswer in French."
	if sys != want {
		t.Fatalf("system instructions want %q, got %q", want, sys)
	}
	// Any content that isn't a text part (e.g. an image on a system message)
	// is dropped and must mark the event truncated so operators know the
	// captured value isn't the whole system message.
	if v, ok := eventAttr(t, stub, AttrTruncated); !ok || v != truncatedTrue {
		t.Error("expected llm_d.payload.truncated=true when non-text system content is dropped")
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
