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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

// setupSpanRecorder installs an in-memory span recorder as the global tracer
// provider and returns it, restoring the previous provider on cleanup.
func setupSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })
	return recorder
}

func tokenizeSpan(t *testing.T, recorder *tracetest.SpanRecorder) sdktrace.ReadOnlySpan {
	t.Helper()
	for _, s := range recorder.Ended() {
		if s.Name() == "tokenize" {
			return s
		}
	}
	t.Fatal("no tokenize span recorded")
	return nil
}

func spanAttrs(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	attrs := make(map[attribute.Key]attribute.Value)
	for _, kv := range span.Attributes() {
		attrs[kv.Key] = kv.Value
	}
	return attrs
}

func chatRequest() *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID:   "req-1",
		TargetModel: "model-a",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hi"}}},
			},
			Payload: fwkrh.PayloadMap{},
		},
	}
}

func TestProduce_EmitsTokenizeSpan(t *testing.T) {
	recorder := setupSpanRecorder(t)
	p := newTestPlugin(&mockTokenizer{
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return []uint32{1, 2, 3}, nil, nil
		},
	})

	require.NoError(t, p.Produce(context.Background(), chatRequest(), nil))

	attrs := spanAttrs(tokenizeSpan(t, recorder))
	assert.Equal(t, backendVLLM, attrs["llm_d.epp.token_producer.backend"].AsString())
	assert.Equal(t, int64(3), attrs["llm_d.epp.token_producer.token_count"].AsInt64())
	assert.Equal(t, "model-a", attrs["gen_ai.request.model"].AsString())
	assert.Equal(t, "req-1", attrs["gen_ai.request.id"].AsString())
	assert.Equal(t, "none", attrs["mm.modality"].AsString())
	assert.Equal(t, int64(0), attrs["mm.hash_count"].AsInt64())
}

// The span must parent to the caller's span so tokenization appears under the
// request trace rather than as a detached root.
func TestProduce_TokenizeSpanParentedToCaller(t *testing.T) {
	recorder := setupSpanRecorder(t)
	p := newTestPlugin(&mockTokenizer{
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return []uint32{1}, nil, nil
		},
	})

	ctx, parent := otel.Tracer("test").Start(context.Background(), "parent")
	require.NoError(t, p.Produce(ctx, chatRequest(), nil))
	parent.End()

	span := tokenizeSpan(t, recorder)
	assert.Equal(t, parent.SpanContext().TraceID(), span.SpanContext().TraceID())
	assert.Equal(t, parent.SpanContext().SpanID(), span.Parent().SpanID())
}

func TestProduce_TokenizeSpanRecordsMultiModal(t *testing.T) {
	recorder := setupSpanRecorder(t)
	p := newTestPlugin(&mockTokenizer{
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return []uint32{1, 2}, &tokenization.MultiModalFeatures{
				MMHashes:       map[string][]string{"image": {"h1"}},
				MMPlaceholders: map[string][]kvblock.PlaceholderRange{"image": {{Offset: 0, Length: 2}}},
			}, nil
		},
	})

	require.NoError(t, p.Produce(context.Background(), chatRequest(), nil))

	attrs := spanAttrs(tokenizeSpan(t, recorder))
	assert.Equal(t, "image", attrs["mm.modality"].AsString())
	assert.Equal(t, int64(1), attrs["mm.hash_count"].AsInt64())
}

func TestProduce_TokenizeSpanRecordsError(t *testing.T) {
	recorder := setupSpanRecorder(t)
	p := newTestPlugin(&mockTokenizer{
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return nil, nil, errors.New("render failed")
		},
	})

	require.Error(t, p.Produce(context.Background(), chatRequest(), nil))

	span := tokenizeSpan(t, recorder)
	assert.Equal(t, codes.Error, span.Status().Code)
	assert.Contains(t, span.Status().Description, "render failed")
}

// The skip path does no tokenization work, so it must not emit a span.
func TestProduce_NoSpanWhenAlreadyTokenized(t *testing.T) {
	recorder := setupSpanRecorder(t)
	p := newTestPlugin(&mockTokenizer{})

	req := chatRequest()
	req.Body.TokenizedRequest = &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{7}}}}
	require.NoError(t, p.Produce(context.Background(), req, nil))

	assert.Empty(t, recorder.Ended())
}

// A backend that returns no tokens still did work, so the span must say why it
// carries no token count rather than looking like an unattributed span.
func TestProduce_TokenizeSpanRecordsSkippedNoTokens(t *testing.T) {
	recorder := setupSpanRecorder(t)
	p := newTestPlugin(&mockTokenizer{
		renderChatFunc: func(_ fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			return nil, nil, nil
		},
	})

	require.NoError(t, p.Produce(context.Background(), chatRequest(), nil))

	attrs := spanAttrs(tokenizeSpan(t, recorder))
	assert.Equal(t, resultSkippedNoTokens, attrs["llm_d.epp.token_producer.result"].AsString())
	_, hasCount := attrs["llm_d.epp.token_producer.token_count"]
	assert.False(t, hasCount, "no token count on the skipped path")
}
