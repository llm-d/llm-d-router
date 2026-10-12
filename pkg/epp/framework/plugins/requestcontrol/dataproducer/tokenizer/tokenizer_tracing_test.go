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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
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
		TargetModel: "model-a",
		Headers:     map[string]string{reqcommon.RequestIDHeaderKey: "req-1"},
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
	assert.Equal(t, BackendVLLM, attrs[semconv.LLMDEPPTokenProducerBackendKey].AsString())
	assert.Equal(t, int64(3), attrs[semconv.LLMDEPPTokenProducerTokenCountKey].AsInt64())
	assert.Equal(t, "model-a", attrs[semconv.GenAIRequestModelKey].AsString())
	assert.Equal(t, "req-1", attrs[semconv.GenAIRequestIDKey].AsString())
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

func TestProduce_RenderStatusErrorPrivacy(t *testing.T) {
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
		raw := `{"messages":[{"role":"user","content":"private_render_arguments"}],"tools":[{"type":"function","function":{"name":"private_render_tool","parameters":{"description":"private_render_schema"}}}],"tool_choice":"auto"}`
		if api == reqcommon.APITypeMessages {
			raw = `{"messages":[{"role":"user","content":"private_render_arguments"}],"tools":[{"name":"private_render_tool","input_schema":{"description":"private_render_schema"}}],"tool_choice":{"type":"auto"}}`
		}
		for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound, http.StatusUnprocessableEntity, http.StatusServiceUnavailable} {
			t.Run(fmt.Sprintf("%s/%d", api, status), func(t *testing.T) {
				recorder := setupSpanRecorder(t)
				forwarded := make(chan string, 1)
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					body, err := io.ReadAll(r.Body)
					assert.NoError(t, err)
					forwarded <- string(body)
					w.WriteHeader(status)
					_, _ = w.Write(body)
				}))
				defer srv.Close()
				renderer := newHTTPRenderer(t, srv)
				renderer.pluginName = t.Name()
				p := newTestPlugin(renderer)
				request := chatRequest()
				if api == reqcommon.APITypeMessages {
					request.Body.ChatCompletions = nil
					request.Body.Messages = &fwkrh.MessagesRequest{}
				}
				request.Body.Payload = fwkrh.RawPayload(raw)
				logs := &logCapture{}
				err := p.Produce(logs.context(context.Background()), request, nil)
				var statusErr *renderStatusError
				require.ErrorAs(t, err, &statusErr)
				require.Equal(t, status, statusErr.StatusCode)
				require.Equal(t, status == http.StatusUnauthorized || status == http.StatusForbidden, isRenderAuthError(err))
				require.Contains(t, err.Error(), strconv.Itoa(status))
				require.Nil(t, request.Body.TokenizedRequest)
				require.Equal(t, raw, <-forwarded)
				require.Equal(t, fwkrh.RawPayload(raw), request.Body.WirePayload())
				require.Equal(t, float64(1), renderFailures(renderer.pluginName, renderReasonStatus))
				require.NotEmpty(t, logs.snapshot())
				for _, secret := range []string{"private_render_tool", "private_render_schema", "private_render_arguments"} {
					assert.NotContains(t, err.Error(), secret)
					assert.NotContains(t, strings.Join(logs.snapshot(), "\n"), secret)
					for _, span := range recorder.Ended() {
						assert.NotContains(t, span.Status().Description, secret)
						for _, attr := range span.Attributes() {
							assert.NotContains(t, attr.Value.String(), secret)
						}
					}
				}
				require.Equal(t, codes.Error, tokenizeSpan(t, recorder).Status().Code)
			})
		}
	}
}

func TestProduce_RenderDecodeErrorPrivacy(t *testing.T) {
	const privateValue = "123456789012345678901234567890e999"
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
		t.Run(api.String(), func(t *testing.T) {
			recorder := setupSpanRecorder(t)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintf(w, `{"token_ids":[%s]}`, privateValue)
			}))
			defer srv.Close()
			renderer := newHTTPRenderer(t, srv)
			renderer.pluginName = t.Name()
			request := chatRequest()
			if api == reqcommon.APITypeMessages {
				request.Body.ChatCompletions = nil
				request.Body.Messages = &fwkrh.MessagesRequest{}
			}
			logs := &logCapture{}
			err := newTestPlugin(renderer).Produce(logs.context(context.Background()), request, nil)
			require.ErrorIs(t, err, errRenderDecode)
			var typeErr *json.UnmarshalTypeError
			require.ErrorAs(t, err, &typeErr)
			require.Contains(t, typeErr.Value, privateValue)
			require.Nil(t, request.Body.TokenizedRequest)
			require.Equal(t, float64(1), renderFailures(renderer.pluginName, renderReasonDecode))
			require.NotEmpty(t, logs.snapshot())
			assert.NotContains(t, err.Error(), privateValue)
			assert.NotContains(t, strings.Join(logs.snapshot(), "\n"), privateValue)
			for _, span := range recorder.Ended() {
				assert.NotContains(t, span.Status().Description, privateValue)
				for _, attr := range span.Attributes() {
					assert.NotContains(t, attr.Value.String(), privateValue)
				}
			}
			require.Equal(t, codes.Error, tokenizeSpan(t, recorder).Status().Code)
		})
	}
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
	assert.Equal(t, resultSkippedNoTokens, attrs[semconv.LLMDEPPTokenProducerResultKey].AsString())
	_, hasCount := attrs[semconv.LLMDEPPTokenProducerTokenCountKey]
	assert.False(t, hasCount, "no token count on the skipped path")
}
