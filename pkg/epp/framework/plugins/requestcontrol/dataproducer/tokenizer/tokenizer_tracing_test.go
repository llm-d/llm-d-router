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
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

// setupSpanRecorder installs an in-memory span recorder as the global tracer
// provider and returns it, restoring the previous provider on cleanup.
func setupSpanRecorder(t *testing.T, options ...sdktrace.TracerProviderOption) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(append(options, sdktrace.WithSpanProcessor(recorder))...)
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

func TestProduce_HTTPToolCallingSummary(t *testing.T) {
	originalPropagator := otel.GetTextMapPropagator()
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() { otel.SetTextMapPropagator(originalPropagator) })
	for _, tc := range []struct {
		name   string
		api    reqcommon.APIType
		raw    string
		choice string
		count  string
		warmup bool
	}{
		{"chat named", reqcommon.APITypeChatCompletions, `{"tools":[{"type":"function","function":{"name":"private_tool","parameters":{"description":"private_schema"}}}],"tool_choice":{"type":"function","function":{"name":"private_tool"}}}`, "named", "1", false},
		{"messages any", reqcommon.APITypeMessages, `{"tools":[{"name":"private_tool","input_schema":{"description":"private_schema"}}],"tool_choice":{"type":"any"}}`, "required", "1", false},
		{"responses named", reqcommon.APITypeResponses, `{"tools":[{"type":"function","name":"private_tool","parameters":{"description":"private_schema"}}],"tool_choice":{"type":"function","name":"private_tool"}}`, "named", "1", false},
		{"parallel only", reqcommon.APITypeChatCompletions, `{"parallel_tool_calls":false}`, "unknown", "", false},
		{"non tool", reqcommon.APITypeChatCompletions, `{"messages":[{"role":"user","content":"private_prompt tools"}]}`, "", "", false},
		{"structured output only", reqcommon.APITypeChatCompletions, `{"response_format":{"type":"json_schema","json_schema":{"name":"private_schema"}}}`, "", "", false},
		{"malformed", reqcommon.APITypeChatCompletions, `{"tools":[`, "", "", false},
		{"completions", reqcommon.APITypeCompletions, `{"tools":[],"tool_choice":"auto"}`, "", "", false},
		{"warmup", reqcommon.APITypeChatCompletions, `{"tools":[],"tool_choice":"auto"}`, "", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := setupSpanRecorder(t)
			forwarded := make(chan string, 1)
			forwardedTrace := make(chan trace.SpanContext, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				assert.NoError(t, err)
				forwarded <- string(body)
				forwardedTrace <- trace.SpanContextFromContext(propagation.TraceContext{}.Extract(r.Context(), propagation.HeaderCarrier(r.Header)))
				response := `{"token_ids":[1]}`
				if tc.api == reqcommon.APITypeCompletions {
					response = "[" + response + "]"
				}
				_, _ = io.WriteString(w, response)
			}))
			defer srv.Close()
			request := chatRequest()
			if tc.api != reqcommon.APITypeChatCompletions {
				request.Body.ChatCompletions = nil
			}
			switch tc.api {
			case reqcommon.APITypeMessages:
				request.Body.Messages = &fwkrh.MessagesRequest{}
			case reqcommon.APITypeResponses:
				request.Body.Responses = &fwkrh.ResponsesRequest{}
			case reqcommon.APITypeCompletions:
				request.Body.Completions = &fwkrh.CompletionsRequest{}
			}
			request.Body.Payload = fwkrh.RawPayload(tc.raw)
			ctx := context.Background()
			ctx = toolcalling.WithTelemetryEnabled(ctx)
			if tc.warmup {
				ctx = withUnobservedRender(ctx)
			} else if tc.api != reqcommon.APITypeCompletions && json.Valid([]byte(tc.raw)) {
				snapshot, err := toolcalling.CaptureRequestJSON(tc.api, []byte(tc.raw))
				require.NoError(t, err)
				ctx = toolcalling.WithRequestSnapshot(ctx, []byte(tc.raw), snapshot)
			}
			require.NoError(t, newTestPlugin(newHTTPRenderer(t, srv)).Produce(ctx, request, nil))
			require.Equal(t, tc.raw, <-forwarded)
			require.Equal(t, fwkrh.RawPayload(tc.raw), request.Body.WirePayload())
			want := map[attribute.Key]attribute.Value{}
			if tc.choice != "" {
				want[semconv.LLMDToolCallingAPISurfaceKey] = attribute.StringValue(tc.api.String())
				want[semconv.LLMDToolCallingPresentKey] = attribute.BoolValue(true)
				if tc.name != "parallel only" {
					want[semconv.LLMDToolCallingToolChoiceKey] = attribute.StringValue(tc.choice)
				}
				if tc.count != "" {
					want[semconv.LLMDToolCallingToolCountKey] = attribute.StringValue(tc.count)
					want[semconv.LLMDToolCallingFieldToolsStatusKey] = attribute.StringValue("preserved")
				}
				if tc.choice == "named" || tc.choice == "required" {
					want[semconv.LLMDToolCallingFieldToolChoiceStatusKey] = attribute.StringValue("preserved")
				}
				if tc.name == "parallel only" {
					want[semconv.LLMDToolCallingFieldParallelToolCallsStatusKey] = attribute.StringValue("preserved")
				}
			}
			if tc.name == "structured output only" {
				want[semconv.LLMDToolCallingAPISurfaceKey] = attribute.StringValue(tc.api.String())
				want[semconv.LLMDToolCallingPresentKey] = attribute.BoolValue(false)
				want[semconv.LLMDToolCallingFieldResponseFormatStatusKey] = attribute.StringValue("preserved")
			}
			wantSpans := 2
			if tc.warmup {
				wantSpans = 1
			}
			require.Len(t, recorder.Ended(), wantSpans)
			tokenize := tokenizeSpan(t, recorder)
			for _, span := range recorder.Ended() {
				if span.SpanKind() == trace.SpanKindClient {
					assert.Equal(t, tokenize.SpanContext().TraceID(), span.SpanContext().TraceID())
					assert.Equal(t, tokenize.SpanContext().SpanID(), span.Parent().SpanID())
					remote := <-forwardedTrace
					assert.Equal(t, span.SpanContext().TraceID(), remote.TraceID())
					assert.Equal(t, span.SpanContext().SpanID(), remote.SpanID())
					assert.Equal(t, span.SpanContext().TraceFlags(), remote.TraceFlags())
				}
				got := map[attribute.Key]attribute.Value{}
				for key, value := range spanAttrs(span) {
					if strings.HasPrefix(string(key), "llm_d.tool_calling.") {
						got[key] = value
					}
					for _, secret := range []string{"private_tool", "private_schema", "private_prompt"} {
						assert.NotContains(t, value.String(), secret)
					}
				}
				assert.Equal(t, want, got, span.Name())
			}
		})
	}
}

func TestProduce_HTTPTracingDisabled(t *testing.T) {
	recorder := setupSpanRecorder(t, sdktrace.WithSampler(sdktrace.NeverSample()))
	raw := `{"tools":[{"type":"function","function":{"name":"private_tool"}}],"tool_choice":"auto"}`
	forwarded := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		forwarded <- string(body)
		_, _ = io.WriteString(w, `{"token_ids":[1]}`)
	}))
	defer srv.Close()
	request := chatRequest()
	request.Body.Payload = fwkrh.RawPayload(raw)
	require.NoError(t, newTestPlugin(newHTTPRenderer(t, srv)).Produce(t.Context(), request, nil))
	require.Equal(t, raw, <-forwarded)
	require.Equal(t, fwkrh.RawPayload(raw), request.Body.WirePayload())
	require.NotNil(t, request.Body.TokenizedRequest)
	assert.Empty(t, recorder.Ended())
}

func TestCompareRenderToolFields(t *testing.T) {
	for _, tc := range []struct {
		name, outbound, status string
		wantError              bool
	}{
		{"unchanged", `{"tools":[],"max_tokens":4}`, "preserved", false},
		{"budget only", `{"max_tokens":1,"tools":[],"return_mm_kwargs":false}`, "preserved", false},
		{"changed", `{"tools":[{"type":"function","function":{"name":"private_tool"}}]}`, "changed", false},
		{"dropped", `{}`, "dropped", false},
		{"invalid outbound", `{"tools":[`, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			snapshot, statuses, err := compareRenderToolFields(t.Context(), reqcommon.APITypeChatCompletions, []byte(`{"tools":[],"max_tokens":4}`), []byte(tc.outbound), false)
			require.Equal(t, tc.wantError, err != nil)
			attrs := map[attribute.Key]attribute.Value{}
			for _, attr := range snapshot.SpanAttributes(statuses) {
				attrs[attr.Key] = attr.Value
			}
			assert.True(t, attrs[semconv.LLMDToolCallingPresentKey].AsBool())
			assert.Equal(t, tc.status, attrs[semconv.LLMDToolCallingFieldToolsStatusKey].AsString())
		})
	}
}

func TestCompareRenderToolFieldsWithRequestSnapshot(t *testing.T) {
	inbound := []byte(`{"tools":[]}`)
	snapshot, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, inbound)
	require.NoError(t, err)
	ctx := toolcalling.WithRequestSnapshot(t.Context(), inbound, snapshot)

	for _, tc := range []struct {
		name, inbound, outbound, status string
		wantError                       bool
	}{
		{"unchanged", `{"tools":[]}`, `{"tools":[]}`, "preserved", false},
		{"render copy drops tools", `{"tools":[]}`, `{}`, "dropped", false},
		{"same-length input mutation", `{"tools":{}}`, `{"tools":[]}`, "changed", false},
		{"malformed input", `{"tools":[`, `{}`, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, statuses, err := compareRenderToolFields(ctx, reqcommon.APITypeChatCompletions, []byte(tc.inbound), []byte(tc.outbound), false)
			require.Equal(t, tc.wantError, err != nil)
			if err == nil {
				attrs := map[attribute.Key]attribute.Value{}
				for _, attr := range got.SpanAttributes(statuses) {
					attrs[attr.Key] = attr.Value
				}
				require.Equal(t, tc.status, attrs[semconv.LLMDToolCallingFieldToolsStatusKey].AsString())
			}
		})
	}
}

func TestCompareRenderToolFieldsWithEmptySnapshot(t *testing.T) {
	inbound := []byte(`{"messages":[]}`)
	snapshot, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, inbound)
	require.NoError(t, err)
	ctx := toolcalling.WithRequestSnapshot(t.Context(), inbound, snapshot)
	for _, tc := range []struct {
		outbound string
		tool     bool
	}{
		{`{"messages":[],"max_tokens":1}`, false},
		{`{"messages":[],"tools":[]}`, true},
		{`{"messages":[],"to\u006fls":[]}`, true},
	} {
		_, statuses, err := compareRenderToolFields(ctx, reqcommon.APITypeChatCompletions, inbound, []byte(tc.outbound), true)
		require.NoError(t, err, tc.outbound)
		if tc.tool {
			require.Contains(t, statuses, toolcalling.FieldStatus{Field: toolcalling.FieldTools, Status: toolcalling.FieldStatusChanged, Observed: true})
		} else {
			require.Empty(t, statuses)
		}
	}
	_, _, err = compareRenderToolFields(ctx, reqcommon.APITypeChatCompletions, inbound, []byte(`{"messages":`), false)
	require.Error(t, err)
}

func BenchmarkCompareRenderToolFieldsNoTools(b *testing.B) {
	inbound := []byte(`{"messages":[{"role":"user","content":"` + strings.Repeat("x", 1<<20) + `"}]}`)
	snapshot, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, inbound)
	if err != nil {
		b.Fatal(err)
	}
	ctx := toolcalling.WithRequestSnapshot(b.Context(), inbound, snapshot)
	rewritten := append([]byte(`{"max_tokens":1,`), inbound[1:]...)
	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := compareRenderToolFields(ctx, reqcommon.APITypeChatCompletions, inbound, rewritten, true); err != nil {
			b.Fatal(err)
		}
	}
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

func TestProduce_RenderRequestErrorPrivacy(t *testing.T) {
	const privateValue = "123456789012345678901234567890e999"
	for _, tc := range []struct {
		name        string
		payload     fwkrh.RequestPayload
		prefillOnly bool
		wantError   string
	}{
		{"marshal", fwkrh.PayloadMap{"tools": json.Number("private_render_tool")}, false, "marshal request"},
		{"budget", fwkrh.RawPayload(privateValue), true, "apply render-only output budget"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := setupSpanRecorder(t)
			renderer := &vllmHTTPRenderer{prefillOnly: tc.prefillOnly}
			request := chatRequest()
			request.Body.Payload = tc.payload
			err := newTestPlugin(renderer).Produce(context.Background(), request, nil)
			require.ErrorContains(t, err, tc.wantError)
			if tc.prefillOnly {
				var typeErr *json.UnmarshalTypeError
				require.ErrorAs(t, err, &typeErr)
				require.Equal(t, "number", typeErr.Value)
			} else {
				var payloadErr *renderPayloadError
				require.ErrorAs(t, err, &payloadErr)
				require.Contains(t, payloadErr.cause.Error(), "private_render_tool")
			}
			require.Nil(t, request.Body.TokenizedRequest)
			require.Equal(t, tc.payload, request.Body.WirePayload())
			span := tokenizeSpan(t, recorder)
			require.Equal(t, codes.Error, span.Status().Code)
			for _, secret := range []string{privateValue, "private_render_tool"} {
				assert.NotContains(t, err.Error(), secret)
				assert.NotContains(t, span.Status().Description, secret)
			}
		})
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
