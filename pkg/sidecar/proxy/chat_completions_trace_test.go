/*
Copyright 2025 The llm-d Authors.

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

package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

func TestPDToolCallingStageSpans(t *testing.T) {
	for _, connector := range []string{"nixl", "parallel", "p2p", "mooncake", "sglang"} {
		for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
			for _, toolRequest := range []bool{true, false} {
				t.Run(fmt.Sprintf("%s/%s/tools=%t", connector, api, toolRequest), func(t *testing.T) {
					recorder := tracetest.NewSpanRecorder()
					provider := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()), sdktrace.WithSpanProcessor(recorder))
					t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
					useTracerProviderForTest(t, provider)
					upstreamContexts := make(chan trace.SpanContext, 2)
					prefill := httptest.NewServer(statusHandler(http.StatusOK, `{"kv_transfer_params":{}}`))
					defer prefill.Close()
					prefillURL, err := url.Parse(prefill.URL)
					require.NoError(t, err)
					s := NewProxy(Config{DecoderURL: prefillURL, KVConnector: constants.KVConnectorNIXLV2,
						MoRIIOWriteMode: connector == "parallel", MoRIIOParallelDispatch: connector == "parallel",
						MoRIIODecodePodIP: "127.0.0.1", MoRIIOParallelDecodeWaitTimeout: time.Second})
					s.allowlistValidator = &AllowlistValidator{}
					s.prefillerProxies.Add(prefillURL.Host, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						upstreamContexts <- trace.SpanFromContext(r.Context()).SpanContext()
						statusHandler(http.StatusOK, `{"kv_transfer_params":{}}`).ServeHTTP(w, r)
					}))
					s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						upstreamContexts <- trace.SpanFromContext(r.Context()).SpanContext()
						statusHandler(http.StatusOK, `{"choices":[]}`).ServeHTTP(w, r)
					})
					body := map[string]any{reqcommon.FieldMessages: []any{}}
					if toolRequest {
						body[string(toolcalling.FieldTools)] = []any{map[string]any{"name": "private_tool_name", "description": "private_schema"}}
						choice := map[string]any{"type": "function", "function": map[string]any{"name": "private_tool_name"}}
						if api == reqcommon.APITypeMessages {
							choice = map[string]any{"type": "tool", "name": "private_tool_name"}
						}
						body[string(toolcalling.FieldToolChoice)] = choice
					}
					encoded, err := json.Marshal(body)
					require.NoError(t, err)
					ctx, root := tracing.Tracer(tracerScope).Start(context.Background(), "forward_request")
					r := httptest.NewRequest(http.MethodPost, api.Path(), strings.NewReader(string(encoded))).WithContext(ctx)
					rw := httptest.NewRecorder()
					switch connector {
					case "nixl", "parallel":
						s.handleNIXLV2(rw, r, prefillURL.Host, "", api)
					case "p2p":
						s.handleP2P(rw, r, prefillURL.Host, "", api)
					case "mooncake":
						s.mooncakeEngineIDs.Add(prefillURL.Host, map[string]string{"0": "engine"})
						s.handleMooncake(rw, r, prefillURL.Host, api)
					case "sglang":
						s.handleSGLang(rw, r, prefillURL.Host)
					}
					root.End()
					require.Equal(t, http.StatusOK, rw.Code, rw.Body.String())
					require.Eventually(t, func() bool { return len(recorder.Ended()) == 3 }, time.Second, time.Millisecond)
					received := map[trace.SpanID]bool{}
					for len(upstreamContexts) > 0 {
						upstream := <-upstreamContexts
						require.Equal(t, root.SpanContext().TraceID(), upstream.TraceID())
						received[upstream.SpanID()] = true
					}
					for _, span := range recorder.Ended() {
						if span.Name() == "forward_request" {
							continue
						}
						require.True(t, received[span.SpanContext().SpanID()], "%s request must carry its stage span", span.Name())
						attrs := map[string]any{}
						for _, attr := range span.Attributes() {
							require.NotContains(t, attr.Value.String(), "private_")
							if strings.HasPrefix(string(attr.Key), "llm_d.tool_calling.") {
								attrs[string(attr.Key)] = attr.Value.AsInterface()
							}
						}
						want := map[string]any{}
						if toolRequest {
							want = map[string]any{
								string(semconv.LLMDToolCallingAPISurfaceKey):            api.String(),
								string(semconv.LLMDToolCallingPresentKey):               true,
								string(semconv.LLMDToolCallingToolChoiceKey):            "named",
								string(semconv.LLMDToolCallingToolCountKey):             "1",
								string(semconv.LLMDToolCallingFieldToolsStatusKey):      "preserved",
								string(semconv.LLMDToolCallingFieldToolChoiceStatusKey): "preserved",
							}
						}
						require.Equal(t, want, attrs, "%s attributes", span.Name())
					}
				})
			}
		}
	}
}

func TestToolCallingStageSpanComparisons(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()), sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	const original = `{"tools":[],"tool_choice":"auto"}`
	request, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(original))
	require.NoError(t, err)
	for _, stage := range []struct {
		name   string
		body   string
		tools  string
		choice string
	}{
		{name: "prefill", body: `{"tool_choice":"none"}`, tools: "dropped", choice: "changed"},
		{name: "decode", body: original, tools: "preserved", choice: "preserved"},
		{name: "invalid outbound capture", body: `{`},
	} {
		_, span := provider.Tracer(tracerScope).Start(context.Background(), stage.name)
		request.compareDispatchedForSpan([]byte(stage.body), span)
		span.End()
		ended := recorder.Ended()
		attrs := map[string]any{}
		for _, attr := range ended[len(ended)-1].Attributes() {
			attrs[string(attr.Key)] = attr.Value.AsInterface()
		}
		if stage.tools == "" {
			require.Empty(t, attrs)
		} else {
			require.Equal(t, stage.tools, attrs[string(semconv.LLMDToolCallingFieldToolsStatusKey)])
			require.Equal(t, stage.choice, attrs[string(semconv.LLMDToolCallingFieldToolChoiceStatusKey)])
		}
	}
	assertToolCallingStatuses(t, request.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
		toolcalling.FieldTools: toolcalling.FieldStatusDropped, toolcalling.FieldToolChoice: toolcalling.FieldStatusChanged,
	})
	_, span := provider.Tracer(tracerScope).Start(context.Background(), "failed inbound capture")
	(*toolCallingRequest)(nil).compareDispatchedForSpan([]byte(original), span)
	span.End()
	ended := recorder.Ended()
	require.Empty(t, ended[len(ended)-1].Attributes())
	request, err = captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(original))
	require.NoError(t, err)
	_, unrecorded := noop.NewTracerProvider().Tracer(tracerScope).Start(context.Background(), "unrecorded")
	request.compareDispatchedForSpan([]byte(`{}`), unrecorded)
	unrecorded.End()
	assertToolCallingStatuses(t, request.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
		toolcalling.FieldTools: toolcalling.FieldStatusDropped, toolcalling.FieldToolChoice: toolcalling.FieldStatusDropped,
	})
}

func TestConcurrentPDStageSpansDoNotWaitForPrefill(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()), sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	useTracerProviderForTest(t, provider)
	decoderURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)
	s := NewProxy(Config{DecoderURL: decoderURL})
	started, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	s.prefillerProxies.Add("prefill:8000", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-release
		assert.NoError(t, r.Context().Err(), "asynchronous prefill must retain its cancellation policy")
		statusHandler(http.StatusBadRequest, `{}`).ServeHTTP(w, r)
	}))
	s.decoderProxy = statusHandler(http.StatusOK, `{"choices":[]}`)
	const original = `{"tools":[],"tool_choice":"auto"}`
	request, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(original))
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, nil).WithContext(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.runConcurrentPD(httptest.NewRecorder(), r, []byte(`{"tools":[]}`), []byte(original), "prefill:8000", constants.KVConnectorSGLang, nil, request)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("decode waited for blocked prefill")
	}
	cancel()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("prefill was not dispatched")
	}
	ended := recorder.Ended()
	require.Len(t, ended, 1)
	require.Equal(t, "decode", ended[0].Name())
	foundChoice := false
	for _, attr := range ended[0].Attributes() {
		if attr.Key == semconv.LLMDToolCallingFieldToolChoiceStatusKey {
			foundChoice = true
			require.Equal(t, "preserved", attr.Value.AsString(), "decode status must not inherit the prefill drop")
		}
	}
	require.True(t, foundChoice)
	assertToolCallingStatuses(t, request.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
		toolcalling.FieldTools: toolcalling.FieldStatusPreserved, toolcalling.FieldToolChoice: toolcalling.FieldStatusDropped,
	})
	counts := toolFieldCounters(t)
	unblock()
	require.Eventually(t, func() bool { return len(recorder.Ended()) == 2 }, time.Second, time.Millisecond)
	require.Equal(t, counts, toolFieldCounters(t), "late prefill completion must not duplicate request metrics")
	for _, span := range recorder.Ended() {
		if span.Name() == "prefill" {
			require.Equal(t, codes.Error, span.Status().Code)
			for _, attr := range span.Attributes() {
				if attr.Key == semconv.LLMDToolCallingFieldToolChoiceStatusKey {
					require.Equal(t, "dropped", attr.Value.AsString())
				}
			}
		}
	}
}

// useTracerProviderForTest installs tp for the duration of the test.
func useTracerProviderForTest(t *testing.T, tp trace.TracerProvider) {
	t.Helper()

	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
}

// runPrefillHandler drives disaggregatedPrefillHandler with the prefill header
// set, appending every logged line to logged and returning the logger the
// downstream connector received through the request context.
func runPrefillHandler(t *testing.T, logged *[]string) (downstream logr.Logger) {
	t.Helper()

	capture := funcr.New(func(prefix, args string) {
		*logged = append(*logged, prefix+" "+args)
	}, funcr.Options{Verbosity: 5})

	s := NewProxy(Config{Port: "8000"})
	s.logger = capture
	s.allowlistValidator = &AllowlistValidator{}
	s.dataParallelProxies = make(map[string]http.Handler)
	s.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ string, _ reqcommon.APIType) {
		downstream = log.FromContext(r.Context())
	}

	req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, http.NoBody)
	req.Header.Set(routing.PrefillEndpointHeader, "prefill-pod:8000")
	s.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(httptest.NewRecorder(), req)

	return downstream
}

// The handler's own log lines carry the forward_request trace, and the request
// context handed to connectors carries the same correlated logger.
func TestPrefillHandlerCorrelatesLogsWithTrace(t *testing.T) {
	useTracerProviderForTest(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

	var logged []string
	downstream := runPrefillHandler(t, &logged)

	var handlerLine string
	for _, rec := range logged {
		if strings.Contains(rec, "using P/D protocol") {
			handlerLine = rec
		}
	}
	if handlerLine == "" {
		t.Fatalf("handler log line not found in %v", logged)
	}
	if !strings.Contains(handlerLine, tracing.LogKeyTraceID) || !strings.Contains(handlerLine, tracing.LogKeySpanID) {
		t.Errorf("handler line missing correlation fields: %q", handlerLine)
	}
	if got := strings.Count(handlerLine, tracing.LogKeyTraceID); got != 1 {
		t.Errorf("trace_id appears %d times, want 1: %q", got, handlerLine)
	}

	// A connector that adopts the request context gets the same trace.
	before := len(logged)
	downstream.Info("connector line")
	if len(logged) != before+1 {
		t.Fatalf("expected the downstream logger to emit one line, got %d", len(logged)-before)
	}
	if !strings.Contains(logged[before], tracing.LogKeyTraceID) {
		t.Errorf("downstream logger is not correlated: %q", logged[before])
	}
}

// With tracing off the span context is invalid, so nothing is added.
func TestPrefillHandlerOmitsCorrelationWhenTracingDisabled(t *testing.T) {
	useTracerProviderForTest(t, noop.NewTracerProvider())

	var logged []string
	runPrefillHandler(t, &logged)

	for _, rec := range logged {
		if strings.Contains(rec, tracing.LogKeyTraceID) || strings.Contains(rec, tracing.LogKeySpanID) {
			t.Errorf("unexpected correlation fields with tracing disabled: %q", rec)
		}
	}
}
