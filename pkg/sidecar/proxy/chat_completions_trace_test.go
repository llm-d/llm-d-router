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
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

func TestSidecarToolCallingOTLP(t *testing.T) {
	const toolSecret, schemaSecret, argumentSecret, responseSecret = "private_tool_otlp", "private_schema_otlp", "private_arguments_otlp", "private_response_otlp"
	for _, connector := range []string{"direct", "nixl", "sglang"} {
		for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
			for _, outcome := range []string{"success", "error", "interrupted"} {
				for _, toolRequest := range []bool{true, false} {
					t.Run(fmt.Sprintf("%s/%s/%s/tools=%t", connector, api, outcome, toolRequest), func(t *testing.T) {
						var mu sync.Mutex
						var logged []string
						logger := funcr.New(func(prefix, args string) {
							mu.Lock()
							defer mu.Unlock()
							logged = append(logged, prefix+" "+args)
						}, funcr.Options{Verbosity: logging.TRACE})
						exports, shutdown := startSidecarOTLPCollector(t, logger)
						recorder := tracetest.NewSpanRecorder()
						otel.GetTracerProvider().(*sdktrace.TracerProvider).RegisterSpanProcessor(recorder)
						upstreamContexts := make(chan struct {
							stage   string
							context trace.SpanContext
						}, 2)
						readRequest := func(r *http.Request, stage string) {
							_, err := io.Copy(io.Discard, r.Body)
							assert.NoError(t, err)
							ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))
							upstreamContexts <- struct {
								stage   string
								context trace.SpanContext
							}{stage, trace.SpanContextFromContext(ctx)}
						}
						prefillReceived := make(chan struct{})
						prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							readRequest(r, "prefill")
							close(prefillReceived)
							statusHandler(http.StatusOK, `{"kv_transfer_params":{}}`).ServeHTTP(w, r)
						}))
						defer prefill.Close()
						prefillURL, err := url.Parse(prefill.URL)
						require.NoError(t, err)
						decoder := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							stage := "decode"
							if connector == "direct" {
								stage = "forward_request"
							}
							readRequest(r, stage)
							if connector == "sglang" {
								<-prefillReceived
							}
							if outcome == "interrupted" {
								w.Header().Set("Content-Length", "1000")
								_, _ = io.WriteString(w, responseSecret)
								return
							}
							code := http.StatusOK
							if outcome == "error" {
								code = http.StatusBadRequest
							}
							statusHandler(code, fmt.Sprintf(`{"content":"%s"}`, responseSecret)).ServeHTTP(w, r)
						}))
						defer decoder.Close()
						decoderURL, err := url.Parse(decoder.URL)
						require.NoError(t, err)
						kvConnector := constants.KVConnectorNIXLV2
						if connector == "sglang" {
							kvConnector = constants.KVConnectorSGLang
						}
						s := NewProxy(Config{Port: "8000", DecoderURL: decoderURL, KVConnector: kvConnector, EnableToolCallingTelemetry: true})
						s.logger = logger
						s.allowlistValidator = &AllowlistValidator{}
						handler := otelhttp.NewHandler(s.createRoutes(), "sidecar-http", otelhttp.WithSpanNameFormatter(func(_ string, r *http.Request) string {
							return r.Method + " " + r.URL.Path
						}))
						completed := make(chan struct{})
						sidecar := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							defer close(completed)
							handler.ServeHTTP(w, r)
						}))
						defer sidecar.Close()
						body := map[string]any{reqcommon.FieldMessages: []any{map[string]any{"content": argumentSecret}}}
						if toolRequest {
							body[string(toolcalling.FieldTools)] = []any{map[string]any{"name": toolSecret, "description": schemaSecret}}
							choice := map[string]any{"type": "function", "function": map[string]any{"name": toolSecret}}
							if api == reqcommon.APITypeMessages {
								choice = map[string]any{"type": "tool", "name": toolSecret}
							}
							body[string(toolcalling.FieldToolChoice)] = choice
						}
						encoded, err := json.Marshal(body)
						require.NoError(t, err)
						r, err := http.NewRequest(http.MethodPost, sidecar.URL+api.Path(), strings.NewReader(string(encoded)))
						require.NoError(t, err)
						traceState, err := trace.ParseTraceState("test=tool-telemetry")
						require.NoError(t, err)
						incoming := trace.NewSpanContext(trace.SpanContextConfig{TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2}, TraceFlags: trace.FlagsSampled, TraceState: traceState, Remote: true})
						otel.GetTextMapPropagator().Inject(trace.ContextWithRemoteSpanContext(context.Background(), incoming), propagation.HeaderCarrier(r.Header))
						if connector != "direct" {
							r.Header.Set(routing.PrefillEndpointHeader, prefillURL.Host)
						}
						before := toolFieldCounters(t)
						resp, requestErr := sidecar.Client().Do(r)
						if requestErr == nil {
							response, readErr := io.ReadAll(resp.Body)
							require.NoError(t, resp.Body.Close())
							if outcome == "interrupted" {
								require.Error(t, readErr)
							} else {
								require.NoError(t, readErr)
								require.Contains(t, string(response), responseSecret)
								code := http.StatusOK
								if outcome == "error" {
									code = http.StatusBadRequest
								}
								require.Equal(t, code, resp.StatusCode)
							}
						} else {
							require.Equal(t, "interrupted", outcome, requestErr)
						}
						select {
						case <-completed:
						case <-time.After(2 * time.Second):
							t.Fatal("request handler did not finish")
						}
						if connector == "sglang" {
							require.Eventually(t, func() bool {
								for _, span := range recorder.Ended() {
									if span.Name() == "prefill" {
										return true
									}
								}
								return false
							}, 2*time.Second, time.Millisecond)
						}
						require.NoError(t, shutdown(context.Background()))
						spans := map[string]*tracepb.Span{}
						for len(exports) > 0 {
							exported := <-exports
							wireJSON, err := protojson.Marshal(exported)
							assert.NoError(t, err)
							for _, secret := range []string{toolSecret, schemaSecret, argumentSecret, responseSecret} {
								require.NotContains(t, string(wireJSON), secret)
							}
							for _, resource := range exported.GetResourceSpans() {
								for _, scope := range resource.GetScopeSpans() {
									for _, span := range scope.GetSpans() {
										require.Equal(t, incoming.TraceID().String(), hex.EncodeToString(span.GetTraceId()))
										spans[hex.EncodeToString(span.GetSpanId())] = span
									}
								}
							}
						}
						require.NotEmpty(t, spans)
						wantAttrs := map[string]any{}
						if toolRequest {
							wantAttrs = map[string]any{
								string(semconv.LLMDToolCallingAPISurfaceKey): api.String(), string(semconv.LLMDToolCallingPresentKey): true,
								string(semconv.LLMDToolCallingToolChoiceKey): "named", string(semconv.LLMDToolCallingToolCountKey): "1",
								string(semconv.LLMDToolCallingFieldToolsStatusKey): "preserved", string(semconv.LLMDToolCallingFieldToolChoiceStatusKey): "preserved",
							}
						}
						byName := map[string]*tracepb.Span{}
						for _, span := range spans {
							byName[span.GetName()] = span
							attrs := map[string]any{}
							for _, attr := range span.GetAttributes() {
								if strings.HasPrefix(attr.GetKey(), "llm_d.tool_calling.") {
									if value, ok := attr.GetValue().GetValue().(*commonpb.AnyValue_BoolValue); ok {
										attrs[attr.GetKey()] = value.BoolValue
									} else {
										attrs[attr.GetKey()] = attr.GetValue().GetStringValue()
									}
								}
							}
							if span.GetName() == "forward_request" || span.GetName() == "prefill" || span.GetName() == "decode" {
								require.Equal(t, wantAttrs, attrs, span.GetName())
							} else {
								require.Empty(t, attrs, span.GetName())
							}
						}
						serverSpanName := http.MethodPost + " " + api.Path()
						require.Contains(t, byName, serverSpanName)
						require.Contains(t, byName, "forward_request")
						require.Equal(t, incoming.SpanID().String(), hex.EncodeToString(byName[serverSpanName].GetParentSpanId()))
						require.Equal(t, hex.EncodeToString(byName[serverSpanName].GetSpanId()), hex.EncodeToString(byName["forward_request"].GetParentSpanId()))
						count := 1
						if connector != "direct" {
							count = 2
							for _, stage := range []string{"prefill", "decode"} {
								require.Contains(t, byName, stage)
							}
							rootID, prefillID := hex.EncodeToString(byName["forward_request"].GetSpanId()), hex.EncodeToString(byName["prefill"].GetSpanId())
							require.Equal(t, rootID, hex.EncodeToString(byName["prefill"].GetParentSpanId()))
							require.Contains(t, []string{rootID, prefillID}, hex.EncodeToString(byName["decode"].GetParentSpanId()))
						}
						require.Len(t, upstreamContexts, count)
						for len(upstreamContexts) > 0 {
							upstream := <-upstreamContexts
							require.Equal(t, incoming.TraceID(), upstream.context.TraceID())
							require.Equal(t, incoming.TraceState(), upstream.context.TraceState())
							require.Equal(t, incoming.TraceFlags(), upstream.context.TraceFlags())
							clientSpan := spans[upstream.context.SpanID().String()]
							require.NotNil(t, clientSpan)
							require.Equal(t, tracepb.Span_SPAN_KIND_CLIENT, clientSpan.GetKind())
							parent := spans[hex.EncodeToString(clientSpan.GetParentSpanId())]
							require.NotNil(t, parent)
							require.Equal(t, upstream.stage, parent.GetName())
						}
						for key, value := range toolFieldCounters(t) {
							change := value - before[key]
							if toolRequest && (key[0] == string(toolcalling.FieldTools) || key[0] == string(toolcalling.FieldToolChoice)) && key[1] == string(toolcalling.FieldStatusPreserved) {
								require.Equal(t, float64(1), change)
							} else {
								require.Zero(t, change)
							}
						}
						mu.Lock()
						defer mu.Unlock()
						for _, secret := range []string{toolSecret, schemaSecret, argumentSecret, responseSecret} {
							require.NotContains(t, strings.Join(logged, "\n"), secret)
							require.NotContains(t, fmt.Sprint(gatherFamily(t, metricToolFieldStatus)), secret)
						}
					})
				}
			}
		}
	}
}

func startSidecarOTLPCollector(t *testing.T, logger logr.Logger) (<-chan *coltracepb.ExportTraceServiceRequest, func(context.Context) error) {
	t.Helper()
	exports := make(chan *coltracepb.ExportTraceServiceRequest, 64)
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/v1/traces", r.URL.Path)
		assert.Equal(t, "application/x-protobuf", r.Header.Get("Content-Type"))
		encoded, err := io.ReadAll(r.Body)
		if !assert.NoError(t, err) {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		exported := &coltracepb.ExportTraceServiceRequest{}
		if !assert.NoError(t, proto.Unmarshal(encoded, exported)) {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		exports <- exported
		w.Header().Set("Content-Type", "application/x-protobuf")
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(collector.Close)
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(key, "OTEL_") {
			t.Setenv(key, "")
		}
	}
	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
	t.Setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", collector.URL+"/v1/traces")
	t.Setenv("OTEL_TRACES_SAMPLER", "always_on")
	previousProvider, previousPropagator, previousErrorHandler := otel.GetTracerProvider(), otel.GetTextMapPropagator(), otel.GetErrorHandler()
	shutdown, err := tracing.InitTracing(context.Background(), logger, "sidecar-test")
	require.NoError(t, err)
	t.Cleanup(func() {
		assert.NoError(t, shutdown(context.Background()))
		otel.SetTracerProvider(previousProvider)
		otel.SetTextMapPropagator(previousPropagator)
		otel.SetErrorHandler(previousErrorHandler)
	})
	return exports, shutdown
}

func TestToolCallingPassthroughBody(t *testing.T) {
	const body = `{"tools":[],"tool_choice":"auto"}`
	for _, tt := range []struct {
		name    string
		length  int64
		read    int
		readErr error
		want    bool
	}{
		{name: "known length without EOF", length: int64(len(body)), read: len(body), want: true},
		{name: "unknown length with EOF", length: -1, read: len(body), readErr: io.EOF, want: true},
		{name: "zero length treated as unknown", read: len(body), readErr: io.EOF, want: true},
		{name: "unknown length without EOF", length: -1, read: len(body)},
		{name: "partial known length", length: int64(len(body)), read: 4},
		{name: "short body EOF", length: int64(len(body)) + 1, read: len(body), readErr: io.EOF},
		{name: "read failure with bytes", length: int64(len(body)), read: len(body), readErr: io.ErrUnexpectedEOF},
		{name: "canceled read with bytes", length: int64(len(body)), read: len(body), readErr: context.Canceled},
		{name: "unread body", length: int64(len(body))},
	} {
		t.Run(tt.name, func(t *testing.T) {
			closeErr := errors.New("close failure")
			recorder := tracetest.NewSpanRecorder()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
			t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
			ctx, span := provider.Tracer(tracerScope).Start(context.Background(), "forward_request")
			ctx = context.WithValue(ctx, toolCallingRequestSpanKey, span)
			ctx = context.WithValue(ctx, toolCallingTelemetryEnabledKey{}, true)
			source := &toolCallingTestBody{body: body[:tt.read], err: tt.readErr, closeErr: closeErr}
			r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, nil).WithContext(ctx)
			r.Body, r.ContentLength = source, tt.length
			before := toolFieldCounters(t)
			observeToolCallingPassthrough(r, reqcommon.APITypeChatCompletions)
			require.Zero(t, source.reads, "observation must not pre-read")
			require.Equal(t, tt.length, r.ContentLength)
			if tt.read > 0 {
				buf := make([]byte, len(body))
				n, err := r.Body.Read(buf)
				require.Equal(t, body[:tt.read], string(buf[:n]))
				require.Equal(t, tt.readErr, err)
				_, _ = r.Body.Read(buf) // A second EOF must not duplicate metrics.
			}
			require.ErrorIs(t, r.Body.Close(), closeErr)
			require.Equal(t, 1, source.closes)
			want := map[[2]string]float64{}
			if tt.want {
				want[[2]string{string(toolcalling.FieldTools), string(toolcalling.FieldStatusPreserved)}] = 1
				want[[2]string{string(toolcalling.FieldToolChoice), string(toolcalling.FieldStatusPreserved)}] = 1
			}
			delta := toolFieldDelta(before, toolFieldCounters(t))
			require.Equal(t, want, delta)
			span.End()
			if tt.want {
				snapshot, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(body))
				require.NoError(t, err)
				statuses, err := toolcalling.CompareRequests(snapshot, snapshot)
				require.NoError(t, err)
				require.ElementsMatch(t, snapshot.SpanAttributes(statuses), recorder.Ended()[0].Attributes())
			} else {
				require.Empty(t, recorder.Ended()[0].Attributes())
			}
		})
	}
}

type toolCallingTestBody struct {
	body     string
	err      error
	closeErr error
	reads    int
	closes   int
}

func (b *toolCallingTestBody) Read(p []byte) (int, error) {
	b.reads++
	n := copy(p, b.body)
	b.body = b.body[n:]
	return n, b.err
}

func (b *toolCallingTestBody) Close() error {
	b.closes++
	return b.closeErr
}

func TestToolCallingPassthroughConcurrentClose(t *testing.T) {
	reader, writer := io.Pipe()
	r := withToolCallingTelemetry(httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, nil))
	r.Body, r.ContentLength = reader, -1
	observeToolCallingPassthrough(r, reqcommon.APITypeChatCompletions)
	done := make(chan error, 1)
	go func() {
		_, err := io.Copy(io.Discard, r.Body)
		done <- err
	}()
	_, err := writer.Write([]byte(`{"tools":`))
	require.NoError(t, err)
	require.NoError(t, r.Body.Close())
	require.ErrorIs(t, <-done, io.ErrClosedPipe)
	require.NoError(t, writer.Close())
}

func TestToolCallingPassthroughSkipsUnsupportedOrEmptyBody(t *testing.T) {
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses, reqcommon.APITypeCompletions} {
		for _, body := range []io.ReadCloser{nil, http.NoBody, io.NopCloser(strings.NewReader(`{"tools":[]}`))} {
			if api == reqcommon.APITypeChatCompletions && body != nil && body != http.NoBody {
				continue
			}
			r := &http.Request{Body: body}
			observeToolCallingPassthrough(r, api)
			require.True(t, body == r.Body)
		}
	}
}

func TestToolCallingRequestSpans(t *testing.T) {
	for _, route := range []string{"direct", "shared", "encoder", "chunks", "fallback"} {
		for _, fields := range []string{`"tools":[{"name":"private_tool"}],"tool_choice":"auto"`, `"response_format":{"type":"json_object"}`, `"messages":[]`} {
			t.Run(route+"/"+fields, func(t *testing.T) {
				recorder := tracetest.NewSpanRecorder()
				provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
				t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
				useTracerProviderForTest(t, provider)
				prefill := httptest.NewServer(statusHandler(http.StatusOK, `{"kv_transfer_params":{}}`))
				defer prefill.Close()
				prefillURL, err := url.Parse(prefill.URL)
				require.NoError(t, err)
				s := NewProxy(Config{Port: "8000", DecoderURL: prefillURL, EnableToolCallingTelemetry: true})
				s.allowlistValidator = &AllowlistValidator{}
				s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_, err := io.Copy(io.Discard, r.Body)
					require.NoError(t, err)
					statusHandler(http.StatusOK, chatResponse("ok", "stop", 1, 1)).ServeHTTP(w, r)
				})
				if route == "chunks" || route == "fallback" {
					s.config.DecodeChunkSize = 3
				}
				budget := 6
				if route == "fallback" {
					budget = 2
				}
				r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions,
					strings.NewReader(fmt.Sprintf(`{%s,"max_tokens":%d}`, fields, budget)))
				if route == "shared" {
					r.Header.Set(routing.PrefillEndpointHeader, prefillURL.Host)
					s.handlePDConnector = func(w http.ResponseWriter, r *http.Request, target, _ string, api reqcommon.APIType) {
						s.handleSharedStorage(w, r, target, api)
					}
				}
				if route == "encoder" {
					r.Header.Set(routing.EncoderEndpointsHeader, prefillURL.Host)
					s.handleECConnector = s.handleECNIXL
				}
				rw := httptest.NewRecorder()
				s.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(rw, r)
				require.Equal(t, http.StatusOK, rw.Code, rw.Body.String())
				snapshot, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte("{"+fields+"}"))
				require.NoError(t, err)
				statuses, err := toolcalling.CompareRequests(snapshot, snapshot)
				require.NoError(t, err)
				want := snapshot.SpanAttributes(statuses)
				foundRoot := false
				for _, span := range recorder.Ended() {
					if span.Name() != "forward_request" && span.Name() != "chunked_decode" {
						continue
					}
					foundRoot = foundRoot || span.Name() == "forward_request"
					var got []attribute.KeyValue
					for _, attr := range span.Attributes() {
						require.NotContains(t, attr.Value.String(), "private_tool")
						if strings.HasPrefix(string(attr.Key), "llm_d.tool_calling.") {
							got = append(got, attr)
						}
					}
					require.ElementsMatch(t, want, got, span.Name())
				}
				require.True(t, foundRoot)
			})
		}
	}
}

func toolCallingSpanAttributes(span sdktrace.ReadOnlySpan) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	for _, attr := range span.Attributes() {
		if strings.HasPrefix(string(attr.Key), "llm_d.tool_calling.") {
			attrs = append(attrs, attr)
		}
	}
	return attrs
}

func TestChunkedDecodeSpanExcludesPrefillOutcomes(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	useTracerProviderForTest(t, provider)
	const original = `{"messages":[],"max_tokens":6,"tools":[],"tool_choice":"auto"}`
	toolFields, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(original))
	require.NoError(t, err)
	_, err = toolFields.compareDispatched([]byte(`{"tools":[]}`))
	require.NoError(t, err)
	s := NewProxy(Config{DecodeChunkSize: 3})
	s.decoderProxy = statusHandler(http.StatusOK, chatResponse("ok", "stop", 1, 1))
	body, err := decodeRequestBody([]byte(original))
	require.NoError(t, err)
	ctx, root := provider.Tracer(tracerScope).Start(context.Background(), "forward_request")
	ctx = context.WithValue(ctx, toolCallingRequestSpanKey, root)
	ctx, decode := provider.Tracer(tracerScope).Start(ctx, "decode")
	r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(original)).WithContext(ctx)
	s.runChunkedDecodeFromMap(httptest.NewRecorder(), r, body, []byte(original), toolFields)
	decode.End()
	toolFields.report(ctx)
	root.End()
	require.Len(t, recorder.Ended(), 3)
	for _, span := range recorder.Ended()[:2] {
		require.Contains(t, toolCallingSpanAttributes(span), semconv.LLMDToolCallingFieldToolChoiceStatusKey.String("preserved"))
	}
	require.Contains(t, toolCallingSpanAttributes(recorder.Ended()[2]), semconv.LLMDToolCallingFieldToolChoiceStatusKey.String("dropped"))
	assertToolCallingStatuses(t, toolFields.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
		toolcalling.FieldTools: toolcalling.FieldStatusPreserved, toolcalling.FieldToolChoice: toolcalling.FieldStatusDropped,
	})
}

func TestChunkedDecodeSpanWithoutDispatchedChunk(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	useTracerProviderForTest(t, provider)
	const original = `{"messages":[],"max_tokens":6,"tools":[],"tool_choice":"auto"}`
	body, err := decodeRequestBody([]byte(original))
	require.NoError(t, err)
	s := NewProxy(Config{DecodeChunkSize: 3})
	s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("decode chunk must not be dispatched")
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx, root := provider.Tracer(tracerScope).Start(ctx, "forward_request")
	ctx = context.WithValue(ctx, toolCallingRequestSpanKey, root)
	ctx = context.WithValue(ctx, toolCallingTelemetryEnabledKey{}, true)
	ctx, decode := provider.Tracer(tracerScope).Start(ctx, "decode")
	r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(original)).WithContext(ctx)
	toolFields, err := captureInferenceToolCallingRequest(r, reqcommon.APITypeChatCompletions, []byte(original))
	require.NoError(t, err)
	cancel()
	s.runChunkedDecodeFromMap(httptest.NewRecorder(), r, body, []byte(original), toolFields)
	decode.End()
	root.End()
	require.Len(t, recorder.Ended(), 3)
	require.Empty(t, toolCallingSpanAttributes(recorder.Ended()[0]))
	require.Empty(t, toolCallingSpanAttributes(recorder.Ended()[1]))
	require.Contains(t, toolCallingSpanAttributes(recorder.Ended()[2]), semconv.LLMDToolCallingToolChoiceKey.String("auto"))
}

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
					s := NewProxy(Config{DecoderURL: prefillURL, KVConnector: constants.KVConnectorNIXLV2, EnableToolCallingTelemetry: true,
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
					ctx = context.WithValue(ctx, toolCallingRequestSpanKey, root)
					ctx = context.WithValue(ctx, toolCallingTelemetryEnabledKey{}, true)
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
						if span.Name() != "forward_request" {
							require.True(t, received[span.SpanContext().SpanID()], "%s request must carry its stage span", span.Name())
						}
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

func TestPDToolCallingSummaryOnSetupFailure(t *testing.T) {
	bootstrap := httptest.NewServer(statusHandler(http.StatusServiceUnavailable, `{}`))
	defer bootstrap.Close()
	bootstrapURL, err := url.Parse(bootstrap.URL)
	require.NoError(t, err)
	bootstrapPort, err := strconv.Atoi(bootstrapURL.Port())
	require.NoError(t, err)

	for _, connector := range []string{"mooncake", "nixl"} {
		for _, tt := range []struct {
			api  reqcommon.APIType
			body string
		}{
			{reqcommon.APITypeChatCompletions, `{"tools":[{}],"tool_choice":"auto"}`},
			{reqcommon.APITypeMessages, `{"tools":[{}],"tool_choice":{"type":"auto"}}`},
		} {
			t.Run(connector+"/"+tt.api.String(), func(t *testing.T) {
				recorder := tracetest.NewSpanRecorder()
				provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
				t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
				useTracerProviderForTest(t, provider)
				s := NewProxy(Config{MooncakeBootstrapPort: bootstrapPort, EnableToolCallingTelemetry: true})
				if connector == "nixl" {
					s.nixlRequestIDFn = func() (string, error) { return "", errors.New("request ID failure") }
				}
				ctx, root := tracing.Tracer(tracerScope).Start(context.Background(), "forward_request")
				ctx = context.WithValue(ctx, toolCallingRequestSpanKey, root)
				ctx = context.WithValue(ctx, toolCallingTelemetryEnabledKey{}, true)
				r := httptest.NewRequest(http.MethodPost, tt.api.Path(), strings.NewReader(tt.body)).WithContext(ctx)
				rw := httptest.NewRecorder()
				before := toolFieldCounters(t)
				if connector == "mooncake" {
					s.handleMooncake(rw, r, bootstrapURL.Host, tt.api)
				} else {
					s.handleNIXLV2(rw, r, bootstrapURL.Host, "", tt.api)
				}
				root.End()
				require.Equal(t, http.StatusBadGateway, rw.Code)
				require.Empty(t, toolFieldDelta(before, toolFieldCounters(t)))
				require.Len(t, recorder.Ended(), 1)
				require.ElementsMatch(t, []attribute.KeyValue{
					semconv.LLMDToolCallingAPISurfaceKey.String(tt.api.String()),
					semconv.LLMDToolCallingPresentKey.Bool(true),
					semconv.LLMDToolCallingToolChoiceKey.String("auto"),
					semconv.LLMDToolCallingToolCountKey.String("1"),
				}, toolCallingSpanAttributes(recorder.Ended()[0]))
			})
		}
	}
}

func TestToolCallingStageSpanComparisons(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()), sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
	const original = `{"tools":[],"tool_choice":"auto"}`
	ctx, root := provider.Tracer(tracerScope).Start(context.Background(), "forward_request")
	ctx = context.WithValue(ctx, toolCallingRequestSpanKey, root)
	ctx = context.WithValue(ctx, toolCallingTelemetryEnabledKey{}, true)
	inbound := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, nil).WithContext(ctx)
	request, err := captureInferenceToolCallingRequest(inbound, reqcommon.APITypeChatCompletions, []byte(original))
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
	request.report(ctx)
	root.End()
	rootAttrs := recorder.Ended()[len(recorder.Ended())-1].Attributes()
	require.Contains(t, rootAttrs, semconv.LLMDToolCallingFieldToolsStatusKey.String("dropped"))
	require.Contains(t, rootAttrs, semconv.LLMDToolCallingFieldToolChoiceStatusKey.String("changed"))
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
	ctx, root := provider.Tracer(tracerScope).Start(ctx, "forward_request")
	ctx = context.WithValue(ctx, toolCallingRequestSpanKey, root)
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
	root.End()
	cancel()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("prefill was not dispatched")
	}
	ended := recorder.Ended()
	require.Len(t, ended, 2)
	require.Contains(t, toolCallingSpanAttributes(ended[1]), semconv.LLMDToolCallingFieldToolChoiceStatusKey.String("dropped"))
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
	require.Eventually(t, func() bool { return len(recorder.Ended()) == 3 }, time.Second, time.Millisecond)
	require.Equal(t, ended[1].Attributes(), recorder.Ended()[1].Attributes(), "late prefill must not rewrite the ended request span")
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
