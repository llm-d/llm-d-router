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

package handlers

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	grpcmetadata "google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	upstreamTraceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0c9902b7-01"
	// upstreamTraceID is the trace ID of upstreamTraceparent: the trace the EPP
	// must join rather than starting a fresh one.
	upstreamTraceID = "4bf92f3577b34da6a3ce929d0e0e4736"
)

// scriptedProcessServer replays its messages in order, then reports EOF so
// Process returns cleanly.
type scriptedProcessServer struct {
	mockProcessServer
	ctx  context.Context
	reqs []*extProcPb.ProcessingRequest
}

func (m *scriptedProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(m.reqs) == 0 {
		return nil, io.EOF
	}
	req := m.reqs[0]
	m.reqs = m.reqs[1:]
	return req, nil
}

func (m *scriptedProcessServer) Context() context.Context { return m.ctx }

func newRequestHeaders(headers map[string]string) *extProcPb.ProcessingRequest {
	values := make([]*configPb.HeaderValue, 0, len(headers))
	for key, value := range headers {
		values = append(values, &configPb.HeaderValue{Key: key, RawValue: []byte(value)})
	}
	return &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &configPb.HeaderMap{Headers: values},
				// EndOfStream would route to a random endpoint and need a director.
				EndOfStream: false,
			},
		},
	}
}

// runProcess drives Process over a single RequestHeaders message and returns the
// lines it logged.
func runProcess(t *testing.T, headers map[string]string) []string {
	return runProcessWithContext(context.Background(), t, headers)
}

func runProcessWithContext(ctx context.Context, t *testing.T, headers map[string]string) []string {
	t.Helper()

	var logged []string
	capture := funcr.New(func(prefix, args string) {
		logged = append(logged, prefix+" "+args)
	}, funcr.Options{Verbosity: 2})

	srv := &scriptedProcessServer{
		ctx:  log.IntoContext(ctx, capture),
		reqs: []*extProcPb.ProcessingRequest{newRequestHeaders(headers)},
	}
	require.NoError(t, NewStreamingServer(nil, nil, nil, 0).Process(srv))

	return logged
}

func TestProcessCorrelatesRequestLogsWithGRPCMetadata(t *testing.T) {
	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

	const metadataTraceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0c9902b7-01"
	ctx := grpcmetadata.NewIncomingContext(context.Background(), grpcmetadata.Pairs("traceparent", metadataTraceparent))
	entry := entryLine(t, runProcessWithContext(ctx, t, map[string]string{
		"x-request-id": "req-grpc-metadata",
	}))

	require.Contains(t, entry, upstreamTraceID, "EPP should join the trace from incoming gRPC metadata")
}

func entryLine(t *testing.T, logged []string) string {
	t.Helper()

	for _, rec := range logged {
		if strings.Contains(rec, "EPP received request") {
			return rec
		}
	}
	t.Fatalf("entry-point log line not found in %v", logged)
	return ""
}

// Pins the entry-point wiring rather than the helper: this fails if server.go
// stops installing the enriched logger.
func TestProcessCorrelatesRequestLogsWithTrace(t *testing.T) {
	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

	entry := entryLine(t, runProcess(t, map[string]string{
		"traceparent":  upstreamTraceparent,
		"x-request-id": "req-correlation-1",
	}))

	require.Contains(t, entry, tracing.LogKeyTraceID)
	require.Contains(t, entry, tracing.LogKeySpanID)
	require.Contains(t, entry, upstreamTraceID, "must join the upstream trace, not start a fresh one")
	require.Contains(t, entry, "req-correlation-1", "correlation must not displace the request ID")
	require.Equal(t, 1, strings.Count(entry, tracing.LogKeyTraceID), "trace_id must appear once: %q", entry)
}

// With tracing off the span context is invalid, so request logs are unchanged.
func TestProcessOmitsCorrelationWhenTracingDisabled(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())

	entry := entryLine(t, runProcess(t, map[string]string{"x-request-id": "req-correlation-2"}))

	require.Contains(t, entry, "req-correlation-2")
	require.NotContains(t, entry, tracing.LogKeyTraceID)
	require.NotContains(t, entry, tracing.LogKeySpanID)
}

// agentIdentityDirector resolves the fairness identity inside HandleRequest, as
// the real Director does after the request span opened.
type agentIdentityDirector struct {
	mockDirector
	err error
}

func (d *agentIdentityDirector) HandleRequest(ctx context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	tracing.SetRequestAttribution(ctx, "agent-7", tracing.AttributionSourceAgentIdentity)
	return reqCtx, d.err
}

// The request span opens before the Director resolves the fairness identity, so
// Process must refresh it afterwards on both the success and error paths.
func TestProcessRefreshesRequestSpanAfterDirectorResolvesFairness(t *testing.T) {
	for name, directorErr := range map[string]error{
		"success": nil,
		"error":   errors.New("stop after attribution"),
	} {
		t.Run(name, func(t *testing.T) {
			recorder := tracetest.NewSpanRecorder()
			useTracerProvider(t, sdktrace.NewTracerProvider(
				sdktrace.WithSpanProcessor(tracing.NewRequestAttributionProcessor()),
				sdktrace.WithSpanProcessor(recorder),
			))

			srv := &scriptedProcessServer{
				ctx: context.Background(),
				reqs: []*extProcPb.ProcessingRequest{
					newRequestHeaders(map[string]string{":path": "/v1/completions"}),
					{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
						Body:        []byte(`{"model":"m","prompt":"hi"}`),
						EndOfStream: true,
					}}},
				},
			}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			_ = NewStreamingServer(nil, &agentIdentityDirector{err: directorErr}, registry, 0).Process(srv)

			var requestSpans []sdktrace.ReadOnlySpan
			for _, span := range recorder.Ended() {
				if span.Name() == "request" {
					requestSpans = append(requestSpans, span)
				}
			}
			require.Len(t, requestSpans, 1)

			attrs := attribute.NewSet(requestSpans[0].Attributes()...)
			id, hasID := attrs.Value(semconv.LLMDEPPFairnessIDKey)
			source, hasSource := attrs.Value(semconv.LLMDEPPFairnessSourceKey)
			require.True(t, hasID && hasSource, "fairness attribution must be paired")
			require.Equal(t, "agent-7", id.AsString())
			require.Equal(t, tracing.AttributionSourceAgentIdentity, source.AsString())
		})
	}
}

func TestProcessExportsToolCallingTelemetryOverOTLP(t *testing.T) {
	exported := make(chan *coltracepb.ExportTraceServiceRequest, 16)
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/traces" {
			t.Errorf("unexpected export path %s", r.URL.Path)
		}
		payload, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read exported traces: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		request := new(coltracepb.ExportTraceServiceRequest)
		if err := proto.Unmarshal(payload, request); err != nil {
			t.Errorf("decode exported traces: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		select {
		case exported <- request:
		default:
			t.Error("unexpected number of trace exports")
		}
		w.Header().Set("Content-Type", "application/x-protobuf")
	}))
	t.Cleanup(collector.Close)

	previousProvider, previousPropagator, previousHandler := otel.GetTracerProvider(), otel.GetTextMapPropagator(), otel.GetErrorHandler()
	t.Cleanup(func() {
		otel.SetTracerProvider(previousProvider)
		otel.SetTextMapPropagator(previousPropagator)
		otel.SetErrorHandler(previousHandler)
	})
	// Keep exporter configuration independent of the developer's collector settings.
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
	shutdown, err := tracing.InitTracing(t.Context(), logr.Discard(), "tool-calling-test")
	require.NoError(t, err)
	flush := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		require.NoError(t, shutdown(ctx))
	}
	t.Cleanup(flush)

	eppmetrics.Register()
	expectedBySpanID := make(map[string][]attribute.KeyValue)
	for _, api := range []struct {
		surface  reqcommon.APIType
		path     string
		tools    string
		choice   any
		fields   []toolcalling.Field
		response string
		event    string
	}{
		{
			surface: reqcommon.APITypeChatCompletions, path: "/v1/chat/completions",
			tools:  `[{"type":"function","function":{"name":"sentinel_name","parameters":{"type":"object","properties":{"sentinel_schema":{"type":"string"}}}}}]`,
			choice: "auto", fields: []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice, toolcalling.FieldParallelToolCalls, toolcalling.FieldResponseFormat},
			response: `{"choices":[{"message":{"tool_calls":[{"function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}`,
			event:    `data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}` + "\n\n",
		},
		{
			surface: reqcommon.APITypeMessages, path: "/v1/messages",
			tools:  `[{"name":"sentinel_name","input_schema":{"type":"object","properties":{"sentinel_schema":{"type":"string"}}}}]`,
			choice: map[string]any{"type": "auto"}, fields: []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice},
			response: `{"content":[{"type":"tool_use","name":"sentinel_name","input":{"secret":"sentinel_arguments"}}]}`,
			event:    `data: {"type":"content_block_start","content_block":{"type":"tool_use","name":"sentinel_name","input":{"secret":"sentinel_arguments"}}}` + "\n\n",
		},
	} {
		for _, mode := range []string{"JSON", "SSE", "non-tool"} {
			t.Run(api.surface.String()+"/"+mode, func(t *testing.T) {
				eppmetrics.Reset()
				body := map[string]any{"model": "m", "max_tokens": 16, "messages": []any{map[string]any{"role": "user", "content": "sentinel_prompt"}}}
				if mode != "non-tool" {
					body[string(toolcalling.FieldTools)] = json.RawMessage(api.tools)
					body[string(toolcalling.FieldToolChoice)] = api.choice
					if api.surface == reqcommon.APITypeChatCompletions {
						body[string(toolcalling.FieldParallelToolCalls)] = false
						body[string(toolcalling.FieldResponseFormat)] = map[string]any{"type": "json_object"}
					}
				}
				requestBody, err := json.Marshal(body)
				require.NoError(t, err)
				contentType, chunks := "application/json", []string{api.response}
				if mode == "SSE" {
					contentType = "text/event-stream"
					chunks = []string{api.event[:3], api.event[3:]}
				}
				srv := &replayProcessServer{
					ctx: context.Background(),
					reqs: []*extProcPb.ProcessingRequest{
						newRequestHeaders(map[string]string{":path": api.path, "traceparent": upstreamTraceparent, reqcommon.RequestIDHeaderKey: api.surface.String() + "-" + mode}),
						{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{Body: requestBody, EndOfStream: true}}},
						{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{{Key: "content-type", RawValue: []byte(contentType)}}}}}},
					},
				}
				for i, chunk := range chunks {
					srv.reqs = append(srv.reqs, &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{Body: []byte(chunk), EndOfStream: i == len(chunks)-1}}})
				}
				registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser(), anthropic.NewAnthropicParser()}, logr.Discard())
				require.NoError(t, NewStreamingServer(nil, &mockDirector{}, registry, 0).Process(srv))

				var forwardedRequest, forwardedResponse []byte
				outboundTraceparent := ""
				for _, response := range srv.sentResponses {
					if header := response.GetRequestHeaders(); header != nil {
						for _, option := range header.GetResponse().GetHeaderMutation().GetSetHeaders() {
							if option.GetHeader().GetKey() == "traceparent" {
								outboundTraceparent = string(option.GetHeader().GetRawValue())
							}
						}
					}
					if chunk := response.GetRequestBody(); chunk != nil {
						forwardedRequest = append(forwardedRequest, chunk.GetResponse().GetBodyMutation().GetStreamedResponse().GetBody()...)
					}
					if chunk := response.GetResponseBody(); chunk != nil {
						forwardedResponse = append(forwardedResponse, chunk.GetResponse().GetBodyMutation().GetStreamedResponse().GetBody()...)
					}
				}
				require.Equal(t, requestBody, forwardedRequest)
				require.Equal(t, strings.Join(chunks, ""), string(forwardedResponse))
				parts := strings.Split(outboundTraceparent, "-")
				require.Len(t, parts, 4)
				require.Equal(t, upstreamTraceID, parts[1])

				snapshot, err := toolcalling.CaptureRequestJSON(api.surface, requestBody)
				require.NoError(t, err)
				statuses, err := toolcalling.CompareRequests(snapshot, snapshot)
				require.NoError(t, err)
				attrs := snapshot.SpanAttributes(statuses)
				attrs = append(attrs, (toolcalling.ResponseSummary{ToolCallingRequested: mode != "non-tool", UpstreamToolCallPresent: true, ForwardedToolCallPresent: true}).SpanAttributes()...)
				expectedBySpanID[parts[2]] = attrs

				families, err := ctrlmetrics.Registry.Gather()
				require.NoError(t, err)
				foundMetric := false
				for _, family := range families {
					if family.GetName() != "llm_d_epp_tool_calling_field_status_total" {
						continue
					}
					foundMetric = true
					if mode == "non-tool" {
						require.Empty(t, family.GetMetric())
						continue
					}
					require.Len(t, family.GetMetric(), len(api.fields))
					for _, sample := range family.GetMetric() {
						labels := make(map[string]string)
						for _, label := range sample.GetLabel() {
							labels[label.GetName()] = label.GetValue()
						}
						require.Len(t, labels, 4)
						require.Equal(t, toolcalling.ComponentEPP, labels[toolcalling.MetricLabelComponent])
						require.Equal(t, toolcalling.DirectionRequest, labels[toolcalling.MetricLabelDirection])
						require.Equal(t, string(toolcalling.FieldStatusPreserved), labels[toolcalling.MetricLabelStatus])
						require.Contains(t, api.fields, toolcalling.Field(labels[toolcalling.MetricLabelField]))
						require.Equal(t, float64(1), sample.GetCounter().GetValue())
					}
				}
				require.Equal(t, mode != "non-tool", foundMetric, "only tool requests should emit field metrics")
			})
		}
	}
	flush()
	for len(exported) > 0 {
		request := <-exported
		require.NotContains(t, request.String(), "sentinel_")
		for _, resourceSpans := range request.GetResourceSpans() {
			for _, scopeSpans := range resourceSpans.GetScopeSpans() {
				for _, span := range scopeSpans.GetSpans() {
					id := hex.EncodeToString(span.GetSpanId())
					expected, exists := expectedBySpanID[id]
					require.True(t, exists, "unexpected exported span %s", span.GetName())
					delete(expectedBySpanID, id)
					require.Equal(t, upstreamTraceID, hex.EncodeToString(span.GetTraceId()))
					require.Equal(t, strings.Split(upstreamTraceparent, "-")[2], hex.EncodeToString(span.GetParentSpanId()))
					actual := make(map[string]*commonpb.AnyValue)
					for _, attr := range span.GetAttributes() {
						if strings.HasPrefix(attr.GetKey(), "llm_d.tool_calling.") {
							actual[attr.GetKey()] = attr.GetValue()
						}
					}
					require.Len(t, actual, len(expected))
					for _, want := range expected {
						require.Contains(t, actual, string(want.Key))
						value := new(commonpb.AnyValue)
						switch typed := want.Value.AsInterface().(type) {
						case string:
							value.Value = &commonpb.AnyValue_StringValue{StringValue: typed}
						case bool:
							value.Value = &commonpb.AnyValue_BoolValue{BoolValue: typed}
						default:
							t.Fatalf("unexpected attribute type for %s", want.Key)
						}
						require.True(t, proto.Equal(value, actual[string(want.Key)]), "attribute %s", want.Key)
					}
				}
			}
		}
	}
	require.Empty(t, expectedBySpanID, "every request span must reach the OTLP collector")
}

// useTracerProvider installs tp and the W3C propagator for the duration of the
// test, restoring the previous globals afterwards.
func useTracerProvider(t *testing.T, tp trace.TracerProvider) {
	t.Helper()

	prevTP, prevProp := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() {
		otel.SetTracerProvider(prevTP)
		otel.SetTextMapPropagator(prevProp)
	})
}
