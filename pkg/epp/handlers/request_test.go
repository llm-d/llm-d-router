/*
Copyright 2025 The Kubernetes Authors.
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

package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	grpcmetadata "google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/structpb"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

func TestExtractTraceContext(t *testing.T) {
	otel.SetTextMapPropagator(propagation.TraceContext{})

	const (
		traceID = "0af7651916cd43dd8448eb211c80319c"
		spanID  = "b7ad6b7169203331"
	)

	tests := []struct {
		name         string
		headers      []*configPb.HeaderValue
		wantTraceID  string
		wantRemote   bool
		wantHasTrace bool
	}{
		{
			name: "extracts upstream traceparent",
			headers: []*configPb.HeaderValue{
				{Key: "traceparent", Value: "00-" + traceID + "-" + spanID + "-01"},
			},
			wantTraceID:  traceID,
			wantRemote:   true,
			wantHasTrace: true,
		},
		{
			name: "case-insensitive header key",
			headers: []*configPb.HeaderValue{
				{Key: "TraceParent", RawValue: []byte("00-" + traceID + "-" + spanID + "-01")},
			},
			wantTraceID:  traceID,
			wantRemote:   true,
			wantHasTrace: true,
		},
		{
			name: "no traceparent yields no remote span context",
			headers: []*configPb.HeaderValue{
				{Key: "x-test", Value: "val"},
			},
			wantHasTrace: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &extProcPb.ProcessingRequest_RequestHeaders{
				RequestHeaders: &extProcPb.HttpHeaders{
					Headers: &configPb.HeaderMap{Headers: tc.headers},
				},
			}

			ctx := extractTraceContext(context.Background(), req)
			sc := trace.SpanContextFromContext(ctx)

			assert.Equal(t, tc.wantHasTrace, sc.IsValid(), "span context validity should match")
			if tc.wantHasTrace {
				assert.Equal(t, tc.wantTraceID, sc.TraceID().String(), "trace ID should match upstream")
				assert.Equal(t, tc.wantRemote, sc.IsRemote(), "span context should be remote")
			}
		})
	}
}

func TestExtractTraceContextPrefersHTTPHeadersOverGRPCMetadata(t *testing.T) {
	otel.SetTextMapPropagator(propagation.TraceContext{})

	const (
		metadataTraceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0c9902b7-01"
		headerTraceparent   = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
	)
	ctx := grpcmetadata.NewIncomingContext(context.Background(), grpcmetadata.Pairs("traceparent", metadataTraceparent))
	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{{Key: "traceparent", Value: headerTraceparent}}},
		},
	}

	sc := trace.SpanContextFromContext(extractTraceContext(ctx, req))

	assert.True(t, sc.IsValid())
	assert.Equal(t, "0af7651916cd43dd8448eb211c80319c", sc.TraceID().String())
}

func TestHandleRequestHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		headers       []*configPb.HeaderValue
		wantHeaders   map[string]string
		wantAbsent    []string
		wantObjective string
		wantTarget    string
	}{
		{
			name: "Lowercases mixed-case header keys",
			headers: []*configPb.HeaderValue{
				{Key: "X-Test", Value: "val"},
			},
			wantHeaders: map[string]string{"x-test": "val"},
		},
		{
			name: "Prefers RawValue over Value",
			headers: []*configPb.HeaderValue{
				{Key: metadata.ObjectiveKey, RawValue: []byte("binary-id"), Value: "wrong-id"},
			},
			wantObjective: "binary-id",
		},
		{
			name: "Prefers new control headers over old aliases",
			headers: []*configPb.HeaderValue{
				{Key: metadata.OldObjectiveKey, Value: "old-objective"},
				{Key: metadata.ObjectiveKey, Value: "new-objective"},
				{Key: metadata.OldModelNameRewriteKey, Value: "old-model"},
				{Key: metadata.ModelNameRewriteKey, Value: "new-model"},
			},
			wantObjective: "new-objective",
			wantTarget:    "new-model",
		},
		{
			name: "Drops client-supplied routing headers",
			headers: []*configPb.HeaderValue{
				{Key: "X-Prefiller-Host-Port", Value: "10.0.0.1:9090"},
				{Key: routing.EncoderEndpointsHeader, Value: "10.0.0.2:9090"},
				{Key: routing.DataParallelEndpointHeader, Value: "10.0.0.3:9090"},
				{Key: routing.KVCacheSourceHeader, Value: "10.0.0.4:9090"},
				{Key: "x-test", Value: "val"},
			},
			wantHeaders: map[string]string{"x-test": "val"},
			wantAbsent: []string{
				routing.PrefillEndpointHeader,
				routing.EncoderEndpointsHeader,
				routing.DataParallelEndpointHeader,
				routing.KVCacheSourceHeader,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{}
			reqCtx := &RequestContext{
				Request: &Request{Headers: make(map[string]string)},
			}
			req := &extProcPb.ProcessingRequest_RequestHeaders{
				RequestHeaders: &extProcPb.HttpHeaders{
					Headers: &configPb.HeaderMap{Headers: tc.headers},
				},
			}

			err := server.HandleRequestHeaders(context.Background(), reqCtx, req)
			assert.NoError(t, err, "HandleRequestHeaders should not return an error")

			assert.Equal(t, tc.wantObjective, reqCtx.ObjectiveKey, "ObjectiveKey should match expected value")
			assert.Equal(t, tc.wantTarget, reqCtx.TargetModelName, "TargetModelName should match expected value")

			if tc.wantHeaders != nil {
				for k, v := range tc.wantHeaders {
					assert.Equal(t, v, reqCtx.Request.Headers[k], "Header %q should match expected value", k)
				}
			}
			for _, k := range tc.wantAbsent {
				assert.NotContains(t, reqCtx.Request.Headers, k)
			}
		})
	}
}

// generateHeaders must inject W3C trace context for Envoy to forward to vLLM,
// and must not re-forward a client-supplied traceparent after injection.
func TestGenerateHeaders_InjectsTraceContext(t *testing.T) {
	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

	const clientTraceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

	tracer := tracing.Tracer("test")
	ctx, span := tracer.Start(context.Background(), "request", trace.WithSpanKind(trace.SpanKindServer))
	defer span.End()

	server := &StreamingServer{}
	reqCtx := &RequestContext{
		TargetEndpoint: "1.2.3.4:8080",
		Request: &Request{
			Headers: map[string]string{
				"traceparent": clientTraceparent,
				"x-user-data": "important",
			},
		},
	}

	results := server.generateHeaders(ctx, reqCtx)

	gotHeaders := make(map[string]string)
	traceparentCount := 0
	for _, h := range results {
		key := strings.ToLower(h.Header.Key)
		gotHeaders[key] = string(h.Header.RawValue)
		if key == "traceparent" {
			traceparentCount++
		}
	}

	traceparent := gotHeaders["traceparent"]
	require.NotEmpty(t, traceparent, "expected traceparent to be injected into outbound headers")
	require.Equal(t, 1, traceparentCount, "expected exactly one traceparent header")
	require.Contains(t, traceparent, span.SpanContext().TraceID().String(),
		"outbound traceparent must carry the active EPP span trace ID")
	require.NotContains(t, traceparent, "0af7651916cd43dd8448eb211c80319c",
		"client traceparent must not be re-forwarded after injection")
	assert.Equal(t, "important", gotHeaders["x-user-data"])
}

// With tracing disabled the noop tracer keeps the extracted remote span context,
// so the client's trace headers are forwarded unchanged.
func TestGenerateHeaders_ForwardsTraceContextWhenTracingDisabled(t *testing.T) {
	useTracerProvider(t, noop.NewTracerProvider())
	tracing.InitTextMapPropagator()

	const clientTraceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0c9902b7-01"
	req := &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
		Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
			{Key: "traceparent", RawValue: []byte(clientTraceparent)},
			{Key: "tracestate", RawValue: []byte("vendor=abc")},
		}},
	}}
	ctx := extractTraceContext(context.Background(), req)
	ctx, span := tracing.Tracer("test").Start(ctx, "request", trace.WithSpanKind(trace.SpanKindServer))
	defer span.End()

	server := &StreamingServer{}
	reqCtx := &RequestContext{
		TargetEndpoint: "1.2.3.4:8080",
		Request: &Request{Headers: map[string]string{
			"traceparent": clientTraceparent,
			"tracestate":  "vendor=abc",
		}},
	}
	got := map[string]string{}
	for _, h := range server.generateHeaders(ctx, reqCtx) {
		got[strings.ToLower(h.Header.Key)] = string(h.Header.RawValue)
	}
	assert.Equal(t, clientTraceparent, got["traceparent"])
	assert.Equal(t, "vendor=abc", got["tracestate"])
}

func TestGenerateHeaders_Sanitization(t *testing.T) {
	server := &StreamingServer{}
	reqCtx := &RequestContext{
		TargetEndpoint: "1.2.3.4:8080",
		RequestSize:    123,
		Request: &Request{
			Headers: map[string]string{
				"x-user-data":                   "important",                  // should passthrough
				metadata.ObjectiveKey:           "sensitive-objective-id",     // should be stripped
				metadata.OldObjectiveKey:        "old-sensitive-objective-id", // should be stripped
				metadata.DestinationEndpointKey: "1.1.1.1:666",                // should be stripped
				"content-length":                "99999",                      // should be stripped (re-added by logic)
				"traceparent":                   "00-deadbeefdeadbeefdeadbeefdeadbeef-deadbeefdeadbeef-01",
			},
		},
	}

	results := server.generateHeaders(context.Background(), reqCtx)

	gotHeaders := make(map[string]string)
	for _, h := range results {
		gotHeaders[h.Header.Key] = string(h.Header.RawValue)
	}

	assert.Contains(t, gotHeaders, "x-user-data")
	assert.NotContains(t, gotHeaders, metadata.ObjectiveKey)
	assert.NotContains(t, gotHeaders, metadata.OldObjectiveKey)
	assert.Equal(t, "1.2.3.4:8080", gotHeaders[metadata.DestinationEndpointKey])
	assert.Equal(t, "123", gotHeaders["Content-Length"])
	assert.NotContains(t, gotHeaders, "traceparent")
}

func TestGenerateRequestHeaderResponse_MergeMetadata(t *testing.T) {
	t.Parallel()

	server := &StreamingServer{}
	reqCtx := &RequestContext{
		TargetEndpoint: "1.2.3.4:8080",
		Request: &Request{
			Headers: make(map[string]string),
		},
		Response: &Response{
			DynamicMetadata: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"existing_namespace": {
						Kind: &structpb.Value_StructValue{
							StructValue: &structpb.Struct{
								Fields: map[string]*structpb.Value{
									"existing_key": {Kind: &structpb.Value_StringValue{StringValue: "existing_value"}},
								},
							},
						},
					},
				},
			},
		},
	}

	resp := server.generateRequestHeaderResponse(context.Background(), reqCtx)

	// Check that the existing metadata is preserved
	existingNamespace, ok := resp.DynamicMetadata.Fields["existing_namespace"]
	assert.True(t, ok, "Expected existing_namespace to be in DynamicMetadata")
	existingKey, ok := existingNamespace.GetStructValue().Fields["existing_key"]
	assert.True(t, ok, "Expected existing_key to be in existing_namespace")
	assert.Equal(t, "existing_value", existingKey.GetStringValue(), "Unexpected value for existing_key")

	// Check that the new metadata is added
	endpointNamespace, ok := resp.DynamicMetadata.Fields[metadata.DestinationEndpointNamespace]
	assert.True(t, ok, "Expected DestinationEndpointNamespace to be in DynamicMetadata")
	endpointKey, ok := endpointNamespace.GetStructValue().Fields[metadata.DestinationEndpointKey]
	assert.True(t, ok, "Expected DestinationEndpointKey to be in DestinationEndpointNamespace")
	assert.Equal(t, "1.2.3.4:8080", endpointKey.GetStringValue(), "Unexpected value for DestinationEndpointKey")
}

func TestGenerateRequestHeaderResponse_EndpointScores(t *testing.T) {
	t.Parallel()

	scores := map[string]float64{
		"1.2.3.4:8080": 0.91,
		"5.6.7.8:8080": 0.74,
	}

	tests := []struct {
		name               string
		emitEndpointScores bool
		targetScores       map[string]float64
		wantScores         map[string]float64
	}{
		{
			name:               "enabled with scores",
			emitEndpointScores: true,
			targetScores:       scores,
			wantScores:         scores,
		},
		{
			name:               "enabled without scores",
			emitEndpointScores: true,
		},
		{
			name:         "disabled with scores",
			targetScores: scores,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{}
			server.SetEmitEndpointScores(tc.emitEndpointScores)
			reqCtx := &RequestContext{
				TargetEndpoint:       "1.2.3.4:8080,5.6.7.8:8080",
				TargetEndpointScores: tc.targetScores,
				Request: &Request{
					Headers: make(map[string]string),
				},
				Response: &Response{},
			}

			resp := server.generateRequestHeaderResponse(context.Background(), reqCtx)

			endpointNamespace, ok := resp.DynamicMetadata.Fields[metadata.DestinationEndpointNamespace]
			assert.True(t, ok, "Expected DestinationEndpointNamespace to be in DynamicMetadata")
			endpointKey, ok := endpointNamespace.GetStructValue().Fields[metadata.DestinationEndpointKey]
			assert.True(t, ok, "Expected DestinationEndpointKey to be in DestinationEndpointNamespace")
			assert.Equal(t, "1.2.3.4:8080,5.6.7.8:8080", endpointKey.GetStringValue(), "Unexpected value for DestinationEndpointKey")

			scoresValue, ok := endpointNamespace.GetStructValue().Fields[metadata.DestinationEndpointScoresKey]
			if tc.wantScores == nil {
				assert.False(t, ok, "Expected DestinationEndpointScoresKey to be absent from DestinationEndpointNamespace")
				return
			}
			assert.True(t, ok, "Expected DestinationEndpointScoresKey to be in DestinationEndpointNamespace")
			gotScores := make(map[string]float64)
			for endpoint, score := range scoresValue.GetStructValue().Fields {
				gotScores[endpoint] = score.GetNumberValue()
			}
			assert.Equal(t, tc.wantScores, gotScores, "Unexpected values for DestinationEndpointScoresKey")
		})
	}
}

func TestGenerateRequestHeaderResponse_RemovesUnsetRoutingHeaders(t *testing.T) {
	t.Parallel()

	// Every spelling Envoy must strip, deprecated aliases included: a client
	// supplying either name must not reach the sidecar.
	canonical := []string{
		routing.PrefillEndpointHeader,
		routing.EncoderEndpointsHeader,
		routing.DataParallelEndpointHeader,
		routing.KVCacheSourceHeader,
		routing.EndpointPinHeader,
	}
	allRoutingHeaders := make([]string, 0, 2*len(canonical))
	for _, h := range canonical {
		allRoutingHeaders = append(allRoutingHeaders, routing.HeaderNames(h)...)
	}

	tests := []struct {
		name        string
		headers     map[string]string
		wantSet     map[string]string
		wantRemoved []string
	}{
		{
			name:        "decode-only removes every routing header",
			headers:     map[string]string{},
			wantRemoved: allRoutingHeaders,
		},
		{
			// The disagg handler writes both spellings (routing.SetRoutingHeader), so
			// a sidecar on either side of the rename reads a prefill target.
			name: "prefill selected sets both spellings and removes the rest",
			headers: map[string]string{
				routing.PrefillEndpointHeader:       "10.0.0.1:8000",
				routing.LegacyPrefillEndpointHeader: "10.0.0.1:8000",
			},
			wantSet: map[string]string{
				routing.PrefillEndpointHeader:       "10.0.0.1:8000",
				routing.LegacyPrefillEndpointHeader: "10.0.0.1:8000",
			},
			wantRemoved: []string{
				routing.EncoderEndpointsHeader,
				routing.LegacyEncoderEndpointsHeader,
				routing.DataParallelEndpointHeader,
				routing.KVCacheSourceHeader,
				routing.LegacyKVCacheSourceHeader,
				routing.EndpointPinHeader,
			},
		},
		{
			// An older EPP in a mixed fleet, or a hand-set legacy value: the canonical
			// spelling is unset and must still be stripped.
			name:    "legacy spelling only still strips the canonical name",
			headers: map[string]string{routing.LegacyPrefillEndpointHeader: "10.0.0.1:8000"},
			wantSet: map[string]string{routing.LegacyPrefillEndpointHeader: "10.0.0.1:8000"},
			wantRemoved: []string{
				routing.PrefillEndpointHeader,
				routing.EncoderEndpointsHeader,
				routing.LegacyEncoderEndpointsHeader,
				routing.DataParallelEndpointHeader,
				routing.KVCacheSourceHeader,
				routing.LegacyKVCacheSourceHeader,
				routing.EndpointPinHeader,
			},
		},
		{
			// The screener reads the client pin, so it stays on the request;
			// the model server must not see it.
			name:        "a client pin is removed and not set",
			headers:     map[string]string{routing.EndpointPinHeader: "10.0.0.1:8000"},
			wantRemoved: allRoutingHeaders,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{}
			reqCtx := &RequestContext{
				TargetEndpoint: "1.2.3.4:8080",
				Request:        &Request{Headers: tc.headers},
				Response:       &Response{},
			}

			mutation := server.generateRequestHeaderResponse(context.Background(), reqCtx).
				GetRequestHeaders().GetResponse().GetHeaderMutation()

			gotSet := make(map[string]string)
			for _, h := range mutation.GetSetHeaders() {
				gotSet[h.Header.Key] = string(h.Header.RawValue)
			}
			for k, v := range tc.wantSet {
				assert.Equal(t, v, gotSet[k])
			}
			assert.ElementsMatch(t, tc.wantRemoved, mutation.GetRemoveHeaders())
		})
	}
}

func TestFallbackToRandomEndpoint(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name               string
		endpoint           *datalayer.EndpointMetadata
		requestSize        int
		wantTargetEndpoint string
		wantBodyRespLen    int
	}{
		{
			name: "IPv4 endpoint without body",
			endpoint: &datalayer.EndpointMetadata{
				Address: "1.2.3.4",
				Port:    "80",
			},
			requestSize:        0,
			wantTargetEndpoint: "1.2.3.4:80",
			wantBodyRespLen:    0,
		},
		{
			name: "IPv6 endpoint without body",
			endpoint: &datalayer.EndpointMetadata{
				Address: "fd99:0:0:8::bec5",
				Port:    "8000",
			},
			requestSize:        0,
			wantTargetEndpoint: "[fd99:0:0:8::bec5]:8000",
			wantBodyRespLen:    0,
		},
		{
			name: "With body",
			endpoint: &datalayer.EndpointMetadata{
				Address: "1.2.3.4",
				Port:    "80",
			},
			requestSize:        9,
			wantTargetEndpoint: "1.2.3.4:80",
			wantBodyRespLen:    1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{
				director: &mockDirectorRequest{endpoint: tc.endpoint},
			}
			reqCtx := &RequestContext{
				Request:  &Request{Headers: make(map[string]string), RawBody: []byte("test body")},
				Response: &Response{Headers: make(map[string]string)},
			}

			err := server.fallbackToRandomEndpoint(context.Background(), reqCtx, tc.requestSize)
			assert.NoError(t, err)
			assert.Equal(t, tc.wantTargetEndpoint, reqCtx.TargetEndpoint)

			if tc.wantBodyRespLen > 0 {
				assert.NotNil(t, reqCtx.reqBodyResp)
				assert.Len(t, reqCtx.reqBodyResp, tc.wantBodyRespLen)
				bodyResp := reqCtx.reqBodyResp[0].GetRequestBody().GetResponse()
				assert.NotNil(t, bodyResp.BodyMutation)
				streamedResp := bodyResp.BodyMutation.GetStreamedResponse()
				assert.NotNil(t, streamedResp)
				assert.Equal(t, []byte("test body"), streamedResp.Body)
			} else {
				assert.Nil(t, reqCtx.reqBodyResp)
			}
		})
	}
}

type mockDirectorRequest struct {
	Director
	endpoint *datalayer.EndpointMetadata
}

func (m *mockDirectorRequest) GetRandomEndpoint() *datalayer.EndpointMetadata {
	if m.endpoint != nil {
		return m.endpoint
	}
	return &datalayer.EndpointMetadata{
		Address: "1.2.3.4",
		Port:    "80",
	}
}

// Attribution must be established from the incoming headers before the Director
// runs, so a request that fails or returns early is still attributed.
func TestRequestAttributionAtIngress(t *testing.T) {
	previous := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	// Mirror InitTracing: the processor is what attributes spans in production.
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(tracing.NewRequestAttributionProcessor()),
		sdktrace.WithSpanProcessor(recorder),
	)
	otel.SetTracerProvider(provider)
	t.Cleanup(func() { otel.SetTracerProvider(previous); _ = provider.Shutdown(context.Background()) })

	for _, tc := range []struct {
		name               string
		headers            []*configPb.HeaderValue
		wantID, wantSource string
	}{
		{"canonical header", []*configPb.HeaderValue{{Key: metadata.FlowFairnessIDKey, Value: "team-a"}}, "team-a", tracing.AttributionSourceHeader},
		{"deprecated alias", []*configPb.HeaderValue{{Key: metadata.OldFlowFairnessIDKey, Value: "team-b"}}, "team-b", tracing.AttributionSourceHeader},
		{"empty canonical shadows alias", []*configPb.HeaderValue{{Key: metadata.FlowFairnessIDKey, Value: ""}, {Key: metadata.OldFlowFairnessIDKey, Value: "team-b"}}, reqcommon.DefaultFairnessID, tracing.AttributionSourceDefault},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := extractTraceContext(context.Background(), &extProcPb.ProcessingRequest_RequestHeaders{
				RequestHeaders: &extProcPb.HttpHeaders{
					Headers:     &configPb.HeaderMap{Headers: tc.headers},
					EndOfStream: true,
				},
			})

			_, span := tracing.Tracer().Start(ctx, "request")
			span.End()

			ended := recorder.Ended()
			attrs := attribute.NewSet(ended[len(ended)-1].Attributes()...)
			id, hasID := attrs.Value(semconv.LLMDEPPFairnessIDKey)
			source, hasSource := attrs.Value(semconv.LLMDEPPFairnessSourceKey)

			require.True(t, hasID && hasSource, "identity and source are recorded together")
			assert.Equal(t, tc.wantID, id.AsString())
			assert.Equal(t, tc.wantSource, source.AsString())
		})
	}
}

func TestCompareEPPToolCallingSnapshotToBody(t *testing.T) {
	inboundBody := []byte(`{"tools":[{"function":{"name":"private","parameters":{"type":"object"}}}],"tool_choice":"required","parallel_tool_calls":true}`)
	outbound := []byte(`{"parallel_tool_calls":true,"tool_choice":"auto","tools":[{"function":{"parameters":{"type":"object"},"name":"private"}}]}`)

	parserBody := bytes.Clone(inboundBody)
	inbound, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, parserBody)
	require.NoError(t, err)
	parserBody[0] = 'x' // The captured snapshot remains stable if the parser's copy is changed.
	statuses, err := compareEPPToolCallingSnapshotToBody(reqcommon.APITypeChatCompletions, inbound, inboundBody, outbound)
	require.NoError(t, err)
	require.Equal(t, toolcalling.FieldStatusPreserved, statusForField(t, statuses, toolcalling.FieldTools).Status)
	require.Equal(t, toolcalling.FieldStatusChanged, statusForField(t, statuses, toolcalling.FieldToolChoice).Status)
	require.True(t, statusForField(t, statuses, toolcalling.FieldToolChoice).Observed)

	statuses, err = compareEPPToolCallingSnapshotToBody(reqcommon.APITypeChatCompletions, inbound, inboundBody, []byte(`{"tools":[`))
	require.Error(t, err)
	require.Empty(t, statuses, "a failed capture does not establish field rejection")
}

func TestCompareEPPToolCallingUnchangedBodyAllocations(t *testing.T) {
	body := []byte(`{"tools":[{"type":"function","function":{"name":"private","parameters":{"type":"object"}}}],"tool_choice":"required"}`)
	snapshot, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, body)
	require.NoError(t, err)
	outbound := bytes.Clone(body)
	baseline := testing.AllocsPerRun(20, func() {
		_, compareErr := toolcalling.CompareRequests(snapshot, snapshot)
		require.NoError(t, compareErr)
	})
	unchanged := testing.AllocsPerRun(20, func() {
		_, compareErr := compareEPPToolCallingSnapshotToBody(reqcommon.APITypeChatCompletions, snapshot, body, outbound)
		require.NoError(t, compareErr)
	})
	require.LessOrEqual(t, unchanged, baseline, "unchanged bodies should reuse the captured fields")
}

func BenchmarkCompareEPPToolCallingSnapshotToBody(b *testing.B) {
	for _, size := range []int{1024, 64 * 1024, 1024 * 1024} {
		plain := []byte(`{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("x", size) + `"}]}`)
		toolBody := append(bytes.Clone(plain[:len(plain)-1]), []byte(`,"tools":[{"type":"function","function":{"name":"private","parameters":{"type":"object"}}}],"tool_choice":"auto"}`)...)
		for _, tt := range []struct {
			name     string
			inbound  []byte
			outbound []byte
		}{
			{name: "unchanged_tool", inbound: toolBody, outbound: bytes.Clone(toolBody)},
			{name: "changed_tool", inbound: toolBody, outbound: bytes.Replace(toolBody, []byte(`"auto"`), []byte(`"none"`), 1)},
			{name: "unchanged_non_tool", inbound: plain, outbound: bytes.Clone(plain)},
		} {
			b.Run(tt.name+"/"+strconv.Itoa(size), func(b *testing.B) {
				snapshot, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, tt.inbound)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					if _, err := compareEPPToolCallingSnapshotToBody(reqcommon.APITypeChatCompletions, snapshot, tt.inbound, tt.outbound); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func TestToolCallingAPIForPath(t *testing.T) {
	tests := []struct {
		path   string
		want   reqcommon.APIType
		wantOK bool
	}{
		{path: "/v1/chat/completions", want: reqcommon.APITypeChatCompletions, wantOK: true},
		{path: "/prefix/v1/messages", want: reqcommon.APITypeMessages, wantOK: true},
		{path: "/v1/responses", want: reqcommon.APITypeResponses, wantOK: true},
		{path: "/v1/projects/demo/locations/us/endpoints/model/chat/completions", want: reqcommon.APITypeChatCompletions, wantOK: true},
		{path: "/provider/messages/", want: reqcommon.APITypeMessages, wantOK: true},
		{path: "/provider/responses?stream=true", want: reqcommon.APITypeResponses, wantOK: true},
		{path: "/v1/chat/completions/?stream=true", want: reqcommon.APITypeChatCompletions, wantOK: true},
		{path: "/v1/chat/completions/render", wantOK: false},
		{path: "/v1/messages/render", wantOK: false},
		{path: "/v1/messages/count_tokens", wantOK: false},
		{path: "/v1/responses/response-id", wantOK: false},
		{path: "/v1/notchat/completions", wantOK: false},
		{path: "/v1/notmessages", wantOK: false},
		{path: "/v1/chat/completions-extra", wantOK: false},
		{path: "/v1/embeddings?route=/v1/chat/completions", wantOK: false},
		{path: "/unknown", wantOK: false},
		{path: "", wantOK: false},
		{path: "/v1/embeddings", wantOK: false},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got, ok := toolCallingAPIForPath(tt.path)
			require.Equal(t, tt.wantOK, ok)
			if ok {
				require.Equal(t, tt.want, got)
			}
		})
	}
}

func statusForField(t *testing.T, statuses []toolcalling.FieldStatus, field toolcalling.Field) toolcalling.FieldStatus {
	t.Helper()
	for _, status := range statuses {
		if status.Field == field {
			return status
		}
	}
	t.Fatalf("field %q missing from status list", field)
	return toolcalling.FieldStatus{}
}

type requestIntegrityDirector struct {
	mockDirector
	outbound string
	inPlace  bool
	err      error
	called   bool
}

func (d *requestIntegrityDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	d.called = true
	if d.outbound != "" {
		if d.inPlace {
			copy(reqCtx.Request.RawBody, d.outbound)
		} else {
			reqCtx.Request.RawBody = []byte(d.outbound)
		}
		reqCtx.RequestSize = len(d.outbound)
	}
	return reqCtx, d.err
}

func TestProcessRequestToolCallingIntegrity(t *testing.T) {
	const (
		chatBody      = `{"model":"m","messages":[{"role":"user","content":"private_prompt_content"}],"tools":[{"type":"function","function":{"name":"private_tool_name","parameters":{"type":"object","description":"private_schema_content"}}}],"tool_choice":"required","parallel_tool_calls":true,"response_format":{"type":"json_object"}}`
		plainBody     = `{"model":"m","messages":[{"role":"user","content":"hello"}]}`
		responsesBody = `{"model":"m","input":"private_prompt_content","tools":[{"type":"function","name":"private_tool_name","parameters":{"type":"object","description":"private_schema_content"}}],"tool_choice":{"type":"function","name":"private_tool_name"},"parallel_tool_calls":true,"response_format":null}`
	)
	for _, tt := range []struct {
		name                  string
		path                  string
		body                  string
		outbound              string
		inPlace               bool
		directorErr           error
		wantErrorStatus       envoyTypePb.StatusCode
		wantErrorMessage      string
		wantDirector          bool
		wantSurface           string
		wantChoice            string
		wantToolBucket        string
		wantFields            map[string]string
		wantToolCallingAbsent bool
	}{
		{
			name: "chat fields preserved", path: reqcommon.PathChatCompletions,
			body: chatBody, wantDirector: true, wantSurface: "chat_completions",
			wantChoice: "required", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "preserved", "tool_choice": "preserved", "parallel_tool_calls": "preserved", "response_format": "preserved"},
		},
		{
			name: "in-place tool choice change", path: reqcommon.PathChatCompletions,
			body:     `{"model":"m","messages":[{"role":"user","content":"hello"}],"tool_choice":"auto"}`,
			outbound: `{"model":"m","messages":[{"role":"user","content":"hello"}],"tool_choice":"none"}`, inPlace: true,
			wantDirector: true, wantSurface: "chat_completions", wantChoice: "auto",
			wantFields: map[string]string{"tool_choice": "changed"},
		},
		{
			name: "introduced tool fields", path: reqcommon.PathChatCompletions,
			body: plainBody, outbound: chatBody,
			wantDirector: true, wantSurface: "chat_completions", wantToolCallingAbsent: true,
			wantFields: map[string]string{"tools": "changed", "tool_choice": "changed", "parallel_tool_calls": "changed", "response_format": "changed"},
		},
		{
			name: "model rewrite preserves tool fields", path: reqcommon.PathChatCompletions,
			body: chatBody, outbound: strings.Replace(chatBody, `"model":"m"`, `"model":"other"`, 1),
			wantDirector: true, wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "preserved", "tool_choice": "preserved", "parallel_tool_calls": "preserved", "response_format": "preserved"},
		},
		{
			name: "equivalent reserialization preserves tool fields", path: reqcommon.PathChatCompletions,
			body: chatBody, outbound: strings.ReplaceAll(chatBody, `,`, ",\n "),
			wantDirector: true, wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "preserved", "tool_choice": "preserved", "parallel_tool_calls": "preserved", "response_format": "preserved"},
		},
		{
			name: "provider chat route records request telemetry", path: "/v1/projects/demo/locations/us/endpoints/model/chat/completions?stream=true",
			body: chatBody, wantDirector: true, wantSurface: "chat_completions",
			wantChoice: "required", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "preserved", "tool_choice": "preserved", "parallel_tool_calls": "preserved", "response_format": "preserved"},
		},
		{
			name: "Responses preserves supported fields and sanitizes named choice", path: reqcommon.PathResponses,
			body: responsesBody, wantDirector: true, wantSurface: "responses",
			wantChoice: "named", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "preserved", "tool_choice": "preserved", "parallel_tool_calls": "preserved"},
		},
		{
			name: "Responses compares serialized changes and drops", path: "/provider/responses/?stream=true",
			body: responsesBody, outbound: `{"model":"m","input":"hello","tools":[],"tool_choice":"auto","response_format":null}`,
			wantDirector: true, wantSurface: "responses", wantChoice: "named", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "changed", "tool_choice": "changed", "parallel_tool_calls": "dropped"},
		},
		{
			name: "Responses structured output emits no tool telemetry", path: reqcommon.PathResponses,
			body:         `{"model":"m","input":"hello","text":{"format":{"type":"json_object"}},"response_format":null}`,
			wantDirector: true,
		},
		{
			name: "chat render does not emit inference tool telemetry", path: reqcommon.PathChatCompletions + "/render",
			body: chatBody, wantDirector: true,
		},
		{
			name: "Messages render does not emit inference tool telemetry", path: reqcommon.PathMessages + "/render",
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":{"type":"any"}}`, wantDirector: true,
		},
		{
			name: "Messages count_tokens does not emit inference tool telemetry", path: reqcommon.PathMessages + "/count_tokens",
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":{"type":"any"}}`, wantDirector: true,
		},
		{
			name: "serialized body changes and drops fields", path: reqcommon.PathChatCompletions,
			body:         chatBody,
			outbound:     `{"model":"m","messages":[{"role":"user","content":"private_prompt_content"}],"tools":[{"function":{"parameters":{"description":"private_schema_content","type":"object"},"name":"private_tool_name"},"type":"function"}],"tool_choice":"auto","response_format":{"type":"json_object"}}`,
			wantDirector: true, wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "preserved", "tool_choice": "changed", "parallel_tool_calls": "dropped", "response_format": "preserved"},
		},
		{
			name: "all tool fields dropped", path: reqcommon.PathChatCompletions,
			body: chatBody, outbound: plainBody, wantDirector: true,
			wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "dropped", "tool_choice": "dropped", "parallel_tool_calls": "dropped", "response_format": "dropped"},
		},
		{
			name: "structured output preserved without tool presence", path: reqcommon.PathChatCompletions,
			body:         `{"model":"m","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_object"}}`,
			wantDirector: true, wantSurface: "chat_completions", wantToolCallingAbsent: true,
			wantFields: map[string]string{"response_format": "preserved"},
		},
		{
			name: "structured output changed without tool presence", path: reqcommon.PathChatCompletions,
			body:         `{"model":"m","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_object"}}`,
			outbound:     `{"model":"m","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"text"}}`,
			wantDirector: true, wantSurface: "chat_completions", wantToolCallingAbsent: true,
			wantFields: map[string]string{"response_format": "changed"},
		},
		{
			name: "structured output dropped without tool presence", path: reqcommon.PathChatCompletions,
			body:     `{"model":"m","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_object"}}`,
			outbound: plainBody, wantDirector: true, wantSurface: "chat_completions", wantToolCallingAbsent: true,
			wantFields: map[string]string{"response_format": "dropped"},
		},
		{
			name: "null structured output without tool presence", path: reqcommon.PathChatCompletions,
			body:         `{"model":"m","messages":[{"role":"user","content":"hello"}],"response_format":null}`,
			wantDirector: true, wantSurface: "chat_completions", wantToolCallingAbsent: true,
			wantFields: map[string]string{"response_format": "preserved"},
		},
		{
			name: "missing messages does not reject tool fields", path: reqcommon.PathChatCompletions,
			body: `{"model":"m","tools":[],"tool_choice":"required"}`, wantErrorStatus: envoyTypePb.StatusCode_BadRequest,
			wantSurface: "chat_completions", wantChoice: "required",
		},
		{
			name: "chat invalid tools type rejects only tools", path: reqcommon.PathChatCompletions,
			body:             `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":"private_tool_value","tool_choice":"required","parallel_tool_calls":true}`,
			wantErrorStatus:  envoyTypePb.StatusCode_BadRequest,
			wantErrorMessage: "error extracting request body: invalid chat completions request: must have valid messages field",
			wantSurface:      "chat_completions", wantChoice: "required",
			wantFields: map[string]string{"tools": "rejected"},
		},
		{
			name: "chat invalid message role does not reject tools", path: reqcommon.PathChatCompletions,
			body:             `{"model":"m","messages":[{"role":123,"content":"hello"}],"tools":[],"tool_choice":"required"}`,
			wantErrorStatus:  envoyTypePb.StatusCode_BadRequest,
			wantErrorMessage: "error extracting request body: invalid chat completions request: must have valid messages field",
			wantSurface:      "chat_completions", wantChoice: "required",
		},
		{
			name: "Messages invalid tools type rejects only tools", path: reqcommon.PathMessages,
			body:            `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":false,"tool_choice":{"type":"any"}}`,
			wantErrorStatus: envoyTypePb.StatusCode_BadRequest,
			wantSurface:     "messages", wantChoice: "required",
			wantFields: map[string]string{"tools": "rejected"},
		},
		{
			name: "Messages invalid tool name type rejects only tools", path: reqcommon.PathMessages,
			body:            `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[{"name":123}],"tool_choice":{"type":"any"}}`,
			wantErrorStatus: envoyTypePb.StatusCode_BadRequest,
			wantSurface:     "messages", wantChoice: "required", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "rejected"},
		},
		{
			name: "Messages invalid tool strict type rejects only tools", path: reqcommon.PathMessages,
			body:            `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[{"name":"private_tool_name","strict":"private_invalid_value"}],"tool_choice":{"type":"any"}}`,
			wantErrorStatus: envoyTypePb.StatusCode_BadRequest,
			wantSurface:     "messages", wantChoice: "required", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "rejected"},
		},
		{
			name: "Messages invalid message role does not reject tools", path: reqcommon.PathMessages,
			body:            `{"model":"m","messages":[{"role":123,"content":"hello"}],"tools":[],"tool_choice":{"type":"any"}}`,
			wantErrorStatus: envoyTypePb.StatusCode_BadRequest,
			wantSurface:     "messages", wantChoice: "required",
		},
		{
			name: "unvalidated tool choice remains accepted", path: reqcommon.PathChatCompletions,
			body:         `{"model":"m","messages":[{"role":"user","content":"hello"}],"tool_choice":123}`,
			wantDirector: true, wantSurface: "chat_completions", wantChoice: "unknown",
			wantFields: map[string]string{"tool_choice": "preserved"},
		},
		{
			name: "generic director BadRequest does not reject tool fields", path: reqcommon.PathChatCompletions,
			body: chatBody, directorErr: errcommon.Error{Code: errcommon.BadRequest, Msg: "request rejected"},
			wantErrorStatus: envoyTypePb.StatusCode_BadRequest, wantDirector: true,
			wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
		},
		{
			name: "no endpoints does not reject tool fields", path: reqcommon.PathChatCompletions,
			body: chatBody, directorErr: errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "no endpoints available"},
			wantErrorStatus: envoyTypePb.StatusCode_ServiceUnavailable, wantDirector: true,
			wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
		},
		{
			name: "capacity shedding does not reject tool fields", path: reqcommon.PathChatCompletions,
			body: chatBody, directorErr: errcommon.Error{Code: errcommon.ResourceExhausted, Msg: "no request capacity"},
			wantErrorStatus: envoyTypePb.StatusCode_TooManyRequests, wantDirector: true,
			wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
		},
		{
			name: "internal director error does not reject tool fields", path: reqcommon.PathChatCompletions,
			body: chatBody, directorErr: errcommon.Error{Code: errcommon.Internal, Msg: "request processing failed"},
			wantErrorStatus: envoyTypePb.StatusCode_InternalServerError, wantDirector: true,
			wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
		},
		{
			name: "non-tool scheduling failure emits no tool telemetry", path: reqcommon.PathChatCompletions,
			body: plainBody, directorErr: errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "no endpoints available"},
			wantErrorStatus: envoyTypePb.StatusCode_ServiceUnavailable, wantDirector: true,
		},
		{
			name: "failed outbound capture emits no guessed field outcomes", path: reqcommon.PathChatCompletions,
			body: chatBody, outbound: `{"tools":[`, wantDirector: true,
			wantSurface: "chat_completions", wantChoice: "required", wantToolBucket: "1",
		},
		{
			name: "Messages missing messages does not reject tool fields", path: reqcommon.PathMessages,
			body:            `{"model":"m","tools":[],"tool_choice":{"type":"any"}}`,
			wantErrorStatus: envoyTypePb.StatusCode_BadRequest,
			wantSurface:     "messages", wantChoice: "required",
		},
		{
			name: "malformed body has no guessed statuses", path: reqcommon.PathChatCompletions,
			body: `{"model":"m","tools":[`, wantErrorStatus: envoyTypePb.StatusCode_BadRequest,
		},
		{
			name: "non-tool request", path: reqcommon.PathChatCompletions,
			body: plainBody, wantDirector: true,
		},
		{
			name: "nested tool fields are not request fields", path: reqcommon.PathChatCompletions,
			body: `{"model":"m","messages":[{"role":"user","content":"hello","tools":[],"tool_choice":"required"}]}`, wantDirector: true,
		},
		{
			name: "Messages counts only supported fields", path: reqcommon.PathMessages,
			body:         `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[{"name":"private_tool_name","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"private_tool_name"},"parallel_tool_calls":true,"response_format":null}`,
			outbound:     `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[{"name":"private_tool_name","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"private_tool_name"}}`,
			wantDirector: true, wantSurface: "messages", wantChoice: "named", wantToolBucket: "1",
			wantFields: map[string]string{"tools": "preserved", "tool_choice": "preserved"},
		},
		{
			name: "Messages unsupported fields emit no telemetry", path: reqcommon.PathMessages,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"parallel_tool_calls":true,"response_format":null}`, wantDirector: true,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			metrics.Register()
			metrics.Reset()
			t.Cleanup(metrics.Reset)
			exporter := tracetest.NewInMemoryExporter()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter), sdktrace.WithSampler(sdktrace.AlwaysSample()))
			useTracerProvider(t, provider)
			t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
			if tt.inPlace {
				require.Len(t, tt.outbound, len(tt.body))
			}
			director := &requestIntegrityDirector{outbound: tt.outbound, inPlace: tt.inPlace, err: tt.directorErr}
			srv := &scriptedProcessServer{
				ctx: context.Background(),
				reqs: []*extProcPb.ProcessingRequest{
					newRequestHeaders(map[string]string{":path": tt.path, "traceparent": upstreamTraceparent}),
					{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
						Body: []byte(tt.body), EndOfStream: true,
					}}},
				},
			}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser(), anthropic.NewAnthropicParser()}, logr.Discard())
			require.NoError(t, NewStreamingServer(nil, director, registry, 0).Process(srv))
			require.Equal(t, tt.wantDirector, director.called)
			if tt.wantErrorStatus != 0 {
				require.Len(t, srv.sentResponses, 1)
				require.Equal(t, tt.wantErrorStatus, srv.sentResponses[0].GetImmediateResponse().GetStatus().GetCode())
				wantErrorMessage := tt.wantErrorMessage
				if tt.path == reqcommon.PathMessages {
					var messages fwkrh.MessagesRequest
					if decodeErr := json.Unmarshal([]byte(tt.body), &messages); decodeErr != nil {
						wantErrorMessage = "error parsing messages request: " + decodeErr.Error()
					}
				}
				if wantErrorMessage != "" {
					var errorBody struct {
						Error struct {
							Message string `json:"message"`
						} `json:"error"`
					}
					require.NoError(t, json.Unmarshal(srv.sentResponses[0].GetImmediateResponse().GetBody(), &errorBody))
					require.Equal(t, wantErrorMessage, errorBody.Error.Message)
				}
			} else {
				require.Len(t, srv.sentResponses, 2)
				wantBody := tt.body
				if tt.outbound != "" {
					wantBody = tt.outbound
				}
				require.Equal(t, wantBody, string(srv.sentResponses[1].GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse().GetBody()))
			}

			gotFields := make(map[string]string)
			families, err := ctrlmetrics.Registry.Gather()
			require.NoError(t, err)
			for _, family := range families {
				if family.GetName() != "llm_d_epp_tool_calling_field_status_total" {
					continue
				}
				for _, metric := range family.GetMetric() {
					labels := make(map[string]string)
					for _, label := range metric.GetLabel() {
						labels[label.GetName()] = label.GetValue()
					}
					require.Len(t, labels, 4)
					require.Equal(t, toolcalling.ComponentEPP, labels[toolcalling.MetricLabelComponent])
					require.Equal(t, toolcalling.DirectionRequest, labels[toolcalling.MetricLabelDirection])
					require.Equal(t, float64(1), metric.GetCounter().GetValue())
					field := labels[toolcalling.MetricLabelField]
					require.NotContains(t, gotFields, field, "a field must be counted only once")
					gotFields[field] = labels[toolcalling.MetricLabelStatus]
				}
			}
			require.Len(t, gotFields, len(tt.wantFields))
			for field, status := range tt.wantFields {
				require.Equal(t, status, gotFields[field])
			}

			var requestSpans tracetest.SpanStubs
			for _, span := range exporter.GetSpans() {
				if span.Name == "request" {
					requestSpans = append(requestSpans, span)
				}
			}
			require.Len(t, requestSpans, 1)
			require.Equal(t, upstreamTraceID, requestSpans[0].SpanContext.TraceID().String())
			require.Equal(t, "00f067aa0c9902b7", requestSpans[0].Parent.SpanID().String())
			gotAttrs := make(map[string]any)
			for _, attr := range requestSpans[0].Attributes {
				if strings.HasPrefix(string(attr.Key), "llm_d.tool_calling.") {
					gotAttrs[string(attr.Key)] = attr.Value.AsInterface()
				}
			}
			wantAttrs := make(map[string]any)
			if tt.wantSurface != "" {
				wantAttrs["llm_d.tool_calling.api_surface"] = tt.wantSurface
				wantAttrs["llm_d.tool_calling.present"] = !tt.wantToolCallingAbsent
				if !tt.wantToolCallingAbsent {
					wantAttrs["llm_d.tool_calling.tool_choice"] = tt.wantChoice
				}
				if tt.wantToolBucket != "" {
					wantAttrs["llm_d.tool_calling.tool_count"] = tt.wantToolBucket
				}
				for field, status := range tt.wantFields {
					wantAttrs["llm_d.tool_calling.field."+field+".status"] = status
				}
			}
			require.Equal(t, wantAttrs, gotAttrs)
		})
	}
}
