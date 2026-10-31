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

package handlers

import (
	"context"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Pins the wiring in Process rather than the recorder: the error is counted with
// the request's context, so its exemplar names the request's trace.
func TestProcessAttachesTraceExemplarToRequestError(t *testing.T) {
	useTracerProvider(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	srv := &replayProcessServer{
		ctx: logutil.NewTestLoggerIntoContext(context.Background()),
		reqs: []*extProcPb.ProcessingRequest{
			newRequestHeaders(map[string]string{"traceparent": upstreamTraceparent, "x-request-id": "req-error-exemplar"}),
			{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
				Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{{Key: ":status", RawValue: []byte("502")}}},
			}}},
		},
	}
	registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
	require.NoError(t, NewStreamingServer(nil, &mockDirector{}, registry, 0).Process(srv))

	exemplar := modelServerErrorExemplar(t)
	require.NotNil(t, exemplar, "the error must link to the request's trace")
	labels := map[string]string{}
	for _, label := range exemplar.GetLabel() {
		labels[label.GetName()] = label.GetValue()
	}
	require.Equal(t, upstreamTraceID, labels[metricsutil.ExemplarTraceIDLabel], "must name the trace the EPP joined")
	require.NotEmpty(t, labels[metricsutil.ExemplarSpanIDLabel], "the EPP's request span is still recording when the error is counted")
}

// modelServerErrorExemplar returns the exemplar on the request_error_total series
// for ModelServerError, or nil when it has none.
func modelServerErrorExemplar(t *testing.T) *dto.Exemplar {
	t.Helper()

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != "llm_d_epp_request_error_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			if metricHasLabels(metric, map[string]string{"error_code": errcommon.ModelServerError}) {
				return metric.GetCounter().GetExemplar()
			}
		}
	}
	return nil
}
