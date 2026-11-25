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

package epp

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

// TestFairnessIDHeaderContract covers fairness ID resolution from the request
// headers through ext_proc to the metric labels and span attributes.
func TestFairnessIDHeaderContract(t *testing.T) {
	const (
		resolvedTotal = "llm_d_epp_fairness_id_resolved_total"
		overflowTotal = "llm_d_epp_fairness_id_overflow_total"
		requestTotal  = "llm_d_epp_request_total"
	)
	oversized := strings.Repeat("t", metricsutil.MaxLabelValueLength+1)
	// Envoy sends header values as raw bytes, so a value that is not valid
	// UTF-8 reaches the EPP.
	notUTF8 := "contract-\xff"

	cases := []struct {
		name       string
		headers    map[string]string
		wantID     string
		wantSource string
		// folded marks an ID the metrics report as "other".
		folded bool
	}{
		{
			name:       "current name",
			headers:    map[string]string{metadata.FlowFairnessIDKey: "contract-current"},
			wantID:     "contract-current",
			wantSource: tracing.AttributionSourceHeader,
		},
		{
			name:       "deprecated name only",
			headers:    map[string]string{metadata.OldFlowFairnessIDKey: "contract-deprecated"},
			wantID:     "contract-deprecated",
			wantSource: tracing.AttributionSourceHeader,
		},
		{
			name: "both names, current wins",
			headers: map[string]string{
				metadata.FlowFairnessIDKey:    "contract-both-current",
				metadata.OldFlowFairnessIDKey: "contract-both-deprecated",
			},
			wantID:     "contract-both-current",
			wantSource: tracing.AttributionSourceHeader,
		},
		{
			name: "empty current name falls back to deprecated",
			headers: map[string]string{
				metadata.FlowFairnessIDKey:    "",
				metadata.OldFlowFairnessIDKey: "contract-fallback",
			},
			wantID:     "contract-fallback",
			wantSource: tracing.AttributionSourceHeader,
		},
		{
			name:       "no header",
			wantID:     reqcommon.DefaultFairnessID,
			wantSource: tracing.AttributionSourceDefault,
		},
		{
			name: "empty under both names",
			headers: map[string]string{
				metadata.FlowFairnessIDKey:    "",
				metadata.OldFlowFairnessIDKey: "",
			},
			wantID:     reqcommon.DefaultFairnessID,
			wantSource: tracing.AttributionSourceDefault,
		},
		{
			name:       "client sends the default flow ID",
			headers:    map[string]string{metadata.FlowFairnessIDKey: reqcommon.DefaultFairnessID},
			wantID:     reqcommon.DefaultFairnessID,
			wantSource: tracing.AttributionSourceHeader,
		},
		{
			name:       "client sends the overflow label value",
			headers:    map[string]string{metadata.FlowFairnessIDKey: metricsutil.OverflowValue},
			wantID:     metricsutil.OverflowValue,
			wantSource: tracing.AttributionSourceHeader,
		},
		{
			name:       "longer than the label limit",
			headers:    map[string]string{metadata.FlowFairnessIDKey: oversized},
			wantID:     oversized,
			wantSource: tracing.AttributionSourceHeader,
			folded:     true,
		},
		{
			name:       "not valid UTF-8",
			headers:    map[string]string{metadata.FlowFairnessIDKey: notUTF8},
			wantID:     notUTF8,
			wantSource: tracing.AttributionSourceHeader,
			folded:     true,
		},
	}

	body, err := json.Marshal(map[string]any{
		"max_tokens": 100, "model": modelMyModel, "prompt": "fairness-id-contract", "temperature": 0,
	})
	require.NoError(t, err)

	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithTracing(), eppharness.WithStandardMode()).WithBaseResources()
	h.WithPods([]eppharness.PodState{
		eppharness.P(0, 3, 0.2),
		eppharness.P(1, 0, 0.1),
		eppharness.P(2, 10, 0.2),
	}).WaitForSync(3, modelMyModel).WaitForReadyPodsMetric(3)

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// A trace ID per request selects that request's span below.
			traceID := fmt.Sprintf("%032x", i+1)
			headers := map[string]string{
				reqcommon.RequestIDHeaderKey: fmt.Sprintf("fairness-id-contract-%d", i),
				metadata.ObjectiveKey:        modelMyModel,
				metadata.ModelNameRewriteKey: modelMyModelTarget,
				"traceparent":                "00-" + traceID + "-00f067aa0c9902b7-01",
			}
			for k, v := range tc.headers {
				headers[k] = v
			}
			requests := []*extProcPb.ProcessingRequest{
				{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
					Headers: &envoyCorev3.HeaderMap{Headers: buildEnvoyHeaders(headers)},
				}}},
				{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
					Body: body, EndOfStream: true,
				}}},
			}

			wantLabel := tc.wantID
			wantOverflow := 0.0
			if tc.folded {
				wantLabel = metricsutil.OverflowValue
				wantOverflow = 1
			}
			resolvedLabels := map[string]string{
				"fairness_id": wantLabel, "source": tc.wantSource, "inference_pool": eppharness.TestPoolName,
			}
			overflowLabels := map[string]string{"source": tc.wantSource, "inference_pool": eppharness.TestPoolName}
			// Every label is set so that the read selects one series.
			requestLabels := map[string]string{
				"fairness_id": wantLabel, "model_name": modelMyModel, "target_model_name": modelMyModelTarget,
				"priority": "2", "api_surface": reqcommon.APISurfaceUnknown.String(),
			}
			resolvedBefore := gatheredValue(t, resolvedTotal, resolvedLabels)
			overflowBefore := gatheredValue(t, overflowTotal, overflowLabels)
			requestsBefore := gatheredValue(t, requestTotal, requestLabels)

			client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(t.Context())
			require.NoError(t, err)
			defer func() { _ = client.CloseSend() }()
			responses, err := fwkepp.StreamedRequest(t, client, requests, 2)
			require.NoError(t, err)
			require.Len(t, responses, 2, "the request must be routed")

			assert.InDelta(t, 1, gatheredValue(t, resolvedTotal, resolvedLabels)-resolvedBefore, 0,
				"%s%v", resolvedTotal, resolvedLabels)
			assert.InDelta(t, wantOverflow, gatheredValue(t, overflowTotal, overflowLabels)-overflowBefore, 0,
				"%s%v", overflowTotal, overflowLabels)
			assert.InDelta(t, 1, gatheredValue(t, requestTotal, requestLabels)-requestsBefore, 0,
				"%s%v", requestTotal, requestLabels)

			var spanID, spanSource attribute.Value
			require.Eventually(t, func() bool {
				for _, span := range h.GetSpans() {
					if span.Name != "request_orchestration" || span.SpanContext.TraceID().String() != traceID {
						continue
					}
					attrs := attribute.NewSet(span.Attributes...)
					spanID, _ = attrs.Value(semconv.LLMDEPPFairnessIDKey)
					spanSource, _ = attrs.Value(semconv.LLMDEPPFairnessSourceKey)
					return true
				}
				return false
			}, 5*time.Second, 50*time.Millisecond, "request_orchestration span was not recorded")
			assert.Equal(t, tc.wantSource, spanSource.AsString())
			// The span carries the ID as sent: the label cap applies to metrics
			// only. OTLP cannot encode a string that is not valid UTF-8, so the
			// span value for such an ID is left unspecified.
			if utf8.ValidString(tc.wantID) {
				assert.Equal(t, tc.wantID, spanID.AsString())
			}
		})
	}
}
