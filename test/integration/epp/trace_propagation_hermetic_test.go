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

package epp

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	integration "github.com/llm-d/llm-d-router/test/integration"
)

// TestTraceContextDownstreamPropagation verifies that EPP joins an upstream
// trace, records routing spans, and injects a single updated traceparent for
// Envoy to forward to vLLM.
func TestTraceContextDownstreamPropagation(t *testing.T) {
	const (
		upstreamTraceID     = "4bf92f3577b34da6a3ce929d0e0e4736"
		upstreamSpanID      = "00f067aa0c9902b7"
		upstreamTraceparent = "00-" + upstreamTraceID + "-" + upstreamSpanID + "-01"
	)

	bodyMap := map[string]any{
		"max_tokens": 100, "model": modelMyModelTarget, "prompt": "trace-propagation", "temperature": 0,
	}
	body, err := json.Marshal(bodyMap)
	require.NoError(t, err)

	headers := map[string]string{
		"hi":                         "mom",
		reqcommon.RequestIDHeaderKey: "trace-propagation-req",
		metadata.ObjectiveKey:        modelMyModel,
		metadata.ModelNameRewriteKey: modelMyModelTarget,
		"traceparent":                upstreamTraceparent,
	}
	requests := integration.ReqRaw(headers, string(body))

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithTracing(), WithStandardMode()).WithBaseResources()
	h.WithPods([]PodState{
		P(0, 3, 0.2),
		P(1, 0, 0.1),
		P(2, 10, 0.2),
	}).WaitForSync(3, modelMyModel).WaitForReadyPodsMetric(3)

	responses, err := integration.StreamedRequest(t, h.Client, requests, 2)
	require.NoError(t, err)
	require.Len(t, responses, 2)

	setHeaders := responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	outboundTraceparent := headerValue(setHeaders, "traceparent")
	require.NotEmpty(t, outboundTraceparent, "expected traceparent in outbound header mutation")

	traceparentCount := 0
	for _, h := range setHeaders {
		if strings.EqualFold(h.GetHeader().GetKey(), "traceparent") {
			traceparentCount++
		}
	}
	require.Equal(t, 1, traceparentCount, "expected exactly one traceparent header")
	require.Contains(t, outboundTraceparent, upstreamTraceID,
		"downstream traceparent must stay in the upstream trace")
	require.NotContains(t, outboundTraceparent, upstreamSpanID,
		"downstream traceparent must not reuse the upstream span id")

	_ = h.Client.CloseSend()
	assert.Eventually(t, func() bool {
		recorded := make(map[string]bool)
		for _, span := range h.GetSpans() {
			recorded[span.Name] = true
		}
		for _, want := range []string{"request", "request_orchestration"} {
			if !recorded[want] {
				return false
			}
		}
		return true
	}, 5*time.Second, 50*time.Millisecond, "expected routing spans to be recorded")

	for _, span := range h.GetSpans() {
		if span.Name != "request" && span.Name != "request_orchestration" {
			continue
		}
		assert.Equal(t, upstreamTraceID, span.SpanContext.TraceID().String(),
			"span %q must join the upstream trace", span.Name)
	}

	// Sanity check: the injected header uses the active request span as parent.
	requestSpanID := ""
	for _, span := range h.GetSpans() {
		if span.Name == "request" {
			requestSpanID = span.SpanContext.SpanID().String()
			break
		}
	}
	require.NotEmpty(t, requestSpanID)
	require.Contains(t, outboundTraceparent, requestSpanID,
		"outbound traceparent parent must be the EPP request span")
	require.NotEqual(t, upstreamTraceparent, outboundTraceparent,
		"outbound traceparent must differ from the client value")
}
