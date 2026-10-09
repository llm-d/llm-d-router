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

package metrics

import (
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
)

// sampleCount returns the histogram's observation count. Plain histograms have
// no Reset, so tests compare before/after deltas.
func sampleCount(t *testing.T, h prometheus.Histogram) uint64 {
	t.Helper()
	m := &dto.Metric{}
	require.NoError(t, h.Write(m))
	return m.GetHistogram().GetSampleCount()
}

func TestRecordRequest(t *testing.T) {
	requestsTotal.Reset()

	RecordRequest("chat_completions")
	RecordRequest("chat_completions")
	RecordRequest("responses")

	require.Equal(t, 2.0, promtestutil.ToFloat64(requestsTotal.WithLabelValues("chat_completions")))
	require.Equal(t, 1.0, promtestutil.ToFloat64(requestsTotal.WithLabelValues("responses")))
}

func TestRecordDisagg(t *testing.T) {
	disaggRequestsTotal.Reset()

	RecordDisagg(metricsutil.DisaggPathPrefillDecode)
	RecordDisagg(metricsutil.DisaggPathPrefillDecode)
	RecordDisagg(metricsutil.DisaggPathEncodePrefillDecode)
	RecordDisagg(metricsutil.DisaggPathEncodeDecode)

	require.Equal(t, 2.0, promtestutil.ToFloat64(disaggRequestsTotal.WithLabelValues(metricsutil.DisaggPathPrefillDecode)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(disaggRequestsTotal.WithLabelValues(metricsutil.DisaggPathEncodePrefillDecode)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(disaggRequestsTotal.WithLabelValues(metricsutil.DisaggPathEncodeDecode)))
}

func TestRecordDurations(t *testing.T) {
	encodeBase := sampleCount(t, encodeDuration)
	prefillBase := sampleCount(t, prefillDuration)
	decodeBase := sampleCount(t, decodeDuration)

	RecordEncodeDuration(50 * time.Millisecond)
	RecordPrefillDuration(100 * time.Millisecond)
	RecordDecodeDuration(250 * time.Millisecond)

	require.Equal(t, encodeBase+1, sampleCount(t, encodeDuration))
	require.Equal(t, prefillBase+1, sampleCount(t, prefillDuration))
	require.Equal(t, decodeBase+1, sampleCount(t, decodeDuration))
}

func TestRecordError(t *testing.T) {
	errorsTotal.Reset()

	RecordError(StagePrefill)
	RecordError(StagePrefill)
	RecordError(StageDecode)
	RecordError(StageEncode)

	require.Equal(t, 2.0, promtestutil.ToFloat64(errorsTotal.WithLabelValues(StagePrefill)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(errorsTotal.WithLabelValues(StageDecode)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(errorsTotal.WithLabelValues(StageEncode)))
}

func TestRecordToolCallingFieldStatuses(t *testing.T) {
	toolCallingFieldStatusTotal.Reset()
	t.Cleanup(toolCallingFieldStatusTotal.Reset)
	statuses := []toolcalling.FieldStatus{
		{Field: toolcalling.FieldTools, Status: toolcalling.FieldStatusPreserved, Observed: true},
		{Field: toolcalling.FieldToolChoice, Status: toolcalling.FieldStatusChanged, Observed: true},
		{Field: toolcalling.FieldParallelToolCalls, Status: toolcalling.FieldStatusDropped, Observed: true},
		{Field: toolcalling.FieldResponseFormat, Status: toolcalling.FieldStatusRejected, Observed: true},
	}
	RecordToolCallingFieldStatuses(statuses)
	RecordToolCallingFieldStatuses(statuses[:1])

	require.NoError(t, promtestutil.CollectAndCompare(toolCallingFieldStatusTotal, strings.NewReader(`
# HELP llm_d_disagg_sidecar_tool_calling_field_status_total [ALPHA] Tool-calling request field outcomes by component, direction, field, and status.
# TYPE llm_d_disagg_sidecar_tool_calling_field_status_total counter
llm_d_disagg_sidecar_tool_calling_field_status_total{component="routing_sidecar",direction="request",field="tools",status="preserved"} 2
llm_d_disagg_sidecar_tool_calling_field_status_total{component="routing_sidecar",direction="request",field="tool_choice",status="changed"} 1
llm_d_disagg_sidecar_tool_calling_field_status_total{component="routing_sidecar",direction="request",field="parallel_tool_calls",status="dropped"} 1
llm_d_disagg_sidecar_tool_calling_field_status_total{component="routing_sidecar",direction="request",field="response_format",status="rejected"} 1
`)))
}

func TestRecordToolCallingFieldStatusesSkipsUnobservedAndInvalidValues(t *testing.T) {
	toolCallingFieldStatusTotal.Reset()
	t.Cleanup(toolCallingFieldStatusTotal.Reset)
	RecordToolCallingFieldStatuses(nil)
	RecordToolCallingFieldStatuses([]toolcalling.FieldStatus{
		{Field: toolcalling.FieldTools, Status: toolcalling.FieldStatusPreserved},
		{Field: "private_field_name", Status: toolcalling.FieldStatusChanged, Observed: true},
		{Field: toolcalling.FieldTools, Status: "private_status_value", Observed: true},
	})
	require.Zero(t, promtestutil.CollectAndCount(toolCallingFieldStatusTotal))
}

// Register must be idempotent so repeated calls do not panic on duplicate
// registration with controller-runtime's registry.
func TestRegisterIdempotent(t *testing.T) {
	require.NotPanics(t, func() {
		Register()
		Register()
	})
}

// TestMetricNames pins the fully-qualified names this package exports, so a
// subsystem or Name field rename fails this test instead of silently
// breaking every scrape config/dashboard built on the doc comments' promises.
func TestMetricNames(t *testing.T) {
	reg := prometheus.NewRegistry()
	reg.MustRegister(requestsTotal, disaggRequestsTotal, encodeDuration, prefillDuration, decodeDuration, errorsTotal, toolCallingFieldStatusTotal)
	// Gather omits a CounterVec with no children, so create one in each.
	requestsTotal.WithLabelValues("x")
	disaggRequestsTotal.WithLabelValues("x")
	errorsTotal.WithLabelValues("x")
	toolCallingFieldStatusTotal.WithLabelValues("routing_sidecar", toolcalling.DirectionRequest, string(toolcalling.FieldTools), string(toolcalling.FieldStatusPreserved))
	t.Cleanup(toolCallingFieldStatusTotal.Reset)

	mfs, err := reg.Gather()
	require.NoError(t, err)

	names := make([]string, 0, len(mfs))
	for _, mf := range mfs {
		names = append(names, mf.GetName())
	}

	require.ElementsMatch(t, []string{
		"llm_d_disagg_sidecar_requests_total",
		"llm_d_disagg_sidecar_disagg_requests_total",
		"llm_d_disagg_sidecar_encode_duration_seconds",
		"llm_d_disagg_sidecar_prefill_duration_seconds",
		"llm_d_disagg_sidecar_decode_duration_seconds",
		"llm_d_disagg_sidecar_request_errors_total",
		"llm_d_disagg_sidecar_tool_calling_field_status_total",
	}, names)
}
