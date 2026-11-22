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
	"context"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// sampleCount returns the histogram's observation count. Plain histograms have
// no Reset, so tests compare before/after deltas.
func sampleCount(t *testing.T, h prometheus.Histogram) uint64 {
	t.Helper()
	m := &dto.Metric{}
	require.NoError(t, h.Write(m))
	return m.GetHistogram().GetSampleCount()
}

func TestRecordBuildInfo(t *testing.T) {
	buildInfo.Reset()

	RecordBuildInfo("abc123", "v1.2.3")

	require.Equal(t, 1.0, promtestutil.ToFloat64(buildInfo.WithLabelValues("abc123", "v1.2.3")))
	require.Equal(t, 1, promtestutil.CollectAndCount(buildInfo, "llm_d_disagg_sidecar_info"))
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

	RecordEncodeDuration(context.Background(), 50*time.Millisecond)
	RecordPrefillDuration(context.Background(), 100*time.Millisecond)
	RecordDecodeDuration(context.Background(), 250*time.Millisecond)

	require.Equal(t, encodeBase+1, sampleCount(t, encodeDuration))
	require.Equal(t, prefillBase+1, sampleCount(t, prefillDuration))
	require.Equal(t, decodeBase+1, sampleCount(t, decodeDuration))
}

// stageRecorders pairs each stage duration recorder with the histogram it
// observes into.
var stageRecorders = map[string]struct {
	record    func(context.Context, time.Duration)
	histogram prometheus.Histogram
}{
	routing.StageEncode:  {RecordEncodeDuration, encodeDuration},
	routing.StagePrefill: {RecordPrefillDuration, prefillDuration},
	routing.StageDecode:  {RecordDecodeDuration, decodeDuration},
}

// exemplarsWithTraceID returns the labels of every exemplar on h that carries
// traceID. The histograms are process-global and keep the newest exemplar per
// bucket, so a test looks for its own trace rather than for any exemplar.
func exemplarsWithTraceID(t *testing.T, h prometheus.Histogram, traceID string) []map[string]string {
	t.Helper()
	m := &dto.Metric{}
	require.NoError(t, h.Write(m))

	var found []map[string]string
	for _, bucket := range m.GetHistogram().GetBucket() {
		labels := map[string]string{}
		for _, l := range bucket.GetExemplar().GetLabel() {
			labels[l.GetName()] = l.GetValue()
		}
		if labels[metricsutil.ExemplarTraceIDLabel] == traceID {
			found = append(found, labels)
		}
	}
	return found
}

// recordingSpan returns a context with a real SDK span, like the
// forward_request span the sidecar starts per request.
func recordingSpan(t *testing.T) (context.Context, trace.Span) {
	t.Helper()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	t.Cleanup(func() { require.NoError(t, tp.Shutdown(context.Background())) })
	return tp.Tracer("sidecar-exemplar-test").Start(context.Background(), "forward_request")
}

// The request span is still open when a stage finishes, so the exemplar carries
// the span ID as well as the trace ID.
func TestRecordDurationsAttachTraceExemplar(t *testing.T) {
	for stage, tc := range stageRecorders {
		t.Run(stage, func(t *testing.T) {
			ctx, span := recordingSpan(t)
			defer span.End()
			base := sampleCount(t, tc.histogram)

			tc.record(ctx, 120*time.Millisecond)

			require.Equal(t, base+1, sampleCount(t, tc.histogram))
			sc := span.SpanContext()
			require.Equal(t, []map[string]string{{
				metricsutil.ExemplarTraceIDLabel: sc.TraceID().String(),
				metricsutil.ExemplarSpanIDLabel:  sc.SpanID().String(),
			}}, exemplarsWithTraceID(t, tc.histogram, sc.TraceID().String()),
				"a sampled request must attach exactly one exemplar")
		})
	}
}

// An unsampled trace was never exported, so an exemplar would link to a trace
// that does not exist. The stage is still observed.
func TestRecordDurationsSkipExemplarForUnsampledSpan(t *testing.T) {
	const traceID = "5c0ffee5c0ffee5c0ffee5c0ffee5c0f"
	tid, err := trace.TraceIDFromHex(traceID)
	require.NoError(t, err)
	sid, err := trace.SpanIDFromHex("0102030405060708")
	require.NoError(t, err)
	ctx := trace.ContextWithSpanContext(context.Background(),
		trace.NewSpanContext(trace.SpanContextConfig{TraceID: tid, SpanID: sid}))

	for stage, tc := range stageRecorders {
		t.Run(stage, func(t *testing.T) {
			base := sampleCount(t, tc.histogram)

			tc.record(ctx, 120*time.Millisecond)

			require.Equal(t, base+1, sampleCount(t, tc.histogram),
				"the stage must be observed whether or not the trace is sampled")
			require.Empty(t, exemplarsWithTraceID(t, tc.histogram, traceID),
				"an unsampled request must not attach an exemplar")
		})
	}
}

func TestRecordError(t *testing.T) {
	errorsTotal.Reset()

	RecordError(routing.StagePrefill)
	RecordError(routing.StagePrefill)
	RecordError(routing.StageDecode)
	RecordError(routing.StageEncode)

	require.Equal(t, 2.0, promtestutil.ToFloat64(errorsTotal.WithLabelValues(routing.StagePrefill)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(errorsTotal.WithLabelValues(routing.StageDecode)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(errorsTotal.WithLabelValues(routing.StageEncode)))
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
