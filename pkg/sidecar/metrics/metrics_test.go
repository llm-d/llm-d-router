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
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
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

func TestRecordNIXLPushDispatch(t *testing.T) {
	nixlPushDispatchesTotal.Reset()

	RecordNIXLPushDispatch(NIXLPushReasonCacheHit)
	RecordNIXLPushDispatch(NIXLPushReasonCacheHit)
	RecordNIXLPushDispatch(NIXLPushReasonCacheMiss)
	RecordNIXLPushDispatch(NIXLPushReasonSerialOnly)
	RecordNIXLPushDispatch(NIXLPushReasonPrefillRetry)

	// Each reason has one series, under the mode it implies.
	require.Equal(t, 4, promtestutil.CollectAndCount(nixlPushDispatchesTotal))
	require.Equal(t, 2.0, promtestutil.ToFloat64(nixlPushDispatchesTotal.WithLabelValues(NIXLPushModeParallel, NIXLPushReasonCacheHit)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(nixlPushDispatchesTotal.WithLabelValues(NIXLPushModeSerial, NIXLPushReasonCacheMiss)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(nixlPushDispatchesTotal.WithLabelValues(NIXLPushModeSerial, NIXLPushReasonSerialOnly)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(nixlPushDispatchesTotal.WithLabelValues(NIXLPushModeSerial, NIXLPushReasonPrefillRetry)))
}

func TestRecordNIXLPushIdentityEvents(t *testing.T) {
	// Plain counters have no Reset, so the test compares before/after values.
	mismatches := promtestutil.ToFloat64(nixlPushIdentityMismatchesTotal)
	drops := promtestutil.ToFloat64(nixlPushIdentityDropsTotal)
	marks := promtestutil.ToFloat64(nixlPushSerialOnlyMarksTotal)

	RecordNIXLPushIdentityMismatch()
	RecordNIXLPushIdentityDrop()
	RecordNIXLPushIdentityDrop()
	RecordNIXLPushSerialOnlyMark()

	require.Equal(t, mismatches+1, promtestutil.ToFloat64(nixlPushIdentityMismatchesTotal))
	require.Equal(t, drops+2, promtestutil.ToFloat64(nixlPushIdentityDropsTotal))
	require.Equal(t, marks+1, promtestutil.ToFloat64(nixlPushSerialOnlyMarksTotal))
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
	reg.MustRegister(requestsTotal, disaggRequestsTotal, encodeDuration, prefillDuration, decodeDuration, errorsTotal,
		nixlPushDispatchesTotal, nixlPushIdentityMismatchesTotal, nixlPushIdentityDropsTotal, nixlPushSerialOnlyMarksTotal)
	// Gather omits a CounterVec with no children, so create one in each.
	requestsTotal.WithLabelValues("x")
	disaggRequestsTotal.WithLabelValues("x")
	errorsTotal.WithLabelValues("x")
	nixlPushDispatchesTotal.WithLabelValues("x", "x")

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
		"llm_d_disagg_sidecar_nixl_push_dispatches_total",
		"llm_d_disagg_sidecar_nixl_push_identity_mismatches_total",
		"llm_d_disagg_sidecar_nixl_push_identity_drops_total",
		"llm_d_disagg_sidecar_nixl_push_serial_only_marks_total",
	}, names)
}
