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

	RecordDisagg(DisaggTypePD)
	RecordDisagg(DisaggTypePD)
	RecordDisagg(DisaggTypeEPD)
	RecordDisagg(DisaggTypeED)

	require.Equal(t, 2.0, promtestutil.ToFloat64(disaggRequestsTotal.WithLabelValues(DisaggTypePD)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(disaggRequestsTotal.WithLabelValues(DisaggTypeEPD)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(disaggRequestsTotal.WithLabelValues(DisaggTypeED)))
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

// Register must be idempotent so repeated calls do not panic on duplicate
// registration with controller-runtime's registry.
func TestRegisterIdempotent(t *testing.T) {
	require.NotPanics(t, func() {
		Register()
		Register()
	})
}
