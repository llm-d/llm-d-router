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

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

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

	RecordDisagg("nixlv2")
	RecordDisagg("nixlv2")

	require.Equal(t, 2.0, promtestutil.ToFloat64(disaggRequestsTotal.WithLabelValues("nixlv2")))
}

func TestRecordDurations(t *testing.T) {
	prefillDuration.Reset()
	decodeDuration.Reset()

	RecordPrefillDuration("nixlv2", 100*time.Millisecond)
	RecordDecodeDuration("nixlv2", 250*time.Millisecond)

	require.Equal(t, 1, promtestutil.CollectAndCount(prefillDuration))
	require.Equal(t, 1, promtestutil.CollectAndCount(decodeDuration))
}

func TestRecordError(t *testing.T) {
	errorsTotal.Reset()

	RecordError("sglang", StagePrefill)
	RecordError("sglang", StagePrefill)
	RecordError("sglang", StageDecode)

	require.Equal(t, 2.0, promtestutil.ToFloat64(errorsTotal.WithLabelValues("sglang", StagePrefill)))
	require.Equal(t, 1.0, promtestutil.ToFloat64(errorsTotal.WithLabelValues("sglang", StageDecode)))
}

// Register must be idempotent so repeated calls do not panic on duplicate
// registration with controller-runtime's registry.
func TestRegisterIdempotent(t *testing.T) {
	require.NotPanics(t, func() {
		Register()
		Register()
	})
}
