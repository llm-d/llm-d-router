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

// Package registrationtest holds metric checks that need a process of their
// own. Registration runs once per process through a sync.Once and lands in the
// controller-runtime registry, and the metrics package's own TestMain registers
// the defaults for its whole binary, so a custom-bucket registration can only
// be observed from a separate test binary.
package registrationtest

import (
	"context"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// TestRegisterWithOptionsReachesTheScrape fixes the composed path through the
// real entry point: RegisterWithOptions builds the streaming latency
// histograms from the options, the recording helpers observe into the
// registered collectors, and a gather from the controller-runtime registry
// returns the configured edges.
func TestRegisterWithOptionsReachesTheScrape(t *testing.T) {
	tpotEdges := []float64{0.01, 0.02}
	itlEdges := []float64{0.005, 0.01}

	metrics.RegisterWithOptions(metrics.RegisterOptions{
		RequestTPOTBuckets: tpotEdges,
		RequestITLBuckets:  itlEdges,
	})

	ctx := context.Background()
	received := time.Now()
	// (40ms - 10ms) / (3 - 1) = 15ms, which lands in (0.01, 0.02].
	require.True(t, metrics.RecordRequestTPOT(ctx, "m", "tm", reqcommon.APITypeMessages.Surface(), "f", "p", true,
		received, received.Add(10*time.Millisecond), received.Add(40*time.Millisecond), 3))
	require.True(t, metrics.RecordInterTokenLatency(ctx, "m", "tm", reqcommon.APITypeMessages.Surface(), "f", "p", 0.0075))

	tpot := gatherHistogram(t, "llm_d_epp_request_streaming_tpot_seconds")
	require.Equal(t, tpotEdges, upperBounds(tpot))
	require.Equal(t, []uint64{0, 1}, cumulativeCounts(tpot))

	itl := gatherHistogram(t, "llm_d_epp_request_streaming_itl_seconds")
	require.Equal(t, itlEdges, upperBounds(itl))
	require.Equal(t, []uint64{0, 1}, cumulativeCounts(itl))
}

func gatherHistogram(t *testing.T, name string) *dto.Histogram {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		require.Len(t, family.GetMetric(), 1)
		histogram := family.GetMetric()[0].GetHistogram()
		require.NotNil(t, histogram)
		return histogram
	}
	t.Fatalf("metric family %q not gathered", name)
	return nil
}

func upperBounds(histogram *dto.Histogram) []float64 {
	bounds := make([]float64, 0, len(histogram.GetBucket()))
	for _, bucket := range histogram.GetBucket() {
		bounds = append(bounds, bucket.GetUpperBound())
	}
	return bounds
}

func cumulativeCounts(histogram *dto.Histogram) []uint64 {
	counts := make([]uint64, 0, len(histogram.GetBucket()))
	for _, bucket := range histogram.GetBucket() {
		counts = append(counts, bucket.GetCumulativeCount())
	}
	return counts
}
