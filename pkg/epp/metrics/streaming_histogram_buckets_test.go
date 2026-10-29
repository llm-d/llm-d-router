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

package metrics

import (
	"fmt"
	"math"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

// Interactive decoding sits in this range, so the defaults are held to a
// maximum bucket width across it.
const (
	interactiveRangeLow = 0.005
	interactiveRangeHi  = 0.05
	interactiveMaxWidth = 0.005
	// floatSlack absorbs the representation error of sums such as 0.045-0.04.
	floatSlack = 1e-9
)

// checkInteractiveSpacing reports the first bucket inside the interactive range
// that is wider than interactiveMaxWidth.
func checkInteractiveSpacing(edges []float64) error {
	for i := 1; i < len(edges); i++ {
		lower, upper := edges[i-1], edges[i]
		if upper <= interactiveRangeLow || lower >= interactiveRangeHi {
			continue
		}
		if upper-lower > interactiveMaxWidth+floatSlack {
			return fmt.Errorf("bucket (%v, %v] is %v wide, want at most %v between %v and %v",
				lower, upper, upper-lower, interactiveMaxWidth, interactiveRangeLow, interactiveRangeHi)
		}
	}
	return nil
}

func TestDefaultStreamingBucketsAreWellFormed(t *testing.T) {
	for name, edges := range map[string][]float64{
		"request_streaming_tpot_seconds": DefaultRequestTPOTBuckets,
		"request_streaming_itl_seconds":  DefaultInterTokenLatencyBuckets,
	} {
		t.Run(name, func(t *testing.T) {
			require.NoError(t, ValidateHistogramBuckets(edges))
			require.NoError(t, checkInteractiveSpacing(edges))
		})
	}
}

// TestInteractiveSpacingRejectsCoarseEdges keeps the guard honest: the edges
// these defaults replaced must fail it.
func TestInteractiveSpacingRejectsCoarseEdges(t *testing.T) {
	coarse := []float64{0.0005, 0.00205, 0.005, 0.01, 0.02, 0.04, 0.06}
	require.NoError(t, ValidateHistogramBuckets(coarse))
	require.Error(t, checkInteractiveSpacing(coarse))
}

// TestDefaultSeriesPerLabelCombination pins the cardinality of the defaults.
// A histogram exports one series per edge, one for the implicit +Inf bucket,
// plus _sum and _count, for every label combination. Changing a ladder has to
// move these numbers deliberately.
func TestDefaultSeriesPerLabelCombination(t *testing.T) {
	for name, tc := range map[string]struct {
		histogram *prometheus.HistogramVec
		want      int
	}{
		"request_streaming_tpot_seconds": {newRequestTPOTHistogram(DefaultRequestTPOTBuckets), 28},
		"request_streaming_itl_seconds":  {newInterTokenLatencyHistogram(DefaultInterTokenLatencyBuckets), 24},
	} {
		t.Run(name, func(t *testing.T) {
			got := gatherSingle(t, tc.histogram, 0.01)
			require.Equal(t, tc.want, len(got.GetBucket())+3)
		})
	}
}

func TestValidateHistogramBuckets(t *testing.T) {
	for name, tc := range map[string]struct {
		edges   []float64
		wantErr bool
	}{
		"tpot defaults":               {DefaultRequestTPOTBuckets, false},
		"itl defaults":                {DefaultInterTokenLatencyBuckets, false},
		"single edge":                 {[]float64{0.01}, false},
		"nil":                         {nil, true},
		"empty":                       {[]float64{}, true},
		"zero":                        {[]float64{0, 0.01}, true},
		"negative":                    {[]float64{-0.01, 0.01}, true},
		"nan":                         {[]float64{0.01, math.NaN()}, true},
		"positive inf":                {[]float64{0.01, math.Inf(1)}, true},
		"negative inf":                {[]float64{math.Inf(-1), 0.01}, true},
		"duplicate":                   {[]float64{0.01, 0.01}, true},
		"descending":                  {[]float64{0.02, 0.01}, true},
		"at the maximum bucket count": {ascendingEdges(MaxHistogramBuckets), false},
		"one over the maximum":        {ascendingEdges(MaxHistogramBuckets + 1), true},
	} {
		t.Run(name, func(t *testing.T) {
			err := ValidateHistogramBuckets(tc.edges)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
		})
	}
}

func TestCustomEdgesReachTheScrape(t *testing.T) {
	edges := []float64{0.01, 0.02, 0.03}
	histogram := newRequestTPOTHistogram(edges)

	reg := prometheus.NewRegistry()
	require.NoError(t, reg.Register(histogram))
	histogram.WithLabelValues("m", "tm", "f", "p", "messages").Observe(0.015)
	histogram.WithLabelValues("m", "tm", "f", "p", "messages").Observe(0.5)

	got := gatherFrom(t, reg)
	require.Equal(t, edges, bucketEdges(got))
	// 0.015 lands in (0.01, 0.02]; 0.5 only in the implicit +Inf bucket.
	require.Equal(t, []uint64{0, 1, 1}, bucketCounts(got))
	require.Equal(t, uint64(2), got.GetSampleCount())
	require.InDelta(t, 0.515, got.GetSampleSum(), 1e-9)
}

func TestFactoriesCopyTheirEdges(t *testing.T) {
	edges := []float64{0.01, 0.02, 0.03}
	histogram := newInterTokenLatencyHistogram(edges)
	edges[1] = 99 // a flag's backing array may be reused after construction

	got := gatherSingle(t, histogram, 0.015)
	require.Equal(t, []float64{0.01, 0.02, 0.03}, bucketEdges(got))
}

func gatherSingle(t *testing.T, histogram *prometheus.HistogramVec, observation float64) *dto.Histogram {
	t.Helper()
	reg := prometheus.NewRegistry()
	require.NoError(t, reg.Register(histogram))
	histogram.WithLabelValues("m", "tm", "f", "p", "messages").Observe(observation)
	return gatherFrom(t, reg)
}

func gatherFrom(t *testing.T, reg *prometheus.Registry) *dto.Histogram {
	t.Helper()
	families, err := reg.Gather()
	require.NoError(t, err)
	require.Len(t, families, 1)
	require.Len(t, families[0].GetMetric(), 1)
	histogram := families[0].GetMetric()[0].GetHistogram()
	require.NotNil(t, histogram)
	return histogram
}

func bucketEdges(histogram *dto.Histogram) []float64 {
	edges := make([]float64, 0, len(histogram.GetBucket()))
	for _, bucket := range histogram.GetBucket() {
		edges = append(edges, bucket.GetUpperBound())
	}
	return edges
}

func bucketCounts(histogram *dto.Histogram) []uint64 {
	counts := make([]uint64, 0, len(histogram.GetBucket()))
	for _, bucket := range histogram.GetBucket() {
		counts = append(counts, bucket.GetCumulativeCount())
	}
	return counts
}

// ascendingEdges builds a strictly increasing ladder of n boundaries, for the
// cases where only the number of boundaries is under test.
func ascendingEdges(n int) []float64 {
	edges := make([]float64, 0, n)
	for i := 1; i <= n; i++ {
		edges = append(edges, float64(i)*0.001)
	}
	return edges
}

// TestMaxHistogramBucketsLeavesRoomAboveTheDefaults keeps the limit and the
// default ladders in a fixed relationship, so extending a default ladder stays
// well inside the limit.
func TestMaxHistogramBucketsLeavesRoomAboveTheDefaults(t *testing.T) {
	for name, edges := range map[string][]float64{
		"request_streaming_tpot_seconds": DefaultRequestTPOTBuckets,
		"request_streaming_itl_seconds":  DefaultInterTokenLatencyBuckets,
	} {
		t.Run(name, func(t *testing.T) {
			require.Greater(t, MaxHistogramBuckets, 2*len(edges))
		})
	}
}
