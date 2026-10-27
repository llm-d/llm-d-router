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
	"errors"
	"fmt"
	"math"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// Interactive decoding sits between 5ms and 50ms per output token, so both
// streaming latency histograms keep 5ms spacing across that range. Outside it
// the original edges are preserved.
var (
	// DefaultRequestTPOTBuckets are the default bucket edges, in seconds, of
	// request_streaming_tpot_seconds.
	DefaultRequestTPOTBuckets = []float64{
		0.0005, 0.00205, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
		0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 2,
	}

	// DefaultInterTokenLatencyBuckets are the default bucket edges, in seconds,
	// of request_streaming_itl_seconds.
	DefaultInterTokenLatencyBuckets = []float64{
		0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
		0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 2,
	}
)

// MaxHistogramBuckets caps how many boundaries a configured ladder may have.
// Each boundary is another series per label combination, so an oversized list
// multiplies cardinality. The limit is about 2.5 times the longer default ladder.
const MaxHistogramBuckets = 64

// ValidateHistogramBuckets reports whether edges can be used as Prometheus
// histogram bucket boundaries. Callers pass only explicitly configured values;
// an empty slice means the caller asked for a histogram with no buckets and is
// rejected rather than silently replaced by a default.
func ValidateHistogramBuckets(edges []float64) error {
	if len(edges) == 0 {
		return errors.New("at least one bucket boundary is required")
	}
	if len(edges) > MaxHistogramBuckets {
		return fmt.Errorf("%d bucket boundaries exceed the maximum of %d", len(edges), MaxHistogramBuckets)
	}
	var prev float64
	for i, edge := range edges {
		if math.IsNaN(edge) || math.IsInf(edge, 0) {
			return fmt.Errorf("bucket boundary %d is %v, want a finite value", i, edge)
		}
		if edge <= 0 {
			return fmt.Errorf("bucket boundary %d is %v, want a positive value", i, edge)
		}
		if i > 0 && edge <= prev {
			return fmt.Errorf("bucket boundary %d is %v, want a value greater than the preceding %v", i, edge, prev)
		}
		prev = edge
	}
	return nil
}

// newRequestTPOTHistogram builds request_streaming_tpot_seconds over the given
// bucket edges. The edges are copied so later mutation of the caller's slice,
// such as a flag's backing array, cannot change a registered collector.
func newRequestTPOTHistogram(edges []float64) *prometheus.HistogramVec {
	return prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_streaming_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Average time per output token in seconds for streaming requests, computed as (e2e - TTFT) / (output_tokens - 1).", compbasemetrics.ALPHA),
			Buckets:   append([]float64(nil), edges...),
		},
		modelLabelsWithFairnessPriority,
	)
}

// newInterTokenLatencyHistogram builds request_streaming_itl_seconds over the
// given bucket edges, which are copied for the same reason.
func newInterTokenLatencyHistogram(edges []float64) *prometheus.HistogramVec {
	return prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_streaming_itl_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inter-token latency in seconds for streaming requests, measured as the time between consecutive response body chunks.", compbasemetrics.ALPHA),
			Buckets:   append([]float64(nil), edges...),
		},
		modelLabelsWithFairnessPriority,
	)
}
