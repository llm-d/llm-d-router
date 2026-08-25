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

package utilization

import (
	"context"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

func makePodMetric(name string, queueDepth int, kvUsage float64, updateTime time.Time) fwkdl.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Name: name, Namespace: "ns1"},
	}
	metrics := fwkdl.NewMetrics()
	metrics.WaitingQueueSize = queueDepth
	metrics.KVCacheUsagePercent = kvUsage
	metrics.UpdateTime = updateTime
	return fwkdl.NewEndpoint(meta, metrics)
}

func makeSchedulingEndpoint(
	name string,
	queueDepth int,
	kvUsage float64,
	updateTime time.Time,
) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Name: name, Namespace: "ns1"},
	}
	metrics := fwkdl.NewMetrics()
	metrics.WaitingQueueSize = queueDepth
	metrics.KVCacheUsagePercent = kvUsage
	metrics.UpdateTime = updateTime
	return fwksched.NewEndpoint(meta, metrics, nil)
}

// TestUtilizationDetectorFactory evaluates instantiation properties and config parsing constraints.
// It guards against improper configuration block parameters failing initialization correctly.
func TestUtilizationDetectorFactory(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		configJSON []byte
		wantError  bool
	}{
		{
			name:       "valid configuration",
			configJSON: []byte(`{"queueDepthThreshold": 5, "kvCacheUtilThreshold": 0.8}`),
			wantError:  false,
		},
		{
			name:       "invalid schema",
			configJSON: []byte(`{"queueDepthThreshold": "invalid_type"}`),
			wantError:  true,
		},
		{
			name:       "empty config applies defaults",
			configJSON: []byte(`{}`),
			wantError:  false,
		},
		{
			name:       "invalid queue depth",
			configJSON: []byte(`{"queueDepthThreshold": 0}`),
			wantError:  true,
		},
		{
			name:       "invalid kv cache high",
			configJSON: []byte(`{"kvCacheUtilThreshold": 1.5}`),
			wantError:  true,
		},
		{
			name:       "invalid kv cache low",
			configJSON: []byte(`{"kvCacheUtilThreshold": 0.0}`),
			wantError:  true,
		},
		{
			name:       "invalid metrics staleness",
			configJSON: []byte(`{"metricsStalenessThreshold": "0s"}`),
			wantError:  true,
		},
		{
			name:       "invalid headroom",
			configJSON: []byte(`{"headroom": -0.5}`),
			wantError:  true,
		},
		{
			name:       "high headroom warning",
			configJSON: []byte(`{"headroom": 2.0}`),
			wantError:  false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			plugin, err := UtilizationDetectorFactory("test-util-detector", fwkplugin.StrictDecoder(tc.configJSON), fwkplugin.NewEppHandle(t.Context(), func() []types.NamespacedName { return nil }))
			if tc.wantError {
				require.Error(t, err, "Expected initialization to fail on invalid configuration")
				require.Nil(t, plugin, "Plugin must be nil when initialization fails")
			} else {
				require.NoError(t, err, "Expected initialization to succeed with valid configuration")
				require.NotNil(t, plugin, "Plugin must not be nil on success")
			}
		})
	}
}

// TestDetector_TypedName provides structural assurance that initialization assigns proper types.
func TestDetector_TypedName(t *testing.T) {
	t.Parallel()
	plugin, err := UtilizationDetectorFactory("test-plugin", fwkplugin.StrictDecoder([]byte(`{}`)), fwkplugin.NewEppHandle(
		t.Context(), func() []types.NamespacedName { return nil }))
	require.NoError(t, err, "Plugin initialization should succeed")
	require.Equal(t, "test-plugin", plugin.TypedName().Name,
		"TypedName must match the name provided during initialization")
	require.Equal(t, "utilization-detector", plugin.TypedName().Type,
		"TypedName.Type must be exactly 'utilization-detector'")
}

func TestDetector_Saturation(t *testing.T) {
	t.Parallel()

	baseTime := time.Now()

	// Config: Queue=5, KV=0.9
	config := &Config{
		QueueDepthThreshold:       5,
		KVCacheUtilThreshold:      0.90,
		MetricsStalenessThreshold: 100 * time.Millisecond,
	}

	tests := []struct {
		name           string
		pods           []fwkdl.Endpoint
		wantSaturation float64
	}{
		{
			name:           "No candidate pods",
			pods:           []fwkdl.Endpoint{},
			wantSaturation: 1.0, // Fail closed
		},
		{
			name: "Single pod with good capacity",
			pods: []fwkdl.Endpoint{
				// Q=2/5 (0.4). KV=0.5/0.9 (0.555...).
				// Max(0.4, 0.555...) = 0.555...
				makePodMetric("pod1", 2, 0.5, baseTime),
			},
			wantSaturation: 0.5 / 0.9,
		},
		{
			name: "Single pod with stale metrics",
			pods: []fwkdl.Endpoint{
				// All endpoints stale: fail-open (0.0), not fail-closed (1.0).
				makePodMetric("pod1", 1, 0.1, baseTime.Add(-200*time.Millisecond)),
			},
			wantSaturation: 0.0,
		},
		{
			name: "Single pod with high queue depth",
			pods: []fwkdl.Endpoint{
				// Q=10/5 (2.0). KV=0.1/0.9 (0.11).
				// Max(2.0, 0.11) = 2.0
				makePodMetric("pod1", 10, 0.1, baseTime),
			},
			wantSaturation: 2.0,
		},
		{
			name: "Single pod with high KV cache utilization",
			pods: []fwkdl.Endpoint{
				// Q=1/5 (0.2). KV=0.95/0.90 (1.055...).
				// Max(0.2, 1.055...) = 1.055...
				makePodMetric("pod1", 1, 0.95, baseTime),
			},
			wantSaturation: 0.95 / 0.90,
		},
		{
			name: "Single pod with nil metrics",
			pods: []fwkdl.Endpoint{
				// All endpoints have nil metrics: fail-open (0.0).
				fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
					ID: types.NamespacedName{Name: "pod1", Namespace: "ns1"},
				}, nil),
			},
			wantSaturation: 0.0,
		},
		{
			name: "Multiple pods, all good capacity",
			pods: []fwkdl.Endpoint{
				// Pod1: Q=1/5(0.2), KV=0.1/0.9(0.11). Max=0.2.
				makePodMetric("pod1", 1, 0.1, baseTime),
				// Pod2: Q=0/5(0.0), KV=0.2/0.9(0.22). Max=0.22...
				makePodMetric("pod2", 0, 0.2, baseTime),
			},
			// Avg(0.2, 0.222...) = 0.2111...
			wantSaturation: (0.2 + (0.2 / 0.9)) / 2.0,
		},
		{
			name: "Multiple pods, one good, one stale",
			pods: []fwkdl.Endpoint{
				// Pod1 (Good): Q=1/5(0.2), KV=0.1/0.9(0.11). Max=0.2.
				makePodMetric("pod1", 1, 0.1, baseTime),
				// Pod2 (Stale): 1.0.
				makePodMetric("pod2", 0, 0.2, baseTime.Add(-300*time.Millisecond)),
			},
			// Avg(0.2, 1.0) = 0.6
			wantSaturation: 0.6,
		},
		{
			name: "Multiple pods, one good, one bad (high queue)",
			pods: []fwkdl.Endpoint{
				// Pod1 (Good): Max=0.2.
				makePodMetric("pod1", 1, 0.1, baseTime),
				// Pod2 (Bad): Q=15/5(3.0). Max=3.0.
				makePodMetric("pod2", 15, 0.2, baseTime),
			},
			// Avg(0.2, 3.0) = 1.6
			wantSaturation: 1.6,
		},
		{
			name: "Multiple pods, all bad capacity",
			pods: []fwkdl.Endpoint{
				// Pod1 (Stale): 1.0
				makePodMetric("pod1", 1, 0.1, baseTime.Add(-200*time.Millisecond)),
				// Pod2 (High Q): 20/5 = 4.0
				makePodMetric("pod2", 20, 0.2, baseTime),
				// Pod3 (High KV): 0.99/0.90 = 1.1
				makePodMetric("pod3", 1, 0.99, baseTime),
			},
			// Avg(1.0, 4.0, 1.1) = 6.1 / 3 = 2.033...
			wantSaturation: (1.0 + 4.0 + 1.1) / 3.0,
		},
		{
			name: "Queue depth exactly at threshold",
			pods: []fwkdl.Endpoint{
				// Q=5/5(1.0). KV=Low.
				// Max=1.0
				makePodMetric("pod1", 5, 0.1, baseTime),
			},
			wantSaturation: 1.0,
		},
		{
			name: "Metrics age just over staleness threshold",
			pods: []fwkdl.Endpoint{
				// All endpoints stale: fail-open (0.0).
				makePodMetric("pod1", 1, 0.1, baseTime.Add(-101*time.Millisecond)),
			},
			wantSaturation: 0.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			detector := NewDetector("test-detector", *config, logr.Discard())

			got := detector.Saturation(context.Background(), tc.pods)
			require.InDelta(t, tc.wantSaturation, got, 1e-4, "Saturation mismatch")
		})
	}
}

func TestDetector_Filter(t *testing.T) {
	t.Parallel()

	baseTime := time.Now()

	config := &Config{
		QueueDepthThreshold:       5,
		KVCacheUtilThreshold:      0.80,
		MetricsStalenessThreshold: 100 * time.Millisecond,
		Headroom:                  0.2, // 20% burst
	}

	// Limits: Q = 5 * 1.2 = 6.0, KV = 0.8 * 1.2 = 0.96

	tests := []struct {
		name      string
		endpoints []fwksched.Endpoint
		wantLen   int
	}{
		{
			name: "All pass - under thresholds",
			endpoints: []fwksched.Endpoint{
				makeSchedulingEndpoint("pod1", 1, 0.1, baseTime),
				makeSchedulingEndpoint("pod2", 4, 0.7, baseTime),
			},
			wantLen: 2,
		},
		{
			name: "Pass - at threshold but under burst",
			endpoints: []fwksched.Endpoint{
				makeSchedulingEndpoint("pod1", 5, 0.8, baseTime),
			},
			wantLen: 1,
		},
		{
			name: "Pass - in headroom burst",
			endpoints: []fwksched.Endpoint{
				// Q=5.5 (< 6.0). KV=0.9 (< 0.96).
				makeSchedulingEndpoint("pod1", 5, 0.9, baseTime),
			},
			wantLen: 1,
		},
		{
			name: "Filtered - exceeds queue burst",
			endpoints: []fwksched.Endpoint{
				// Pod1 (Over): Q=10/5=2.0.
				makeSchedulingEndpoint("pod1", 7, 0.1, baseTime),
				// Pod2 (OK): Q=1/5=0.2.
				makeSchedulingEndpoint("pod2", 1, 0.1, baseTime),
			},
			wantLen: 1,
		},
		{
			name: "Filtered - exceeds KV burst",
			endpoints: []fwksched.Endpoint{
				// Pod1 (Over): KV=0.97/0.9=1.07...
				makeSchedulingEndpoint("pod1", 1, 0.97, baseTime),
				// Pod2 (OK): KV=0.5/0.9=0.55...
				makeSchedulingEndpoint("pod2", 1, 0.5, baseTime),
			},
			wantLen: 1,
		},
		{
			name: "Pass - all stale (Fail open at pool level)",
			endpoints: []fwksched.Endpoint{
				makeSchedulingEndpoint("pod1", 1, 0.1, baseTime.Add(-200*time.Millisecond)),
				makeSchedulingEndpoint("pod2", 1, 0.1, baseTime.Add(-200*time.Millisecond)),
			},
			wantLen: 2,
		},
		{
			name: "Pass - all saturated (Fail open at pool level)",
			endpoints: []fwksched.Endpoint{
				makeSchedulingEndpoint("pod1", 10, 0.1, baseTime),
				makeSchedulingEndpoint("pod2", 1, 0.99, baseTime),
			},
			wantLen: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			detector := NewDetector("test-detector", *config, logr.Discard())
			got := detector.Filter(context.Background(), nil, tc.endpoints)
			require.Len(t, got, tc.wantLen)
		})
	}
}

// TestDetector_StaleEndpointObservability verifies that Saturation records the stale-endpoint
// gauge (keyed by detector name) and that the stale-metrics log is time-bounded.
func TestDetector_StaleEndpointObservability(t *testing.T) {
	t.Parallel()

	eppmetrics.Register()

	// A wide staleness threshold keeps fresh pods deterministically fresh on slow CI machines;
	// the stale pod is stamped far past the threshold.
	config := Config{
		QueueDepthThreshold:       5,
		KVCacheUtilThreshold:      0.90,
		MetricsStalenessThreshold: time.Hour,
	}
	// A unique detector name isolates this test's gauge series from parallel tests.
	detectorName := "stale-observability-test"
	detector := NewDetector(detectorName, config, logr.Discard())

	staleGaugeValue := func() float64 {
		families, err := ctrlmetrics.Registry.Gather()
		require.NoError(t, err)
		for _, f := range families {
			if f.GetName() != "llm_d_epp_flow_control_stale_endpoints" {
				continue
			}
			for _, m := range f.GetMetric() {
				for _, l := range m.GetLabel() {
					if l.GetName() == "detector" && l.GetValue() == detectorName {
						return m.GetGauge().GetValue()
					}
				}
			}
		}
		return -1 // Series absent.
	}

	baseTime := time.Now()
	pods := []fwkdl.Endpoint{
		makePodMetric("fresh", 1, 0.1, baseTime),
		makePodMetric("stale", 1, 0.1, baseTime.Add(-2*time.Hour)),
		fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
			ID: types.NamespacedName{Name: "nil-metrics", Namespace: "ns1"},
		}, nil),
	}

	detector.Saturation(context.Background(), pods)
	require.Equal(t, 2.0, staleGaugeValue(), "stale and nil-metrics endpoints should both be counted")
	firstWarn := detector.lastStaleWarnNanos.Load()
	require.NotZero(t, firstWarn, "first stale observation should record a log timestamp")

	// A second observation within staleWarnInterval must not log again.
	detector.Saturation(context.Background(), pods)
	require.Equal(t, firstWarn, detector.lastStaleWarnNanos.Load(),
		"stale-metrics logging must be time-bounded, not per evaluation")

	// An empty candidate list has no stale endpoints; the gauge must not stay pinned at its last
	// value, or an empty-pool stall reads as a metrics collection failure.
	detector.Saturation(context.Background(), []fwkdl.Endpoint{})
	require.Equal(t, 0.0, staleGaugeValue(), "gauge should read zero for an empty candidate list")

	// Re-observe staleness, then confirm fresh metrics clear it.
	detector.Saturation(context.Background(), pods)
	require.Equal(t, 2.0, staleGaugeValue(), "staleness should be re-observed after the empty list")
	detector.Saturation(context.Background(), []fwkdl.Endpoint{makePodMetric("fresh", 1, 0.1, time.Now())})
	require.Equal(t, 0.0, staleGaugeValue(), "gauge should return to zero when staleness clears")
}

// TestDetector_SingleEndpointStaleMetrics_SaturationFilterInconsistency demonstrates the
// behavior gap that causes RHOAIENG-87650: when a single-endpoint pool experiences a brief
// metrics scrape delay, Saturation() returns 1.0 (fail-closed, blocking all dispatch) while
// Filter() returns the endpoint (fail-open fallback). The endpoint is healthy with zero errors;
// only the metrics timestamp is stale.
//
// This inconsistency means the flow control layer blocks all requests via HoL blocking
// (saturation >= usageLimit), even though the scheduler would have routed to the healthy
// endpoint if the request had reached it.
func TestDetector_SingleEndpointStaleMetrics_SaturationFilterInconsistency(t *testing.T) {
	t.Parallel()

	baseTime := time.Now()

	config := &Config{
		QueueDepthThreshold:       5,
		KVCacheUtilThreshold:      0.90,
		MetricsStalenessThreshold: 200 * time.Millisecond,
		Headroom:                  0.2,
	}

	detector := NewDetector("single-ep-stale-test", *config, logr.Discard())

	// A healthy endpoint whose metrics are just barely past the staleness threshold.
	// In production this happens during brief metric scrape delays or concurrent request bursts.
	staleButHealthy := makePodMetric("llama-3-1-8b-instruct", 0, 0.10, baseTime.Add(-250*time.Millisecond))
	dlEndpoints := []fwkdl.Endpoint{staleButHealthy}

	// Saturation must NOT hard-block when every endpoint is stale. When all candidates have
	// stale metrics, Saturation should fail-open (return < 1.0), consistent with Filter's
	// fallback behavior. A stale metric timestamp does not mean the pod is unhealthy.
	saturation := detector.Saturation(context.Background(), dlEndpoints)
	require.Less(t, saturation, 1.0,
		"Saturation must not return 1.0 when all endpoints are stale; fail-open to let the scheduler route to the healthy pod")

	// Filter already handles this correctly: when all endpoints are stale, it returns the
	// full list via its fail-open fallback (len(filtered)==0 triggers fallback).
	schedEndpoints := []fwksched.Endpoint{
		makeSchedulingEndpoint("llama-3-1-8b-instruct", 0, 0.10, baseTime.Add(-250*time.Millisecond)),
	}
	filtered := detector.Filter(context.Background(), nil, schedEndpoints)
	require.Len(t, filtered, 1,
		"Filter returns the stale endpoint via fail-open fallback")

	// Saturation and Filter must agree: both should allow routing to the healthy endpoint
	// when the only signal is a stale metrics timestamp.

	// Sanity check: with fresh metrics, the same endpoint has near-zero saturation.
	freshEndpoint := makePodMetric("llama-3-1-8b-instruct", 0, 0.10, time.Now())
	freshSaturation := detector.Saturation(context.Background(), []fwkdl.Endpoint{freshEndpoint})
	require.Less(t, freshSaturation, 0.2,
		"The same endpoint with fresh metrics has low saturation, confirming the pod is healthy")
}
