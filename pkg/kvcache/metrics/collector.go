// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metrics

import (
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

var (
	Admissions = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "admissions_total",
		Help: "Total number of KV-block admissions",
	})
	Evictions = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "evictions_total",
		Help: "Total number of KV-block evictions",
	})

	// LookupRequests counts how many Lookup() calls have been made.
	LookupRequests = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "lookup_requests_total",
		Help: "Total number of lookup calls",
	})
	// MaxPodHitCount counts the maximum cache hits on a single pod on Lookup().
	MaxPodHitCount = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "max_pod_hit_count_total",
		Help: "Maximum cache hits on a single pod on Lookup()",
	})
	// LookupHits counts how many keys were found in the cache on Lookup().
	LookupHits = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "lookup_hits_total",
		Help: "Number of keys found in the cache on Lookup()",
	})
	// LookupLatency logs latency of lookup calls.
	LookupLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "lookup_latency_seconds",
		Help:    "Latency of Lookup calls in seconds",
		Buckets: prometheus.DefBuckets,
	})
)

// Collectors returns a slice of all registered Prometheus collectors.
func Collectors() []prometheus.Collector {
	return []prometheus.Collector{
		Admissions, Evictions,
		LookupRequests, LookupHits, LookupLatency,
	}
}

var registerMetricsOnce = sync.Once{}

// Register registers all metrics with K8s registry.
func Register() {
	registerMetricsOnce.Do(func() {
		metrics.Registry.MustRegister(Collectors()...)
	})
}

// StartMetricsLogging spawns a goroutine that logs current metric values every
// interval.
func StartMetricsLogging(ctx context.Context, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			logMetrics(ctx)
		}
	}()
}

func logMetrics(ctx context.Context) {
	var m dto.Metric

	err := Admissions.Write(&m)
	if err != nil {
		return
	}
	admissions := m.GetCounter().GetValue()

	err = Evictions.Write(&m)
	if err != nil {
		return
	}
	evictions := m.GetCounter().GetValue()

	err = LookupRequests.Write(&m)
	if err != nil {
		return
	}
	lookups := m.GetCounter().GetValue()

	var hitsMetric dto.Metric
	err = LookupHits.Write(&hitsMetric)
	if err != nil {
		return
	}
	hits := hitsMetric.GetCounter().GetValue()

	var latencyMetric dto.Metric
	err = LookupLatency.Write(&latencyMetric)
	if err != nil {
		return
	}
	latencyCount := latencyMetric.GetHistogram().GetSampleCount()
	latencySum := latencyMetric.GetHistogram().GetSampleSum()

	latencyAvg := 0.0
	if latencyCount > 0 {
		latencyAvg = latencySum / float64(latencyCount)
	}

	klog.FromContext(ctx).WithName("metrics").Info("metrics beat",
		"admissions", admissions,
		"evictions", evictions,
		"lookups", lookups,
		"hits", hits,
		"latency_count", latencyCount,
		"latency_sum", latencySum,
		"latency_avg", latencyAvg,
	)
}
