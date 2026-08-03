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
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// routerSubsystem is the router's standard EPP metrics subsystem. New metrics
// are emitted under it; the legacy kvcache_* names are retained as deprecated
// aliases so existing scrapers keep working during the migration.
const routerSubsystem = "llm_d_router_epp"

// podIdentifierLabel is the label key carried by the per-pod kvevents metrics.
// Keeping it as a constant localizes label-schema changes to this package.
const podIdentifierLabel = "pod_identifier"

// dualCounter emits a value to both the deprecated kvcache_* counter and the
// current llm_d_router_epp_* counter. It satisfies prometheus.Collector so both
// register, and exposes the recording/read methods the call sites use.
type dualCounter struct {
	deprecated, current prometheus.Counter
}

func (d *dualCounter) Inc()                      { d.deprecated.Inc(); d.current.Inc() }
func (d *dualCounter) Add(v float64)             { d.deprecated.Add(v); d.current.Add(v) }
func (d *dualCounter) Desc() *prometheus.Desc    { return d.current.Desc() }
func (d *dualCounter) Write(m *dto.Metric) error { return d.current.Write(m) }
func (d *dualCounter) Describe(c chan<- *prometheus.Desc) {
	d.deprecated.Describe(c)
	d.current.Describe(c)
}
func (d *dualCounter) Collect(c chan<- prometheus.Metric) {
	d.deprecated.Collect(c)
	d.current.Collect(c)
}

// dualHistogram is the histogram analogue of dualCounter.
type dualHistogram struct {
	deprecated, current prometheus.Histogram
}

func (d *dualHistogram) Observe(v float64)         { d.deprecated.Observe(v); d.current.Observe(v) }
func (d *dualHistogram) Desc() *prometheus.Desc    { return d.current.Desc() }
func (d *dualHistogram) Write(m *dto.Metric) error { return d.current.Write(m) }
func (d *dualHistogram) Describe(c chan<- *prometheus.Desc) {
	d.deprecated.Describe(c)
	d.current.Describe(c)
}
func (d *dualHistogram) Collect(c chan<- prometheus.Metric) {
	d.deprecated.Collect(c)
	d.current.Collect(c)
}

func newDualCounter(oldSubsystem, oldName, newName, help string) *dualCounter {
	return &dualCounter{
		deprecated: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "kvcache", Subsystem: oldSubsystem, Name: oldName,
			Help: "Deprecated: use " + routerSubsystem + "_" + newName + ". " + help,
		}),
		current: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: routerSubsystem, Name: newName,
			Help: metricsutil.HelpMsgWithStability(help, compbasemetrics.ALPHA),
		}),
	}
}

func newDualHistogram(oldSubsystem, oldName, newName, help string, buckets []float64) *dualHistogram {
	return &dualHistogram{
		deprecated: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: "kvcache", Subsystem: oldSubsystem, Name: oldName,
			Help:    "Deprecated: use " + routerSubsystem + "_" + newName + ". " + help,
			Buckets: buckets,
		}),
		current: prometheus.NewHistogram(prometheus.HistogramOpts{
			Subsystem: routerSubsystem, Name: newName,
			Help:    metricsutil.HelpMsgWithStability(help, compbasemetrics.ALPHA),
			Buckets: buckets,
		}),
	}
}

var (
	Admissions = newDualCounter("index", "admissions_total",
		"kv_cache_index_admissions_total", "Total number of KV-block admissions")
	Evictions = newDualCounter("index", "evictions_total",
		"kv_cache_index_evictions_total", "Total number of KV-block evictions")

	// LookupRequests counts how many Lookup() calls have been made.
	LookupRequests = newDualCounter("index", "lookup_requests_total",
		"kv_cache_index_lookup_requests_total", "Total number of lookup calls")
	// MaxPodHitCount counts the maximum cache hits on a single pod on Lookup().
	MaxPodHitCount = newDualCounter("index", "max_pod_hit_count_total",
		"kv_cache_index_max_pod_hit_count_total", "Maximum cache hits on a single pod on Lookup()")
	// LookupHits counts how many keys were found in the cache on Lookup().
	LookupHits = newDualCounter("index", "lookup_hits_total",
		"kv_cache_index_lookup_hits_total", "Number of keys found in the cache on Lookup()")
	// LookupLatency logs latency of lookup calls.
	LookupLatency = newDualHistogram("index", "lookup_latency_seconds",
		"kv_cache_index_lookup_latency_seconds", "Latency of Lookup calls in seconds", prometheus.DefBuckets)

	// DedupRemovedHashesSuppressed counts individual block hashes whose removal
	// was suppressed by the kvevents reference-count dedup filter because another
	// announcement still references the block. This counts block hashes, not
	// BlockRemoved events.
	DedupRemovedHashesSuppressed = newDualCounter("kvevents", "dedup_removed_hashes_suppressed_total",
		"kv_cache_events_dedup_removed_hashes_suppressed_total",
		"Block hashes whose removal was suppressed by the KV-event dedup filter (block hashes, not BlockRemoved events)")
	// DedupRemovedHashesForwarded counts individual block hashes forwarded to the
	// index for eviction after passing the kvevents dedup filter. This counts
	// block hashes, not BlockRemoved events.
	DedupRemovedHashesForwarded = newDualCounter("kvevents", "dedup_removed_hashes_forwarded_total",
		"kv_cache_events_dedup_removed_hashes_forwarded_total",
		"Block hashes forwarded for eviction after the KV-event dedup filter (block hashes, not BlockRemoved events)")

	// SubscriberActive tracks the number of ZMQ subscribers currently managed by
	// the SubscriberManager. A subscriber is counted from creation until it is
	// removed, so the gauge includes subscribers that are retrying a failed
	// connection; it is a measure of intent, not of live connectivity. Pair it
	// with SubscriberReconnections and ZMQErrors to spot unhealthy subscribers.
	SubscriberActive = prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_active_subscribers",
		Help: metricsutil.HelpMsgWithStability(
			"Number of ZMQ subscribers currently managed, including those retrying a failed connection",
			compbasemetrics.ALPHA),
	})
	// SubscriberReconnections counts ZMQ subscriber reconnection attempts,
	// labeled by the pod identifier the subscriber is bound to.
	SubscriberReconnections = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_subscriber_reconnections_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of ZMQ subscriber reconnection attempts", compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
	// MessagesReceived counts raw messages received from ZMQ subscribers. The
	// rate of message ingestion can be derived via rate() in Prometheus.
	MessagesReceived = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_messages_received_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of messages received from ZMQ subscribers", compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
	// ZMQErrors counts errors encountered by ZMQ subscribers, labeled by the
	// pod identifier and the operation that failed (e.g. "connect", "recv").
	ZMQErrors = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_zmq_errors_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of ZMQ subscriber errors", compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel, "operation"})
	// PoolQueueDepth tracks the total number of messages queued across all
	// worker shards of the event processing pool.
	PoolQueueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_pool_queue_depth",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of messages queued across all worker shards", compbasemetrics.ALPHA),
	})
	// PoolCapacity tracks the number of worker shards (capacity) configured for
	// the event processing pool.
	PoolCapacity = prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_pool_capacity",
		Help: metricsutil.HelpMsgWithStability(
			"Number of worker shards (capacity) in the event processing pool", compbasemetrics.ALPHA),
	})
)

// Collectors returns a slice of all registered Prometheus collectors.
func Collectors() []prometheus.Collector {
	return []prometheus.Collector{
		Admissions, Evictions,
		LookupRequests, LookupHits, LookupLatency, MaxPodHitCount,
		DedupRemovedHashesSuppressed, DedupRemovedHashesForwarded,
		SubscriberActive, SubscriberReconnections, MessagesReceived, ZMQErrors,
		PoolQueueDepth, PoolCapacity,
	}
}

// CleanupSubscriber drops every per-pod kvevents series for podIdentifier so
// stale time series do not linger after a subscriber is removed. Callers must
// invoke it only once the subscriber's goroutine has exited, otherwise a late
// increment resurrects the series.
func CleanupSubscriber(podIdentifier string) {
	SubscriberReconnections.DeleteLabelValues(podIdentifier)
	MessagesReceived.DeleteLabelValues(podIdentifier)
	ZMQErrors.DeletePartialMatch(prometheus.Labels{podIdentifierLabel: podIdentifier})
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

	var maxPodHitMetric dto.Metric
	err = MaxPodHitCount.Write(&maxPodHitMetric)
	if err != nil {
		return
	}
	maxPodHitCount := maxPodHitMetric.GetCounter().GetValue()

	log.FromContext(ctx).WithName("metrics").Info("metrics beat",
		"admissions", admissions,
		"evictions", evictions,
		"lookups", lookups,
		"hits", hits,
		"max_pod_hit_count", maxPodHitCount,
		"latency_count", latencyCount,
		"latency_sum", latencySum,
		"latency_avg", latencyAvg,
	)
}
