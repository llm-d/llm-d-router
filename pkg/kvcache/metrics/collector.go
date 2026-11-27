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

// routerSubsystem is the router's standard EPP metrics subsystem.
const routerSubsystem = metricsutil.LLMDRouterEndpointPickerSubsystem

// podIdentifierLabel is the label key carried by the per-pod kvevents metrics.
// Keeping it as a constant localizes label-schema changes to this package.
const podIdentifierLabel = "pod_identifier"

const (
	cacheKindLabel = "cache_kind"
	reasonLabel    = "reason"
	tagLabel       = "tag"
)

// maxUnknownTagLabelValues caps the distinct tag label values. Tags are read
// from the engine's wire payload, so their number is not bounded by the router.
const maxUnknownTagLabelValues = 32

var unknownTagLimiter = metricsutil.NewBoundedLabel(maxUnknownTagLabelValues)

// emptyTagLabel is the tag label value for an event whose tag is the empty
// string. Prometheus treats an empty label value as a missing label.
const emptyTagLabel = "empty"

var (
	Admissions = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_index_admissions_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-block admissions",
			compbasemetrics.ALPHA),
	})
	Evictions = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_index_evictions_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-block evictions",
			compbasemetrics.ALPHA),
	})

	// LookupRequests counts how many Lookup() calls have been made.
	LookupRequests = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_index_lookup_requests_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of lookup calls",
			compbasemetrics.ALPHA),
	})
	// MaxPodHitCount counts, per prefix match, the longest contiguous prefix
	// chain any single pod holds counting from the first requested block.
	MaxPodHitCount = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_index_max_pod_hit_count_total",
		Help: metricsutil.HelpMsgWithStability(
			"Longest contiguous per-pod prefix chain observed per lookup",
			compbasemetrics.ALPHA),
	})
	// LookupHits accumulates the same per-match contiguous chain length as
	// MaxPodHitCount.
	LookupHits = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_index_lookup_hits_total",
		Help: metricsutil.HelpMsgWithStability(
			"Contiguous prefix blocks matched by the best pod per lookup",
			compbasemetrics.ALPHA),
	})
	LookupLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_index_lookup_latency_seconds",
		Help: metricsutil.HelpMsgWithStability(
			"Duration of Lookup and WalkKeys calls in seconds, including WalkKeys callbacks",
			compbasemetrics.ALPHA),
		Buckets: prometheus.DefBuckets,
	})

	// DedupRemovedHashesSuppressed counts individual block hashes whose removal
	// was suppressed by the kvevents reference-count dedup filter because another
	// announcement still references the block. This counts block hashes, not
	// BlockRemoved events.
	DedupRemovedHashesSuppressed = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_dedup_removed_hashes_suppressed_total",
		Help: metricsutil.HelpMsgWithStability(
			"Block hashes whose removal was suppressed by the KV-event dedup filter (block hashes, not BlockRemoved events)",
			compbasemetrics.ALPHA),
	})
	// DedupRemovedHashesForwarded counts individual block hashes forwarded to the
	// index for eviction after passing the kvevents dedup filter. This counts
	// block hashes, not BlockRemoved events.
	DedupRemovedHashesForwarded = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_dedup_removed_hashes_forwarded_total",
		Help: metricsutil.HelpMsgWithStability(
			"Block hashes forwarded for eviction after the KV-event dedup filter (block hashes, not BlockRemoved events)",
			compbasemetrics.ALPHA),
	})
	// KVEventStoresSkipped counts group-aware stores that cannot safely
	// contribute to prefix indexing.
	KVEventStoresSkipped = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_stores_skipped_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-cache store events skipped before prefix indexing",
			compbasemetrics.ALPHA),
	}, []string{cacheKindLabel, reasonLabel})
	// KVEventRemovalsSkipped counts removals for groups excluded from prefix
	// indexing.
	KVEventRemovalsSkipped = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_removals_skipped_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-cache removal events skipped for groups excluded from prefix indexing",
			compbasemetrics.ALPHA),
	}, []string{cacheKindLabel, reasonLabel})
	// KVEventUnknownTagsSkipped counts events skipped because the engine
	// adapter does not know their tag.
	KVEventUnknownTagsSkipped = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_unknown_tags_skipped_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-cache events skipped because the engine adapter does not know their tag",
			compbasemetrics.ALPHA),
	}, []string{tagLabel})

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
	// LiveOnlySubscribers counts subscribers indexing live events without a
	// snapshot or replay base after a load failed, from the first such event
	// until the subscriber follows a complete event sequence again or is removed.
	LiveOnlySubscribers = prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_live_only_subscribers",
		Help: metricsutil.HelpMsgWithStability(
			"Number of ZMQ subscribers indexing from live events only after a failed snapshot or replay load",
			compbasemetrics.ALPHA),
	})
	// SnapshotLoadDuration observes successful snapshot loads, from the request
	// until the snapshot's batches are queued for indexing.
	SnapshotLoadDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_snapshot_load_duration_seconds",
		Help: metricsutil.HelpMsgWithStability(
			"Duration of successful snapshot loads in seconds, until the batches are queued for indexing",
			compbasemetrics.ALPHA),
		Buckets: prometheus.ExponentialBuckets(0.05, 2, 10),
	})
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
	// EventsSequenceGaps counts detected ZMQ sequence number gaps per pod.
	EventsSequenceGaps = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_sequence_gaps_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-event sequence gaps detected by ZMQ subscribers", compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
	// EventsMissed counts missed KV-event sequence numbers inferred from gaps.
	EventsMissed = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_missed_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of missed KV-event messages inferred from sequence gaps", compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
	// EventsDropped counts KV-event messages dropped before indexing, labeled by
	// pod identifier and drop reason (e.g. "malformed_frame", "replay_cooldown", "decode_error").
	EventsDropped = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_dropped_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-event messages dropped before indexing", compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel, reasonLabel})
	// EventsProcessingLag observes the elapsed time between the engine-emitted
	// batch timestamp and worker-pool processing.
	EventsProcessingLag = prometheus.NewHistogram(prometheus.HistogramOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_processing_lag_seconds",
		Help: metricsutil.HelpMsgWithStability(
			"Elapsed time between KV-event batch timestamp and worker-pool processing in seconds", compbasemetrics.ALPHA),
		Buckets: prometheus.DefBuckets,
	})
)

// Collectors returns a slice of all registered Prometheus collectors.
func Collectors() []prometheus.Collector {
	return []prometheus.Collector{
		Admissions, Evictions,
		LookupRequests, LookupHits, LookupLatency, MaxPodHitCount,
		DedupRemovedHashesSuppressed, DedupRemovedHashesForwarded,
		KVEventStoresSkipped, KVEventRemovalsSkipped, KVEventUnknownTagsSkipped,
		SubscriberActive, SubscriberReconnections, MessagesReceived, ZMQErrors,
		LiveOnlySubscribers, SnapshotLoadDuration,
		PoolQueueDepth, PoolCapacity,
		EventsSequenceGaps, EventsMissed, EventsDropped, EventsProcessingLag,
	}
}

// RecordUnknownTagSkipped counts one event skipped for an unknown tag and
// returns the tag label value it was counted under, which differs from tag
// when the tag is empty or the limiter folds it.
func RecordUnknownTagSkipped(tag string) string {
	// An empty tag is not passed to the limiter, so a full cap cannot fold it.
	label := emptyTagLabel
	if tag != "" {
		label = unknownTagLimiter.Bound(tag)
	}
	KVEventUnknownTagsSkipped.WithLabelValues(label).Inc()
	return label
}

// CleanupSubscriber drops every per-pod kvevents series for podIdentifier so
// stale time series do not linger after a subscriber is removed. Callers must
// invoke it only once the subscriber's goroutine has exited, otherwise a late
// increment resurrects the series.
func CleanupSubscriber(podIdentifier string) {
	SubscriberReconnections.DeleteLabelValues(podIdentifier)
	MessagesReceived.DeleteLabelValues(podIdentifier)
	ZMQErrors.DeletePartialMatch(prometheus.Labels{podIdentifierLabel: podIdentifier})
	EventsSequenceGaps.DeleteLabelValues(podIdentifier)
	EventsMissed.DeleteLabelValues(podIdentifier)
	EventsDropped.DeletePartialMatch(prometheus.Labels{podIdentifierLabel: podIdentifier})
}

var registerMetricsOnce = sync.Once{}

// Register registers all metrics with K8s registry.
func Register() {
	registerMetricsOnce.Do(func() {
		metrics.Registry.MustRegister(Collectors()...)
	})
}

// StartMetricsLogging spawns a goroutine that logs current metric values every
// interval until ctx ends.
func StartMetricsLogging(ctx context.Context, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				logMetrics(ctx)
			}
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
