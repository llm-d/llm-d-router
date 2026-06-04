package programaware

import (
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
)

const programAwareSubsystem = "program_aware"

// Package-level metrics that the plugin records to directly.
// These are registered at startup via GetCollectors() before the plugin is instantiated.
var (
	requestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: programAwareSubsystem,
			Name:      "requests_total",
			Help:      metricsutil.HelpMsgWithStability("Total requests received per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	dispatchedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: programAwareSubsystem,
			Name:      "dispatched_total",
			Help:      metricsutil.HelpMsgWithStability("Total requests dispatched per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	inputTokensTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: programAwareSubsystem,
			Name:      "input_tokens_total",
			Help:      metricsutil.HelpMsgWithStability("Total input (prompt) tokens consumed per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	outputTokensTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: programAwareSubsystem,
			Name:      "output_tokens_total",
			Help:      metricsutil.HelpMsgWithStability("Total output (completion) tokens produced per program", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	pickLatencyUs = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: programAwareSubsystem,
			Name:      "pick_latency_microseconds",
			Help:      metricsutil.HelpMsgWithStability("Latency of the Pick() call in microseconds", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000},
		},
	)

	fairnessIndex = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: programAwareSubsystem,
			Name:      "jains_fairness_index",
			Help:      metricsutil.HelpMsgWithStability("Jain's fairness index over attained service (weighted tokens) across active programs (1.0 = perfectly fair)", compbasemetrics.ALPHA),
		},
	)

	ewmaWaitTimeMs = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: programAwareSubsystem,
			Name:      "ewma_wait_time_milliseconds",
			Help:      metricsutil.HelpMsgWithStability("Exponentially weighted moving average of flow control queue wait time per program in milliseconds", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	serviceRateTokensPerSec = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: programAwareSubsystem,
			Name:      "service_rate_tokens_per_second",
			Help:      metricsutil.HelpMsgWithStability("EWMA of weighted tokens per second per program (used for Jain's fairness index)", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	queueScore = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: programAwareSubsystem,
			Name:      "queue_score",
			Help:      metricsutil.HelpMsgWithStability("Scheduling priority score computed by the scoring strategy for each program queue during Pick()", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)
)

// GetCollectors returns the shared Prometheus collectors for the program-aware
// plugin. Strategy-owned collectors are exposed via ScoringStrategy.Collectors
// and registered separately by the plugin factory.
func GetCollectors() []prometheus.Collector {
	return []prometheus.Collector{
		requestsTotal,
		dispatchedTotal,
		inputTokensTotal,
		outputTokensTotal,
		pickLatencyUs,
		fairnessIndex,
		ewmaWaitTimeMs,
		serviceRateTokensPerSec,
		queueScore,
	}
}
