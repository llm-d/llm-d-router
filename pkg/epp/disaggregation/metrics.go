package disaggregation

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

// Metric namespace/subsystem. Landed under the shared llm_d_epp subsystem so
// operators can grep for one namespace when debugging disagg deployments.
const (
	metricsSubsystem = "llm_d_epp"
	metricsPrefix    = "disagg"
)

var (
	headerStampedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: metricsSubsystem,
			Name:      metricsPrefix + "_header_stamped_total",
			Help:      "Response headers stamped by the disaggregation controller, by selector name.",
		},
		[]string{"selector"},
	)

	filterOutcomeTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: metricsSubsystem,
			Name:      metricsPrefix + "_filter_outcome_total",
			Help:      "Per-selector filter outcomes: matched, no_match_strict, no_match_prefer_fallback.",
		},
		[]string{"selector", "mode", "outcome"},
	)

	gatingDroppedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: metricsSubsystem,
			Name:      metricsPrefix + "_gating_dropped_total",
			Help:      "Requests where the gating filter dropped at least one candidate, by revision.",
		},
		[]string{"revision"},
	)
)

var registerMetricsOnce sync.Once

// registerMetrics attaches disaggregation collectors to the controller-runtime
// metrics registry. Idempotent — safe to call multiple times.
func registerMetrics() {
	registerMetricsOnce.Do(func() {
		ctrlmetrics.Registry.MustRegister(
			headerStampedTotal,
			filterOutcomeTotal,
			gatingDroppedTotal,
		)
	})
}

// Filter outcome labels attached to disagg_filter_outcome_total. "absent"
// (no header sent) is deliberately NOT recorded — it is the silent default
// on every request that doesn't opt in, so the counter would balloon with
// near-zero-signal increments.
const (
	// filterOutcomeMatched: header matched at least one candidate; survivor
	// set narrowed to the intersection. Fires for both strict and prefer.
	filterOutcomeMatched = "matched"
	// filterOutcomeNoMatchStrict: strict-mode header matched zero candidates;
	// survivor set became empty and the framework will return 503. This is
	// the "no fallback" case operators alert on.
	filterOutcomeNoMatchStrict = "no_match_strict"
	// filterOutcomeNoMatchPreferFallback: prefer-mode header matched zero
	// candidates; survivor set kept unchanged (fallback engaged). Not an
	// error — expected during rollouts before the client updates its header.
	filterOutcomeNoMatchPreferFallback = "no_match_prefer_fallback"
)

// Package-scoped emitters — one call site per collector, tied to the
// disaggregation package rather than a per-controller receiver. The earlier
// per-instance Metrics struct was dropped because it was empty and forwarded
// to these same globals through a nil-guarded wrapper.

func recordHeaderStamped(selectorName string) {
	headerStampedTotal.WithLabelValues(selectorName).Inc()
}

func recordFilterOutcome(selectorName string, mode SelectorMode, outcome string) {
	filterOutcomeTotal.WithLabelValues(selectorName, string(mode), outcome).Inc()
}

func recordGatingDropped(revision string) {
	gatingDroppedTotal.WithLabelValues(revision).Inc()
}
