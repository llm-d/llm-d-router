// Package metrics provides metrics registration for the epp.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	"github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

const (
	// SchedulerSubsystem is the metric prefix of the package.
	SchedulerSubsystem = "llm_d_inference_scheduler"

	// DecisionTypeDecodeOnly is for requests that are routed to decode instance only.
	DecisionTypeDecodeOnly = "decode-only"
	// DecisionTypePrefillDecode is for requests that are gone through P/D or EP/D.
	DecisionTypePrefillDecode = "prefill-decode"
	// DecisionTypeEncodeDecode is for requests that are gone through E/PD.
	DecisionTypeEncodeDecode = "encode-decode"
	// DecisionTypeEncodePrefillDecode is for requests that are gone through E/P/D.
	DecisionTypeEncodePrefillDecode = "encode-prefill-decode"
)

var (
	// SchedulerPDDecisionCount records request P/D decision.
	//
	// Deprecated: Use SchedulerDisaggDecisionCount instead.
	SchedulerPDDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "pd_decision_total",
			Help:      metrics.HelpMsgWithStability("Total number of P/D disaggregation decisions made", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "decision_type"}, // "decode-only" or "prefill-decode"
	)

	// SchedulerDisaggDecisionCount records disaggregation routing decisions,
	// covering all stages: decode-only, prefill-decode, encode-decode, encode-prefill-decode.
	SchedulerDisaggDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "disagg_decision_total",
			Help:      metrics.HelpMsgWithStability("Total number of disaggregation routing decisions made", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "decision_type"},
	)

	// Data-layer counters: label values must be plugin TypedName.Type only —
	// never per-instance or runtime-variable strings (cardinality).

	DataLayerPollErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "datalayer_poll_errors_total",
			Help:      metrics.HelpMsgWithStability("Data-source poll errors per source type.", compbasemetrics.ALPHA),
		},
		[]string{"source_type"},
	)

	DataLayerExtractErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "datalayer_extract_errors_total",
			Help:      metrics.HelpMsgWithStability("Extract errors per source/extractor type.", compbasemetrics.ALPHA),
		},
		[]string{"source_type", "extractor_type"},
	)
)

// GetCollectors returns all custom collectors for the llm-d-inference-scheduler.
func GetCollectors() []prometheus.Collector {
	return []prometheus.Collector{
		SchedulerPDDecisionCount,
		SchedulerDisaggDecisionCount,
		DataLayerPollErrorsTotal,
		DataLayerExtractErrorsTotal,
	}
}

// RecordPDDecision increments the counter for a specific P/D routing decision.
//
// Deprecated: Use RecordDisaggDecision instead.
func RecordPDDecision(modelName, decisionType string) {
	if modelName == "" {
		modelName = "unknown"
	}
	SchedulerPDDecisionCount.WithLabelValues(modelName, decisionType).Inc()
}

// RecordDisaggDecision increments the counter for a disaggregation routing decision.
// The decisionType must be one of the DecisionType* constants (DecisionTypeDecodeOnly,
// DecisionTypePrefillDecode, DecisionTypeEncodeDecode, DecisionTypeEncodePrefillDecode).
// The model parameter should be the target model name; if empty, "unknown" is used.
func RecordDisaggDecision(modelName, decisionType string) {
	if modelName == "" {
		modelName = "unknown"
	}
	SchedulerDisaggDecisionCount.WithLabelValues(modelName, decisionType).Inc()
}

// DisaggDecisionType returns the DecisionType* constant corresponding to which
// disaggregation stages were used for a request.
func DisaggDecisionType(encodeUsed, prefillUsed bool) string {
	switch {
	case encodeUsed && prefillUsed:
		return DecisionTypeEncodePrefillDecode
	case encodeUsed:
		return DecisionTypeEncodeDecode
	case prefillUsed:
		return DecisionTypePrefillDecode
	default:
		return DecisionTypeDecodeOnly
	}
}
