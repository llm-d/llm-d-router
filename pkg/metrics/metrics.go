// Package metrics provides metrics registration for the epp.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/metrics"
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

	// DeciderReasonDisabled indicates disaggregation is disabled (NonCachedTokens=0).
	DeciderReasonDisabled = "disabled"
	// DeciderReasonInputTooShort indicates the total input is shorter than the NonCachedTokens threshold.
	DeciderReasonInputTooShort = "input_too_short"
	// DeciderReasonSuffixCached indicates the non-cached suffix is below the threshold,
	// meaning enough of the prompt is already cached and disaggregation is unnecessary.
	DeciderReasonSuffixCached = "suffix_cached"
	// DeciderReasonError indicates disaggregation was skipped due to an error
	// (e.g. nil endpoint, missing prefix cache state, input parse failure).
	DeciderReasonError = "error"
	// DeciderReasonDisaggregated indicates the decider chose to disaggregate.
	DeciderReasonDisaggregated = "disaggregated"
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

	// DeciderEvaluationCount records the reason for each disaggregation decider
	// evaluation. This complements disagg_decision_total by exposing *why* the
	// decider accepted or rejected disaggregation, enabling operators to
	// distinguish threshold misses from errors or disabled configurations.
	SchedulerDeciderEvaluationCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "decider_evaluation_total",
			Help:      metrics.HelpMsgWithStability("Total number of disaggregation decider evaluations by reason", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "decider", "reason"},
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
		SchedulerDeciderEvaluationCount,
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

// RecordDeciderEvaluation increments the counter for a disaggregation decider evaluation.
// The decider parameter identifies the decider plugin (e.g. "prefix-based-pd-decider").
// The reason parameter must be one of the DeciderReason* constants.
func RecordDeciderEvaluation(modelName, decider, reason string) {
	if modelName == "" {
		modelName = "unknown"
	}
	SchedulerDeciderEvaluationCount.WithLabelValues(modelName, decider, reason).Inc()
}
