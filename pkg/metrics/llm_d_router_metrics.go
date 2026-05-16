// Package metrics provides metrics registration for the epp.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/metrics"
)

const (
	// llmdSubsystem is the standardized metric prefix of the package.
	llmdSubsystem = "llm_d_router_epp"
)

var (
	// LlmdPDDecisionCount records request P/D decision.
	LlmdPDDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: llmdSubsystem,
			Name:      "pd_decision_total",
			Help:      metrics.HelpMsgWithStability("Total number of P/D disaggregation decisions made", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "decision_type"},
	)

	// LlmdDisaggDecisionCount records disaggregation routing decisions.
	LlmdDisaggDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: llmdSubsystem,
			Name:      "disagg_decision_total",
			Help:      metrics.HelpMsgWithStability("Total number of disaggregation routing decisions made", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "decision_type"},
	)

	// LlmdDataLayerPollErrorsTotal records data-source poll errors per source type.
	LlmdDataLayerPollErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: llmdSubsystem,
			Name:      "datalayer_poll_errors_total",
			Help:      metrics.HelpMsgWithStability("Data-source poll errors per source type.", compbasemetrics.ALPHA),
		},
		[]string{"source_type"},
	)

	// LlmdDataLayerExtractErrorsTotal records extract errors per source/extractor type.
	LlmdDataLayerExtractErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: llmdSubsystem,
			Name:      "datalayer_extract_errors_total",
			Help:      metrics.HelpMsgWithStability("Extract errors per source/extractor type.", compbasemetrics.ALPHA),
		},
		[]string{"source_type", "extractor_type"},
	)
)
