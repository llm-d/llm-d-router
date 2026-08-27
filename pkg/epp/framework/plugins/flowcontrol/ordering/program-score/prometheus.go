package programscore

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var (
	decayedTurns = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_score_decayed_turns",
			Help:      metricsutil.HelpMsgWithStability("Time-decayed turn count per program.", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)

	decayedTokens = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_score_decayed_tokens",
			Help:      metricsutil.HelpMsgWithStability("Time-decayed token cost per program.", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)
)

func GetCollectors() []prometheus.Collector {
	return []prometheus.Collector{decayedTurns, decayedTokens}
}

func DeleteSharedSeries(id string) {
	decayedTurns.DeleteLabelValues(id)
	decayedTokens.DeleteLabelValues(id)
}
