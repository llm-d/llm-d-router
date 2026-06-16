package programaware

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var (
	fairnessIndex = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_jains_fairness_index",
			Help:      metricsutil.HelpMsgWithStability("Jain's fairness index over average wait time across active programs.", compbasemetrics.ALPHA),
		},
	)

	avgWaitTimeMs = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_avg_wait_time_milliseconds",
			Help:      metricsutil.HelpMsgWithStability("Cumulative mean of flow-control queue wait time per program in milliseconds.", compbasemetrics.ALPHA),
		},
		[]string{"program_id"},
	)
)

func GetCollectors() []prometheus.Collector {
	return []prometheus.Collector{fairnessIndex, avgWaitTimeMs}
}

func DeleteSharedSeries(id string) {
	avgWaitTimeMs.DeleteLabelValues(id)
}
