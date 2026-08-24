/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
