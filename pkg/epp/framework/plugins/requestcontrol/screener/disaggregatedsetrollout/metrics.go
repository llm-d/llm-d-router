/*
Copyright 2026 The Kubernetes Authors.

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

package disaggregatedsetrollout

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const metricsPrefix = "disaggregatedset"

var (
	strictRevisionNoMatchTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      metricsPrefix + "_strict_revision_no_match_total",
			Help: metricsutil.HelpMsgWithStability(
				"Strict revision selections that matched no endpoint and failed closed.",
				compbasemetrics.ALPHA,
			),
		},
		[]string{"plugin_type", "plugin_name"},
	)

	revisionGatingShare = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      metricsPrefix + "_revision_gating_share",
			Help: metricsutil.HelpMsgWithStability(
				"Current weighted revision gating share from 0 to 1; incomplete revisions have share 0.",
				compbasemetrics.ALPHA,
			),
		},
		[]string{"plugin_type", "plugin_name", "mode", "revision"},
	)
)

var registerMetricsOnce sync.Once

func registerMetrics() {
	registerMetricsOnce.Do(func() {
		ctrlmetrics.Registry.MustRegister(
			strictRevisionNoMatchTotal,
			revisionGatingShare,
		)
	})
}

func recordStrictRevisionNoMatch(pluginName string) {
	strictRevisionNoMatchTotal.WithLabelValues(PluginType, pluginName).Inc()
}

func recordRevisionGatingShares(
	pluginName string,
	mode GatingMode,
	previous revisionDistribution,
	current revisionDistribution,
) {
	for revision := range previous.roleCounts {
		if _, exists := current.roleCounts[revision]; !exists {
			revisionGatingShare.DeleteLabelValues(PluginType, pluginName, string(mode), revision)
		}
	}
	for revision := range current.roleCounts {
		revisionGatingShare.WithLabelValues(PluginType, pluginName, string(mode), revision).Set(current.shares[revision])
	}
}
