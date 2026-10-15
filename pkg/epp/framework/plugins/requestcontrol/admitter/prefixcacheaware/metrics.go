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

package prefixcacheaware

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Decision outcome labels. Each Admit call records exactly one of these.
const (
	// outcomeAdmittedBelowThreshold: the pool is below T; the request is admitted.
	outcomeAdmittedBelowThreshold = "admitted_below_threshold"
	// outcomeAdmittedHot: the pool is at or above T and at least one endpoint has
	// a hit ratio >= H; the request is admitted.
	outcomeAdmittedHot = "admitted_hot"
	// outcomeShedCold: the pool is at or above T and no endpoint has a hit ratio
	// >= H; the request is denied.
	outcomeShedCold = "shed_cold"
	// outcomeAdmittedSmallRequest: the pool is at or above T but the request adds fewer
	// uncached tokens than minUncachedTokens; the request is admitted.
	outcomeAdmittedSmallRequest = "admitted_small_request"
	// outcomeNotApplicable: the admitter received no endpoints.
	outcomeNotApplicable = "not_applicable"
)

var (
	decisions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "prefix_cache_aware_admitter_decisions_total",
			Help:      metricsutil.HelpMsgWithStability("Prefix-cache-aware admitter decisions, by outcome.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "outcome"},
	)
	observedSaturation = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "prefix_cache_aware_admitter_saturation",
			Help:      metricsutil.HelpMsgWithStability("Pool saturation the prefix-cache-aware admitter read on its last decision.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name"},
	)
)

// registerMetrics registers the admitter collectors on the given registerer. A nil
// registerer is a no-op. Repeated registration of the same collector is tolerated.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{decisions, observedSaturation} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register prefix-cache-aware admitter metric: %w", err)
		}
	}
	return nil
}

func recordDecision(pluginName, outcome string) {
	decisions.WithLabelValues(pluginName, outcome).Inc()
}

func recordSaturation(pluginName string, saturation float64) {
	observedSaturation.WithLabelValues(pluginName).Set(saturation)
}

// resetMetrics clears the collectors between tests.
func resetMetrics() {
	decisions.Reset()
	observedSaturation.Reset()
}
