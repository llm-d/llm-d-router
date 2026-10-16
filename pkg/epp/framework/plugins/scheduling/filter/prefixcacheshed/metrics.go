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

package prefixcacheshed

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Decision outcome labels. Each Filter call records exactly one of these.
const (
	// outcomePassBelowThreshold: the pool is below T (or inside the hysteresis
	// band after shedding stopped); the input is returned unchanged.
	outcomePassBelowThreshold = "pass_below_threshold"
	// outcomeAdmittedHot: shedding is active and at least one endpoint with room
	// has a hit ratio >= H; the set is narrowed to those endpoints.
	outcomeAdmittedHot = "admitted_hot"
	// outcomeShedCold: shedding is active and no endpoint has a hit ratio >= H.
	outcomeShedCold = "shed_cold"
	// outcomeShedDisplacedHot: shedding is active, some endpoint has a hit ratio
	// >= H, but none of those endpoints has room.
	outcomeShedDisplacedHot = "shed_displaced_hot"
	// outcomeBypassSmallRequest: shedding is active but the request adds fewer
	// uncached tokens than minUncachedTokens; the input is returned unchanged.
	outcomeBypassSmallRequest = "bypass_small_request"
	// outcomeNotApplicable: the filter received no endpoints.
	outcomeNotApplicable = "not_applicable"
)

var (
	filterDecisions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "prefix_cache_shed_filter_decisions_total",
			Help:      metricsutil.HelpMsgWithStability("Prefix cache shed filter decisions, by outcome.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "outcome"},
	)
	shedActive = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "prefix_cache_shed_filter_active",
			Help:      metricsutil.HelpMsgWithStability("1 while the prefix cache shed filter is shedding, 0 otherwise.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name"},
	)
	observedSaturation = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "prefix_cache_shed_filter_saturation",
			Help:      metricsutil.HelpMsgWithStability("Pool saturation the prefix cache shed filter read on its last decision.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name"},
	)
)

// registerMetrics registers the filter collectors on the given registerer. A nil
// registerer is a no-op. Repeated registration of the same collector is tolerated.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{filterDecisions, shedActive, observedSaturation} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register prefix cache shed filter metric: %w", err)
		}
	}
	return nil
}

func recordDecision(pluginName, outcome string) {
	filterDecisions.WithLabelValues(pluginName, outcome).Inc()
}

func recordState(pluginName string, active bool, saturation float64) {
	value := 0.0
	if active {
		value = 1.0
	}
	shedActive.WithLabelValues(pluginName).Set(value)
	observedSaturation.WithLabelValues(pluginName).Set(saturation)
}

// resetMetrics clears the collectors between tests.
func resetMetrics() {
	filterDecisions.Reset()
	shedActive.Reset()
	observedSaturation.Reset()
}
