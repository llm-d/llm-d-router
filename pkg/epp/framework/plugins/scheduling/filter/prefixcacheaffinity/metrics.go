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

package prefixcacheaffinity

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Decision outcome labels. Each Filter call records exactly one of these,
// capturing whether kv-cache affinity or load decided the candidate set.
const (
	// outcomeSticky: the affinity gate narrowed the set to endpoints whose
	// prefix cache score met the threshold, and the load gate kept them.
	outcomeSticky = "sticky"
	// outcomeNoMatch: no endpoint met the affinity threshold, so all were kept.
	outcomeNoMatch = "no_match"
	// outcomeLoadOverride: the TTFT load gate discarded a non-empty sticky set
	// because those endpoints were too slow, reopening all endpoints.
	outcomeLoadOverride = "load_override"
	// outcomeExploration: the gate was skipped for exploration.
	outcomeExploration = "exploration"
	// outcomeNotApplicable: the filter had nothing to decide (a single
	// candidate, or the affinity threshold disabled).
	outcomeNotApplicable = "not_applicable"
)

var filterDecisions = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_cache_affinity_filter_decisions_total",
		Help:      metricsutil.HelpMsgWithStability("Prefix cache affinity filter decisions, by outcome.", compbasemetrics.ALPHA),
	},
	[]string{"plugin_name", "outcome"},
)

// registerMetrics registers the filter collectors on the given registerer. A nil
// registerer is a no-op, so the filter runs without metrics when the handle
// supplies no recorder. Repeated registration of the same collector is tolerated.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	if err := registerer.Register(filterDecisions); err != nil {
		var alreadyRegistered prometheus.AlreadyRegisteredError
		if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == filterDecisions {
			return nil
		}
		return fmt.Errorf("register prefix cache affinity filter metric: %w", err)
	}
	return nil
}

func recordDecision(pluginName, outcome string) {
	filterDecisions.WithLabelValues(pluginName, outcome).Inc()
}

// resetMetrics clears the collectors between tests.
func resetMetrics() {
	filterDecisions.Reset()
}
