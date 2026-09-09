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

package loraresidency

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// outcomeSticky: at least one home had room, requests stay on the homes.
	outcomeSticky = "sticky"
	// outcomeNoHome: the adapter is resident nowhere, every endpoint passes.
	outcomeNoHome = "no_home"
	// outcomeSpread: every home was saturated, a new copy is allowed on an endpoint with room.
	outcomeSpread = "spread"
	// outcomeCapBlocked: every home was saturated but maxReplicas forbids another copy.
	outcomeCapBlocked = "cap_blocked"
	// outcomeFleetSaturated: every home and every other endpoint was saturated.
	outcomeFleetSaturated = "fleet_saturated"
	// outcomeNotApplicable: no target model or a single candidate.
	outcomeNotApplicable = "not_applicable"
)

var filterDecisions = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "lora_residency_filter_decisions_total",
		Help:      metricsutil.HelpMsgWithStability("LoRA residency filter decisions, by outcome.", compbasemetrics.ALPHA),
	},
	[]string{"plugin_name", "outcome"},
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	if err := registerer.Register(filterDecisions); err != nil {
		var alreadyRegistered prometheus.AlreadyRegisteredError
		if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == filterDecisions {
			return nil
		}
		return fmt.Errorf("register lora residency filter metric: %w", err)
	}
	return nil
}

func recordDecision(pluginName, outcome string) {
	filterDecisions.WithLabelValues(pluginName, outcome).Inc()
}

func resetMetrics() {
	filterDecisions.Reset()
}
