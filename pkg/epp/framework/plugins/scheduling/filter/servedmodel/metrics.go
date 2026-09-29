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

package servedmodel

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Each Filter call records exactly one outcome. README.md describes them.
const (
	outcomeListed        = "listed"
	outcomeUnlisted      = "unlisted"
	outcomeFallback      = "fallback"
	outcomeEmpty         = "empty"
	outcomeNotApplicable = "not_applicable"
)

var outcomes = []string{outcomeListed, outcomeUnlisted, outcomeFallback, outcomeEmpty, outcomeNotApplicable}

var (
	filterDecisions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "served_model_filter_decisions_total",
			Help:      metricsutil.HelpMsgWithStability("Served model filter decisions, by outcome.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "outcome"},
	)
	// A rate ratio of these counters holds across profiles that share an
	// instance and across EPP replicas.
	candidates = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "served_model_filter_candidates_total",
			Help:      metricsutil.HelpMsgWithStability("Candidate endpoints evaluated by the served model filter.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name"},
	)
	unlistedCandidates = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "served_model_filter_unlisted_candidates_total",
			Help:      metricsutil.HelpMsgWithStability("Candidate endpoints evaluated by the served model filter without a stored model list.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name"},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{filterDecisions, candidates, unlistedCandidates} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register served model filter metric: %w", err)
		}
	}
	return nil
}

// initMetrics creates the series at zero, so rate() counts their first
// increment.
func initMetrics(pluginName string) {
	for _, outcome := range outcomes {
		filterDecisions.WithLabelValues(pluginName, outcome)
	}
	candidates.WithLabelValues(pluginName)
	unlistedCandidates.WithLabelValues(pluginName)
}

func recordDecision(pluginName, outcome string) {
	filterDecisions.WithLabelValues(pluginName, outcome).Inc()
}

func recordCandidates(pluginName string, total, unlisted int) {
	candidates.WithLabelValues(pluginName).Add(float64(total))
	unlistedCandidates.WithLabelValues(pluginName).Add(float64(unlisted))
}

// resetMetrics clears the collectors between tests.
func resetMetrics() {
	filterDecisions.Reset()
	candidates.Reset()
	unlistedCandidates.Reset()
}
