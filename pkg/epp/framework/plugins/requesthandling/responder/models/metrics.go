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

package models

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Screen outcome labels, described in the README.
const (
	outcomeAdapter       = "adapter"
	outcomeBase          = "base"
	outcomeUnreported    = "unreported"
	outcomeNotFound      = "not_found"
	outcomeNotApplicable = "not_applicable"
)

var screenDecisions = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "models_responder_screen_decisions_total",
		Help:      metricsutil.HelpMsgWithStability("Requests screened by the models responder, by target model and by which endpoints it kept.", compbasemetrics.ALPHA),
	},
	[]string{"plugin_name", "model_name", "outcome"},
)

// registerMetrics registers the responder collector on the given registerer. A
// nil registerer is a no-op. Repeated registration of the same collector is
// tolerated.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	if err := registerer.Register(screenDecisions); err != nil {
		var alreadyRegistered prometheus.AlreadyRegisteredError
		if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == screenDecisions {
			return nil
		}
		return fmt.Errorf("register models responder metric: %w", err)
	}
	return nil
}

// boundModelName caps the model_name label. Tests swap it for a small limiter so
// they don't fill the shared cap.
var boundModelName = eppmetrics.BoundModelName

// recordDecision counts one screened request. The target model goes through the
// EPP model label cap and folds to "other" once that cap is full.
func recordDecision(pluginName, model, outcome string) {
	model = boundModelName(model)
	if model == "" {
		model = "unknown"
	}
	screenDecisions.WithLabelValues(pluginName, model, outcome).Inc()
}
