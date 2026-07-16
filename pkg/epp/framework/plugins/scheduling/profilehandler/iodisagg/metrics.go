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

package iodisagg

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// DecisionTypePreprocessOnly is for requests that are gone through preprocess disaggregation.
	DecisionTypePreprocessOnly = "preprocess-only"
	// DecisionTypePostprocessOnly is for requests that are gone through postprocess disaggregation.
	DecisionTypePostprocessOnly = "postprocess-only"

	// DecisionTypeNoDisaggregation is for requests that are not disaggregated.
	DecisionTypeNoDisaggregation = "no-disaggregation"
	// DecisionTypeFull is for requests that are gone through full IO disaggregation.
	DecisionTypeFull = "preprocess-inference-postprocess"
)

var (
	// LlmdDisaggDecisionCount records disaggregation routing decisions.
	LlmdIODisaggDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "io_disagg_decision_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of disaggregation routing decisions made", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "model_name", "decision_type"},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("disagg metrics registerer is required")
	}
	for _, collector := range []prometheus.Collector{
		LlmdIODisaggDecisionCount,
	} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register disagg metric: %w", err)
		}
	}
	return nil
}

// RecordIODisaggDecision increments the counter for a disaggregation routing decision.
// The decisionType must be one of the DecisionType* constants (DecisionTypePreprocessOnly,
// DecisionTypePostprocessOnly, DecisionTypeNoDisaggregation, DecisionTypeFull).
// The model parameter should be the target model name; if empty, "unknown" is used.
func RecordIODisaggDecision(pluginName, pluginType, modelName, decisionType string) {
	if modelName == "" {
		modelName = "unknown"
	}
	LlmdIODisaggDecisionCount.WithLabelValues(pluginName, pluginType, modelName, decisionType).Inc()
}

// IODisaggDecisionType returns the DecisionType* constant corresponding to which
// disaggregation stages were used for a request.
func IODisaggDecisionType(preprocessUsed, inferenceUsed, postprocessUsed bool) string {
	switch {
	case preprocessUsed && inferenceUsed && postprocessUsed:
		return DecisionTypeFull
	case preprocessUsed:
		return DecisionTypePreprocessOnly // Not supported yeet
	case postprocessUsed:
		return DecisionTypePreprocessOnly // Not supported yeet
	default:
		return DecisionTypeNoDisaggregation // Not supported yet
	}
}
