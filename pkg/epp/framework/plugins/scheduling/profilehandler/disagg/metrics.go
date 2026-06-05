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

package disagg

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// DecisionTypeDecodeOnly is for requests that are routed to decode instance only.
	DecisionTypeDecodeOnly = "decode-only"
	// DecisionTypePrefillDecode is for requests that are gone through P/D or EP/D.
	DecisionTypePrefillDecode = "prefill-decode"
	// DecisionTypeEncodeDecode is for requests that are gone through E/PD.
	DecisionTypeEncodeDecode = "encode-decode"
	// DecisionTypeEncodePrefillDecode is for requests that are gone through E/P/D.
	DecisionTypeEncodePrefillDecode = "encode-prefill-decode"
)

var (
	// SchedulerPDDecisionCount records request P/D decision.
	//
	// Deprecated: Use LlmdPDDecisionCount instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	SchedulerPDDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.SchedulerSubsystem,
			Name:      "pd_decision_total",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_pd_decision_total] Total number of P/D disaggregation decisions made", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "decision_type"}, // "decode-only" or "prefill-decode"
	)

	// LlmdPDDecisionCount records request P/D decision.
	LlmdPDDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "pd_decision_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of P/D disaggregation decisions made", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "model_name", "decision_type"},
	)

	// SchedulerDisaggDecisionCount records disaggregation routing decisions,
	// covering all stages: decode-only, prefill-decode, encode-decode, encode-prefill-decode.
	//
	// Deprecated: Use llm_d_router_epp_disagg_decision_total instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	SchedulerDisaggDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.SchedulerSubsystem,
			Name:      "disagg_decision_total",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_disagg_decision_total] Total number of disaggregation routing decisions made", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "decision_type"},
	)

	// LlmdDisaggDecisionCount records disaggregation routing decisions.
	LlmdDisaggDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "disagg_decision_total",
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
		SchedulerPDDecisionCount,
		LlmdPDDecisionCount,
		SchedulerDisaggDecisionCount,
		LlmdDisaggDecisionCount,
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

// RecordPDDecision increments the counter for a specific P/D routing decision.
//
// Deprecated: Use RecordDisaggDecision instead.
func RecordPDDecision(pluginName, pluginType, modelName, decisionType string) {
	if modelName == "" {
		modelName = "unknown"
	}
	SchedulerPDDecisionCount.WithLabelValues(modelName, decisionType).Inc()
	LlmdPDDecisionCount.WithLabelValues(pluginName, pluginType, modelName, decisionType).Inc()
}

// RecordDisaggDecision increments the counter for a disaggregation routing decision.
// The decisionType must be one of the DecisionType* constants (DecisionTypeDecodeOnly,
// DecisionTypePrefillDecode, DecisionTypeEncodeDecode, DecisionTypeEncodePrefillDecode).
// The model parameter should be the target model name; if empty, "unknown" is used.
func RecordDisaggDecision(pluginName, pluginType, modelName, decisionType string) {
	if modelName == "" {
		modelName = "unknown"
	}
	SchedulerDisaggDecisionCount.WithLabelValues(modelName, decisionType).Inc()
	LlmdDisaggDecisionCount.WithLabelValues(pluginName, pluginType, modelName, decisionType).Inc()
}

// DisaggDecisionType returns the DecisionType* constant corresponding to which
// disaggregation stages were used for a request.
func DisaggDecisionType(encodeUsed, prefillUsed bool) string {
	switch {
	case encodeUsed && prefillUsed:
		return DecisionTypeEncodePrefillDecode
	case encodeUsed:
		return DecisionTypeEncodeDecode
	case prefillUsed:
		return DecisionTypePrefillDecode
	default:
		return DecisionTypeDecodeOnly
	}
}
