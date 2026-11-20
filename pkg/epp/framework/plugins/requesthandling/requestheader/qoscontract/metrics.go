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

package qoscontract

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var (
	contractDecisions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_contract_decisions_total",
			Help:      metricsutil.HelpMsgWithStability("QoS contract decisions per team, committed class, served class and the limit that moved the request out of its class.", compbasemetrics.ALPHA),
		},
		[]string{"team", "class", "served_class", "decision", "reason", "mode"},
	)

	chargedTokens = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_charged_tokens_total",
			Help:      metricsutil.HelpMsgWithStability("Weighted tokens consumed by a team, after refunds, by served class.", compbasemetrics.ALPHA),
		},
		[]string{"team", "served_class"},
	)

	inflightRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_inflight_requests",
			Help:      metricsutil.HelpMsgWithStability("Requests per team counted against its concurrency commit, from the request header to completion.", compbasemetrics.ALPHA),
		},
		[]string{"team"},
	)

	sloPromiseStatus = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_slo_status",
			Help:      metricsutil.HelpMsgWithStability("Last validation of a team's latency promise, 1 for the current status: met, violated, insufficient_data or unknown.", compbasemetrics.ALPHA),
		},
		[]string{"team", "class", "slo", "status"},
	)

	sloObservedSeconds = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_slo_observed_seconds",
			Help:      metricsutil.HelpMsgWithStability("Latency at a team's promised percentile over the validation window, for its traffic within the contract (scope guaranteed) and all of it (scope all).", compbasemetrics.ALPHA),
		},
		[]string{"team", "class", "slo", "scope"},
	)

	sloWindowRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_slo_window_requests",
			Help:      metricsutil.HelpMsgWithStability("Requests within the contract a team's promise was validated on, over the validation window.", compbasemetrics.ALPHA),
		},
		[]string{"team", "class", "slo"},
	)

	sloValidationErrors = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_slo_validation_errors_total",
			Help:      metricsutil.HelpMsgWithStability("Validation cycles that failed to read the EPP latency metrics.", compbasemetrics.ALPHA),
		},
	)

	sloAdmissions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_slo_admissions_total",
			Help:      metricsutil.HelpMsgWithStability("Requests within a team's contract, by the status of its promise when they were admitted.", compbasemetrics.ALPHA),
		},
		[]string{"team", "class", "status"},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{contractDecisions, chargedTokens, inflightRequests,
		sloPromiseStatus, sloObservedSeconds, sloWindowRequests, sloValidationErrors, sloAdmissions} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register qos contract metric: %w", err)
		}
	}
	return nil
}
