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

package qosadmission

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var (
	admissionDecisions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_admission_decisions_total",
			Help:      metricsutil.HelpMsgWithStability("QoS admission decisions per team, committed class and served class.", compbasemetrics.ALPHA),
		},
		[]string{"team", "class", "served_class", "decision", "mode"},
	)

	chargedTokens = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_charged_tokens_total",
			Help:      metricsutil.HelpMsgWithStability("Weighted tokens charged to a team's quota, after refunds, by served class.", compbasemetrics.ALPHA),
		},
		[]string{"team", "served_class"},
	)

	inflightRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_inflight_requests",
			Help:      metricsutil.HelpMsgWithStability("Requests per team counted against its concurrency cap, from admission to completion.", compbasemetrics.ALPHA),
		},
		[]string{"team"},
	)

	bucketTokens = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "qos_bucket_tokens",
			Help:      metricsutil.HelpMsgWithStability("Weighted tokens available in a team's token bucket at its last update.", compbasemetrics.ALPHA),
		},
		[]string{"team"},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{admissionDecisions, chargedTokens, inflightRequests, bucketTokens} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register qos admission metric: %w", err)
		}
	}
	return nil
}
