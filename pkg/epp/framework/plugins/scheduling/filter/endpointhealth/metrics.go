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

package endpointhealth

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Exclusion reason labels.
const (
	// reasonPoll: the endpoint's metric polls kept failing.
	reasonPoll = "poll"
	// reasonResponse: the endpoint returned consecutive failed responses.
	reasonResponse = "response"
)

var (
	excludedEndpoints = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "endpoint_health_excluded_endpoints",
			Help:      metricsutil.HelpMsgWithStability("Endpoints the endpoint health filter currently excludes from scheduling.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name"},
	)

	ejections = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "endpoint_health_ejections_total",
			Help:      metricsutil.HelpMsgWithStability("Endpoints marked unhealthy by the endpoint health filter, by reason.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "reason"},
	)

	readmissions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "endpoint_health_readmissions_total",
			Help:      metricsutil.HelpMsgWithStability("Unhealthy marks cleared by the endpoint health filter, by reason.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "reason"},
	)

	failOpens = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "endpoint_health_fail_open_total",
			Help:      metricsutil.HelpMsgWithStability("Filter calls in which every candidate was unhealthy and all candidates were kept.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name"},
	)
)

// registerMetrics registers the filter collectors on the given registerer. A nil
// registerer is a no-op, so the filter runs without metrics when the handle
// supplies no recorder. Repeated registration of the same collector is tolerated.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{excludedEndpoints, ejections, readmissions, failOpens} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register endpoint health filter metric: %w", err)
		}
	}
	return nil
}

// resetMetrics clears the collectors between tests.
func resetMetrics() {
	excludedEndpoints.Reset()
	ejections.Reset()
	readmissions.Reset()
	failOpens.Reset()
}
