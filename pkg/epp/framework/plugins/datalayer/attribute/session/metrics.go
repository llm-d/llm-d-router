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

package session

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// affinityStaleBindingTotal counts requests where a session was bound to an
// endpoint that is no longer in the candidate set, observed independently by
// the filter and the scorer. A non-trivial rate signals churn (pod restarts,
// scale-down) that is invalidating session pinning faster than sessions can
// rebind.
var affinityStaleBindingTotal = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "session_affinity_stale_binding_total",
		Help:      metricsutil.HelpMsgWithStability("Total session-affinity requests where the bound endpoint was not among the candidates", compbasemetrics.ALPHA),
	},
	[]string{"plugin_name", "plugin_type"},
)

// RegisterAffinityMetrics registers session-affinity metrics on the given
// registerer. Multiple plugin instances sharing one registerer call this
// idempotently: re-registration with the same collector is treated as a
// no-op so the second filter or scorer instantiation does not error.
func RegisterAffinityMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("session-affinity metrics registerer is required")
	}
	for _, collector := range []prometheus.Collector{affinityStaleBindingTotal} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register session-affinity metric: %w", err)
		}
	}
	return nil
}

// RecordStaleBinding increments the stale-binding counter for the given
// consumer. pluginType is the consumer's TypedName.Type (e.g.
// "session-affinity-filter", "session-affinity-scorer").
func RecordStaleBinding(pluginName, pluginType string) {
	affinityStaleBindingTotal.WithLabelValues(pluginName, pluginType).Inc()
}
