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

package sessionmanager

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

type managerMetrics struct {
	identityOutcomes *prometheus.CounterVec
	requestOutcomes  *prometheus.CounterVec
	eventOutcomes    *prometheus.CounterVec
	activeBindings   prometheus.GaugeFunc
}

func newManagerMetrics(name string, bindings *bindingStore, registerer fwkplugin.MetricsRecorder) (*managerMetrics, error) {
	constLabels := prometheus.Labels{"plugin_name": name}
	metrics := &managerMetrics{
		identityOutcomes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:        "session_manager_identity_total",
			Help:        "Total session-manager identity attempts by bounded outcome.",
			ConstLabels: constLabels,
		}, []string{"outcome"}),
		requestOutcomes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:        "session_manager_cache_request_total",
			Help:        "Total session-manager cache-request attempts by bounded outcome.",
			ConstLabels: constLabels,
		}, []string{"outcome"}),
		eventOutcomes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:        "session_manager_event_total",
			Help:        "Total session-manager event-correlation attempts by bounded outcome.",
			ConstLabels: constLabels,
		}, []string{"outcome"}),
		activeBindings: prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:        "session_manager_active_bindings",
			Help:        "Current number of bounded request-stamp bindings.",
			ConstLabels: constLabels,
		}, func() float64 { return float64(bindings.len()) }),
	}
	if registerer == nil {
		return metrics, nil
	}
	for _, collector := range []prometheus.Collector{
		metrics.identityOutcomes,
		metrics.requestOutcomes,
		metrics.eventOutcomes,
		metrics.activeBindings,
	} {
		if err := registerer.Register(collector); err != nil {
			return nil, fmt.Errorf("register session-manager metrics: %w", err)
		}
	}
	return metrics, nil
}
