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

package sessionidentity

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

type identityMetrics struct {
	outcomes *prometheus.CounterVec
}

func newIdentityMetrics(name string, registerer fwkplugin.MetricsRecorder) (*identityMetrics, error) {
	metrics := &identityMetrics{
		outcomes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:        "session_identity_total",
			Help:        "Total session-identity attempts by bounded outcome.",
			ConstLabels: prometheus.Labels{"plugin_name": name},
		}, []string{"outcome"}),
	}
	if registerer != nil {
		if err := registerer.Register(metrics.outcomes); err != nil {
			return nil, fmt.Errorf("register session-identity metrics: %w", err)
		}
	}
	return metrics, nil
}
