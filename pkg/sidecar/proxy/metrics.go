/*
Copyright 2025 The llm-d Authors.

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

package proxy

import (
	"net/http"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

const sidecarSubsystem = "llm_d_sidecar"

var (
	sidecarRegistry     = prometheus.NewRegistry()
	registerSidecarOnce sync.Once

	toolCallingPreservedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: sidecarSubsystem,
			Name:      "tool_calling_preserved_total",
			Help:      metricsutil.HelpMsgWithStability("Total requests by tool-calling preservation status across the sidecar boundary.", compbasemetrics.ALPHA),
		},
		[]string{"preserved"},
	)
)

func registerSidecarMetrics() {
	registerSidecarOnce.Do(func() {
		sidecarRegistry.MustRegister(toolCallingPreservedTotal)
	})
}

func metricsHandler() http.Handler {
	registerSidecarMetrics()
	return promhttp.HandlerFor(sidecarRegistry, promhttp.HandlerOpts{})
}

func recordToolCallingPreservation(preserved string) {
	toolCallingPreservedTotal.WithLabelValues(preserved).Inc()
}
