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

package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

var (
	configReloadTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "llm_d",
		Subsystem: "epp",
		Name:      "config_reload_total",
		Help:      metricsutil.HelpMsgWithStability("Configuration reload activity by result.", compbasemetrics.ALPHA),
	}, []string{"result"})
	configActiveGeneration = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "llm_d",
		Subsystem: "epp",
		Name:      "config_active_generation",
		Help:      metricsutil.HelpMsgWithStability("Active configuration generation.", compbasemetrics.ALPHA),
	})
	configLastReloadSuccess = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "llm_d",
		Subsystem: "epp",
		Name:      "config_last_reload_success_timestamp_seconds",
		Help:      metricsutil.HelpMsgWithStability("Unix timestamp of the last successful configuration reload.", compbasemetrics.ALPHA),
	})
)

// RecordConfigReload records a completed reload attempt.
func RecordConfigReload(result string) {
	configReloadTotal.WithLabelValues(result).Inc()
}

// RecordConfigGeneration sets the active configuration generation.
func RecordConfigGeneration(generation uint64) {
	configActiveGeneration.Set(float64(generation))
}

// RecordConfigReloadSuccess records the generation and completion time of a successful reload.
func RecordConfigReloadSuccess(generation uint64) {
	RecordConfigGeneration(generation)
	configLastReloadSuccess.Set(float64(time.Now().Unix()))
}
