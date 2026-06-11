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

package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// Opt-in ext_proc gRPC stream metrics (see --enable-grpc-stream-metrics).
var (
	extProcStreamsInflight = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "extproc_streams_inflight",
			Help:      metricsutil.HelpMsgWithStability("Number of ext_proc gRPC streams currently open.", compbasemetrics.ALPHA),
		},
	)

	extProcStreamDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "extproc_stream_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration an ext_proc gRPC stream stays open, in seconds.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600},
		},
	)

	extProcStreamsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "extproc_streams_total",
			Help:      metricsutil.HelpMsgWithStability("Total ext_proc gRPC streams completed, by gRPC status code.", compbasemetrics.ALPHA),
		},
		[]string{"code"},
	)
)

var registerGRPCStreamMetricsOnce sync.Once

// RegisterGRPCStreamMetrics registers the ext_proc stream metrics; called only when enabled.
func RegisterGRPCStreamMetrics() {
	registerGRPCStreamMetricsOnce.Do(func() {
		metrics.Registry.MustRegister(extProcStreamsInflight)
		metrics.Registry.MustRegister(extProcStreamDuration)
		metrics.Registry.MustRegister(extProcStreamsTotal)
	})
}

// ExtProcStreamStarted records an ext_proc stream open.
func ExtProcStreamStarted() {
	extProcStreamsInflight.Inc()
}

// ExtProcStreamFinished records an ext_proc stream close with its status and duration.
func ExtProcStreamFinished(code string, seconds float64) {
	extProcStreamsInflight.Dec()
	extProcStreamDuration.Observe(seconds)
	extProcStreamsTotal.WithLabelValues(code).Inc()
}
