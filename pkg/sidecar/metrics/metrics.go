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

// Package metrics defines the Prometheus metrics exposed by the disaggregation
// sidecar proxy. Metrics are registered with controller-runtime's registry and
// served on the sidecar's /metrics route alongside Go runtime metrics, matching
// EPP. The sidecar handles encode, prefill, and decode disaggregation.
package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// subsystem is the Prometheus subsystem for the disaggregation sidecar metrics.
const subsystem = "llm_d_disagg_sidecar"

// Stage labels for encode/prefill/decode error attribution.
const (
	StageEncode  = "encode"
	StagePrefill = "prefill"
	StageDecode  = "decode"
)

// latencyBuckets covers encode, prefill, and decode wall-clock latency in
// seconds, from a few milliseconds to several minutes (decode can stream for a
// long time).
var latencyBuckets = []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300}

var (
	requestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "requests_total",
			Help:      metricsutil.HelpMsgWithStability("Total requests handled by the sidecar, by OpenAI API type.", compbasemetrics.ALPHA),
		},
		[]string{"api_type"},
	)

	disaggRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "disagg_requests_total",
			Help:      metricsutil.HelpMsgWithStability("Total requests routed through disaggregation, by connector.", compbasemetrics.ALPHA),
		},
		[]string{"connector"},
	)

	encodeDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: subsystem,
			Name:      "encode_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Encode stage latency in seconds, by connector.", compbasemetrics.ALPHA),
			Buckets:   latencyBuckets,
		},
		[]string{"connector"},
	)

	prefillDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: subsystem,
			Name:      "prefill_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Prefill stage latency in seconds, by connector.", compbasemetrics.ALPHA),
			Buckets:   latencyBuckets,
		},
		[]string{"connector"},
	)

	decodeDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: subsystem,
			Name:      "decode_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Decode stage latency in seconds, by connector.", compbasemetrics.ALPHA),
			Buckets:   latencyBuckets,
		},
		[]string{"connector"},
	)

	errorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "request_errors_total",
			Help:      metricsutil.HelpMsgWithStability("Total prefill/decode stage errors, by connector and stage.", compbasemetrics.ALPHA),
		},
		[]string{"connector", "stage"},
	)
)

var registerOnce sync.Once

// Register registers the sidecar metrics with controller-runtime's registry,
// which the /metrics endpoint serves. Safe to call more than once.
func Register() {
	registerOnce.Do(func() {
		ctrlmetrics.Registry.MustRegister(
			requestsTotal,
			disaggRequestsTotal,
			encodeDuration,
			prefillDuration,
			decodeDuration,
			errorsTotal,
		)
	})
}

// RecordRequest counts a request handled by the sidecar for the given API type.
func RecordRequest(apiType string) {
	requestsTotal.WithLabelValues(apiType).Inc()
}

// RecordDisagg counts a request routed through disaggregation for the given connector.
func RecordDisagg(connector string) {
	disaggRequestsTotal.WithLabelValues(connector).Inc()
}

// RecordEncodeDuration records encode stage latency for the given connector.
func RecordEncodeDuration(connector string, d time.Duration) {
	encodeDuration.WithLabelValues(connector).Observe(d.Seconds())
}

// RecordPrefillDuration records prefill stage latency for the given connector.
func RecordPrefillDuration(connector string, d time.Duration) {
	prefillDuration.WithLabelValues(connector).Observe(d.Seconds())
}

// RecordDecodeDuration records decode stage latency for the given connector.
func RecordDecodeDuration(connector string, d time.Duration) {
	decodeDuration.WithLabelValues(connector).Observe(d.Seconds())
}

// RecordError counts a stage error (StagePrefill or StageDecode) for the given connector.
func RecordError(connector, stage string) {
	errorsTotal.WithLabelValues(connector, stage).Inc()
}
