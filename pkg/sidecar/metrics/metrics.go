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
// served at /metrics on the sidecar's dedicated --metrics-port, kept separate
// from the data-plane proxy port so the model server's own /metrics stays
// reachable through the proxy. The sidecar handles encode, prefill, and decode
// disaggregation.
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

// Values of the mode label on nixlPushDispatchesTotal.
const (
	NIXLPushModeParallel = "parallel"
	NIXLPushModeSerial   = "serial"
)

// Values of the reason label on nixlPushDispatchesTotal. A cache hit is the
// reason of every parallel dispatch; the other reasons are those of serial
// dispatches.
const (
	NIXLPushReasonCacheHit     = "cache_hit"
	NIXLPushReasonCacheMiss    = "cache_miss"
	NIXLPushReasonSerialOnly   = "serial_only"
	NIXLPushReasonPrefillRetry = "prefill_retry"
)

var (
	requestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "requests_total",
			Help:      metricsutil.HelpMsgWithStability("Requests on the intercepted inference API paths, by API type.", compbasemetrics.ALPHA),
		},
		[]string{"api_type"},
	)

	disaggRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "disagg_requests_total",
			Help:      metricsutil.HelpMsgWithStability("Total requests routed through disaggregation, by disaggregation type.", compbasemetrics.ALPHA),
		},
		[]string{"disagg_type"},
	)

	encodeDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: subsystem,
			Name:      "encode_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Encode stage latency in seconds.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.GeneralLatencyBuckets,
		},
	)

	prefillDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: subsystem,
			Name:      "prefill_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Prefill stage latency in seconds.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.GeneralLatencyBuckets,
		},
	)

	decodeDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: subsystem,
			Name:      "decode_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Decode stage latency in seconds. Not sampled when prefill returns an error status, since decode is then not run or its output is discarded.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.GeneralLatencyBuckets,
		},
	)

	errorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "request_errors_total",
			Help:      metricsutil.HelpMsgWithStability("Total encode/prefill/decode stage errors, by stage.", compbasemetrics.ALPHA),
		},
		[]string{"stage"},
	)

	nixlPushDispatchesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "nixl_push_dispatches_total",
			Help:      metricsutil.HelpMsgWithStability("Total NIXL push dispatches, by mode (parallel or serial) and reason.", compbasemetrics.ALPHA),
		},
		[]string{"mode", "reason"},
	)

	nixlPushIdentityMismatchesTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "nixl_push_identity_mismatches_total",
			Help:      metricsutil.HelpMsgWithStability("Total parallel NIXL push dispatches whose prefill response did not carry the cached NIXL push identity, so the decode request was sent again.", compbasemetrics.ALPHA),
		},
	)

	nixlPushIdentityDropsTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "nixl_push_identity_drops_total",
			Help:      metricsutil.HelpMsgWithStability("Total cached NIXL push identities dropped after a parallel NIXL push dispatch failed.", compbasemetrics.ALPHA),
		},
	)

	nixlPushSerialOnlyMarksTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "nixl_push_serial_only_marks_total",
			Help:      metricsutil.HelpMsgWithStability("Total times a prefill endpoint was marked serial-only because its NIXL push identity kept changing.", compbasemetrics.ALPHA),
		},
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
			nixlPushDispatchesTotal,
			nixlPushIdentityMismatchesTotal,
			nixlPushIdentityDropsTotal,
			nixlPushSerialOnlyMarksTotal,
		)
	})
}

// RecordRequest counts a request on an intercepted inference API path for the
// given API type.
func RecordRequest(apiType string) {
	requestsTotal.WithLabelValues(apiType).Inc()
}

// RecordDisagg counts a request routed through disaggregation for the given
// disaggregation type (metricsutil.DisaggPathPrefillDecode,
// DisaggPathEncodePrefillDecode, or DisaggPathEncodeDecode).
func RecordDisagg(disaggType string) {
	disaggRequestsTotal.WithLabelValues(disaggType).Inc()
}

// RecordEncodeDuration records encode stage latency.
func RecordEncodeDuration(d time.Duration) {
	encodeDuration.Observe(d.Seconds())
}

// RecordPrefillDuration records prefill stage latency.
func RecordPrefillDuration(d time.Duration) {
	prefillDuration.Observe(d.Seconds())
}

// RecordDecodeDuration records decode stage latency.
func RecordDecodeDuration(d time.Duration) {
	decodeDuration.Observe(d.Seconds())
}

// RecordError counts a stage error (StageEncode, StagePrefill, or StageDecode).
func RecordError(stage string) {
	errorsTotal.WithLabelValues(stage).Inc()
}

// RecordNIXLPushDispatch counts a NIXL push dispatch for the given reason
// (NIXLPushReasonCacheHit, NIXLPushReasonCacheMiss, NIXLPushReasonSerialOnly
// or NIXLPushReasonPrefillRetry) under the mode that reason implies.
func RecordNIXLPushDispatch(reason string) {
	mode := NIXLPushModeSerial
	if reason == NIXLPushReasonCacheHit {
		mode = NIXLPushModeParallel
	}
	nixlPushDispatchesTotal.WithLabelValues(mode, reason).Inc()
}

// RecordNIXLPushIdentityMismatch counts a parallel NIXL push dispatch whose
// decode request was sent again because the prefill response did not carry the
// cached identity.
func RecordNIXLPushIdentityMismatch() {
	nixlPushIdentityMismatchesTotal.Inc()
}

// RecordNIXLPushIdentityDrop counts a cached NIXL push identity dropped after a
// failed parallel dispatch.
func RecordNIXLPushIdentityDrop() {
	nixlPushIdentityDropsTotal.Inc()
}

// RecordNIXLPushSerialOnlyMark counts a prefill endpoint marked serial-only.
func RecordNIXLPushSerialOnlyMark() {
	nixlPushSerialOnlyMarksTotal.Inc()
}
