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
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// Request family. Recorded by the inbound request handler; observes every
// client request the coordinator accepts, including bodies that fail
// pre-parse validation (413, unreadable body, invalid JSON).
var (
	requestTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "request_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of inbound client requests, including malformed ones.", compbasemetrics.ALPHA),
		},
		modelLabel,
	)

	requestErrorTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "request_error_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of failed client requests.", compbasemetrics.ALPHA),
		},
		withLabel(modelLabel, "error_code"),
	)

	requestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "request_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("End-to-end request latency distribution in seconds.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.GeneralLatencyBuckets,
		},
		modelLabel,
	)

	requestSize = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "request_size_bytes",
			Help:      metricsutil.HelpMsgWithStability("Incoming request body size distribution in bytes.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.RequestSizeBuckets,
		},
		modelLabel,
	)

	requestInputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "request_input_tokens",
			Help:      metricsutil.HelpMsgWithStability("Prompt token count distribution per client request, measured after render.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.TokenCountBuckets,
		},
		modelLabel,
	)

	requestRunning = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "request_running",
			Help:      metricsutil.HelpMsgWithStability("Requests currently being processed by the coordinator.", compbasemetrics.ALPHA),
		},
		modelLabel,
	)
)

// Pipeline step family. Recorded by the pipeline executor around every
// step it runs; captures each stage's total wall time (local work plus all
// backend calls the stage makes) and the failures attributable to it.
var (
	stepDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "step_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Per-step wall-time latency distribution in seconds.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.GeneralLatencyBuckets,
		},
		stepLabel,
	)

	stepErrorTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "step_errors_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of pipeline-step failures.", compbasemetrics.ALPHA),
		},
		withLabel(stepLabel, "error_code"),
	)

	stepRunning = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "step_running",
			Help:      metricsutil.HelpMsgWithStability("Requests currently executing each pipeline step. For decode this stays high while the response streams.", compbasemetrics.ALPHA),
		},
		stepLabel,
	)
)

// Upstream call family. Recorded once per outbound HTTP call by the step
// that makes it: encode contributes one observation per multimodal entry
// and replace-media-urls one per URL, so the counter multiplies past step_total
// by the fan-out factor. Failures roll up into step_errors_total, so there
// is no upstream_request_error_total.
var (
	upstreamRequestTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "upstream_request_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of outbound calls to upstream services, one per call.", compbasemetrics.ALPHA),
		},
		upstreamLabel,
	)

	upstreamRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "upstream_request_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Latency distribution of a single outbound call to an upstream service.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.GeneralLatencyBuckets,
		},
		upstreamLabel,
	)
)

// Execution path and conditional-decode probe outcome. execution_path_total
// records which set of disaggregation phases actually ran for each client
// request; conditional_decode_probes_total records how the decode worker
// answered the conditional-decode probe (served the request inline vs
// deferred to the full pipeline). The coordinator sees only the probe
// status, not the reason behind it.
var (
	executionPathTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "execution_path_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of client requests by the set of disaggregation phases actually executed.", compbasemetrics.ALPHA),
		},
		withLabel(modelLabel, "path"),
	)

	conditionalDecodeProbesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterCoordinatorSubsystem,
			Name:      "conditional_decode_probes_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of conditional-decode probes by the worker's answer: served inline (2xx/3xx), deferred (HTTP 412) to the full pipeline, or error (any other 4xx/5xx).", compbasemetrics.ALPHA),
		},
		[]string{"result"},
	)
)
