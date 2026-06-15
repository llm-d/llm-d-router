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
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

const (
	// LLMDRouterEndpointPickerSubsystem is the subsystem for llm-d router endpoint picker metrics.
	LLMDRouterEndpointPickerSubsystem = "llm_d_router_epp"
)

var (
	// llmdEndpointLabels replaces the deprecated endpointLabels that used "pod_name".
	llmdEndpointLabels                       = []string{"endpoint_name", "namespace", "port"}
	modelLabelsWithFairnessPriority          = append(append([]string{}, modelLabels...), "fairness_id", "priority")
	modelLabelsWithFairnessPriorityStreaming = append(append([]string{}, modelLabelsWithFairnessPriority...), "streaming")
)

// --- llm-d Inference Objective Metrics ---
var (
	llmdRequestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of processed requests.", compbasemetrics.ALPHA),
		},
		modelLabelsWithFairnessPriority,
	)

	llmdRequestErrCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_error_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of request errors.", compbasemetrics.ALPHA),
		},
		append(modelLabelsWithFairnessPriority, "error_code"),
	)

	llmdRequestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("End-to-end request latency distribution in seconds.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		modelLabelsWithFairnessPriority,
	)

	llmdRequestSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_size_bytes",
			Help:      metricsutil.HelpMsgWithStability("Incoming request body size distribution in bytes.", compbasemetrics.ALPHA),
			Buckets: []float64{
				64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
				131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
				16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
			},
		},
		modelLabelsWithFairnessPriority,
	)

	llmdResponseSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "response_size_bytes",
			Help:      metricsutil.HelpMsgWithStability("Outgoing response body size distribution in bytes.", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536},
		},
		modelLabelsWithFairnessPriority,
	)

	llmdInputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_input_tokens",
			Help:      metricsutil.HelpMsgWithStability("Input token count distribution per request.", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		modelLabelsWithFairnessPriority,
	)

	llmdOutputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_output_tokens",
			Help:      metricsutil.HelpMsgWithStability("Output token count distribution per request.", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
		},
		modelLabelsWithFairnessPriority,
	)

	llmdPromptCachedTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_cached_tokens",
			Help:      metricsutil.HelpMsgWithStability("Distribution of prompt tokens read from cache per request, as reported by the model server in the response.", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		modelLabelsWithFairnessPriority,
	)

	llmdRunningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_running",
			Help:      metricsutil.HelpMsgWithStability("Current number of active running requests.", compbasemetrics.ALPHA),
		},
		modelLabelsWithFairnessPriority,
	)

	llmdNormalizedTimePerOutputToken = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_ntpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Normalized time per output token in seconds (end-to-end latency divided by output token count).", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
			},
		},
		modelLabelsWithFairnessPriority,
	)

	llmdRequestTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Time to first token in seconds, measured from request received to first response byte. For non-streaming requests, this equals total request duration.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6,
				8, 10, 15, 20, 30, 45, 60, 120,
			},
		},
		modelLabelsWithFairnessPriorityStreaming,
	)

	llmdRequestTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_streaming_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Average time per output token in seconds for streaming requests, computed as (e2e - TTFT) / (output_tokens - 1).", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0005, 0.00205, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2,
				0.3, 0.4, 0.5, 0.6, 0.8, 1, 2,
			},
		},
		modelLabelsWithFairnessPriority,
	)

	llmdInterTokenLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "request_streaming_itl_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inter-token latency in seconds for streaming requests, measured as the time between consecutive response body chunks.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 2,
			},
		},
		append(append([]string{}, modelLabels...), "fairness_id", "priority"),
	)
)

// --- llm-d Inference Pool Metrics ---
var (
	llmdInferencePoolAvgKVCache = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "average_kv_cache_utilization",
			Help:      metricsutil.HelpMsgWithStability("The average kv cache utilization for an inference server pool.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	llmdInferencePoolAvgQueueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "average_queue_size",
			Help:      metricsutil.HelpMsgWithStability("The average number of requests pending in the model server queue.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	llmdInferencePoolAvgRunningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "average_running_requests",
			Help:      metricsutil.HelpMsgWithStability("The average number of running requests across model servers in the pool.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	llmdInferencePoolReadyEndpoints = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "ready_endpoints",
			Help:      metricsutil.HelpMsgWithStability("The number of ready endpoints in the inference server pool.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)
)

// --- llm-d Scheduling Metrics ---
var (
	llmdSchedulerE2ELatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "scheduler_e2e_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("End-to-end scheduling latency distribution in seconds.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{},
	)

	llmdSchedulerAttemptsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "scheduler_attempts_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of scheduling attempts.", compbasemetrics.ALPHA),
		},
		append([]string{"status", "target_model_name"}, llmdEndpointLabels...),
	)

	llmdPluginProcessingLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "plugin_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Plugin processing latency distribution in seconds for each extension point, plugin type and plugin name.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{"extension_point", "plugin_type", "plugin_name"},
	)
)

// --- llm-d Info Metrics ---
var llmdInferenceExtensionInfo = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: LLMDRouterEndpointPickerSubsystem,
		Name:      "info",
		Help:      metricsutil.HelpMsgWithStability("General information of the current build of Inference Extension.", compbasemetrics.ALPHA),
	},
	[]string{"commit", "build_ref"},
)

// --- llm-d Flow Control Metrics ---
var (
	llmdFlowControlRequestQueueDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "flow_control_request_queue_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Distribution of total time requests spend in the Flow Control layer (from enqueue to final outcome).", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
			},
		},
		append([]string{"fairness_id", "priority", "outcome", "inference_pool"}, modelLabels...),
	)

	llmdFlowControlDispatchCycleDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "flow_control_dispatch_cycle_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Distribution of time taken for each internal dispatch cycle in the Flow Control layer.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{},
	)

	llmdFlowControlRequestEnqueueDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "flow_control_request_enqueue_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Distribution of time taken to enqueue requests into the Flow Control layer.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{"fairness_id", "priority", "outcome"},
	)

	llmdFlowControlQueueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "flow_control_queue_size",
			Help:      metricsutil.HelpMsgWithStability("Current number of requests actively held in the Flow Control queue.", compbasemetrics.ALPHA),
		},
		append([]string{"fairness_id", "priority", "inference_pool"}, modelLabels...),
	)

	llmdFlowControlQueueBytes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "flow_control_queue_bytes",
			Help:      metricsutil.HelpMsgWithStability("Current total size in bytes of requests actively held in the Flow Control queue.", compbasemetrics.ALPHA),
		},
		append([]string{"fairness_id", "priority", "inference_pool"}, modelLabels...),
	)

	llmdFlowControlPoolSaturation = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "flow_control_pool_saturation",
			Help:      metricsutil.HelpMsgWithStability("Current saturation level of the inference pool (0.0 = empty, 1.0 = fully saturated).", compbasemetrics.ALPHA),
		},
		[]string{"inference_pool"},
	)
)

// --- llm-d Inference Model Rewrite Metrics ---
var llmdInferenceModelRewriteDecisionsTotal = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: LLMDRouterEndpointPickerSubsystem,
		Name:      "model_rewrite_decisions_total",
		Help:      metricsutil.HelpMsgWithStability("Total number of inference model rewrite decisions.", compbasemetrics.ALPHA),
	},
	[]string{"model_rewrite_name", "model_name", "target_model"},
)

// --- llm-d Data-layer Metrics ---
var (
	// LlmdDataLayerPollErrorsTotal records data-source poll errors per source type.
	LlmdDataLayerPollErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "datalayer_poll_errors_total",
			Help:      metricsutil.HelpMsgWithStability("Data-source poll errors per source type.", compbasemetrics.ALPHA),
		},
		[]string{"source_type"},
	)

	// LlmdDataLayerExtractErrorsTotal records extract errors per source/extractor type.
	LlmdDataLayerExtractErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: LLMDRouterEndpointPickerSubsystem,
			Name:      "datalayer_extract_errors_total",
			Help:      metricsutil.HelpMsgWithStability("Extract errors per source/extractor type.", compbasemetrics.ALPHA),
		},
		[]string{"source_type", "extractor_type"},
	)
)

var (
	// DescInferencePoolPerEndpointQueueSize is the standardized exported prometheus descriptor.
	DescInferencePoolPerEndpointQueueSize = prometheus.NewDesc(
		"llm_d_router_epp_per_endpoint_queue_size",
		metricsutil.HelpMsgWithStability("The total number of requests pending in the model server queue for each underlying endpoint.", compbasemetrics.ALPHA),
		[]string{
			"name",
			"model_server_endpoint",
		}, nil,
	)
)
