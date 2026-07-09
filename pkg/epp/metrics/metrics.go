/*
Copyright 2025 The Kubernetes Authors.

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
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/toolcalling"
)

const (
	// --- Subsystems ---
	inferenceObjectiveComponent = "inference_objective"
	inferencePoolComponent      = "inference_pool"
	inferenceExtension          = "inference_extension"

	// InferenceObjectiveSubsystem is the legacy subsystem for inference objective metrics.
	InferenceObjectiveSubsystem = inferenceObjectiveComponent
	// InferenceExtensionSubsystem is the legacy subsystem for inference extension metrics.
	InferenceExtensionSubsystem = inferenceExtension

	// SchedulerSubsystem is the legacy metric prefix for scheduler.
	SchedulerSubsystem = "llm_d_inference_scheduler"
)

var (
	// --- Common Label Sets ---
	modelLabels             = []string{"model_name", "target_model_name"}
	modelWithPriorityLabels = []string{"model_name", "target_model_name", "priority"}
	poolLabels              = []string{"name"}
	endpointLabels          = []string{"pod_name", "namespace", "port"}

	// --- Common Buckets ---

	// generalLatencyBuckets for long running inference from 5ms to 1 hour
	generalLatencyBuckets = []float64{
		0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6,
		8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200,
		1800, 2700, 3600,
	}
)

// --- Inference Objective Metrics ---
var (
	// Deprecated: Use llm_d_epp_request_total instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	requestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "request_total",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_total] Counter of inference objective requests broken out for each model and target model.", compbasemetrics.ALPHA),
		},
		modelWithPriorityLabels,
	)

	// Deprecated: Use llm_d_epp_request_error_total instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	requestErrCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "request_error_total",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_error_total] Counter of inference objective requests errors broken out for each model and target model.", compbasemetrics.ALPHA),
		},
		append(modelLabels, "error_code"),
	)

	// Deprecated: Use llm_d_epp_request_duration_seconds instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	requestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "request_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_duration_seconds] Inference objective response latency distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		modelLabels,
	)

	// Deprecated: Use llm_d_epp_request_size_bytes instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	requestSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "request_sizes",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_size_bytes] Inference objective requests size distribution in bytes for each model and target model.", compbasemetrics.ALPHA),
			// Use buckets ranging from 1000 bytes (1KB) to 10^9 bytes (1GB).
			Buckets: []float64{
				64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, // More fine-grained up to 64KB
				131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, // Exponential up to 8MB
				16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, // Exponential up to 1GB
			},
		},
		modelLabels,
	)

	// Deprecated: Use llm_d_epp_response_size_bytes instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	responseSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "response_sizes",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_response_size_bytes] Inference objective responses size distribution in bytes for each model and target model.", compbasemetrics.ALPHA),
			// Most models have a response token < 8192 tokens. Each token, in average, has 4 characters.
			// 8192 * 4 = 32768.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536},
		},
		modelLabels,
	)

	// Deprecated: Use llm_d_epp_request_input_tokens instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	inputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "input_tokens",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_input_tokens] Inference objective input token count distribution for requests in each model.", compbasemetrics.ALPHA),
			// Most models have a input context window less than 1 million tokens.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		modelLabels,
	)

	// Deprecated: Use llm_d_epp_request_output_tokens instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	outputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "output_tokens",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_output_tokens] Inference objective output token count distribution for requests in each model.", compbasemetrics.ALPHA),
			// Most models generates output less than 8192 tokens.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
		},
		modelLabels,
	)

	// Deprecated: Use llm_d_epp_request_cached_tokens instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	promptCachedTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "prompt_cached_tokens",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_cached_tokens] Inference objective prompt cached token count distribution for requests in each model.", compbasemetrics.ALPHA),
			// Most models have a input context window less than 1 million tokens.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		modelLabels,
	)

	// Deprecated: Use llm_d_epp_request_running instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	runningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "running_requests",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_running] Inference objective number of running requests in each model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name"},
	)

	// Deprecated: Use llm_d_epp_request_ntpot_seconds instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	normalizedTimePerOutputToken = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceObjectiveComponent,
			Name:      "normalized_time_per_output_token_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_request_ntpot_seconds] Inference objective latency divided by number of output tokens in seconds for each model and target model.", compbasemetrics.ALPHA),
			// From few milliseconds per token to multiple seconds per token
			Buckets: []float64{
				0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
			},
		},
		modelLabels,
	)
)

// --- Inference Pool Metrics ---
var (
	// Deprecated: Use llm_d_epp_average_kv_cache_utilization instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	inferencePoolAvgKVCache = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: inferencePoolComponent,
			Name:      "average_kv_cache_utilization",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_average_kv_cache_utilization] The average kv cache utilization for an inference server pool.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	// Deprecated: Use llm_d_epp_average_queue_size instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	inferencePoolAvgQueueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: inferencePoolComponent,
			Name:      "average_queue_size",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_average_queue_size] The average number of requests pending in the model server queue.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	// Deprecated: Use llm_d_epp_average_running_requests instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	inferencePoolAvgRunningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: inferencePoolComponent,
			Name:      "average_running_requests",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_average_running_requests] The average number of running requests across model servers in the pool.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	// Deprecated: Use llm_d_epp_ready_endpoints instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	inferencePoolReadyPods = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: inferencePoolComponent,
			Name:      "ready_pods",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_ready_endpoints] The number of ready pods in the inference server pool.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)
)

// --- Scheduling Metrics ---
var (
	// Deprecated: Use llm_d_epp_scheduler_e2e_duration_seconds instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	schedulerE2ELatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceExtension,
			Name:      "scheduler_e2e_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_scheduler_e2e_duration_seconds] End-to-end scheduling latency distribution in seconds.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{},
	)

	// Deprecated: Use llm_d_epp_scheduler_attempts_total instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	schedulerAttemptsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: inferenceExtension,
			Name:      "scheduler_attempts_total",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_scheduler_attempts_total] Total number of scheduling attempts.", compbasemetrics.ALPHA),
		},
		append([]string{"status", "target_model_name"}, endpointLabels...),
	)

	// Deprecated: Use llm_d_epp_plugin_duration_seconds instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	pluginProcessingLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceExtension,
			Name:      "plugin_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_plugin_duration_seconds] Plugin processing latency distribution in seconds for each extension point, plugin type and plugin name.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{"extension_point", "plugin_type", "plugin_name"},
	)
)

// --- Info Metrics ---
var inferenceExtensionInfo = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: inferenceExtension,
		Name:      "info",
		Help:      metricsutil.HelpMsgWithStability("General information of the current build of Inference Extension.", compbasemetrics.ALPHA),
	},
	[]string{"commit", "build_ref"},
)

// --- Flow Control Metrics ---
var (
	// Deprecated: Use llm_d_epp_flow_control_request_queue_duration_seconds instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	flowControlRequestQueueDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceExtension,
			Name:      "flow_control_request_queue_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_flow_control_request_queue_duration_seconds] Distribution of total time requests spend in the Flow Control layer (from enqueue to final outcome).", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
			},
		},
		append([]string{"fairness_id", "priority", "outcome", "inference_pool"}, modelLabels...),
	)

	// Deprecated: Use llm_d_epp_flow_control_dispatch_cycle_duration_seconds instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	flowControlDispatchCycleDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceExtension,
			Name:      "flow_control_dispatch_cycle_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_flow_control_dispatch_cycle_duration_seconds] Distribution of time taken for each internal dispatch cycle in the Flow Control layer.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{},
	)

	// Deprecated: Use llm_d_epp_flow_control_request_enqueue_duration_seconds instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	flowControlRequestEnqueueDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: inferenceExtension,
			Name:      "flow_control_request_enqueue_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_flow_control_request_enqueue_duration_seconds] Distribution of time taken to enqueue requests into the Flow Control layer.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{"fairness_id", "priority", "outcome"},
	)

	// Deprecated: Use llm_d_epp_flow_control_queue_size instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	flowControlQueueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: inferenceExtension,
			Name:      "flow_control_queue_size",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_flow_control_queue_size] Current number of requests actively held in the Flow Control queue.", compbasemetrics.ALPHA),
		},
		append([]string{"fairness_id", "priority", "inference_pool"}, modelLabels...),
	)

	// Deprecated: Use llm_d_epp_flow_control_queue_bytes instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	flowControlQueueBytes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: inferenceExtension,
			Name:      "flow_control_queue_bytes",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_flow_control_queue_bytes] Current total size in bytes of requests actively held in the Flow Control queue.", compbasemetrics.ALPHA),
		},
		append([]string{"fairness_id", "priority", "inference_pool"}, modelLabels...),
	)

	// Deprecated: Use llm_d_epp_flow_control_pool_saturation instead.
	// Tracked in: https://github.com/llm-d/llm-d-inference-scheduler/issues/1070
	flowControlPoolSaturation = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: inferenceExtension,
			Name:      "flow_control_pool_saturation",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_flow_control_pool_saturation] Current saturation level of the inference pool (0.0 = empty, 1.0 = fully saturated).", compbasemetrics.ALPHA),
		},
		[]string{"inference_pool"},
	)
)

// --- Inference Model Rewrite Metrics ---
var inferenceModelRewriteDecisionsTotal = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: inferenceExtension,
		Name:      "model_rewrite_decisions_total",
		Help:      metricsutil.HelpMsgWithStability("Total number of inference model rewrite decisions.", compbasemetrics.ALPHA),
	},
	[]string{"model_rewrite_name", "model_name", "target_model"},
)

// --- Data-layer Metrics ---

var (
	// DataLayerPollErrorsTotal records data-source poll errors per source type.
	//
	// Deprecated: Use LlmdDataLayerPollErrorsTotal instead.
	DataLayerPollErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "datalayer_poll_errors_total",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_datalayer_poll_errors_total] Data-source poll errors per source type.", compbasemetrics.ALPHA),
		},
		[]string{"source_type"},
	)

	// DataLayerExtractErrorsTotal records extract errors per source/extractor type.
	//
	// Deprecated: Use LlmdDataLayerExtractErrorsTotal instead.
	DataLayerExtractErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "datalayer_extract_errors_total",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_datalayer_extract_errors_total] Extract errors per source/extractor type.", compbasemetrics.ALPHA),
		},
		[]string{"source_type", "extractor_type"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register(customCollectors ...prometheus.Collector) {
	registerMetrics.Do(func() {
		// Register other metrics
		metrics.Registry.MustRegister(requestCounter)
		metrics.Registry.MustRegister(llmdRequestCounter)
		metrics.Registry.MustRegister(requestErrCounter)
		metrics.Registry.MustRegister(llmdRequestErrCounter)
		metrics.Registry.MustRegister(requestLatencies)
		metrics.Registry.MustRegister(llmdRequestLatencies)
		metrics.Registry.MustRegister(requestSizes)
		metrics.Registry.MustRegister(llmdRequestSizes)
		metrics.Registry.MustRegister(responseSizes)
		metrics.Registry.MustRegister(llmdResponseSizes)
		metrics.Registry.MustRegister(inputTokens)
		metrics.Registry.MustRegister(llmdInputTokens)
		metrics.Registry.MustRegister(outputTokens)
		metrics.Registry.MustRegister(llmdOutputTokens)
		metrics.Registry.MustRegister(promptCachedTokens)
		metrics.Registry.MustRegister(llmdPromptCachedTokens)
		metrics.Registry.MustRegister(runningRequests)
		metrics.Registry.MustRegister(llmdRunningRequests)
		metrics.Registry.MustRegister(normalizedTimePerOutputToken)
		metrics.Registry.MustRegister(llmdNormalizedTimePerOutputToken)
		metrics.Registry.MustRegister(llmdRequestTTFT)
		metrics.Registry.MustRegister(llmdRequestTPOT)
		metrics.Registry.MustRegister(llmdInterTokenLatency)
		metrics.Registry.MustRegister(inferencePoolAvgKVCache)
		metrics.Registry.MustRegister(llmdInferencePoolAvgKVCache)
		metrics.Registry.MustRegister(inferencePoolAvgQueueSize)
		metrics.Registry.MustRegister(llmdInferencePoolAvgQueueSize)
		metrics.Registry.MustRegister(inferencePoolAvgRunningRequests)
		metrics.Registry.MustRegister(llmdInferencePoolAvgRunningRequests)
		metrics.Registry.MustRegister(inferencePoolReadyPods)
		metrics.Registry.MustRegister(llmdInferencePoolReadyEndpoints)
		metrics.Registry.MustRegister(schedulerE2ELatency)
		metrics.Registry.MustRegister(llmdSchedulerE2ELatency)
		metrics.Registry.MustRegister(schedulerAttemptsTotal)
		metrics.Registry.MustRegister(llmdSchedulerAttemptsTotal)
		metrics.Registry.MustRegister(pluginProcessingLatencies)
		metrics.Registry.MustRegister(llmdPluginProcessingLatencies)
		metrics.Registry.MustRegister(inferenceExtensionInfo)
		metrics.Registry.MustRegister(llmdInferenceExtensionInfo)
		metrics.Registry.MustRegister(flowControlRequestQueueDuration)
		metrics.Registry.MustRegister(llmdFlowControlRequestQueueDuration)
		metrics.Registry.MustRegister(flowControlDispatchCycleDuration)
		metrics.Registry.MustRegister(llmdFlowControlDispatchCycleDuration)
		metrics.Registry.MustRegister(flowControlQueueSize)
		metrics.Registry.MustRegister(llmdFlowControlQueueSize)
		metrics.Registry.MustRegister(flowControlQueueBytes)
		metrics.Registry.MustRegister(llmdFlowControlQueueBytes)
		metrics.Registry.MustRegister(flowControlPoolSaturation)
		metrics.Registry.MustRegister(llmdFlowControlPoolSaturation)
		metrics.Registry.MustRegister(flowControlRequestEnqueueDuration)
		metrics.Registry.MustRegister(llmdFlowControlRequestEnqueueDuration)
		metrics.Registry.MustRegister(inferenceModelRewriteDecisionsTotal)
		metrics.Registry.MustRegister(llmdInferenceModelRewriteDecisionsTotal)
		metrics.Registry.MustRegister(DataLayerPollErrorsTotal)
		metrics.Registry.MustRegister(LlmdDataLayerPollErrorsTotal)
		metrics.Registry.MustRegister(DataLayerExtractErrorsTotal)
		metrics.Registry.MustRegister(LlmdDataLayerExtractErrorsTotal)
		metrics.Registry.MustRegister(llmdToolCallingRequestsTotal)
		metrics.Registry.MustRegister(llmdToolChoiceTotal)
		metrics.Registry.MustRegister(llmdToolDefinitionsCount)
		metrics.Registry.MustRegister(llmdParallelToolCallsTotal)
		metrics.Registry.MustRegister(llmdToolCallingPreservedTotal)
		for _, collector := range customCollectors {
			metrics.Registry.MustRegister(collector)
		}
	})
}

// Just for integration test
func Reset() {
	// Reset other metrics
	requestCounter.Reset()
	llmdRequestCounter.Reset()
	requestErrCounter.Reset()
	llmdRequestErrCounter.Reset()
	requestLatencies.Reset()
	llmdRequestLatencies.Reset()
	requestSizes.Reset()
	llmdRequestSizes.Reset()
	responseSizes.Reset()
	llmdResponseSizes.Reset()
	inputTokens.Reset()
	llmdInputTokens.Reset()
	outputTokens.Reset()
	llmdOutputTokens.Reset()
	promptCachedTokens.Reset()
	llmdPromptCachedTokens.Reset()
	runningRequests.Reset()
	llmdRunningRequests.Reset()
	normalizedTimePerOutputToken.Reset()
	llmdNormalizedTimePerOutputToken.Reset()
	llmdRequestTTFT.Reset()
	llmdRequestTPOT.Reset()
	llmdInterTokenLatency.Reset()
	inferencePoolAvgKVCache.Reset()
	llmdInferencePoolAvgKVCache.Reset()
	inferencePoolAvgQueueSize.Reset()
	llmdInferencePoolAvgQueueSize.Reset()
	inferencePoolAvgRunningRequests.Reset()
	llmdInferencePoolAvgRunningRequests.Reset()
	inferencePoolReadyPods.Reset()
	llmdInferencePoolReadyEndpoints.Reset()
	schedulerE2ELatency.Reset()
	llmdSchedulerE2ELatency.Reset()
	schedulerAttemptsTotal.Reset()
	llmdSchedulerAttemptsTotal.Reset()
	pluginProcessingLatencies.Reset()
	llmdPluginProcessingLatencies.Reset()
	inferenceExtensionInfo.Reset()
	llmdInferenceExtensionInfo.Reset()
	flowControlRequestQueueDuration.Reset()
	llmdFlowControlRequestQueueDuration.Reset()
	flowControlQueueSize.Reset()
	llmdFlowControlQueueSize.Reset()
	flowControlQueueBytes.Reset()
	llmdFlowControlQueueBytes.Reset()
	flowControlPoolSaturation.Reset()
	llmdFlowControlPoolSaturation.Reset()
	flowControlRequestEnqueueDuration.Reset()
	llmdFlowControlRequestEnqueueDuration.Reset()
	flowControlDispatchCycleDuration.Reset()
	llmdFlowControlDispatchCycleDuration.Reset()
	inferenceModelRewriteDecisionsTotal.Reset()
	llmdInferenceModelRewriteDecisionsTotal.Reset()
	DataLayerPollErrorsTotal.Reset()
	LlmdDataLayerPollErrorsTotal.Reset()
	DataLayerExtractErrorsTotal.Reset()
	LlmdDataLayerExtractErrorsTotal.Reset()
	llmdToolCallingRequestsTotal.Reset()
	llmdToolChoiceTotal.Reset()
	llmdToolDefinitionsCount.Reset()
	llmdParallelToolCallsTotal.Reset()
	llmdToolCallingPreservedTotal.Reset()
}

// RecordRequestCounter records the number of requests.
func RecordRequestCounter(modelName, targetModelName, fairnessID string, priority int) {
	prioStr := strconv.Itoa(priority)
	requestCounter.WithLabelValues(modelName, targetModelName, prioStr).Inc()
	llmdRequestCounter.WithLabelValues(modelName, targetModelName, fairnessID, prioStr).Inc()
}

// RecordRequestErrCounter records the number of error requests.
func RecordRequestErrCounter(modelName, targetModelName, fairnessID, priority string, code string) {
	if code != "" {
		requestErrCounter.WithLabelValues(modelName, targetModelName, code).Inc()
		llmdRequestErrCounter.WithLabelValues(modelName, targetModelName, fairnessID, priority, code).Inc()
	}
}

// RecordRequestSizes records the request sizes.
func RecordRequestSizes(modelName, targetModelName, fairnessID, priority string, reqSize int) {
	requestSizes.WithLabelValues(modelName, targetModelName).Observe(float64(reqSize))
	llmdRequestSizes.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(reqSize))
}

// RecordRequestLatencies records duration of request.
func RecordRequestLatencies(ctx context.Context, modelName, targetModelName, fairnessID, priority string, received time.Time, complete time.Time) bool {
	if !complete.After(received) {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "Request latency values are invalid",
			"modelName", modelName, "targetModelName", targetModelName, "completeTime", complete, "receivedTime", received)
		return false
	}
	elapsedSeconds := complete.Sub(received).Seconds()
	requestLatencies.WithLabelValues(modelName, targetModelName).Observe(elapsedSeconds)
	llmdRequestLatencies.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(elapsedSeconds)
	return true
}

// RecordResponseSizes records the response sizes.
func RecordResponseSizes(modelName, targetModelName, fairnessID, priority string, size int) {
	responseSizes.WithLabelValues(modelName, targetModelName).Observe(float64(size))
	llmdResponseSizes.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(size))
}

// RecordInputTokens records input tokens count.
func RecordInputTokens(modelName, targetModelName, fairnessID, priority string, size int) {
	if size > 0 {
		inputTokens.WithLabelValues(modelName, targetModelName).Observe(float64(size))
		llmdInputTokens.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(size))
	}
}

// RecordOutputTokens records output tokens count.
func RecordOutputTokens(modelName, targetModelName, fairnessID, priority string, size int) {
	if size > 0 {
		outputTokens.WithLabelValues(modelName, targetModelName).Observe(float64(size))
		llmdOutputTokens.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(size))
	}
}

// RecordPromptCachedTokens records prompt cached tokens count.
func RecordPromptCachedTokens(modelName, targetModelName, fairnessID, priority string, size int) {
	promptCachedTokens.WithLabelValues(modelName, targetModelName).Observe(float64(size))
	llmdPromptCachedTokens.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(size))
}

// RecordNormalizedTimePerOutputToken (NTPOT) records the normalized time per output token.
func RecordNormalizedTimePerOutputToken(ctx context.Context, modelName, targetModelName, fairnessID, priority string, received time.Time, complete time.Time, outputTokenCount int) bool {
	if outputTokenCount <= 0 {
		return false
	}

	if !complete.After(received) {
		log.FromContext(ctx).Error(nil, "Request latency values are invalid for NTPOT calculation",
			"modelName", modelName, "targetModelName", targetModelName, "completeTime", complete, "receivedTime", received)
		return false
	}

	elapsedSeconds := complete.Sub(received).Seconds()
	secondsPerToken := elapsedSeconds / float64(outputTokenCount)

	normalizedTimePerOutputToken.WithLabelValues(modelName, targetModelName).Observe(secondsPerToken)
	llmdNormalizedTimePerOutputToken.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(secondsPerToken)
	return true
}

// RecordRequestTTFT records the time to first token.
func RecordRequestTTFT(ctx context.Context, modelName, targetModelName, fairnessID, priority string, streaming bool, received time.Time, firstToken time.Time) bool {
	if firstToken.IsZero() {
		return false
	}
	if !firstToken.After(received) {
		log.FromContext(ctx).Error(nil, "Request latency values are invalid for TTFT calculation",
			"modelName", modelName, "targetModelName", targetModelName, "firstTokenTime", firstToken, "receivedTime", received)
		return false
	}

	streamingLabel := "false"
	if streaming {
		streamingLabel = "true"
	}
	ttftSeconds := firstToken.Sub(received).Seconds()
	llmdRequestTTFT.WithLabelValues(modelName, targetModelName, fairnessID, priority, streamingLabel).Observe(ttftSeconds)
	return true
}

// RecordRequestTPOT records the average time per output token.
func RecordRequestTPOT(ctx context.Context, modelName, targetModelName, fairnessID, priority string, received time.Time, firstToken time.Time, complete time.Time, outputTokenCount int) bool {
	if firstToken.IsZero() || outputTokenCount <= 1 {
		return false
	}
	if !firstToken.After(received) || !complete.After(firstToken) {
		log.FromContext(ctx).Error(nil, "Request latency values are invalid for TPOT calculation",
			"modelName", modelName, "targetModelName", targetModelName,
			"receivedTime", received, "firstTokenTime", firstToken, "completeTime", complete)
		return false
	}

	e2eSeconds := complete.Sub(received).Seconds()
	ttftSeconds := firstToken.Sub(received).Seconds()
	tpotSeconds := (e2eSeconds - ttftSeconds) / float64(outputTokenCount-1)
	llmdRequestTPOT.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(tpotSeconds)
	return true
}

// RecordInterTokenLatency records the time between consecutive response body chunks for streaming requests.
func RecordInterTokenLatency(ctx context.Context, modelName, targetModelName, fairnessID, priority string, itlSeconds float64) bool {
	if itlSeconds < 0 {
		log.FromContext(ctx).Error(nil, "Inter-token latency value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "itlSeconds", itlSeconds)
		return false
	}
	llmdInterTokenLatency.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(itlSeconds)
	return true
}

// IncRunningRequests increases the current running requests.
func IncRunningRequests(modelName, targetModelName, fairnessID, priority string) {
	if modelName != "" {
		runningRequests.WithLabelValues(modelName).Inc()
		llmdRunningRequests.WithLabelValues(modelName, targetModelName, fairnessID, priority).Inc()
	}
}

// DecRunningRequests decreases the current running requests.
func DecRunningRequests(modelName, targetModelName, fairnessID, priority string) {
	if modelName != "" {
		runningRequests.WithLabelValues(modelName).Dec()
		llmdRunningRequests.WithLabelValues(modelName, targetModelName, fairnessID, priority).Dec()
	}
}

func RecordInferencePoolAvgKVCache(name string, utilization float64) {
	inferencePoolAvgKVCache.WithLabelValues(name).Set(utilization)
	llmdInferencePoolAvgKVCache.WithLabelValues(name).Set(utilization)
}

func RecordInferencePoolAvgQueueSize(name string, queueSize float64) {
	inferencePoolAvgQueueSize.WithLabelValues(name).Set(queueSize)
	llmdInferencePoolAvgQueueSize.WithLabelValues(name).Set(queueSize)
}

func RecordInferencePoolAvgRunningRequests(name string, runningRequests float64) {
	inferencePoolAvgRunningRequests.WithLabelValues(name).Set(runningRequests)
	llmdInferencePoolAvgRunningRequests.WithLabelValues(name).Set(runningRequests)
}

func RecordInferencePoolReadyPods(name string, runningPods float64) {
	inferencePoolReadyPods.WithLabelValues(name).Set(runningPods)
	llmdInferencePoolReadyEndpoints.WithLabelValues(name).Set(runningPods)
}

// RecordSchedulerE2ELatency records the end-to-end scheduling latency.
func RecordSchedulerE2ELatency(duration time.Duration) {
	schedulerE2ELatency.WithLabelValues().Observe(duration.Seconds())
	llmdSchedulerE2ELatency.WithLabelValues().Observe(duration.Seconds())
}

// RecordSchedulerAttempt records a scheduling attempt with status and endpoint information.
func RecordSchedulerAttempt(err error, targetModelName string, result *fwksched.SchedulingResult) {
	if err != nil {
		schedulerAttemptsTotal.WithLabelValues(SchedulerStatusFailure, targetModelName, "", "", "").Inc()
		llmdSchedulerAttemptsTotal.WithLabelValues(SchedulerStatusFailure, targetModelName, "", "", "").Inc()
		return
	}

	if result != nil {
		// Collect endpoint information for successful scheduling attempts
		primaryResults := result.ProfileResults[result.PrimaryProfileName]
		if primaryResults != nil {
			// prepareRequest (in director.go) selects the first endpoint. Do the same here.
			if len(primaryResults.TargetEndpoints) > 0 {
				metadata := primaryResults.TargetEndpoints[0].GetMetadata()
				if metadata != nil {
					schedulerAttemptsTotal.WithLabelValues(SchedulerStatusSuccess, targetModelName, metadata.PodName, metadata.NamespacedName.Namespace, metadata.Port).Inc()
					llmdSchedulerAttemptsTotal.WithLabelValues(SchedulerStatusSuccess, targetModelName, metadata.PodName, metadata.NamespacedName.Namespace, metadata.Port).Inc()
					return
				}
			}
		}
	}

	schedulerAttemptsTotal.WithLabelValues(SchedulerStatusSuccess, targetModelName, "", "", "").Inc()
	llmdSchedulerAttemptsTotal.WithLabelValues(SchedulerStatusSuccess, targetModelName, "", "", "").Inc()
}

const (
	SchedulerStatusSuccess = "success"
	SchedulerStatusFailure = "failure"
)

// RecordPluginProcessingLatency records the processing latency for a plugin.
func RecordPluginProcessingLatency(extensionPoint, pluginType, pluginName string, duration time.Duration) {
	pluginProcessingLatencies.WithLabelValues(extensionPoint, pluginType, pluginName).Observe(duration.Seconds())
	llmdPluginProcessingLatencies.WithLabelValues(extensionPoint, pluginType, pluginName).Observe(duration.Seconds())
}

func RecordInferenceExtensionInfo(commitSha, buildRef string) {
	inferenceExtensionInfo.WithLabelValues(commitSha, buildRef).Set(1)
	llmdInferenceExtensionInfo.WithLabelValues(commitSha, buildRef).Set(1)
}

// RecordFlowControlRequestQueueDuration records the duration a request spent in the Flow Control layer.
func RecordFlowControlRequestQueueDuration(
	fairnessID, priority, outcome,
	inferencePool,
	modelName, targetModelName string,
	duration time.Duration,
) {
	flowControlRequestQueueDuration.WithLabelValues(
		fairnessID, priority, outcome,
		inferencePool,
		modelName, targetModelName,
	).Observe(duration.Seconds())

	llmdFlowControlRequestQueueDuration.WithLabelValues(
		fairnessID, priority, outcome,
		inferencePool,
		modelName, targetModelName,
	).Observe(duration.Seconds())
}

// RecordFlowControlDispatchCycleDuration records the duration of a dispatch cycle in the Flow Control layer.
func RecordFlowControlDispatchCycleDuration(duration time.Duration) {
	flowControlDispatchCycleDuration.WithLabelValues().Observe(duration.Seconds())
	llmdFlowControlDispatchCycleDuration.WithLabelValues().Observe(duration.Seconds())
}

// RecordFlowControlRequestEnqueueDuration records the duration a request was in the enqueuing process in the Flow Control layer.
func RecordFlowControlRequestEnqueueDuration(
	fairnessID string, priority string, outcome string,
	duration time.Duration,
) {
	flowControlRequestEnqueueDuration.WithLabelValues(
		fairnessID, priority, outcome,
	).Observe(duration.Seconds())

	llmdFlowControlRequestEnqueueDuration.WithLabelValues(
		fairnessID, priority, outcome,
	).Observe(duration.Seconds())
}

// IncFlowControlQueueSize increments the Flow Control queue size gauge.
func IncFlowControlQueueSize(fairnessID, priority, inferencePool, modelName, targetModelName string) {
	flowControlQueueSize.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Inc()
	llmdFlowControlQueueSize.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Inc()
}

// DecFlowControlQueueSize decrements the Flow Control queue size gauge.
func DecFlowControlQueueSize(fairnessID, priority, inferencePool, modelName, targetModelName string) {
	flowControlQueueSize.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Dec()
	llmdFlowControlQueueSize.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Dec()
}

// AddFlowControlQueueBytes increments the Flow Control queue bytes gauge.
func AddFlowControlQueueBytes(fairnessID, priority, inferencePool, modelName, targetModelName string, bytes uint64) {
	flowControlQueueBytes.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Add(float64(bytes))
	llmdFlowControlQueueBytes.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Add(float64(bytes))
}

// SubFlowControlQueueBytes decrements the Flow Control queue bytes gauge.
func SubFlowControlQueueBytes(fairnessID, priority, inferencePool, modelName, targetModelName string, bytes uint64) {
	flowControlQueueBytes.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Sub(float64(bytes))
	llmdFlowControlQueueBytes.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Sub(float64(bytes))
}

// RecordFlowControlPoolSaturation records the current saturation level for an inference pool.
func RecordFlowControlPoolSaturation(inferencePool string, saturation float64) {
	flowControlPoolSaturation.WithLabelValues(inferencePool).Set(saturation)
	llmdFlowControlPoolSaturation.WithLabelValues(inferencePool).Set(saturation)
}

// RecordInferenceModelRewriteDecision records the routing decision for InferenceModelRewrite.
func RecordInferenceModelRewriteDecision(modelRewriteName, modelName, targetModel string) {
	inferenceModelRewriteDecisionsTotal.WithLabelValues(modelRewriteName, modelName, targetModel).Inc()
	llmdInferenceModelRewriteDecisionsTotal.WithLabelValues(modelRewriteName, modelName, targetModel).Inc()
}

// RecordDataLayerPollError increments the poll error counter for a source type.
func RecordDataLayerPollError(sourceType string) {
	DataLayerPollErrorsTotal.WithLabelValues(sourceType).Inc()
	LlmdDataLayerPollErrorsTotal.WithLabelValues(sourceType).Inc()
}

// RecordDataLayerExtractError increments the extract error counter for a source/extractor type.
func RecordDataLayerExtractError(sourceType, extractorType string) {
	DataLayerExtractErrorsTotal.WithLabelValues(sourceType, extractorType).Inc()
	LlmdDataLayerExtractErrorsTotal.WithLabelValues(sourceType, extractorType).Inc()
}

// RecordToolCallingRequest records tool-calling usage metrics from a snapshot.
func RecordToolCallingRequest(parser string, snapshot *toolcalling.ToolCallingSnapshot) {
	if snapshot == nil {
		llmdToolCallingRequestsTotal.WithLabelValues(parser, "false").Inc()
		return
	}
	llmdToolCallingRequestsTotal.WithLabelValues(parser, strconv.FormatBool(snapshot.Present)).Inc()
	llmdToolChoiceTotal.WithLabelValues(parser, snapshot.ToolChoiceKind).Inc()
	llmdToolDefinitionsCount.WithLabelValues(parser).Observe(float64(snapshot.ToolDefinitionsCount))
	llmdParallelToolCallsTotal.WithLabelValues(parser, snapshot.ParallelToolCalls).Inc()
}

// RecordToolCallingPreservation records whether tool-calling parameters were preserved across the EPP boundary.
func RecordToolCallingPreservation(parser, preserved string) {
	llmdToolCallingPreservedTotal.WithLabelValues(parser, preserved).Inc()
}
