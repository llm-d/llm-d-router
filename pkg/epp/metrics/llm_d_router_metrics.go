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

	metricsutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/metrics"
)

const (
	llmdInferenceSchedulerSubsystem = "llm_d_router_epp"
)

var (
	// llmdEndpointLabels replaces the deprecated endpointLabels that used "pod_name".
	llmdEndpointLabels = []string{"endpoint_name", "namespace", "port"}
)

// --- llm-d Inference Objective Metrics ---
var (
	llmdRequestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of inference objective requests broken out for each model and target model.", compbasemetrics.ALPHA),
		},
		modelWithPriorityLabels,
	)

	llmdRequestErrCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_error_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of inference objective requests errors broken out for each model and target model.", compbasemetrics.ALPHA),
		},
		append(modelLabels, "error_code"),
	)

	llmdInferenceGauges = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "inference_request_metric",
			Help:      metricsutil.HelpMsgWithStability("Consolidated gauge for various inference request metrics including TTFT, TPOT, SLOs, and prediction durations.", compbasemetrics.ALPHA),
		},
		modelTypeLabels,
	)

	llmdRequestTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		modelLabels,
	)

	llmdRequestPredictedTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_predicted_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		modelLabels,
	)

	llmdRequestTTFTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_ttft_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration taken to generate TTFT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   predictionLatencyBuckets,
		},
		modelLabels,
	)

	llmdRequestTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   tpotBuckets,
		},
		modelLabels,
	)

	llmdRequestPredictedTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_predicted_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   tpotBuckets,
		},
		modelLabels,
	)

	llmdRequestTPOTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_tpot_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration taken to generate TPOT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   predictionLatencyBuckets,
		},
		modelLabels,
	)

	llmdSloViolationCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_slo_violation_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of SLO violations for each model, target model, and violation type.", compbasemetrics.ALPHA),
		},
		modelTypeLabels,
	)

	llmdRequestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference objective response latency distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		modelLabels,
	)

	llmdRequestSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "request_sizes",
			Help:      metricsutil.HelpMsgWithStability("Inference objective requests size distribution in bytes for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
				131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
				16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
			},
		},
		modelLabels,
	)

	llmdResponseSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "response_sizes",
			Help:      metricsutil.HelpMsgWithStability("Inference objective responses size distribution in bytes for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536},
		},
		modelLabels,
	)

	llmdInputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "input_tokens",
			Help:      metricsutil.HelpMsgWithStability("Inference objective input token count distribution for requests in each model.", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		modelLabels,
	)

	llmdOutputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "output_tokens",
			Help:      metricsutil.HelpMsgWithStability("Inference objective output token count distribution for requests in each model.", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
		},
		modelLabels,
	)

	llmdPromptCachedTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "prompt_cached_tokens",
			Help:      metricsutil.HelpMsgWithStability("Inference objective prompt cached token count distribution for requests in each model.", compbasemetrics.ALPHA),
			Buckets:   []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		modelLabels,
	)

	llmdRunningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "running_requests",
			Help:      metricsutil.HelpMsgWithStability("Inference objective number of running requests in each model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name"},
	)

	llmdNormalizedTimePerOutputToken = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "normalized_time_per_output_token_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference objective latency divided by number of output tokens in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
			},
		},
		modelLabels,
	)
)

// --- llm-d Inference Pool Metrics ---
var (
	llmdInferencePoolAvgKVCache = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "average_kv_cache_utilization",
			Help:      metricsutil.HelpMsgWithStability("The average kv cache utilization for an inference server pool.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	llmdInferencePoolAvgQueueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "average_queue_size",
			Help:      metricsutil.HelpMsgWithStability("The average number of requests pending in the model server queue.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	llmdInferencePoolAvgRunningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "average_running_requests",
			Help:      metricsutil.HelpMsgWithStability("The average number of running requests across model servers in the pool.", compbasemetrics.ALPHA),
		},
		poolLabels,
	)

	llmdInferencePoolReadyEndpoints = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
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
			Subsystem: llmdInferenceSchedulerSubsystem,
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
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "scheduler_attempts_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of scheduling attempts.", compbasemetrics.ALPHA),
		},
		append([]string{"status", "target_model_name"}, llmdEndpointLabels...),
	)

	llmdPluginProcessingLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "plugin_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Plugin processing latency distribution in seconds for each extension point, plugin type and plugin name.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{"extension_point", "plugin_type", "plugin_name"},
	)

	llmdPrefixCacheSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "prefix_indexer_size",
			Help:      metricsutil.HelpMsgWithStability("Size of the prefix indexer.", compbasemetrics.ALPHA),
		},
		[]string{},
	)

	llmdPrefixCacheHitRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "prefix_indexer_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Ratio of prefix length matched to total prefix length in the cache lookup.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{},
	)

	llmdPrefixCacheHitLength = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "prefix_indexer_hit_bytes",
			Help:      metricsutil.HelpMsgWithStability("Length of the prefix match in number of bytes in the cache lookup.", compbasemetrics.ALPHA),
			Buckets:   []float64{0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536},
		},
		[]string{},
	)
)

// --- llm-d Info Metrics ---
var llmdInferenceExtensionInfo = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: llmdInferenceSchedulerSubsystem,
		Name:      "info",
		Help:      metricsutil.HelpMsgWithStability("General information of the current build of Inference Extension.", compbasemetrics.ALPHA),
	},
	[]string{"commit", "build_ref"},
)

// --- llm-d Flow Control Metrics ---
var (
	llmdFlowControlRequestQueueDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
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
			Subsystem: llmdInferenceSchedulerSubsystem,
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
			Subsystem: llmdInferenceSchedulerSubsystem,
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
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "flow_control_queue_size",
			Help:      metricsutil.HelpMsgWithStability("Current number of requests actively held in the Flow Control queue.", compbasemetrics.ALPHA),
		},
		append([]string{"fairness_id", "priority", "inference_pool"}, modelLabels...),
	)

	llmdFlowControlQueueBytes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "flow_control_queue_bytes",
			Help:      metricsutil.HelpMsgWithStability("Current total size in bytes of requests actively held in the Flow Control queue.", compbasemetrics.ALPHA),
		},
		append([]string{"fairness_id", "priority", "inference_pool"}, modelLabels...),
	)

	llmdFlowControlPoolSaturation = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: llmdInferenceSchedulerSubsystem,
			Name:      "flow_control_pool_saturation",
			Help:      metricsutil.HelpMsgWithStability("Current saturation level of the inference pool (0.0 = empty, 1.0 = fully saturated).", compbasemetrics.ALPHA),
		},
		[]string{"inference_pool"},
	)
)

// --- llm-d Inference Model Rewrite Metrics ---
var llmdInferenceModelRewriteDecisionsTotal = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: llmdInferenceSchedulerSubsystem,
		Name:      "model_rewrite_decisions_total",
		Help:      metricsutil.HelpMsgWithStability("Total number of inference model rewrite decisions.", compbasemetrics.ALPHA),
	},
	[]string{"model_rewrite_name", "model_name", "target_model"},
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
