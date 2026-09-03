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
)

const (
	// --- Subsystems ---
	inferenceExtension = "inference_extension"

	// InferenceObjectiveSubsystem is retained for predicted-latency metrics.
	InferenceObjectiveSubsystem = "inference_objective"

	// InferenceExtensionSubsystem is the legacy subsystem for inference extension metrics.
	InferenceExtensionSubsystem = inferenceExtension

	// SchedulerSubsystem is the legacy metric prefix for scheduler.
	SchedulerSubsystem = "llm_d_inference_scheduler"
)

var (
	// --- Common Label Sets ---
	modelLabels    = []string{"model_name", "target_model_name"}
	poolLabels     = []string{"name"}
	endpointLabels = []string{"pod_name", "namespace", "port"}
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
			Help: metricsutil.HelpMsgWithStability(
				"[Deprecated: Use llm_d_epp_flow_control_pool_saturation] Pool saturation signal gating Flow Control "+
					"dispatch. The stage label partitions by pipeline role: 'prefill' and 'decode' are per-stage "+
					"signals, 'effective' is max(prefill, decode) and is the value used for gating. "+
					"1.0 is the gating set point; values above 1.0 indicate the magnitude of oversubscription "+
					"past it. An empty pool reads as 1.0. With the default utilization detector, endpoints with missing "+
					"or stale metrics score as fully saturated (fail-closed; see "+
					"llm_d_epp_flow_control_stale_endpoints).",
				compbasemetrics.ALPHA),
		},
		[]string{"inference_pool", "stage"},
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
		metrics.Registry.MustRegister(llmdRequestCounter)
		metrics.Registry.MustRegister(llmdRequestErrCounter)
		metrics.Registry.MustRegister(llmdRequestLatencies)
		metrics.Registry.MustRegister(llmdRequestSizes)
		metrics.Registry.MustRegister(llmdResponseSizes)
		metrics.Registry.MustRegister(llmdInputTokens)
		metrics.Registry.MustRegister(llmdOutputTokens)
		metrics.Registry.MustRegister(llmdPromptCachedTokens)
		metrics.Registry.MustRegister(llmdRunningRequests)
		metrics.Registry.MustRegister(llmdNormalizedTimePerOutputToken)
		metrics.Registry.MustRegister(llmdRequestTTFT)
		metrics.Registry.MustRegister(llmdRequestTPOT)
		metrics.Registry.MustRegister(llmdInterTokenLatency)
		metrics.Registry.MustRegister(llmdInferencePoolAvgKVCache)
		metrics.Registry.MustRegister(llmdInferencePoolStdDevKVCache)
		metrics.Registry.MustRegister(llmdInferencePoolAvgQueueSize)
		metrics.Registry.MustRegister(llmdInferencePoolStdDevQueueSize)
		metrics.Registry.MustRegister(llmdInferencePoolAvgRunningRequests)
		metrics.Registry.MustRegister(llmdInferencePoolStdDevRunningRequests)
		metrics.Registry.MustRegister(llmdInferencePoolReadyEndpoints)
		metrics.Registry.MustRegister(schedulerE2ELatency)
		metrics.Registry.MustRegister(llmdSchedulerE2ELatency)
		metrics.Registry.MustRegister(schedulerAttemptsTotal)
		metrics.Registry.MustRegister(llmdSchedulerAttemptsTotal)
		metrics.Registry.MustRegister(pluginProcessingLatencies)
		metrics.Registry.MustRegister(llmdPluginProcessingLatencies)
		metrics.Registry.MustRegister(llmdPluginDataScopeViolations)
		metrics.Registry.MustRegister(llmdRequestProcessingLatency)
		metrics.Registry.MustRegister(llmdResponseProcessingLatency)
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
		// No deprecated inference_extension twin: new flow control metrics are emitted under the
		// llm_d_epp prefix only.
		metrics.Registry.MustRegister(llmdFlowControlStaleEndpoints)
		metrics.Registry.MustRegister(llmdFlowControlCapacityUtilizationRequests)
		metrics.Registry.MustRegister(llmdFlowControlCapacityUtilizationBytes)
		metrics.Registry.MustRegister(llmdFlowControlGlobalCapacityUtilizationRequests)
		metrics.Registry.MustRegister(llmdFlowControlGlobalCapacityUtilizationBytes)
		metrics.Registry.MustRegister(flowControlRequestEnqueueDuration)
		metrics.Registry.MustRegister(llmdFlowControlRequestEnqueueDuration)
		metrics.Registry.MustRegister(llmdFlowControlRequestsTotal)
		metrics.Registry.MustRegister(llmdFlowControlRevocationsIssuedTotal)
		metrics.Registry.MustRegister(llmdFlowControlRevocationsTotal)
		metrics.Registry.MustRegister(llmdFlowControlReclaimTarget)
		metrics.Registry.MustRegister(llmdFlowControlPendingReclaim)
		metrics.Registry.MustRegister(llmdFlowControlRevocationConfirmationDuration)
		metrics.Registry.MustRegister(inferenceModelRewriteDecisionsTotal)
		metrics.Registry.MustRegister(llmdInferenceModelRewriteDecisionsTotal)
		metrics.Registry.MustRegister(DataLayerPollErrorsTotal)
		metrics.Registry.MustRegister(LlmdDataLayerPollErrorsTotal)
		metrics.Registry.MustRegister(DataLayerExtractErrorsTotal)
		metrics.Registry.MustRegister(LlmdDataLayerExtractErrorsTotal)
		for _, collector := range customCollectors {
			metrics.Registry.MustRegister(collector)
		}
	})
}

// Just for integration test
func Reset() {
	// Reset other metrics
	llmdRequestCounter.Reset()
	llmdRequestErrCounter.Reset()
	llmdRequestLatencies.Reset()
	llmdRequestSizes.Reset()
	llmdResponseSizes.Reset()
	llmdInputTokens.Reset()
	llmdOutputTokens.Reset()
	llmdPromptCachedTokens.Reset()
	llmdRunningRequests.Reset()
	llmdNormalizedTimePerOutputToken.Reset()
	llmdRequestTTFT.Reset()
	llmdRequestTPOT.Reset()
	llmdInterTokenLatency.Reset()
	llmdInferencePoolAvgKVCache.Reset()
	llmdInferencePoolStdDevKVCache.Reset()
	llmdInferencePoolAvgQueueSize.Reset()
	llmdInferencePoolStdDevQueueSize.Reset()
	llmdInferencePoolAvgRunningRequests.Reset()
	llmdInferencePoolStdDevRunningRequests.Reset()
	llmdInferencePoolReadyEndpoints.Reset()
	schedulerE2ELatency.Reset()
	llmdSchedulerE2ELatency.Reset()
	schedulerAttemptsTotal.Reset()
	llmdSchedulerAttemptsTotal.Reset()
	pluginProcessingLatencies.Reset()
	llmdPluginProcessingLatencies.Reset()
	llmdPluginDataScopeViolations.Reset()
	llmdRequestProcessingLatency.Reset()
	llmdResponseProcessingLatency.Reset()
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
	llmdFlowControlStaleEndpoints.Reset()
	llmdFlowControlCapacityUtilizationRequests.Reset()
	llmdFlowControlCapacityUtilizationBytes.Reset()
	llmdFlowControlGlobalCapacityUtilizationRequests.Reset()
	llmdFlowControlGlobalCapacityUtilizationBytes.Reset()
	flowControlRequestEnqueueDuration.Reset()
	llmdFlowControlRequestEnqueueDuration.Reset()
	flowControlDispatchCycleDuration.Reset()
	llmdFlowControlDispatchCycleDuration.Reset()
	llmdFlowControlRequestsTotal.Reset()
	llmdFlowControlRevocationsIssuedTotal.Reset()
	llmdFlowControlRevocationsTotal.Reset()
	llmdFlowControlReclaimTarget.Reset()
	llmdFlowControlPendingReclaim.Reset()
	llmdFlowControlRevocationConfirmationDuration.Reset()
	inferenceModelRewriteDecisionsTotal.Reset()
	llmdInferenceModelRewriteDecisionsTotal.Reset()
	DataLayerPollErrorsTotal.Reset()
	LlmdDataLayerPollErrorsTotal.Reset()
	DataLayerExtractErrorsTotal.Reset()
	LlmdDataLayerExtractErrorsTotal.Reset()
}

// RecordRequestCounter records the number of requests.
func RecordRequestCounter(modelName, targetModelName, fairnessID string, priority int) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	prioStr := strconv.Itoa(priority)
	llmdRequestCounter.WithLabelValues(modelName, targetModelName, fairnessID, prioStr).Inc()
}

// RecordRequestErrCounter records the number of error requests.
func RecordRequestErrCounter(modelName, targetModelName, fairnessID, priority string, code string) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	if code != "" {
		llmdRequestErrCounter.WithLabelValues(modelName, targetModelName, fairnessID, priority, code).Inc()
	}
}

// RecordRequestSizes records the request sizes.
func RecordRequestSizes(modelName, targetModelName, fairnessID, priority string, reqSize int) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	llmdRequestSizes.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(reqSize))
}

// RecordRequestLatencies records duration of request.
func RecordRequestLatencies(ctx context.Context, modelName, targetModelName, fairnessID, priority string, received time.Time, complete time.Time) bool {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	if !complete.After(received) {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "Request latency values are invalid",
			"modelName", modelName, "targetModelName", targetModelName, "completeTime", complete, "receivedTime", received)
		return false
	}
	elapsedSeconds := complete.Sub(received).Seconds()
	llmdRequestLatencies.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(elapsedSeconds)
	return true
}

// RecordResponseSizes records the response sizes.
func RecordResponseSizes(modelName, targetModelName, fairnessID, priority string, size int) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	llmdResponseSizes.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(size))
}

// RecordInputTokens records input tokens count.
func RecordInputTokens(modelName, targetModelName, fairnessID, priority string, size int) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	if size > 0 {
		llmdInputTokens.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(size))
	}
}

// RecordOutputTokens records output tokens count.
func RecordOutputTokens(modelName, targetModelName, fairnessID, priority string, size int) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	if size > 0 {
		llmdOutputTokens.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(size))
	}
}

// RecordPromptCachedTokens records prompt cached tokens count.
func RecordPromptCachedTokens(modelName, targetModelName, fairnessID, priority string, size int) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	llmdPromptCachedTokens.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(float64(size))
}

// RecordNormalizedTimePerOutputToken (NTPOT) records the normalized time per output token.
func RecordNormalizedTimePerOutputToken(ctx context.Context, modelName, targetModelName, fairnessID, priority string, received time.Time, complete time.Time, outputTokenCount int) bool {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
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

	llmdNormalizedTimePerOutputToken.WithLabelValues(modelName, targetModelName, fairnessID, priority).Observe(secondsPerToken)
	return true
}

// RecordRequestTTFT records the time to first token.
func RecordRequestTTFT(ctx context.Context, modelName, targetModelName, fairnessID, priority string, streaming bool, received time.Time, firstToken time.Time) bool {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
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

// RecordRequestTPOT records the average time per output token. TPOT is only
// derivable for streaming responses: a non-streaming response arrives as a
// single body chunk, so the first-token and completion timestamps coincide and
// no inter-token timing exists. Such requests are skipped silently instead of
// being logged as invalid (they would otherwise emit an error-level line per
// request on non-streaming workloads).
func RecordRequestTPOT(ctx context.Context, modelName, targetModelName, fairnessID, priority string, streaming bool, received time.Time, firstToken time.Time, complete time.Time, outputTokenCount int) bool {
	if !streaming {
		return false
	}
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
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
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
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
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	if modelName != "" {
		llmdRunningRequests.WithLabelValues(modelName, targetModelName, fairnessID, priority).Inc()
	}
}

// DecRunningRequests decreases the current running requests.
func DecRunningRequests(modelName, targetModelName, fairnessID, priority string) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	if modelName != "" {
		llmdRunningRequests.WithLabelValues(modelName, targetModelName, fairnessID, priority).Dec()
	}
}

func RecordInferencePoolAvgKVCache(name string, utilization float64) {
	llmdInferencePoolAvgKVCache.WithLabelValues(name).Set(utilization)
}

func RecordInferencePoolAvgQueueSize(name string, queueSize float64) {
	llmdInferencePoolAvgQueueSize.WithLabelValues(name).Set(queueSize)
}

func RecordInferencePoolAvgRunningRequests(name string, runningRequests float64) {
	llmdInferencePoolAvgRunningRequests.WithLabelValues(name).Set(runningRequests)
}

func RecordInferencePoolStdDevKVCache(name string, utilization float64) {
	llmdInferencePoolStdDevKVCache.WithLabelValues(name).Set(utilization)
}

func RecordInferencePoolStdDevQueueSize(name string, queueSize float64) {
	llmdInferencePoolStdDevQueueSize.WithLabelValues(name).Set(queueSize)
}

func RecordInferencePoolStdDevRunningRequests(name string, runningRequests float64) {
	llmdInferencePoolStdDevRunningRequests.WithLabelValues(name).Set(runningRequests)
}

func RecordInferencePoolReadyPods(name string, runningPods float64) {
	llmdInferencePoolReadyEndpoints.WithLabelValues(name).Set(runningPods)
}

// RecordSchedulerE2ELatency records the end-to-end scheduling latency.
func RecordSchedulerE2ELatency(duration time.Duration) {
	schedulerE2ELatency.WithLabelValues().Observe(duration.Seconds())
	llmdSchedulerE2ELatency.WithLabelValues().Observe(duration.Seconds())
}

// RecordRequestProcessingLatency records the EPP request processing latency,
// measured from request receipt until the request body has been handled.
func RecordRequestProcessingLatency(duration time.Duration) {
	llmdRequestProcessingLatency.WithLabelValues().Observe(duration.Seconds())
}

// RecordResponseProcessingLatency records the EPP response processing latency
// for a single request.
func RecordResponseProcessingLatency(duration time.Duration) {
	llmdResponseProcessingLatency.WithLabelValues().Observe(duration.Seconds())
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
					schedulerAttemptsTotal.WithLabelValues(SchedulerStatusSuccess, targetModelName, metadata.Name, metadata.ID.Namespace, metadata.Port).Inc()
					llmdSchedulerAttemptsTotal.WithLabelValues(SchedulerStatusSuccess, targetModelName, metadata.Name, metadata.ID.Namespace, metadata.Port).Inc()
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

// Access kinds for RecordPluginDataScopeViolation.
const (
	DataScopeAccessRead  = "read"
	DataScopeAccessWrite = "write"
)

// RecordPluginDataScopeViolation records an endpoint attribute access rejected
// because the plugin did not declare the DataKey.
func RecordPluginDataScopeViolation(extensionPoint, pluginType, pluginName, access string) {
	llmdPluginDataScopeViolations.WithLabelValues(extensionPoint, pluginType, pluginName, access).Inc()
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
	fairnessID = boundFairnessID(fairnessID)
	modelName, targetModelName = boundModels(modelName, targetModelName)
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
	fairnessID = boundFairnessID(fairnessID)
	flowControlRequestEnqueueDuration.WithLabelValues(
		fairnessID, priority, outcome,
	).Observe(duration.Seconds())

	llmdFlowControlRequestEnqueueDuration.WithLabelValues(
		fairnessID, priority, outcome,
	).Observe(duration.Seconds())
}

// IncFlowControlQueueSize increments the Flow Control queue size gauge.
func IncFlowControlQueueSize(fairnessID, priority, inferencePool, modelName, targetModelName string) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	flowControlQueueSize.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Inc()
	llmdFlowControlQueueSize.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Inc()
}

// DecFlowControlQueueSize decrements the Flow Control queue size gauge.
func DecFlowControlQueueSize(fairnessID, priority, inferencePool, modelName, targetModelName string) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	flowControlQueueSize.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Dec()
	llmdFlowControlQueueSize.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Dec()
}

// AddFlowControlQueueBytes increments the Flow Control queue bytes gauge.
func AddFlowControlQueueBytes(fairnessID, priority, inferencePool, modelName, targetModelName string, bytes uint64) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	flowControlQueueBytes.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Add(float64(bytes))
	llmdFlowControlQueueBytes.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Add(float64(bytes))
}

// SubFlowControlQueueBytes decrements the Flow Control queue bytes gauge.
func SubFlowControlQueueBytes(fairnessID, priority, inferencePool, modelName, targetModelName string, bytes uint64) {
	modelName, targetModelName = boundModels(modelName, targetModelName)
	fairnessID = boundFairnessID(fairnessID)
	flowControlQueueBytes.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Sub(float64(bytes))
	llmdFlowControlQueueBytes.WithLabelValues(fairnessID, priority, inferencePool, modelName, targetModelName).Sub(float64(bytes))
}

// RecordFlowControlPoolSaturation records the current saturation level for an inference pool
// partitioned by pipeline stage ("prefill", "decode", or "effective").
func RecordFlowControlPoolSaturation(inferencePool, stage string, saturation float64) {
	flowControlPoolSaturation.WithLabelValues(inferencePool, stage).Set(saturation)
	llmdFlowControlPoolSaturation.WithLabelValues(inferencePool, stage).Set(saturation)
}

// DeleteFlowControlPoolSaturation removes the saturation gauge series for a pool/stage pair.
func DeleteFlowControlPoolSaturation(inferencePool, stage string) {
	flowControlPoolSaturation.DeleteLabelValues(inferencePool, stage)
	llmdFlowControlPoolSaturation.DeleteLabelValues(inferencePool, stage)
}

// RecordFlowControlStaleEndpoints records how many candidate endpoints the given saturation
// detector scored as fully saturated because their metrics were missing or stale.
func RecordFlowControlStaleEndpoints(detector string, count int) {
	llmdFlowControlStaleEndpoints.WithLabelValues(detector).Set(float64(count))
}

// RecordFlowControlCapacityUtilizationRequests sets the request-count capacity utilization ratio
// (occupancy/effective capacity, 0.0-1.0) for a single priority band. The band denominator falls back to a default
// when unconfigured, so every configured band reports a series.
func RecordFlowControlCapacityUtilizationRequests(priority, inferencePool string, ratio float64) {
	llmdFlowControlCapacityUtilizationRequests.WithLabelValues(priority, inferencePool).Set(ratio)
}

// RecordFlowControlCapacityUtilizationBytes sets the byte-size capacity utilization ratio (occupancy/effective
// capacity, 0.0-1.0) for a single priority band. The band denominator falls back to a default when unconfigured, so
// every configured band reports a series.
func RecordFlowControlCapacityUtilizationBytes(priority, inferencePool string, ratio float64) {
	llmdFlowControlCapacityUtilizationBytes.WithLabelValues(priority, inferencePool).Set(ratio)
}

// RecordFlowControlGlobalCapacityUtilizationRequests sets the all-bands request-count capacity utilization ratio
// (occupancy/global capacity, 0.0-1.0). Global capacity is optional, so callers only invoke this when one is
// configured; otherwise no series is emitted rather than a misleading 0.
func RecordFlowControlGlobalCapacityUtilizationRequests(inferencePool string, ratio float64) {
	llmdFlowControlGlobalCapacityUtilizationRequests.WithLabelValues(inferencePool).Set(ratio)
}

// RecordFlowControlGlobalCapacityUtilizationBytes sets the all-bands byte-size capacity utilization ratio
// (occupancy/global capacity, 0.0-1.0). Global capacity is optional, so callers only invoke this when one is
// configured; otherwise no series is emitted rather than a misleading 0.
func RecordFlowControlGlobalCapacityUtilizationBytes(inferencePool string, ratio float64) {
	llmdFlowControlGlobalCapacityUtilizationBytes.WithLabelValues(inferencePool).Set(ratio)
}

// IncFlowControlRequestsTotal increments the total request counter for a given outcome.
func IncFlowControlRequestsTotal(outcome, priority, inferencePool string) {
	llmdFlowControlRequestsTotal.WithLabelValues(outcome, priority, inferencePool).Inc()
}

// Terminal revocation outcomes for the flow control revocations counter. Every issued revocation
// eventually increments exactly one outcome.
const (
	RevocationOutcomeConfirmed = "confirmed"
	RevocationOutcomeTimedOut  = "timed_out"
)

// RecordFlowControlRevocationsIssued counts revocations at issue time, labeled by the demand
// band's priority.
func RecordFlowControlRevocationsIssued(inferencePool, priority string, n int) {
	llmdFlowControlRevocationsIssuedTotal.WithLabelValues(priority, inferencePool).Add(float64(n))
}

// RecordFlowControlRevocations increments the revocation counter for a terminal outcome.
func RecordFlowControlRevocations(inferencePool, outcome string, n int) {
	llmdFlowControlRevocationsTotal.WithLabelValues(outcome, inferencePool).Add(float64(n))
}

// RecordFlowControlReclaimTarget records the last computed reclamation deficit.
func RecordFlowControlReclaimTarget(inferencePool string, target float64) {
	llmdFlowControlReclaimTarget.WithLabelValues(inferencePool).Set(target)
}

// RecordFlowControlPendingReclaim records the capacity debited for unconfirmed revocations.
func RecordFlowControlPendingReclaim(inferencePool string, pending float64) {
	llmdFlowControlPendingReclaim.WithLabelValues(inferencePool).Set(pending)
}

// RecordFlowControlRevocationConfirmationDuration records issue-to-confirmation latency.
func RecordFlowControlRevocationConfirmationDuration(inferencePool string, duration time.Duration) {
	llmdFlowControlRevocationConfirmationDuration.WithLabelValues(inferencePool).Observe(duration.Seconds())
}

// DeleteFlowControlFlowSeries removes every flow-control series labeled with the given fairness ID
// and priority, across both the deprecated and the llm_d_epp metric families. The fairness ID is
// derived from client input, so its cardinality is unbounded; the flow registry calls this when it
// garbage-collects an idle flow so that the metric vectors track live flows instead of growing
// monotonically with every fairness ID ever observed.
//
// Pruning is not synchronized with recording: a request reviving the flow concurrently with GC can
// have a queue gauge increment deleted here while its paired decrement lands afterwards, leaving
// the queue size/bytes gauges negative until the flow's next collection deletes the series again.
func DeleteFlowControlFlowSeries(fairnessID, priority string) {
	// The overflow value aggregates every capped-out fairness ID, so a flow whose client-chosen ID
	// equals it must not delete the shared series.
	if fairnessID == metricsutil.OverflowValue {
		return
	}
	labels := prometheus.Labels{"fairness_id": fairnessID, "priority": priority}
	flowControlRequestQueueDuration.DeletePartialMatch(labels)
	flowControlRequestEnqueueDuration.DeletePartialMatch(labels)
	flowControlQueueSize.DeletePartialMatch(labels)
	flowControlQueueBytes.DeletePartialMatch(labels)
	llmdFlowControlRequestQueueDuration.DeletePartialMatch(labels)
	llmdFlowControlRequestEnqueueDuration.DeletePartialMatch(labels)
	llmdFlowControlQueueSize.DeletePartialMatch(labels)
	llmdFlowControlQueueBytes.DeletePartialMatch(labels)
}

// RecordInferenceModelRewriteDecision records the routing decision for InferenceModelRewrite.
// The rewrite name and target come from configuration; only the source model name is
// request-derived and needs bounding (a generic rule matches arbitrary model names).
func RecordInferenceModelRewriteDecision(modelRewriteName, modelName, targetModel string) {
	modelName = boundModel(modelName)
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
