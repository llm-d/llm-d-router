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

package predictedlatency

import (
	"context"
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	typeTPOT                   = "tpot"
	typePredictedTPOT          = "predicted_tpot"
	typeTPOTPredictionDuration = "tpot_prediction_duration"
	typeTPOTSLOViolation       = "tpot_slo_violation"

	typeTTFT                   = "ttft"
	typePredictedTTFT          = "predicted_ttft"
	typeTTFTPredictionDuration = "ttft_prediction_duration"
	typeTTFTSLOViolation       = "ttft_slo_violation"
)

var (
	modelLabels     = []string{"model_name", "target_model_name"}
	modelTypeLabels = []string{"model_name", "target_model_name", "type"}

	generalLatencyBuckets = []float64{
		0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6,
		8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200,
		1800, 2700, 3600,
	}

	tpotBuckets = []float64{
		0.0005, 0.00205, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2,
		0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4.5, 6, 12, 18, 24, 30, 36, 48, 60,
		90, 120, 180, 270, 360,
	}

	predictionLatencyBuckets = []float64{
		0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
	}
)

var (
	inferenceGauges = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.InferenceObjectiveSubsystem,
			Name:      "inference_request_metric",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_inference_request_metric] Consolidated gauge for various inference request metrics including TTFT, TPOT, SLOs, and prediction durations.", compbasemetrics.ALPHA),
		},
		modelTypeLabels,
	)

	llmdInferenceGauges = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "inference_request_metric",
			Help:      metricsutil.HelpMsgWithStability("Consolidated gauge for various inference request metrics including TTFT, TPOT, SLOs, and prediction durations.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "model_name", "target_model_name", "type"},
	)

	requestTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.InferenceObjectiveSubsystem,
			Name:      "request_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_request_ttft_seconds] Inference model TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		modelLabels,
	)

	llmdRequestTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "request_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		[]string{"plugin_name", "plugin_type", "model_name", "target_model_name"},
	)

	requestPredictedTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.InferenceObjectiveSubsystem,
			Name:      "request_predicted_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_request_predicted_ttft_seconds] Inference model Predicted TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		modelLabels,
	)

	llmdRequestPredictedTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "request_predicted_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   generalLatencyBuckets,
		},
		[]string{"plugin_name", "plugin_type", "model_name", "target_model_name"},
	)

	requestTTFTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.InferenceObjectiveSubsystem,
			Name:      "request_ttft_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_request_ttft_prediction_duration_seconds] Duration taken to generate TTFT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   predictionLatencyBuckets,
		},
		modelLabels,
	)

	llmdRequestTTFTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "request_ttft_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration taken to generate TTFT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   predictionLatencyBuckets,
		},
		[]string{"plugin_name", "plugin_type", "model_name", "target_model_name"},
	)

	requestTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.InferenceObjectiveSubsystem,
			Name:      "request_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_request_tpot_seconds] Inference model TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   tpotBuckets,
		},
		modelLabels,
	)

	llmdRequestTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "request_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   tpotBuckets,
		},
		[]string{"plugin_name", "plugin_type", "model_name", "target_model_name"},
	)

	requestPredictedTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.InferenceObjectiveSubsystem,
			Name:      "request_predicted_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_request_predicted_tpot_seconds] Inference model Predicted TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   tpotBuckets,
		},
		modelLabels,
	)

	llmdRequestPredictedTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "request_predicted_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   tpotBuckets,
		},
		[]string{"plugin_name", "plugin_type", "model_name", "target_model_name"},
	)

	requestTPOTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.InferenceObjectiveSubsystem,
			Name:      "request_tpot_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_request_tpot_prediction_duration_seconds] Duration taken to generate TPOT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   predictionLatencyBuckets,
		},
		modelLabels,
	)

	llmdRequestTPOTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "request_tpot_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration taken to generate TPOT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   predictionLatencyBuckets,
		},
		[]string{"plugin_name", "plugin_type", "model_name", "target_model_name"},
	)

	sloViolationCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.InferenceObjectiveSubsystem,
			Name:      "request_slo_violation_total",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_router_epp_request_slo_violation_total] Counter of SLO violations for each model, target model, and violation type.", compbasemetrics.ALPHA),
		},
		modelTypeLabels,
	)

	llmdSloViolationCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "request_slo_violation_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of SLO violations for each model, target model, and violation type.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "model_name", "target_model_name", "type"},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("predicted latency metrics registerer is required")
	}
	for _, collector := range []prometheus.Collector{
		inferenceGauges,
		llmdInferenceGauges,
		requestTTFT,
		llmdRequestTTFT,
		requestPredictedTTFT,
		llmdRequestPredictedTTFT,
		requestTTFTPredictionDuration,
		llmdRequestTTFTPredictionDuration,
		requestTPOT,
		llmdRequestTPOT,
		requestPredictedTPOT,
		llmdRequestPredictedTPOT,
		requestTPOTPredictionDuration,
		llmdRequestTPOTPredictionDuration,
		sloViolationCounter,
		llmdSloViolationCounter,
	} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register predicted latency metric: %w", err)
		}
	}
	return nil
}

func recordRequestTPOT(ctx context.Context, pluginName, pluginType, modelName, targetModelName string, tpot float64) bool {
	if tpot < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TPOT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "tpot", tpot)
		return false
	}
	requestTPOT.WithLabelValues(modelName, targetModelName).Observe(tpot)
	llmdRequestTPOT.WithLabelValues(pluginName, pluginType, modelName, targetModelName).Observe(tpot)
	inferenceGauges.WithLabelValues(modelName, targetModelName, typeTPOT).Set(tpot)
	llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTPOT).Set(tpot)
	return true
}

func recordRequestTPOTWithSLO(ctx context.Context, pluginName, pluginType, modelName, targetModelName string, tpot float64, sloThreshold float64) bool {
	if tpot < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TPOT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "tpot", tpot)
		return false
	}

	if tpot > sloThreshold {
		inferenceGauges.WithLabelValues(modelName, targetModelName, typeTPOTSLOViolation).Set(1)
		llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTPOTSLOViolation).Set(1)
		sloViolationCounter.WithLabelValues(modelName, targetModelName, typeTPOT).Inc()
		llmdSloViolationCounter.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTPOT).Inc()
		log.FromContext(ctx).V(logutil.DEFAULT).Info("TPOT SLO violation detected",
			"modelName", modelName, "targetModelName", targetModelName, "tpot", tpot, "threshold", sloThreshold)
	} else {
		inferenceGauges.WithLabelValues(modelName, targetModelName, typeTPOTSLOViolation).Set(0)
		llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTPOTSLOViolation).Set(0)
	}

	return true
}

func recordRequestPredictedTPOT(ctx context.Context, pluginName, pluginType, modelName, targetModelName string, predictedTPOT float64) bool {
	if predictedTPOT < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "Predicted TPOT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "tpot", predictedTPOT)
		return false
	}
	requestPredictedTPOT.WithLabelValues(modelName, targetModelName).Observe(predictedTPOT)
	llmdRequestPredictedTPOT.WithLabelValues(pluginName, pluginType, modelName, targetModelName).Observe(predictedTPOT)
	inferenceGauges.WithLabelValues(modelName, targetModelName, typePredictedTPOT).Set(predictedTPOT)
	llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typePredictedTPOT).Set(predictedTPOT)
	return true
}

func recordRequestTPOTPredictionDuration(ctx context.Context, pluginName, pluginType, modelName, targetModelName string, duration float64) bool {
	if duration < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TPOT prediction duration must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "duration", duration)
		return false
	}
	requestTPOTPredictionDuration.WithLabelValues(modelName, targetModelName).Observe(duration)
	llmdRequestTPOTPredictionDuration.WithLabelValues(pluginName, pluginType, modelName, targetModelName).Observe(duration)
	inferenceGauges.WithLabelValues(modelName, targetModelName, typeTPOTPredictionDuration).Set(duration)
	llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTPOTPredictionDuration).Set(duration)
	return true
}

func recordRequestTTFT(ctx context.Context, pluginName, pluginType, modelName, targetModelName string, ttft float64) bool {
	if ttft < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TTFT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "ttft", ttft)
		return false
	}
	requestTTFT.WithLabelValues(modelName, targetModelName).Observe(ttft)
	llmdRequestTTFT.WithLabelValues(pluginName, pluginType, modelName, targetModelName).Observe(ttft)
	inferenceGauges.WithLabelValues(modelName, targetModelName, typeTTFT).Set(ttft)
	llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTTFT).Set(ttft)
	return true
}

func recordRequestTTFTWithSLO(ctx context.Context, pluginName, pluginType, modelName, targetModelName string, ttft float64, sloThreshold float64) bool {
	if ttft < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TTFT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "ttft", ttft)
		return false
	}

	if ttft > sloThreshold {
		inferenceGauges.WithLabelValues(modelName, targetModelName, typeTTFTSLOViolation).Set(1)
		llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTTFTSLOViolation).Set(1)
		sloViolationCounter.WithLabelValues(modelName, targetModelName, typeTTFT).Inc()
		llmdSloViolationCounter.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTTFT).Inc()
		log.FromContext(ctx).V(logutil.DEFAULT).Info("TTFT SLO violation detected",
			"modelName", modelName, "targetModelName", targetModelName, "ttft", ttft, "threshold", sloThreshold)
	} else {
		inferenceGauges.WithLabelValues(modelName, targetModelName, typeTTFTSLOViolation).Set(0)
		llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTTFTSLOViolation).Set(0)
	}

	return true
}

func recordRequestPredictedTTFT(ctx context.Context, pluginName, pluginType, modelName, targetModelName string, predictedTTFT float64) bool {
	if predictedTTFT < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "Predicted TTFT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "ttft", predictedTTFT)
		return false
	}
	requestPredictedTTFT.WithLabelValues(modelName, targetModelName).Observe(predictedTTFT)
	llmdRequestPredictedTTFT.WithLabelValues(pluginName, pluginType, modelName, targetModelName).Observe(predictedTTFT)
	inferenceGauges.WithLabelValues(modelName, targetModelName, typePredictedTTFT).Set(predictedTTFT)
	llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typePredictedTTFT).Set(predictedTTFT)
	return true
}

func recordRequestTTFTPredictionDuration(ctx context.Context, pluginName, pluginType, modelName, targetModelName string, duration float64) bool {
	if duration < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TTFT prediction duration must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "duration", duration)
		return false
	}
	requestTTFTPredictionDuration.WithLabelValues(modelName, targetModelName).Observe(duration)
	llmdRequestTTFTPredictionDuration.WithLabelValues(pluginName, pluginType, modelName, targetModelName).Observe(duration)
	inferenceGauges.WithLabelValues(modelName, targetModelName, typeTTFTPredictionDuration).Set(duration)
	llmdInferenceGauges.WithLabelValues(pluginName, pluginType, modelName, targetModelName, typeTTFTPredictionDuration).Set(duration)
	return true
}
