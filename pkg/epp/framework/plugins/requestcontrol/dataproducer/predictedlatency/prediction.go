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

// Package requestcontrol contains helpers to decouple latency-predictor logic.
package predictedlatency

import (
	"context"

	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type endpointPredictionResult struct {
	Endpoint         fwksched.Endpoint
	TTFT             float64
	TPOT             float64
	TTFTValid        bool
	TPOTValid        bool
	IsValid          bool
	Error            error
	Headroom         float64 // Headroom for the pod, if applicable
	TTFTHeadroom     float64 // TTFT headroom for the pod
	PrefixCacheScore float64 // Prefix cache score for the pod
}

// generatePredictions creates prediction results for all candidate pods
func (pl *PredictedLatency) generatePredictions(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx, candidateEndpoints []fwksched.Endpoint) ([]endpointPredictionResult, error) {
	logger := log.FromContext(ctx)
	predictions := make([]endpointPredictionResult, 0, len(candidateEndpoints))

	// Prepare inputs for bulk prediction
	metricsStates := make([]*fwkdl.Metrics, len(candidateEndpoints))
	targetEndpointsMetadatas := make([]*fwkdl.EndpointMetadata, len(candidateEndpoints))
	prompts := make([]string, len(candidateEndpoints))
	generatedTokenCounts := make([]int, len(candidateEndpoints))
	prefixCacheScores := make([]float64, len(candidateEndpoints))
	prefillTokensInFlights := make([]int64, len(candidateEndpoints))

	for i, endpoint := range candidateEndpoints {
		logger.V(logutil.TRACE).Info("Candidate pod for scheduling", "endpoint", endpoint.GetMetadata().String(), "metrics", endpoint.GetMetrics().String())

		// Get prefix cache score for the pod
		prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[endpoint.GetMetadata().NamespacedName.Name]

		logger.V(logutil.DEBUG).Info("Prefix cache score for pod", "pod", endpoint.GetMetadata().String(), "prefixCacheScore", prefixCacheScore)

		metricsStates[i] = endpoint.GetMetrics()
		targetEndpointsMetadatas[i] = endpoint.GetMetadata()
		prompts[i] = predictedLatencyCtx.promptText
		generatedTokenCounts[i] = 1
		prefixCacheScores[i] = prefixCacheScore

		podKey := endpoint.GetMetadata().NamespacedName.String()
		prefillTokensInFlights[i] = pl.endpointCounter(&pl.prefillTokensInFlight, podKey).Load()
	}

	// Bulk predict
	bulkPredictions, err := bulkPredictWithMetrics(ctx, predictedLatencyCtx, pl.latencypredictor, metricsStates, pl.config.EndpointRoleLabel, targetEndpointsMetadatas, prompts, generatedTokenCounts, prefixCacheScores, prefillTokensInFlights)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "Bulk prediction failed")
		return nil, err
	}

	// Process results
	for i, endpoint := range candidateEndpoints {
		prediction := bulkPredictions[i]
		predResult := endpointPredictionResult{Endpoint: endpoint}

		predResult.PrefixCacheScore = prefixCacheScores[i]
		predResult.TTFT = prediction.TTFT
		predResult.TPOT = prediction.TPOT

		podMinTPOTSLO := pl.getEndpointMinTPOTSLO(endpoint)
		predResult.TTFTValid, predResult.TPOTValid, predResult.IsValid, predResult.Headroom, predResult.TTFTHeadroom = pl.validatePrediction(prediction, predictedLatencyCtx, podMinTPOTSLO)

		// Neutralize TPOT when it's not meaningful:
		// - Non-streaming mode: TPOT is never trained (no per-token observations)
		// - Disaggregated prefill: prefill pods don't generate tokens
		// Setting TPOTValid=true and Headroom=0 prevents untrained TPOT
		// predictions from polluting scoring, tier classification, or admission.
		if !pl.config.StreamingMode || hasPrefillRole(pl.config.EndpointRoleLabel, endpoint) {
			predResult.TPOTValid = true
			predResult.Headroom = 0
			predResult.IsValid = predResult.TTFTValid
		}

		logger.V(logutil.DEBUG).Info("Prediction for scheduling",
			"endpoint", endpoint.GetMetadata().String(),
			"prefixCacheScore", predResult.PrefixCacheScore,
			"TTFT", prediction.TTFT,
			"TPOT", prediction.TPOT,
			"buffer", pl.config.SLOBufferFactor,
			"podMinTPOTSLO", podMinTPOTSLO,
			"ttftSLO", predictedLatencyCtx.ttftSLO,
			"requestTPOTSLO", predictedLatencyCtx.avgTPOTSLO,
			"tpotHeadroom", predResult.Headroom,
			"ttftHeadroom", predResult.TTFTHeadroom,
			"tpotValid", predResult.TPOTValid,
			"ttftValid", predResult.TTFTValid,
		)

		predictions = append(predictions, predResult)
	}

	return predictions, nil
}

// updateRequestContextWithPredictions updates the request context with prediction data
func (pl *PredictedLatency) updateRequestContextWithPredictions(predictedLatencyCtx *predictedLatencyCtx, predictions []endpointPredictionResult) {
	predMap := make(map[string]endpointPredictionResult, len(predictions))
	for _, pred := range predictions {
		if pred.Endpoint != nil && pred.Endpoint.GetMetadata() != nil {
			predMap[pred.Endpoint.GetMetadata().NamespacedName.Name] = pred
		}
	}
	predictedLatencyCtx.predictionsForScheduling = predMap
}

func (pl *PredictedLatency) validatePrediction(
	pred *latencypredictor.PredictionResponse,
	predictedLatencyCtx *predictedLatencyCtx,
	podMinTPOTSLO float64,
) (ttftOk, tpotOk, isValid bool, headroom float64, ttftHeadroom float64) {

	ttftOk = pred.TTFT < predictedLatencyCtx.ttftSLO
	ttftHeadroom = predictedLatencyCtx.ttftSLO - pred.TTFT

	tpotOk = true
	headroom = 0.0

	if pl.config.StreamingMode {
		bufferedTPOT := predictedLatencyCtx.avgTPOTSLO * pl.config.SLOBufferFactor
		// a podMinTPOTSLO of 0 means no either no requests, or no TPOT SLOs specified on running requests
		if podMinTPOTSLO > 0 {
			if podMinTPOTSLO < predictedLatencyCtx.avgTPOTSLO {
				log.FromContext(context.Background()).V(logutil.DEBUG).Info("Endpoint min TPOT SLO is less than the req SLO, adjusting", "podMinTPOTSLO", podMinTPOTSLO, "bufferedTPOT", predictedLatencyCtx.avgTPOTSLO)
			}
			bufferedTPOT = min(bufferedTPOT, podMinTPOTSLO*pl.config.SLOBufferFactor)
		}

		tpotOk = pred.TPOT < bufferedTPOT
		headroom = bufferedTPOT - pred.TPOT
	}

	isValid = ttftOk && tpotOk

	return
}

// hasPrefillRole returns true if the endpoint has the prefill role label set.
func hasPrefillRole(roleLabel string, endpoint fwksched.Endpoint) bool {
	if roleLabel == "" {
		return false
	}
	labels := endpoint.GetMetadata().Labels
	return labels != nil && labels[roleLabel] == "prefill"
}
