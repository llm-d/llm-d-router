/*
Copyright 2025 The Kubernetes Authors.
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

package predictedlatency

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// recordTTFTTrainingData sends a TTFT training entry to the predictor sidecar.
func recordTTFTTrainingData(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	state *predictedLatencyCtx,
	metrics *fwkdl.Metrics,
	metadata *fwkdl.EndpointMetadata,
	now time.Time,
) {
	features := composeFeatures(state, metrics, endpointFeaturesFor(state, metadata), ttftFeatures)
	entry := latencypredictor.TrainingFromFeatures(features, latencypredictor.TrainingEntry{
		ActualTTFT:    state.ttft,
		Timestamp:     now,
		PredictedTTFT: predictedLatencyMSPtr(state.predictedTTFT),
	})
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		log.FromContext(ctx).V(logutil.DEBUG).Error(err, "record TTFT training failed")
	}
}

// predictedLatencyMSPtr returns a pointer for optional predicted_* JSON fields.
// Zero or non-finite values mean no usable selected prediction and must be omitted.
func predictedLatencyMSPtr(ms float64) *float64 {
	if ms <= 0 || math.IsNaN(ms) || math.IsInf(ms, 0) {
		return nil
	}
	return &ms
}

// refreshLastSeenMetrics updates predictedLatencyCtx.lastSeenMetrics from scheduling results.
func refreshLastSeenMetrics(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx) {
	if sr := predictedLatencyCtx.schedulingResult; sr != nil {
		for profileName, profileResult := range sr.ProfileResults {
			if profileResult != nil && profileResult.TargetEndpoints != nil && len(profileResult.TargetEndpoints) > 0 {
				predictedLatencyCtx.lastSeenMetrics[profileName] = profileResult.TargetEndpoints[0].GetMetrics().Clone()
			}
		}
	} else {
		log.FromContext(ctx).V(logutil.DEBUG).Info("No scheduling result found, skipping metrics refresh")
	}
}

// getLatestMetricsForProfile retrieves the latest metrics for the specified profile.
func getLatestMetricsForProfile(predictedLatencyCtx *predictedLatencyCtx, profileName string) (*fwkdl.Metrics, error) {
	if len(predictedLatencyCtx.lastSeenMetrics) == 0 {
		return nil, errors.New("no last seen metrics available for prediction")
	}

	if profileName == "" && predictedLatencyCtx.schedulingResult != nil {
		profileName = predictedLatencyCtx.schedulingResult.PrimaryProfileName
	}

	if m, exists := predictedLatencyCtx.lastSeenMetrics[profileName]; exists {
		return m, nil
	}

	return nil, fmt.Errorf("no metrics found for profile %s", profileName)
}

// bulkPredict sends registry-composed requests and records prediction duration.
func bulkPredict(
	ctx context.Context,
	pluginName, pluginType string,
	predictedLatencyContext *predictedLatencyCtx,
	predictor latencypredictor.PredictorInterface,
	bulkRequests []latencypredictor.PredictionRequest,
) ([]*latencypredictor.PredictionResponse, error) {
	logger := log.FromContext(ctx)
	if len(bulkRequests) == 0 {
		return []*latencypredictor.PredictionResponse{}, nil
	}
	if predictor == nil {
		return nil, errPredictorUnavailable
	}
	start := time.Now()
	bulkResponse, err := predictor.PredictBulkStrict(ctx, bulkRequests)
	duration := time.Since(start)

	if err != nil {
		logger.V(logutil.DEBUG).Info("bulk prediction failed",
			"error", err,
			"duration_ms", duration.Milliseconds(),
			"request_count", len(bulkRequests))
		return nil, err
	}

	if bulkResponse == nil {
		logger.V(logutil.DEBUG).Info("bulk prediction returned nil",
			"duration_ms", duration.Milliseconds())
		return nil, errNilPredictionResponse
	}

	if len(bulkResponse.Predictions) != len(bulkRequests) {
		return nil, fmt.Errorf("%w: got %d, want %d", latencypredictor.ErrResponseLengthMismatch, len(bulkResponse.Predictions), len(bulkRequests))
	}

	if predictedLatencyContext != nil {
		recordRequestTTFTPredictionDuration(ctx, pluginName, pluginType, predictedLatencyContext.incomingModelName, predictedLatencyContext.targetModel, duration.Seconds())
		recordRequestTPOTPredictionDuration(ctx, pluginName, pluginType, predictedLatencyContext.incomingModelName, predictedLatencyContext.targetModel, duration.Seconds())
	}

	results := make([]*latencypredictor.PredictionResponse, len(bulkRequests))
	for i := range bulkRequests {
		results[i] = &bulkResponse.Predictions[i]
	}

	logger.V(logutil.DEBUG).Info("bulk prediction succeeded",
		"duration_ms", duration.Milliseconds(),
		"request_count", len(bulkRequests),
		"successful_predictions", bulkResponse.SuccessfulPredictions,
		"failed_predictions", bulkResponse.FailedPredictions,
		"processing_time_ms", bulkResponse.ProcessingTimeMs)

	if logger.V(logutil.TRACE).Enabled() {
		for i, request := range bulkRequests {
			result := results[i]
			logger.V(logutil.TRACE).Info("bulk prediction result",
				"index", i,
				"ttft_ms", result.TTFT,
				"tpot_ms", result.TPOT,
				"input_tokens", request.InputTokenLength,
				"generated_tokens", request.NumTokensGenerated,
				"kv_cache_percent", request.KVCachePercentage,
				"waiting_queue", request.NumRequestWaiting,
				"running_requests", request.NumRequestRunning,
				"prefix_cache_score", request.PrefixCacheScore)
		}
	}

	return results, nil
}

// calculateRunningAverage calculates the running average efficiently.
func calculateRunningAverage(currentAvg float64, newValue float64, count int) float64 {
	if count == 0 {
		return 0
	}
	if count == 1 {
		return newValue
	}
	return currentAvg + (newValue-currentAvg)/float64(count)
}
