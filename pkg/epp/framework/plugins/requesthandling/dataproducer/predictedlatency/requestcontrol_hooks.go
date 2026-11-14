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
	"time"

	"github.com/go-logr/logr"
	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

var _ fwkrh.PreRequest = &PredictedLatency{}
var _ fwkrh.ResponseHeaderProcessor = &PredictedLatency{}
var _ fwkrh.ResponseBodyProcessor = &PredictedLatency{}

// --- RequestControl Hooks ---

func (pl *PredictedLatency) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.PreRequest: request is nil, skipping")
		return nil
	}

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping PreRequest because no scheduling result was provided.")
		return nil
	}

	targetMetadata := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName].TargetEndpoints[0].GetMetadata()
	if !pl.checkPredictor(logger, targetMetadata) {
		return nil
	}

	endpointName := types.NamespacedName{
		Name:      targetMetadata.ID.Name,
		Namespace: targetMetadata.ID.Namespace,
	}

	logger.V(logutil.TRACE).Info("request ID for SLO tracking", "requestID", request.RequestID, "endpointName", endpointName)
	if request.RequestID == "" {
		logger.V(logutil.DEBUG).Error(errors.New("missing request ID"), "PredictedLatency.PreRequest: Request is missing request ID")
		return nil
	}

	id := request.RequestID

	actual, _ := pl.runningRequestLists.LoadOrStore(endpointName, newRequestPriorityQueue())
	endpointRequestList := actual.(*requestPriorityQueue)

	predictedLatencyCtx, err := pl.getPredictedLatencyContextForRequest(request)
	if err != nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.PreRequest: Failed to get SLO context for request", "error", err, "requestID", id)
		return nil
	}

	added := endpointRequestList.Add(id, predictedLatencyCtx.avgTPOTSLO)
	if !added {
		logger.V(logutil.TRACE).Info("PredictedLatency: Item already exists in queue", "endpointName", endpointName, "requestID", id)
	}

	predictedLatencyCtx.targetMetadata = targetMetadata
	var prefillEndpoint fwksched.Endpoint
	if prefillResult, exists := schedulingResult.ProfileResults[ExperimentalDefaultPrefillProfile]; exists && prefillResult != nil && len(prefillResult.TargetEndpoints) > 0 {
		prefillEndpoint = prefillResult.TargetEndpoints[0]
		prefillMetadata := prefillEndpoint.GetMetadata()
		predictedLatencyCtx.prefillTargetMetadata = prefillMetadata
		logger.V(logutil.DEBUG).Info("Prefill target identified for request", "requestID", id, "prefillEndpoint", prefillMetadata.ID.String())
	} else {
		logger.V(logutil.DEBUG).Info("No prefill target identified for request", "requestID", id)
	}
	predictedLatencyCtx.schedulingResult = schedulingResult
	predictedLatencyCtx.requestReceivedTimestamp = time.Now()
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)

	processPreRequestForLatencyPrediction(ctx, predictedLatencyCtx)
	return nil
}

func (pl *PredictedLatency) ResponseHeader(ctx context.Context, request *fwksched.InferenceRequest, response *fwkrh.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseReceived: request is nil, skipping")
		return
	}
}

// ResponseBody handles both per-chunk processing and request completion logic.
func (pl *PredictedLatency) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest, response *fwkrh.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseBody: request is nil, skipping")
		return
	}
	if !pl.checkPredictor(logger, targetMetadata) {
		return
	}

	// The handler synthesizes completion records for truncated streams.
	// These callbacks carry no new data event and must not add a token.
	completionOnly := response.EndOfStream &&
		response.TerminationCause != "" &&
		response.TerminationCause != fwkrh.TerminationCauseNatural
	now := time.Now()
	predictedLatencyCtx, err := pl.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.RequestID
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseBody: Failed to get SLO context", "error", err, "requestID", id)
		return
	}

	if predictedLatencyCtx.ttft == 0 {
		if pl.config.StreamingMode && !response.EndOfStream {
			processFirstTokenForLatencyPrediction(ctx, pl.latencypredictor, pl.config.StreamingMode, predictedLatencyCtx, now)
		}
	} else if !completionOnly {
		processTokenForLatencyPrediction(ctx, predictedLatencyCtx, now)
	}

	if response.EndOfStream {
		if !pl.config.StreamingMode && !completionOnly {
			processFirstTokenForLatencyPrediction(ctx, pl.latencypredictor, pl.config.StreamingMode, predictedLatencyCtx, now)
		}

		if predictedLatencyCtx.ttft > 0 {
			// In non-streaming mode, TTFT represents full e2e latency.
			logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTTFT", predictedLatencyCtx.ttft, "avgPredictedTTFT", predictedLatencyCtx.predictedTTFT)
			recordRequestTTFT(ctx, pl.typedName.Name, pl.typedName.Type, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.ttft/1000)
			recordRequestPredictedTTFT(ctx, pl.typedName.Name, pl.typedName.Type, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.predictedTTFT/1000)
			if predictedLatencyCtx.ttftSLO > 0 {
				recordRequestTTFTWithSLO(ctx, pl.typedName.Name, pl.typedName.Type, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.ttft, predictedLatencyCtx.ttftSLO)
			}
		}

		if predictedLatencyCtx.ttft > 0 && predictedLatencyCtx.generatedTokenCount > 1 {
			e2eMs := float64(predictedLatencyCtx.lastTokenTimestamp.Sub(predictedLatencyCtx.requestReceivedTimestamp).Milliseconds())
			predictedLatencyCtx.avgTPOT = (e2eMs - predictedLatencyCtx.ttft) / float64(predictedLatencyCtx.generatedTokenCount-1)
		}

		if predictedLatencyCtx.avgTPOT > 0 {
			logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTPOT", predictedLatencyCtx.avgTPOT, "avgPredictedTPOT", predictedLatencyCtx.avgPredictedTPOT)
			recordRequestTPOT(ctx, pl.typedName.Name, pl.typedName.Type, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgTPOT/1000)
			recordRequestPredictedTPOT(ctx, pl.typedName.Name, pl.typedName.Type, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgPredictedTPOT/1000)
			if predictedLatencyCtx.avgTPOTSLO > 0 {
				recordRequestTPOTWithSLO(ctx, pl.typedName.Name, pl.typedName.Type, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgTPOT, predictedLatencyCtx.avgTPOTSLO)
			}

			if m, err := getLatestMetricsForProfile(predictedLatencyCtx, ""); err == nil {
				features := composeFeatures(predictedLatencyCtx, m, endpointFeaturesFor(predictedLatencyCtx, predictedLatencyCtx.targetMetadata), tpotFeatures)
				entry := latencypredictor.TrainingFromFeatures(features, latencypredictor.TrainingEntry{
					ActualTPOT:    predictedLatencyCtx.avgTPOT,
					Timestamp:     now,
					PredictedTPOT: predictedLatencyMSPtr(predictedLatencyCtx.avgPredictedTPOT),
				})
				// Request-average actual TPOT is paired with the scheduling-time
				// selected decode/primary prediction (avgPredictedTPOT).
				if err := pl.latencypredictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
					logger.V(logutil.DEBUG).Error(err, "record TPOT training failed")
				}
			}
		}

		id := request.RequestID
		pl.removeRequestFromQueue(id, predictedLatencyCtx)
		pl.deletePredictedLatencyContextForRequest(request)
	}
}

func (pl *PredictedLatency) checkPredictor(logger logr.Logger, metadata *fwkdl.EndpointMetadata) bool {
	if metadata == nil {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping hook because no target metadata was provided.")
		return false
	}
	if pl.latencypredictor == nil {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping hook because predictor missing")
		return false
	}
	return true
}

// processPreRequestForLatencyPrediction looks up the stored prediction for the target endpoint.
func processPreRequestForLatencyPrediction(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx) {
	logger := log.FromContext(ctx)
	targetName := predictedLatencyCtx.targetMetadata.ID.Name
	if m := predictedLatencyCtx.prefillTargetMetadata; m != nil {
		targetName = m.ID.Name
	}
	if storedPred, ok := predictedLatencyCtx.predictionsForScheduling[targetName]; ok {
		logger.V(logutil.DEBUG).Info("PreRequest TTFT from stored prediction", "value_ms", storedPred.TTFT, "endpoint", targetName)
		predictedLatencyCtx.predictedTTFT = storedPred.TTFT
	} else {
		logger.V(logutil.DEBUG).Info("PreRequest: no stored prediction found for target endpoint", "endpoint", targetName)
		predictedLatencyCtx.predictedTTFT = 0
	}
	predictedLatencyCtx.lastTokenTimestamp = time.Now()
}

// processFirstTokenForLatencyPrediction records actual TTFT, trains, predicts first TPOT.
func processFirstTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	streamingMode bool,
	predictedLatencyCtx *predictedLatencyCtx,
	now time.Time,
) {
	logger := log.FromContext(ctx)

	predictedLatencyCtx.ttft = float64(now.Sub(predictedLatencyCtx.requestReceivedTimestamp).Milliseconds())
	predictedLatencyCtx.generatedTokenCount = 1

	if prefillTargetMetadata := predictedLatencyCtx.prefillTargetMetadata; prefillTargetMetadata != nil {
		prefillMetrics, err := getLatestMetricsForProfile(predictedLatencyCtx, ExperimentalDefaultPrefillProfile)
		if err == nil {
			recordTTFTTrainingData(ctx, predictor, predictedLatencyCtx, prefillMetrics, prefillTargetMetadata, now)
		}
	} else {
		m, err := getLatestMetricsForProfile(predictedLatencyCtx, "")
		if err != nil {
			logger.V(logutil.DEBUG).Info("Skipping TTFT training due to missing metrics or schedulingResult", "error", err)
			return
		}
		targetEndpointMetadata := predictedLatencyCtx.targetMetadata
		recordTTFTTrainingData(ctx, predictor, predictedLatencyCtx, m, targetEndpointMetadata, now)
	}

	if streamingMode {
		predictFirstTPOT(ctx, predictedLatencyCtx)
	}

	predictedLatencyCtx.lastTokenTimestamp = now
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
}

func predictFirstTPOT(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx) {
	logger := log.FromContext(ctx)
	targetName := predictedLatencyCtx.targetMetadata.ID.Name
	if storedPred, ok := predictedLatencyCtx.predictionsForScheduling[targetName]; ok {
		logger.V(logutil.DEBUG).Info("first TPOT from stored prediction", "value_ms", storedPred.TPOT)
		predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, storedPred.TPOT)
		predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, storedPred.TPOT, len(predictedLatencyCtx.predictedTPOTObservations))
	} else {
		logger.V(logutil.DEBUG).Info("first TPOT: no stored prediction found for target endpoint", "endpoint", targetName)
		predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, 0)
		predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, 0, len(predictedLatencyCtx.predictedTPOTObservations))
	}
}

// processTokenForLatencyPrediction records the actual TPOT for the token and advances the timestamp.
func processTokenForLatencyPrediction(
	ctx context.Context,
	predictedLatencyCtx *predictedLatencyCtx,
	now time.Time,
) {
	logger := log.FromContext(ctx)

	latencyMs := float64(now.Sub(predictedLatencyCtx.lastTokenTimestamp).Milliseconds())
	predictedLatencyCtx.generatedTokenCount++

	if predictedLatencyCtx.generatedTokenCount == 2 {
		logger.V(logutil.DEBUG).Info("First inter-token latency observed",
			"actual_tpot_ms", latencyMs)
	}

	predictedLatencyCtx.lastTokenTimestamp = now
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
}
