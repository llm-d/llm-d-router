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

package predictedlatency

import (
	"context"
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

var _ requestcontrol.DataProducer = &PredictedLatency{}

// Produce prepares the SLO context for the request, including
// parsing SLO headers, gathering prefix cache scores, and generating predictions.
func (pl *PredictedLatency) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	logger := log.FromContext(ctx)
	predictedLatencyCtx := pl.getOrMakePredictedLatencyContextForRequest(request)

	pl.parseSLOHeaders(ctx, request, predictedLatencyCtx)
	var prefixCacheScore float64
	for _, endpoint := range endpoints {

		if prefixCacheInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey); ok {
			prefixCacheInfo := prefixCacheInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
			prefixCacheScore = float64(prefixCacheInfo.MatchBlocks()) / float64(prefixCacheInfo.TotalBlocks())
			if !math.IsNaN(prefixCacheScore) {
				logger.V(logutil.DEBUG).Info("Found prefix cache score in pod attribute", "pod", endpoint.GetMetadata().NamespacedName.Name, "score", prefixCacheScore)
			} else {
				prefixCacheScore = 0.0
				logger.V(logutil.DEBUG).Info("Prefix cache score is NaN, defaulting to 0", "pod", endpoint.GetMetadata().NamespacedName.Name)
			}
		} else {
			logger.V(logutil.DEBUG).Info("No prefix cache score found in pod attribute, defaulting to 0", "pod", endpoint.GetMetadata().NamespacedName.Name)
			prefixCacheScore = 0.0
		}
		predictedLatencyCtx.prefixCacheScoresForEndpoints[endpoint.GetMetadata().NamespacedName.Name] = prefixCacheScore
	}
	if !pl.config.PredictInProduce {
		logger.V(logutil.DEBUG).Info("PredictInProduce disabled, skipping predictions")
		if err := ctx.Err(); err != nil {
			return err
		}
		pl.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)
		return nil
	}

	predictions, err := pl.generatePredictions(ctx, predictedLatencyCtx, endpoints)
	if err == nil && len(predictions) == len(endpoints) {
		pl.updateRequestContextWithPredictions(predictedLatencyCtx, predictions)

		// Store predictions in endpoint attributes
		for _, pred := range predictions {
			if pred.Endpoint != nil {
				latencyInfo := attrlatency.NewLatencyPredictionInfo(
					pred.TTFTValid,
					pred.TPOTValid,
					pred.TTFTHeadroom,
					pred.Headroom, // Maps to TPOTHeadroom
					pred.TTFT,
					pred.TPOT,
					pl.getEndpointRunningRequestCount(pred.Endpoint),
				)
				pred.Endpoint.Put(attrlatency.LatencyPredictionInfoKey, latencyInfo)
				logger.V(logutil.DEBUG).Info("Stored latency prediction in endpoint",
					"pod", pred.Endpoint.GetMetadata().NamespacedName.Name,
					"ttft", pred.TTFT,
					"tpot", pred.TPOT,
					"ttftValid", pred.TTFTValid,
					"tpotValid", pred.TPOTValid,
					"ttftHeadroom", pred.TTFTHeadroom,
					"tpotHeadroom", pred.Headroom)
			}
		}
	}

	// Don't publish the SLO context after the director's Produce window has closed.
	// If we did, PreRequest has already run (and skipped incrementing counters because the
	// context wasn't yet present), but ResponseBody would later find the context and issue
	// an orphan decrement — drifting prefillTokensInFlight negative under sustained load.
	if err := ctx.Err(); err != nil {
		return err
	}
	pl.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)
	return nil
}

func (pl *PredictedLatency) Produces() map[string]any {
	return map[string]any{
		attrlatency.LatencyPredictionInfoKey: attrlatency.LatencyPredictionInfo{},
	}
}

func (pl *PredictedLatency) Consumes() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}
