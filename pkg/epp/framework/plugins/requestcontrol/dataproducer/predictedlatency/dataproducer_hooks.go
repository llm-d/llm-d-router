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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
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

		if prefixCacheInfoRaw, ok := endpoint.Get(pl.prefixMatchDataKey.String()); ok {
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

		if pl.config.UseEncoderCacheFeatures {
			pl.captureEncoderCacheSizes(ctx, predictedLatencyCtx, endpoint)
		}

		// Capture the in-flight load here, in the DAG-ordered Produce hook, and
		// reuse it for both the prediction features and the dispatch-time training
		// features. Re-reading the live attribute in PreRequest would make the
		// value depend on undefined PreRequest hook ordering.
		predictedLatencyCtx.inFlightLoadForEndpoints[endpoint.GetMetadata().NamespacedName.String()] = pl.readInFlightLoad(endpoint)
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
				pred.Endpoint.Put(pl.latencyPredictionInfoDataKey.String(), latencyInfo)
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

// captureEncoderCacheSizes stores the request's total multimodal encoder item
// size and the endpoint's matched portion in the request context. The match
// data is attached by the multimodal encoder-cache producer, which the DAG
// orders before this plugin; absence means a text-only request and leaves the
// sizes at 0.
func (pl *PredictedLatency) captureEncoderCacheSizes(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx, endpoint fwksched.Endpoint) {
	logger := log.FromContext(ctx)
	raw, ok := endpoint.Get(pl.encoderCacheDataKey.String())
	if !ok {
		return
	}
	matchInfo, ok := raw.(*attrmm.EncoderCacheMatchInfo)
	if !ok || matchInfo == nil {
		return
	}
	inputSize := sumItemSizes(matchInfo.RequestItems())
	matchedSize := sumItemSizes(matchInfo.MatchedItems())
	// The predictor rejects matched > input, so clamp inconsistent match data
	// rather than dropping the whole prediction or training entry.
	if matchedSize > inputSize {
		logger.V(logutil.DEBUG).Info("Encoder cache matched size exceeds input size, clamping",
			"pod", endpoint.GetMetadata().String(),
			"encoderInputSize", inputSize,
			"encoderMatchedSize", matchedSize)
		matchedSize = inputSize
	}
	predictedLatencyCtx.encoderInputSize = inputSize
	predictedLatencyCtx.encoderMatchedSizeForEndpoints[endpoint.GetMetadata().NamespacedName.Name] = matchedSize
	logger.V(logutil.DEBUG).Info("Encoder cache sizes for pod",
		"pod", endpoint.GetMetadata().String(),
		"encoderInputSize", inputSize,
		"encoderMatchedSize", matchedSize)
}

func sumItemSizes(items []attrmm.MatchItem) int {
	total := 0
	for _, item := range items {
		total += item.Size
	}
	return total
}

func (pl *PredictedLatency) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{
		pl.latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfo{},
	}
}

func (pl *PredictedLatency) Consumes() plugin.DataDependencies {
	required := map[plugin.DataKey]any{
		pl.prefixMatchDataKey:                attrprefix.PrefixCacheMatchInfo{},
		pl.inFlightLoadDataKey:               attrconcurrency.InFlightLoad{},
		tokenproducer.TokenizedPromptDataKey: fwksched.TokenizedPrompt{},
	}
	// Required (not Optional) because only Required dependencies create DAG
	// ordering edges; the encoder-cache producer must run before this plugin.
	if pl.config.UseEncoderCacheFeatures {
		required[pl.encoderCacheDataKey] = attrmm.EncoderCacheMatchInfo{}
	}
	return plugin.DataDependencies{Required: required}
}
