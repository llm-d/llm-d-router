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

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

var _ fwkrh.DataProducer = &PredictedLatency{}

// Produce prepares the SLO context for the request, including
// parsing SLO headers, gathering prefix cache scores, and generating predictions.
func (pl *PredictedLatency) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	logger := log.FromContext(ctx)
	predictedLatencyCtx := pl.getOrMakePredictedLatencyContextForRequest(request)

	pl.parseSLOHeaders(ctx, request, predictedLatencyCtx)
	for _, endpoint := range endpoints {
		predictedLatencyCtx.endpointFeatures[endpoint.GetMetadata().ID.String()] = pl.captureEndpointFeatures(endpoint)
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
	if err == nil {
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
				pred.Endpoint.Put(pl.latencyPredictionInfoDataKey, latencyInfo)
				logger.V(logutil.DEBUG).Info("Stored latency prediction in endpoint",
					"pod", pred.Endpoint.GetMetadata().ID.Name,
					"ttft", pred.TTFT,
					"tpot", pred.TPOT,
					"ttftValid", pred.TTFTValid,
					"tpotValid", pred.TPOTValid,
					"ttftHeadroom", pred.TTFTHeadroom,
					"tpotHeadroom", pred.Headroom)
			}
		}
	} else if reason := classifyPredictionError(err); reason != "" {
		llmdRequestPredictionFailures.WithLabelValues(pl.typedName.Name, pl.typedName.Type, reason).Inc()
		pl.failureLog.Do(func() {
			logger.Error(err, "Latency prediction failed", "reason", reason, "endpoints", len(endpoints))
		})
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
	required := make(map[plugin.DataKey]any)
	for _, provider := range featureProviders {
		if provider.dependency != nil && (provider.enabled == nil || provider.enabled(pl)) {
			key, witness := provider.dependency(pl)
			required[key] = witness
		}
	}
	return plugin.DataDependencies{Required: required}
}
