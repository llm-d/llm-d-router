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
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// recordCacheLocality compares the prompt-cached fraction the router predicted for
// the endpoint that served this request against the fraction the engine reported in
// the response's usage block, and records exactly one outcome per request.
//
// It is called only from the end-of-stream branch of ResponseBody: the actual
// fraction is final only once the engine has finished reporting usage, and observing
// it per chunk would both double count the request and compare against a partial
// usage block.
//
// The prediction is the value Produce captured before the routing decision, not a
// fresh read of the endpoint attribute — that attribute is per-pod state the next
// request's Produce overwrites, so re-reading it here would judge this request's
// usage against some later request's match.
//
// A response that reported no cached-token detail is recorded as missing_usage
// rather than as an actual fraction of 0, so an engine that never reports reuse
// stays distinguishable from a genuine 0% cache hit. Non-observed outcomes carry no
// histogram observation at all; the predictions and actuals therefore share one
// denominator, the outcome=observed series of the observations counter.
func (pl *PredictedLatency) recordCacheLocality(
	ctx context.Context,
	request *fwksched.InferenceRequest,
	response *requestcontrol.Response,
	predictedLatencyCtx *predictedLatencyCtx,
) {
	logger := log.FromContext(ctx)
	if request == nil || response == nil {
		return
	}

	endpoint := localityTargetEndpoint(predictedLatencyCtx)
	if endpoint == nil || endpoint.ID.Name == "" {
		// PreRequest captured no target: the request was never dispatched through a
		// scheduling result, so there is no selected endpoint to judge.
		logger.V(logutil.TRACE).Info("Skipping cache locality observation: no target endpoint captured")
		eppmetrics.RecordCacheLocalityObservation("", request.TargetModel, 0, 0, eppmetrics.CacheLocalityMissingPrediction)
		return
	}

	endpointName := endpoint.ID.Name
	targetModel := request.TargetModel
	if !predictedLatencyCtx.prefixCacheScoreKnownForEndpoints[endpointName] {
		logger.V(logutil.TRACE).Info("Skipping cache locality observation: no prediction captured for target endpoint",
			"pod", endpointName)
		eppmetrics.RecordCacheLocalityObservation(endpointName, targetModel, 0, 0, eppmetrics.CacheLocalityMissingPrediction)
		return
	}
	predicted := predictedLatencyCtx.prefixCacheScoresForEndpoints[endpointName]

	// The parsed type carries no presence bit (requesthandling.PromptTokenDetails
	// holds a plain int CachedTokens), so a nil pointer is the only way a parser can
	// say "the engine never reported cache reuse" — and each parser must leave it nil
	// in that case, rather than allocating the struct for a details block that
	// reports other counters but not cached_tokens. A non-nil pointer with
	// CachedTokens 0 is therefore read as a genuine 0% hit. The vLLM gRPC parser
	// cannot make that distinction: its cached_tokens is a proto3 uint32, where
	// unset and 0 are the same wire value, so it always reports a definite number.
	usage := response.Usage
	if usage.PromptTokenDetails == nil {
		logger.V(logutil.TRACE).Info("Skipping cache locality observation: response reported no cached tokens",
			"pod", endpointName)
		eppmetrics.RecordCacheLocalityObservation(endpointName, targetModel, 0, 0, eppmetrics.CacheLocalityMissingUsage)
		return
	}
	cached := usage.PromptTokenDetails.CachedTokens
	prompt := usage.PromptTokens
	if prompt <= 0 || cached < 0 || cached > prompt ||
		math.IsNaN(predicted) || predicted < 0 || predicted > 1 {
		logger.V(logutil.DEBUG).Info("Ignoring cache locality observation: usage or prediction is not a valid ratio",
			"pod", endpointName, "promptTokens", prompt, "cachedTokens", cached, "predicted", predicted)
		eppmetrics.RecordCacheLocalityObservation(endpointName, targetModel, 0, 0, eppmetrics.CacheLocalityInvalidUsage)
		return
	}

	actual := float64(cached) / float64(prompt)
	eppmetrics.RecordCacheLocalityObservation(endpointName, targetModel, predicted, actual, eppmetrics.CacheLocalityObserved)
	if debug := logger.V(logutil.DEBUG); debug.Enabled() {
		debug.Info("Recorded cache locality observation",
			"pod", endpointName, "predicted", predicted, "actual", actual,
			"promptTokens", prompt, "cachedTokens", cached)
	}
}

// localityTargetEndpoint returns the endpoint whose prefix-cache match the request's
// predicted fraction describes: the prefill target under disaggregation (the endpoint
// the prompt was matched against), otherwise the decode/primary target — the same
// choice the TTFT training record makes.
func localityTargetEndpoint(predictedLatencyCtx *predictedLatencyCtx) *fwkdl.EndpointMetadata {
	if predictedLatencyCtx == nil {
		return nil
	}
	if metadata := predictedLatencyCtx.prefillTargetMetadata; metadata != nil {
		return metadata
	}
	return predictedLatencyCtx.targetMetadata
}
