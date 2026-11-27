/*
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

package preciseprefixcache

import (
	"context"
	"fmt"
	"maps"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/common/observability/multimodal"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	rhplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixmetrics"
)

const (
	defaultSpeculativeTTL = 2 * time.Second
	blockKeysStateKey     = plugin.StateKey("precise-prefix-cache-producer.block-keys")
	bestAvailableStateKey = plugin.StateKey("precise-prefix-cache-producer.best-available")
)

var _ fwkrh.PreRequest = &Producer{}

// parseSpeculativeTTL parses the configured hint lifetime. Empty and
// non-positive values select defaultSpeculativeTTL.
func parseSpeculativeTTL(raw string) (time.Duration, error) {
	if raw == "" {
		return defaultSpeculativeTTL, nil
	}
	parsed, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid speculativeTTL %q: %w", raw, err)
	}
	if parsed <= 0 {
		return defaultSpeculativeTTL, nil
	}
	return parsed, nil
}

// blockKeysState carries the block keys computed in Produce to PreRequest
// via PluginState, avoiding a second hash on the same request.
type blockKeysState struct {
	perFormatKeys map[kvblock.CacheFormat][][]kvblock.BlockHash
}

// Clone implements plugin.StateData.
func (s *blockKeysState) Clone() plugin.StateData {
	cp := make(map[kvblock.CacheFormat][][]kvblock.BlockHash, len(s.perFormatKeys))
	for format, prompts := range s.perFormatKeys {
		keys := make([][]kvblock.BlockHash, len(prompts))
		for i, prompt := range prompts {
			keys[i] = append([]kvblock.BlockHash(nil), prompt...)
		}
		cp[format] = keys
	}
	return &blockKeysState{perFormatKeys: cp}
}

// bestAvailableState carries candidate predictions and their maxima from
// Produce to PreRequest, before the scheduler's filters narrow the candidates.
type bestAvailableState struct {
	cachedTokens          int
	predictedCachedTokens map[datalayer.ID]int
	gpuCachedTokens       map[datalayer.ID]int
	maxGPUCachedTokens    int
	maxConfirmedTokens    int
}

// Clone implements plugin.StateData.
func (s *bestAvailableState) Clone() plugin.StateData {
	cp := *s
	cp.predictedCachedTokens = maps.Clone(s.predictedCachedTokens)
	cp.gpuCachedTokens = maps.Clone(s.gpuCachedTokens)
	return &cp
}

// predictedCachedTokens converts a match into the prompt tokens the index
// expects the endpoint to serve from its prefix cache. It reads the unweighted
// cached-block count rather than the tier-weighted match score, so a RAM-tier
// hit contributes its full token count, and it counts speculative entries
// because those are part of what the router acted on. The token processor drops
// a prompt's trailing partial block, so the conversion cannot exceed the prompt
// length.
func predictedCachedTokens(info *attrprefix.PrefixCacheMatchInfo) int {
	return info.CachedBlockCount() * info.BlockSizeTokens()
}

// gpuCachedTokens converts a match into the contiguous prompt tokens the index
// holds in the endpoint's GPU memory.
func gpuCachedTokens(info *attrprefix.PrefixCacheMatchInfo) int {
	return info.CachedBlocksByTier()[kvcache.GPUTier] * info.BlockSizeTokens()
}

// confirmedCachedTokens converts a match into the contiguous prompt tokens the
// engine confirmed in any device tier.
func confirmedCachedTokens(info *attrprefix.PrefixCacheMatchInfo) int {
	return info.ConfirmedCachedBlockCount() * info.BlockSizeTokens()
}

// matchInfo reads the match the producer attached to an endpoint.
func (p *Producer) matchInfo(endpoint scheduling.Endpoint) (*attrprefix.PrefixCacheMatchInfo, bool) {
	raw, ok := endpoint.Get(p.dk)
	if !ok {
		return nil, false
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	return info, ok
}

// recordPrediction reports the prediction for the endpoint chosen by
// prefixmetrics.PredictionTarget against the best its profile's picker could
// have chosen and the best any candidate held before filtering, so the reuse a
// routing decision left behind is separable from the reuse filtering put out of
// reach.
func (p *Producer) recordPrediction(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	profile, role := prefixmetrics.PredictionTarget(schedulingResult)
	if profile == nil {
		return
	}
	info, ok := p.matchInfo(profile.TargetEndpoints[0])
	if !ok {
		return
	}
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return
	}
	selected := predictedCachedTokens(info)
	// A profile that reports no scored candidates leaves only the chosen
	// endpoint to go on, so selected stands in for both maxima. That keeps the
	// histograms on the same requests, at the cost of reading as a perfect
	// routing decision.
	bestPredicted, bestAvailable := selected, selected
	if len(profile.ScoredCandidates) > 0 {
		state, err := plugin.ReadPluginStateKey[*bestAvailableState](
			p.pluginState, request.RequestID, bestAvailableStateKey)
		for _, candidate := range profile.ScoredCandidates {
			if err == nil {
				if md := candidate.GetMetadata(); md != nil {
					bestPredicted = max(bestPredicted, state.predictedCachedTokens[md.ID])
				}
			} else if candidateInfo, ok := p.matchInfo(candidate.Endpoint); ok {
				bestPredicted = max(bestPredicted, predictedCachedTokens(candidateInfo))
			}
		}
		bestAvailable = bestPredicted
		if err == nil {
			bestAvailable = max(bestAvailable, state.cachedTokens)
		}
	}

	modality, _ := mmobs.Summary(request)
	prefixmetrics.RecordPrediction(p.typedName.Name, p.typedName.Type, role, modality, prefixmetrics.Prediction{
		Selected:      selected,
		BestPredicted: bestPredicted,
		BestAvailable: bestAvailable,
		PromptTokens:  request.Body.TokenizedRequest.TokenCount(),
	})
	p.recordGPUMiss(request, profile, role, info)
	p.recordMMPrediction(ctx, request, role, info)
}

// recordGPUMiss attributes the prompt tokens the chosen endpoint lacks in GPU
// memory to where the index holds them. Without scored candidates or Produce
// state the chosen endpoint stands in for the maxima, as in recordPrediction.
func (p *Producer) recordGPUMiss(request *scheduling.InferenceRequest, profile *scheduling.ProfileRunResult,
	role string, info *attrprefix.PrefixCacheMatchInfo,
) {
	selected, confirmed := gpuCachedTokens(info), confirmedCachedTokens(info)
	miss := prefixmetrics.GPUMiss{
		Selected:               selected,
		BestPredicted:          selected,
		BestAvailable:          selected,
		SelectedConfirmed:      confirmed,
		BestAvailableConfirmed: confirmed,
		PromptTokens:           request.Body.TokenizedRequest.TokenCount(),
	}
	state, err := plugin.ReadPluginStateKey[*bestAvailableState](
		p.pluginState, request.RequestID, bestAvailableStateKey)
	if err == nil && len(profile.ScoredCandidates) > 0 {
		for _, candidate := range profile.ScoredCandidates {
			if md := candidate.GetMetadata(); md != nil {
				miss.BestPredicted = max(miss.BestPredicted, state.gpuCachedTokens[md.ID])
			}
		}
		miss.BestAvailable = max(miss.BestPredicted, state.maxGPUCachedTokens)
		miss.BestAvailableConfirmed = max(confirmed, state.maxConfirmedTokens)
	}
	prefixmetrics.RecordGPUMiss(p.typedName.Name, p.typedName.Type, role, miss)
}

// recordMMPrediction reports the multimodal prompt tokens the index expects
// the endpoint to serve from its prefix cache, measured against the request's
// multimodal token total. Match info without multimodal attribution covers a
// text-only request, which records nothing, so a zero observation means a
// multimodal request matched no blocks. The producer counts each feature's
// tokens inside the matched prefix, so a feature that starts or ends
// mid-block contributes only the tokens it holds.
func (p *Producer) recordMMPrediction(ctx context.Context, request *scheduling.InferenceRequest, role string, info *attrprefix.PrefixCacheMatchInfo) {
	mm := info.MM()
	if mm == nil {
		return
	}
	if span := trace.SpanFromContext(ctx); span.IsRecording() {
		span.SetAttributes(
			mmMatchedBlocksKey.Int(mm.MatchBlocks),
			mmTotalBlocksKey.Int(totalMMBlocks(request, info.BlockSizeTokens())),
		)
	}
	mmPromptTokens := 0
	for _, prompt := range request.Body.TokenizedRequest.Prompts {
		for _, feature := range prompt.MultiModalFeatures {
			mmPromptTokens += feature.Length
		}
	}
	if mmPromptTokens == 0 {
		return
	}
	prefixmetrics.RecordMMPrediction(p.typedName.Name, p.typedName.Type, role,
		mm.MatchTokens, mmPromptTokens)
}

// PreRequest records the prefix-cache hit predicted for the selected endpoint,
// then hints the request's block keys as held by the endpoint(s) the scheduler
// selected, so the next same-prefix request matches them without waiting for
// confirmed KV-events from the engine. Hints expire after speculativeTTL;
// hinting is skipped when speculativeIndexing is disabled.
func (p *Producer) PreRequest(ctx context.Context,
	request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult,
) error {
	// Produce writes state on every request, so the cleanup cannot sit behind
	// the speculative-indexing gate.
	defer p.pluginState.Delete(request.RequestID)

	ctx, span := tracing.Tracer(rhplugins.TracerScope).Start(ctx, "pre_request_precise_prefix_cache",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	p.recordPrediction(ctx, request, schedulingResult)

	if p.speculative == nil {
		return nil
	}

	logger := log.FromContext(ctx).WithName(p.typedName.String())

	state, err := plugin.ReadPluginStateKey[*blockKeysState](
		p.pluginState, request.RequestID, blockKeysStateKey)
	if err != nil {
		logger.V(logging.TRACE).Info("No plugin state for PreRequest, skipping speculative indexing",
			"requestID", request.RequestID)
		return nil
	}
	if len(state.perFormatKeys) == 0 {
		return nil
	}

	targetEndpoint := schedulingResult.PrimaryEndpoint()
	if targetEndpoint == nil || targetEndpoint.GetMetadata() == nil {
		return nil
	}
	targets := []scheduling.Endpoint{targetEndpoint}
	if prefill := schedulingResult.PrefillProfile().FirstEndpoint(); prefill != nil {
		if p.seedDecode {
			targets = append(targets, prefill)
		} else {
			targets = []scheduling.Endpoint{prefill}
		}
	}
	for _, target := range targets {
		md := target.GetMetadata()
		if md == nil {
			continue
		}
		podID := p.indexPodIdentifier(md)
		format := p.kvCacheIndexer.CacheFormat(podID)
		engineBlockSize := 0
		if metrics := target.GetMetrics(); metrics != nil {
			engineBlockSize = metrics.CacheBlockSize
		}
		kept, dropped := 0, 0
		// Each prompt's keys form their own chain.
		for _, promptKeys := range state.perFormatKeys[format] {
			n := engineReusableKeys(len(promptKeys), format.BlockSize, engineBlockSize)
			kept += n
			dropped += len(promptKeys) - n
			p.speculative.Add(promptKeys[:n], podID)
		}
		logger.V(logging.TRACE).Info("Speculative keys within whole engine blocks",
			"requestID", request.RequestID,
			"pod", podID,
			"blockSize", format.BlockSize,
			"engineBlockSize", engineBlockSize,
			"keptKeys", kept,
			"droppedKeys", dropped)
	}
	return nil
}

// engineReusableKeys returns how many of numKeys leading blocks of blockSize
// tokens fall within whole engine blocks of engineBlockSize tokens. The engine
// reuses only whole blocks, so an entry past the last one credits tokens the
// engine recomputes. An engineBlockSize of 0, from an endpoint that reports no
// block size, keeps every key.
func engineReusableKeys(numKeys, blockSize, engineBlockSize int) int {
	if engineBlockSize <= 0 {
		return numKeys
	}
	return numKeys * blockSize / engineBlockSize * engineBlockSize / blockSize
}

// RequestAbandoned releases the per-request state when the request is abandoned
// before reaching PreRequest.
func (p *Producer) RequestAbandoned(_ context.Context, request *scheduling.InferenceRequest) {
	p.pluginState.Delete(request.RequestID)
}
