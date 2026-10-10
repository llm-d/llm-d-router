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
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/observability/multimodal"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	rcplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixmetrics"
)

const (
	defaultSpeculativeTTL      = 2 * time.Second
	experimentalPrefillProfile = "prefill"
	blockKeysStateKey          = plugin.StateKey("precise-prefix-cache-producer.block-keys")
	bestAvailableStateKey      = plugin.StateKey("precise-prefix-cache-producer.best-available")
)

var _ requestcontrol.PreRequest = &Producer{}

// speculativeEntries records the speculative rows added on a routing decision
// so the TTL-eviction callback can roll them back.
type speculativeEntries struct {
	perPromptKeys [][]kvblock.BlockHash
	podEntries    []kvblock.PodEntry
}

// blockKeysState carries the block keys computed in Produce to PreRequest
// via PluginState, avoiding a second hash on the same request.
// perPromptKeys holds one slice of block keys per prompt; single-prompt
// requests use a length-1 outer slice.
type blockKeysState struct {
	perPromptKeys [][]kvblock.BlockHash
}

// Clone implements plugin.StateData.
func (s *blockKeysState) Clone() plugin.StateData {
	cp := make([][]kvblock.BlockHash, len(s.perPromptKeys))
	for i, keys := range s.perPromptKeys {
		cp[i] = make([]kvblock.BlockHash, len(keys))
		copy(cp[i], keys)
	}
	return &blockKeysState{perPromptKeys: cp}
}

// bestAvailableState carries the highest prediction across the request's
// candidate endpoints from Produce to PreRequest. Produce is the only stage
// that sees those candidates before the scheduler's filters narrow them.
type bestAvailableState struct {
	cachedTokens int
}

// Clone implements plugin.StateData.
func (s *bestAvailableState) Clone() plugin.StateData {
	cp := *s
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
	profile, role := prefixmetrics.PredictionTarget(schedulingResult, experimentalPrefillProfile)
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
	bestPredicted := selected
	for _, candidate := range profile.ScoredCandidates {
		if candidateInfo, ok := p.matchInfo(candidate.Endpoint); ok {
			bestPredicted = max(bestPredicted, predictedCachedTokens(candidateInfo))
		}
	}

	bestAvailable := bestPredicted
	if len(profile.ScoredCandidates) > 0 {
		if state, err := plugin.ReadPluginStateKey[*bestAvailableState](
			p.pluginState, request.RequestID, bestAvailableStateKey); err == nil {
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
	p.recordMMPrediction(ctx, request, role, info)
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

// buildSpeculativeCache constructs the TTL cache used to evict speculative
// index entries. Returns (nil, 0, nil) when speculative indexing is disabled.
// The cache and its background goroutine are bound to ctx.
func buildSpeculativeCache(ctx context.Context, config PluginConfig,
	index kvblock.Index,
) (*ttlcache.Cache[string, *speculativeEntries], time.Duration, error) {
	if !config.SpeculativeIndexing {
		return nil, 0, nil
	}

	ttl := defaultSpeculativeTTL
	if config.SpeculativeTTL != "" {
		parsed, err := time.ParseDuration(config.SpeculativeTTL)
		if err != nil {
			return nil, 0, fmt.Errorf("invalid speculativeTTL %q: %w", config.SpeculativeTTL, err)
		}
		if parsed > 0 {
			ttl = parsed
		}
	}

	cache := ttlcache.New[string, *speculativeEntries](
		ttlcache.WithTTL[string, *speculativeEntries](ttl),
	)
	logger := log.FromContext(ctx).WithName(PluginType)
	cache.OnEviction(func(_ context.Context, reason ttlcache.EvictionReason,
		item *ttlcache.Item[string, *speculativeEntries],
	) {
		if reason != ttlcache.EvictionReasonExpired {
			return
		}
		entries := item.Value()
		keys := make([]kvblock.BlockHash, 0)
		for _, promptKeys := range entries.perPromptKeys {
			keys = append(keys, promptKeys...)
		}
		if len(keys) == 0 || len(entries.podEntries) == 0 {
			return
		}
		// On failure the entries stay until the pod's next Clear, or until the same
		// prefix is routed to the same pod again and that TTL expiry succeeds.
		if err := index.Evict(ctx, kvblock.RequestKey, keys, entries.podEntries); err != nil {
			logger.Error(err, "Failed to evict speculative entries on TTL expiry",
				"requestID", item.Key(), "keys", len(keys))
		}
	})
	go cache.Start()
	go func() {
		<-ctx.Done()
		cache.Stop()
	}()

	return cache, ttl, nil
}

// PreRequest records the prefix-cache hit predicted for the selected endpoint,
// then seeds speculative KV-block index entries for the endpoint(s) selected by
// the scheduler, so the next same-prefix request hits without waiting for
// confirmed KV-events from the engine. Speculative entries are tracked in a TTL
// cache and evicted automatically; seeding is skipped when speculativeIndexing
// is disabled.
func (p *Producer) PreRequest(ctx context.Context,
	request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult,
) error {
	// Produce writes state on every request, so the cleanup cannot sit behind
	// the speculative-indexing gate.
	defer p.pluginState.Delete(request.RequestID)

	ctx, span := tracing.Tracer(rcplugins.TracerScope).Start(ctx, "pre_request_precise_prefix_cache",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	p.recordPrediction(ctx, request, schedulingResult)

	if !p.speculativeEnabled {
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

	hasKeys := false
	for _, pk := range state.perPromptKeys {
		if len(pk) > 0 {
			hasKeys = true
			break
		}
	}
	if !hasKeys {
		return nil
	}

	targetEndpoint := schedulingResult.PrimaryEndpoint()
	if targetEndpoint == nil {
		return nil
	}
	targetMeta := targetEndpoint.GetMetadata()
	if targetMeta == nil {
		return nil
	}
	speculativePod := kvblock.PodEntry{
		PodIdentifier: fmt.Sprintf("%s:%s", targetMeta.Address, targetMeta.Port),
		Speculative:   true,
	}

	index := p.kvCacheIndexer.KVBlockIndex()
	// Insert per-prompt keys separately to preserve correct block adjacency.
	for _, promptKeys := range state.perPromptKeys {
		if err := index.Add(ctx, nil, promptKeys, []kvblock.PodEntry{speculativePod}); err != nil {
			logger.Error(err, "Failed to add speculative entries to index",
				"pod", speculativePod.PodIdentifier)
		}
	}

	allPodEntries := []kvblock.PodEntry{speculativePod}

	// P/D disagg: seed the prefill endpoint too.
	if prefill := schedulingResult.ProfileResults[experimentalPrefillProfile].FirstEndpoint(); prefill != nil {
		if prefillMeta := prefill.GetMetadata(); prefillMeta != nil {
			prefillPod := kvblock.PodEntry{
				PodIdentifier: fmt.Sprintf("%s:%s", prefillMeta.Address, prefillMeta.Port),
				Speculative:   true,
			}
			for _, promptKeys := range state.perPromptKeys {
				if err := index.Add(ctx, nil, promptKeys, []kvblock.PodEntry{prefillPod}); err != nil {
					logger.Error(err, "Failed to add speculative entries for prefill endpoint",
						"pod", prefillPod.PodIdentifier)
				}
			}
			allPodEntries = append(allPodEntries, prefillPod)
		}
	}

	p.speculativeCache.Set(request.RequestID, &speculativeEntries{
		perPromptKeys: state.perPromptKeys,
		podEntries:    allPodEntries,
	}, p.speculativeTTL)

	logger.V(logging.TRACE).Info("Added speculative entries",
		"requestID", request.RequestID,
		"pod", speculativePod.PodIdentifier,
		"prompts", len(state.perPromptKeys),
		"ttl", p.speculativeTTL)
	return nil
}
