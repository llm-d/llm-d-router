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
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	defaultSpeculativeTTL      = 2 * time.Second
	experimentalPrefillProfile = "prefill"
	blockKeysStateKey          = plugin.StateKey("precise-prefix-cache-producer.block-keys")
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
	repairMatches map[string]repairMatch
}

type repairMatch struct {
	total     int
	confirmed int
}

// Clone implements plugin.StateData.
func (s *blockKeysState) Clone() plugin.StateData {
	cp := make([][]kvblock.BlockHash, len(s.perPromptKeys))
	for i, keys := range s.perPromptKeys {
		cp[i] = make([]kvblock.BlockHash, len(keys))
		copy(cp[i], keys)
	}
	matches := make(map[string]repairMatch, len(s.repairMatches))
	for endpoint, match := range s.repairMatches {
		matches[endpoint] = match
	}
	return &blockKeysState{perPromptKeys: cp, repairMatches: matches}
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
	cache.OnEviction(func(_ context.Context, reason ttlcache.EvictionReason,
		item *ttlcache.Item[string, *speculativeEntries],
	) {
		if reason != ttlcache.EvictionReasonExpired {
			return
		}
		entries := item.Value()
		for _, promptKeys := range entries.perPromptKeys {
			for _, reqKey := range promptKeys {
				//nolint:errcheck // best-effort cleanup on TTL expiry
				index.Evict(ctx, reqKey, kvblock.RequestKey, entries.podEntries)
			}
		}
	})
	go cache.Start()
	go func() {
		<-ctx.Done()
		cache.Stop()
	}()

	return cache, ttl, nil
}

// PreRequest may request a bounded full KV-cache report for an eligible,
// materially under-indexed selected endpoint, then seeds speculative entries
// when speculative indexing is enabled.
func (p *Producer) PreRequest(ctx context.Context,
	request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult,
) error {
	if !p.speculativeEnabled && p.fullReportRepair == nil {
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
	p.pluginState.Delete(request.RequestID)
	// Decide from the confirmed lookup captured before speculative entries are added.
	p.requestFullReportIfNeeded(ctx, request, schedulingResult, state)

	if !p.speculativeEnabled {
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

	primary := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return nil
	}
	targetEndpoint := primary.TargetEndpoints[0]
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
	if pr, exists := schedulingResult.ProfileResults[experimentalPrefillProfile]; exists && len(pr.TargetEndpoints) > 0 {
		if prefillMeta := pr.TargetEndpoints[0].GetMetadata(); prefillMeta != nil {
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

func (p *Producer) requestFullReportIfNeeded(ctx context.Context, request *scheduling.InferenceRequest,
	schedulingResult *scheduling.SchedulingResult, state *blockKeysState,
) {
	if p.fullReportRepair == nil || request == nil || request.Body == nil ||
		request.Body.Payload == nil || schedulingResult == nil {
		return
	}
	// A disaggregated prefill endpoint owns the cache state being repaired.
	selected := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if prefill, ok := schedulingResult.ProfileResults[experimentalPrefillProfile]; ok &&
		prefill != nil && len(prefill.TargetEndpoints) > 0 {
		selected = prefill
	}
	if selected == nil || len(selected.TargetEndpoints) == 0 {
		return
	}
	metadata := selected.TargetEndpoints[0].GetMetadata()
	if metadata == nil {
		return
	}
	endpoint := fmt.Sprintf("%s:%s", metadata.Address, metadata.Port)
	match, ok := state.repairMatches[endpoint]
	if !ok {
		return
	}
	payload, ok := request.Body.Payload.AsMap()
	if !ok {
		return
	}
	// Preserve any caller-supplied vLLM arguments.
	var xargs map[string]any
	setXArgs := false
	switch existing := payload["vllm_xargs"].(type) {
	case nil:
		xargs = map[string]any{}
		setXArgs = true
	case map[string]any:
		xargs = existing
	case fwkrh.PayloadMap:
		xargs = map[string]any(existing)
	default:
		log.FromContext(ctx).V(logging.DEBUG).Info("Skipping full report repair for malformed vllm_xargs",
			"requestID", request.RequestID, "endpoint", endpoint)
		return
	}
	requestFull, reason := p.fullReportRepair.shouldRequest(endpoint, match)
	if !requestFull {
		return
	}
	// Use the body mutator so every protocol serializer sees the new argument.
	request.Body.MutatePayloadMap(func(payload fwkrh.PayloadMap) {
		if setXArgs {
			payload["vllm_xargs"] = xargs
		}
		xargs["kv_cache_report_mode"] = "full"
	})
	log.FromContext(ctx).V(logging.DEBUG).Info("Requested full KV-cache report",
		"requestID", request.RequestID, "endpoint", endpoint, "reason", reason,
		"totalBlocks", match.total, "confirmedBlocks", match.confirmed)
}
