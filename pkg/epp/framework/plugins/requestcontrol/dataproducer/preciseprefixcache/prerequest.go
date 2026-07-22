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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	approximateprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix"
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
// prefixCacheMatches is a map of pod identifier to its longest prefix cache match length.
type blockKeysState struct {
	perPromptKeys           [][]kvblock.BlockHash
	prefixCacheMatchLengths map[string]float64
}

// Clone implements plugin.StateData.
func (s *blockKeysState) Clone() plugin.StateData {
	cp := make([][]kvblock.BlockHash, len(s.perPromptKeys))
	pm := make(map[string]float64, len(s.prefixCacheMatchLengths))
	for k, v := range s.prefixCacheMatchLengths {
		pm[k] = v
	}
	for i, keys := range s.perPromptKeys {
		cp[i] = make([]kvblock.BlockHash, len(keys))
		copy(cp[i], keys)
	}
	return &blockKeysState{perPromptKeys: cp, prefixCacheMatchLengths: pm}
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

// PreRequest seeds speculative KV-block index entries for the endpoint(s)
// selected by the scheduler, so the next same-prefix request hits without
// waiting for confirmed KV-events from the engine. Entries are tracked in
// a TTL cache and evicted automatically. No-op when speculativeIndexing
// is disabled.
func (p *Producer) PreRequest(ctx context.Context,
	request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult,
) {
	if !p.speculativeEnabled {
		return
	}

	logger := log.FromContext(ctx).WithName(p.typedName.String())

	state, err := plugin.ReadPluginStateKey[*blockKeysState](
		p.pluginState, request.RequestID, blockKeysStateKey)
	if err != nil {
		logger.V(logging.TRACE).Info("No plugin state for PreRequest, skipping speculative indexing",
			"requestID", request.RequestID)
		return
	}
	p.pluginState.Delete(request.RequestID)

	hasKeys := false
	for _, pk := range state.perPromptKeys {
		if len(pk) > 0 {
			hasKeys = true
			break
		}
	}
	if !hasKeys {
		return
	}

	primary := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return
	}
	targetEndpoint := primary.TargetEndpoints[0]
	targetMeta := targetEndpoint.GetMetadata()
	if targetMeta == nil {
		return
	}
	speculativePod := kvblock.PodEntry{
		PodIdentifier: fmt.Sprintf("%s:%s", targetMeta.Address, targetMeta.Port),
		Speculative:   true,
	}
	allPodEntries := []kvblock.PodEntry{speculativePod}
	totalBlocks := 0
	index := p.kvCacheIndexer.KVBlockIndex()
	// Insert per-prompt keys separately to preserve correct block adjacency.
	for _, promptKeys := range state.perPromptKeys {
		if err := index.Add(ctx, nil, promptKeys, allPodEntries); err != nil {
			logger.Error(err, "Failed to add speculative entries to index",
				"pod", speculativePod.PodIdentifier)
		}
		totalBlocks += len(promptKeys)
	}
	matchLen := state.prefixCacheMatchLengths[speculativePod.PodIdentifier]
	approximateprefix.RecordPrefixCacheMatch(p.typedName.Name, p.typedName.Type, int(matchLen), totalBlocks)

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
}
