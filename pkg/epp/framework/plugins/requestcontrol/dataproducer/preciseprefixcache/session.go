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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/utils/ptr"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// sessionEventConsumer keeps engine-block availability in the producer's index.
// Association policy and report completeness belong to the configured manager.
type sessionEventConsumer struct {
	index     kvblock.Index
	manager   fwkrc.SessionCacheManager
	mu        sync.RWMutex
	locations map[string][]sessionLocation
}

type sessionLocation struct {
	scope           kvblock.EngineScope
	blockSizeTokens int
}

func (c *sessionEventConsumer) ProcessEvents(ctx context.Context, source kvevents.EventSource, batch kvevents.EventBatch) error {
	entries := []kvblock.PodEntry{{PodIdentifier: source.Endpoint, DeviceTier: "gpu"}}
	for _, event := range batch.Events {
		scope := kvblock.EngineScope{Endpoint: source.Endpoint, DataParallelRank: batch.DataParallelRank}
		switch ev := event.(type) {
		case *kvevents.BlockStoredEvent:
			if !sessionStoreIndexable(ev) || len(ev.BlockHashes) == 0 {
				continue
			}
			scope.GroupIdx = ev.GroupIdx
			scope.CacheNamespace = c.manager.CacheNamespace(source, ev.GroupIdx)
			if scope.CacheNamespace == "" {
				continue
			}
			if err := c.index.Add(ctx, nil, scope.Keys(ev.BlockHashes), entries); err != nil {
				return err
			}
			c.rememberLocation(sessionLocation{scope: scope, blockSizeTokens: ev.BlockSize})
		case *kvevents.BlockRemovedEvent:
			if !localGPUEvent(ev.DeviceTier, ev.Locality, ev.Ownership) {
				continue
			}
			scope.GroupIdx = ev.GroupIdx
			for _, location := range c.sourceLocations(source.Endpoint) {
				if !sameEngineLocation(scope, location.scope) {
					continue
				}
				for _, key := range location.scope.Keys(ev.BlockHashes) {
					if err := c.index.Evict(ctx, key, kvblock.RequestKey, entries); err != nil {
						return err
					}
				}
			}
		case *kvevents.AllBlocksClearedEvent:
			if err := c.clearResidency(ctx, source.Endpoint); err != nil {
				return err
			}
		}
	}
	return c.manager.ProcessEvents(ctx, source, batch)
}

func (c *sessionEventConsumer) Reset(ctx context.Context, endpoint string) error {
	return errors.Join(c.clearResidency(ctx, endpoint), c.manager.Reset(ctx, endpoint))
}

func (c *sessionEventConsumer) clearResidency(ctx context.Context, endpoint string) error {
	c.mu.Lock()
	delete(c.locations, endpoint)
	c.mu.Unlock()
	return c.index.Clear(ctx, endpoint)
}

func (c *sessionEventConsumer) rememberLocation(location sessionLocation) {
	c.mu.Lock()
	defer c.mu.Unlock()
	locations := c.locations[location.scope.Endpoint]
	for i := range locations {
		if sameEngineLocation(locations[i].scope, location.scope) {
			locations[i] = location
			return
		}
	}
	c.locations[location.scope.Endpoint] = append(locations, location)
}

func (c *sessionEventConsumer) sourceLocations(endpoint string) []sessionLocation {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]sessionLocation(nil), c.locations[endpoint]...)
}

func sameEngineLocation(a, b kvblock.EngineScope) bool {
	return a.Endpoint == b.Endpoint && ptr.Equal(a.DataParallelRank, b.DataParallelRank) && ptr.Equal(a.GroupIdx, b.GroupIdx)
}

func (c *sessionEventConsumer) prefixLocations(prefix fwkrc.SessionCachePrefix, endpoints sets.Set[string]) []sessionLocation {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var result []sessionLocation
	for endpoint := range endpoints {
		for _, location := range c.locations[endpoint] {
			if location.scope.CacheNamespace == prefix.CacheNamespace && location.blockSizeTokens == prefix.BlockSizeTokens {
				result = append(result, location)
			}
		}
	}
	return result
}

func sessionStoreIndexable(event *kvevents.BlockStoredEvent) bool {
	if !localGPUEvent(event.DeviceTier, event.Locality, event.Ownership) || event.BlockSize <= 0 {
		return false
	}
	switch event.KVCacheSpecKind {
	case kvevents.KVCacheSpecKindFullAttention, kvevents.KVCacheSpecKindMlaAttention:
		return true
	case "":
		return event.GroupIdx == nil
	default:
		return false
	}
}

func localGPUEvent(tier, locality, ownership string) bool {
	return strings.EqualFold(tier, "gpu") && ownership == "" &&
		(locality == "" || strings.EqualFold(locality, "local"))
}

func (p *Producer) produceFromSession(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	if request == nil {
		return nil
	}
	lookup, ok := scheduling.ReadRequestAttribute[fwkrc.SessionCacheRequest](request, p.sessionDataKey)
	if !ok || lookup.Stamp == "" {
		return nil
	}
	if lookup.TotalTokens < 0 {
		return errors.New("session cache lookup totalTokens must be nonnegative")
	}
	endpointSet := extractEndpointSet(endpoints)
	infos := make(map[string]*attrprefix.PrefixCacheMatchInfo)
	for _, prefix := range lookup.Prefixes {
		if prefix.CacheNamespace == "" || len(prefix.BlockHashes) == 0 {
			continue
		}
		if prefix.BlockSizeTokens <= 0 {
			return errors.New("session cache prefix blockSizeTokens must be positive")
		}
		for _, location := range p.sessionEvents.prefixLocations(prefix, endpointSet) {
			info, err := p.matchSessionPrefix(ctx, prefix, location.scope, lookup.TotalTokens)
			if err != nil {
				return err
			}
			previous := infos[location.scope.Endpoint]
			if previous == nil || info.MatchBlocks() > previous.MatchBlocks() ||
				(info.MatchBlocks() == previous.MatchBlocks() && info.CachedBlockCount() > previous.CachedBlockCount()) {
				infos[location.scope.Endpoint] = info
			}
		}
	}
	results := make([]endpointResult, 0, len(endpoints))
	for _, ep := range endpoints {
		md := ep.GetMetadata()
		if md == nil {
			continue
		}
		info := infos[fmt.Sprintf("%s:%s", md.Address, md.Port)]
		if info == nil {
			info = attrprefix.NewPrefixCacheMatchInfo(0, lookup.TotalTokens, 1).
				WithCachedBlocksByTier(map[string]int{}).
				WithInputTokenCount(lookup.TotalTokens).
				WithObservedTokenCount(0)
		}
		results = append(results, endpointResult{endpoint: ep, info: info})
	}
	return p.publishEndpointResults(ctx, results)
}

func (p *Producer) matchSessionPrefix(ctx context.Context, prefix fwkrc.SessionCachePrefix, scope kvblock.EngineScope, totalTokens int) (*attrprefix.PrefixCacheMatchInfo, error) {
	hashes := prefix.BlockHashes
	if prefix.Exact {
		hashes = hashes[:min(len(hashes), totalTokens/prefix.BlockSizeTokens)]
	}
	matches, err := p.kvCacheIndexer.MatchBlockKeys(ctx, scope.Keys(hashes), sets.New(scope.Endpoint))
	if err != nil {
		return nil, fmt.Errorf("match session cache prefix: %w", err)
	}
	match := matches[scope.Endpoint]
	matchedTokens := min(totalTokens, match.MatchedBlocks*prefix.BlockSizeTokens)
	score := min(totalTokens, int(match.WeightedScore*float64(prefix.BlockSizeTokens)))
	// Unit-size blocks preserve token coverage across engine block sizes.
	info := attrprefix.NewPrefixCacheMatchInfo(score, totalTokens, 1).
		WithCachedBlockCount(0).
		WithCachedBlocksByTier(map[string]int{}).
		WithInputTokenCount(totalTokens).
		WithObservedTokenCount(match.MatchedBlocks * prefix.BlockSizeTokens)
	if prefix.Exact {
		info.WithCachedBlockCount(matchedTokens).
			WithCachedBlocksByTier(map[string]int{"gpu": matchedTokens})
	}
	return info, nil
}

func (p *Producer) prepareSessionRequest(request *scheduling.InferenceRequest) error {
	if request == nil {
		return nil
	}
	lookup, ok := scheduling.ReadRequestAttribute[fwkrc.SessionCacheRequest](request, p.sessionDataKey)
	if !ok || lookup.Stamp == "" {
		return nil
	}
	if request.Body == nil {
		return errors.New("session cache stamping requires a JSON request body")
	}
	payload, ok := request.Body.Payload.(fwkrh.PayloadMap)
	if !ok {
		return errors.New("session cache stamping requires a JSON request envelope")
	}
	xargs := map[string]any{}
	switch existing := payload["vllm_xargs"].(type) {
	case nil:
	case map[string]any:
		xargs = maps.Clone(existing)
	case fwkrh.PayloadMap:
		xargs = maps.Clone(map[string]any(existing))
	case json.RawMessage:
		decoder := json.NewDecoder(bytes.NewReader(existing))
		decoder.UseNumber()
		if err := decoder.Decode(&xargs); err != nil || xargs == nil {
			return errors.New("session cache stamping requires an object for vllm_xargs")
		}
	default:
		return errors.New("session cache stamping requires an object for vllm_xargs")
	}
	if xargs == nil {
		xargs = map[string]any{}
	}
	mode := "incremental"
	if lookup.FullReport {
		mode = "full"
	}
	xargs["kv_cache_report_mode"] = mode
	request.Body.MutatePayloadMap(func(payload fwkrh.PayloadMap) {
		payload["session_id"] = lookup.Stamp
		payload["vllm_xargs"] = xargs
	})
	return nil
}
