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

package sessionprefixcache

import (
	"context"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

const gpuTier = "gpu"

var _ kvevents.EventConsumer = &eventConsumer{}

// eventConsumer keeps engine-block residency in the producer's index, keyed by
// the physical cache that reported each block, and remembers the caches seen
// per endpoint so lookups can query each one. It is the pool's consumer:
// events arrive ordered per endpoint and concurrently across endpoints.
type eventConsumer struct {
	index kvblock.Index

	mu       sync.RWMutex
	observed map[string][]observedScope // by endpoint
}

// observedScope is one physical cache seen in events and the block size it reports.
type observedScope struct {
	scope           kvblock.EngineScope
	blockSizeTokens int
}

func newEventConsumer(index kvblock.Index) *eventConsumer {
	return &eventConsumer{index: index, observed: make(map[string][]observedScope)}
}

// ProcessEvents records local GPU block stores as resident in their scope,
// removes residency for local GPU block removals, and drops the endpoint's
// residency on AllBlocksCleared. Reports for other tiers, remote or offloaded
// blocks, and cache kinds other than full or MLA attention leave residency
// unchanged.
func (c *eventConsumer) ProcessEvents(ctx context.Context, source kvevents.EventSource, batch kvevents.EventBatch) error {
	entries := []kvblock.PodEntry{{PodIdentifier: source.Endpoint, DeviceTier: gpuTier}}
	for _, event := range batch.Events {
		scope := kvblock.EngineScope{Endpoint: source.Endpoint, DataParallelRank: batch.DataParallelRank}
		switch ev := event.(type) {
		case *kvevents.BlockStoredEvent:
			if !indexableStore(ev) {
				continue
			}
			scope.GroupIdx = ev.GroupIdx
			if err := c.index.Add(ctx, nil, scope.Keys(ev.BlockHashes), entries); err != nil {
				return err
			}
			c.remember(observedScope{scope: scope, blockSizeTokens: ev.BlockSize})
		case *kvevents.BlockRemovedEvent:
			if !localGPU(ev.DeviceTier, ev.Locality, ev.Ownership) {
				continue
			}
			scope.GroupIdx = ev.GroupIdx
			for _, key := range scope.Keys(ev.BlockHashes) {
				if err := c.index.Evict(ctx, key, kvblock.RequestKey, entries); err != nil {
					return err
				}
			}
		case *kvevents.AllBlocksClearedEvent:
			if err := c.clear(ctx, source.Endpoint); err != nil {
				return err
			}
		}
	}
	return nil
}

// Reset drops the endpoint's residency and remembered caches.
func (c *eventConsumer) Reset(ctx context.Context, endpoint string) error {
	return c.clear(ctx, endpoint)
}

func (c *eventConsumer) clear(ctx context.Context, endpoint string) error {
	c.mu.Lock()
	delete(c.observed, endpoint)
	c.mu.Unlock()
	return c.index.Clear(ctx, endpoint)
}

func (c *eventConsumer) remember(observed observedScope) {
	c.mu.Lock()
	defer c.mu.Unlock()
	known := c.observed[observed.scope.Endpoint]
	for i := range known {
		if sameScope(known[i].scope, observed.scope) {
			known[i] = observed
			return
		}
	}
	c.observed[observed.scope.Endpoint] = append(known, observed)
}

// scopes returns the observed caches of the candidate endpoints that report
// blocks of blockSizeTokens.
func (c *eventConsumer) scopes(endpoints sets.Set[string], blockSizeTokens int) []kvblock.EngineScope {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var result []kvblock.EngineScope
	for endpoint := range endpoints {
		for _, observed := range c.observed[endpoint] {
			if observed.blockSizeTokens == blockSizeTokens {
				result = append(result, observed.scope)
			}
		}
	}
	return result
}

// residentPrefix returns how many leading hashes are resident in scope.
func (c *eventConsumer) residentPrefix(ctx context.Context, scope kvblock.EngineScope, hashes []uint64) (int, error) {
	if len(hashes) == 0 {
		return 0, nil
	}
	keys := scope.Keys(hashes)
	found, err := c.index.Lookup(ctx, keys, sets.New(scope.Endpoint))
	if err != nil {
		return 0, err
	}
	for i, key := range keys {
		if !holds(found[key], scope.Endpoint) {
			return i, nil
		}
	}
	return len(keys), nil
}

func holds(entries []kvblock.PodEntry, endpoint string) bool {
	for _, entry := range entries {
		if entry.PodIdentifier == endpoint {
			return true
		}
	}
	return false
}

func sameScope(a, b kvblock.EngineScope) bool {
	return a.Endpoint == b.Endpoint && ptr.Equal(a.DataParallelRank, b.DataParallelRank) && ptr.Equal(a.GroupIdx, b.GroupIdx)
}

// indexableStore reports whether a store event describes blocks this producer
// tracks: local GPU blocks of a full-attention or MLA cache group, or of an
// engine that reports no group. Sliding-window and other cache kinds hold
// partial context, so their blocks do not establish prefix residency.
func indexableStore(ev *kvevents.BlockStoredEvent) bool {
	if !localGPU(ev.DeviceTier, ev.Locality, ev.Ownership) || ev.BlockSize <= 0 || len(ev.BlockHashes) == 0 {
		return false
	}
	switch ev.KVCacheSpecKind {
	case kvevents.KVCacheSpecKindFullAttention, kvevents.KVCacheSpecKindMlaAttention:
		return true
	case "":
		return ev.GroupIdx == nil
	default:
		return false
	}
}

// localGPU reports whether an event describes blocks the engine itself holds
// in GPU memory, as opposed to remote, offloaded, or lower-tier copies.
func localGPU(tier, locality, ownership string) bool {
	return strings.EqualFold(tier, gpuTier) && ownership == "" &&
		(locality == "" || strings.EqualFold(locality, "local"))
}
