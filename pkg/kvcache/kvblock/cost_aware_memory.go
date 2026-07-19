/*
Copyright 2025 The llm-d Authors.

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

package kvblock

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/dgraph-io/ristretto/v2"
	"github.com/dustin/go-humanize"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

const (
	defaultNumCounters = 1e8 // 100M keys
	defaultBufferItems = 64  // default buffer size for ristretto
)

// CostAwareMemoryIndexConfig holds the configuration for the CostAwareMemoryIndex.
type CostAwareMemoryIndexConfig struct {
	// Size is the maximum memory size that can be used by the index.
	// Supports human-readable formats like "2GiB", "500MiB", "1GB", etc.
	Size string `json:"size,omitempty"`
}

func DefaultCostAwareMemoryIndexConfig() *CostAwareMemoryIndexConfig {
	return &CostAwareMemoryIndexConfig{
		Size: "2GiB", // 2GiB default size
	}
}

// NewCostAwareMemoryIndex creates a new CostAwareMemoryIndex instance.
func NewCostAwareMemoryIndex(cfg *CostAwareMemoryIndexConfig) (*CostAwareMemoryIndex, error) {
	if cfg == nil {
		cfg = DefaultCostAwareMemoryIndexConfig()
	}

	// Parse the size string to get byte value using go-humanize
	sizeBytes, err := humanize.ParseBytes(cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize cost aware index: %w", err)
	}

	requestKeys, err := lru.New[BlockHash, []BlockHash](defaultNumCounters)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory engine key map: %w", err)
	}

	index := &CostAwareMemoryIndex{
		requestKeys: requestKeys,
		keyIndex:    make(map[string]struct{}),
	}

	// OnEvict/OnReject fire from ristretto's processing goroutine whenever an entry
	// leaves the cost cache (cost-based eviction or admission rejection). Pruning
	// keyIndex there keeps it bounded to the live cost-cache set instead of growing
	// with every key ever added. The callback takes keyIndexMu only — never mu —
	// because Add holds mu while blocking in data.Wait(), which is what drains the
	// buffer that triggers these callbacks; taking mu here would deadlock.
	cache, err := ristretto.NewCache(&ristretto.Config[string, *CostPodCache]{
		NumCounters: defaultNumCounters, // number of keys to track.
		MaxCost:     int64(sizeBytes),   // #nosec G115 , maximum cost of cache
		BufferItems: defaultBufferItems, // number of keys per Get buffer.
		OnEvict:     index.onCostCacheRemoval,
		OnReject:    index.onCostCacheRemoval,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize cost aware index: %w", err)
	}
	index.data = cache

	return index, nil
}

// CostAwareMemoryIndex implements the Index interface using Ristretto cache for cost-aware memory management.
// The two caches below are kept in sync:
//   - data: requestKey -> pod cache (cost-bound by Ristretto MaxCost)
//   - requestKeys: engineKey -> requestKey (LRU to cap mapping size)
//
// Add always writes both maps; Evict removes pods and, when empty, removes
// both the requestKey entry and its engineKey mapping to avoid dangling keys.
type CostAwareMemoryIndex struct {
	// data holds the mapping of request keys to sets of pod identifiers.
	data *ristretto.Cache[string, *CostPodCache]
	// requestKeys holds the mapping of engine keys to request keys.
	requestKeys *lru.Cache[BlockHash, []BlockHash]
	// keyIndex tracks live request-key strings so Clear can enumerate them
	// (ristretto exposes no iteration). Kept bounded to the live cost-cache set by
	// onCostCacheRemoval, which prunes a key when ristretto evicts or rejects it.
	// Guarded by keyIndexMu (not mu) so the ristretto callback can prune without
	// deadlocking against Add's data.Wait().
	keyIndex map[string]struct{}
	// keyIndexMu guards keyIndex independently of mu.
	keyIndexMu sync.Mutex
	// mu protects concurrent access to the index operations
	mu sync.RWMutex
}

// onCostCacheRemoval prunes keyIndex when ristretto evicts or rejects an entry.
// It runs on ristretto's processing goroutine, so it must take keyIndexMu only —
// never mu. The Item carries only the hashed key, so it recovers the original
// request-key string from the cached value's key field.
func (m *CostAwareMemoryIndex) onCostCacheRemoval(item *ristretto.Item[*CostPodCache]) {
	if item == nil || item.Value == nil || item.Value.key == "" {
		return
	}
	m.removeKeyIndex(item.Value.key)
}

func (m *CostAwareMemoryIndex) addKeyIndex(keyStr string) {
	m.keyIndexMu.Lock()
	m.keyIndex[keyStr] = struct{}{}
	m.keyIndexMu.Unlock()
}

func (m *CostAwareMemoryIndex) removeKeyIndex(keyStr string) {
	m.keyIndexMu.Lock()
	delete(m.keyIndex, keyStr)
	m.keyIndexMu.Unlock()
}

// snapshotKeyIndex returns a copy of the live request keys so Clear can scan them
// without holding keyIndexMu (or mu) for the whole pass.
func (m *CostAwareMemoryIndex) snapshotKeyIndex() []string {
	m.keyIndexMu.Lock()
	defer m.keyIndexMu.Unlock()
	keys := make([]string, 0, len(m.keyIndex))
	for k := range m.keyIndex {
		keys = append(keys, k)
	}
	return keys
}

func (m *CostAwareMemoryIndex) MaxCost() int64 {
	return m.data.MaxCost()
}

// CostPodCache wraps a sync.Map of PodEntry and provides cost calculation for memory usage estimation.
type CostPodCache struct {
	cache sync.Map // map[PodEntry]struct{}
	// size tracks the number of entries in cache for O(1) Len().
	size atomic.Int64
	// key is the request-key string this cache is stored under. It is captured so
	// the ristretto OnEvict/OnReject callback can prune keyIndex — the callback's
	// Item carries only the hashed key, not the original string.
	key string
}

// Add adds a PodEntry to the cache.
func (c *CostPodCache) Add(entry PodEntry) {
	if _, loaded := c.cache.LoadOrStore(entry, struct{}{}); !loaded {
		c.size.Add(1)
	}
}

// Delete removes a PodEntry from the cache.
func (c *CostPodCache) Delete(entry PodEntry) {
	if _, loaded := c.cache.LoadAndDelete(entry); loaded {
		c.size.Add(-1)
	}
}

// Len returns the number of entries in the cache.
func (c *CostPodCache) Len() int {
	return int(c.size.Load())
}

// CalculateByteSize estimates memory usage for ristretto cost calculation.
// This is an approximation used for cache eviction decisions.
func (c *CostPodCache) CalculateByteSize(keyStr string) int64 {
	var totalBytes int64
	var entryCount int64

	// Key string memory usage
	totalBytes += int64(len(keyStr))

	// CostPodCache struct overhead (sync.Map overhead)
	totalBytes += 64 // approximate sync.Map overhead

	// Count entries and calculate their size
	c.cache.Range(func(key, value interface{}) bool {
		entry, ok := key.(PodEntry)
		if !ok {
			return true
		}

		entryCount++
		totalBytes += int64(len(entry.PodIdentifier)) // PodIdentifier string content
		totalBytes += int64(len(entry.DeviceTier))    // DeviceTier string content
		totalBytes += 32                              // string headers (16 bytes each for 2 strings)
		totalBytes += 8                               // struct padding/alignment
		return true
	})

	// sync.Map overhead estimation
	if entryCount > 0 {
		// Map overhead: assuming 24 bytes per entry (key+value+metadata in sync.Map)
		totalBytes += entryCount * 24
	}

	return totalBytes
}

var _ Index = &CostAwareMemoryIndex{}

// Add adds a set of keys and their associated pod entries to the index backend.
// If engineKeys is nil, only requestKey -> PodEntry mappings are created (no engineKey -> requestKey mapping).
// This is used for speculative entries where engine keys are not yet known.
// When engineKeys is non-nil, the mapping type is inferred from the ratio of array lengths.
func (m *CostAwareMemoryIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CostAwareMemoryIndex.Add")

	// Build engine->request mappings when engine keys are provided.
	// The ratio of array lengths determines the mapping type:
	//   equal  (4 eng, 4 req) -> 1:1   E0->R0, E1->R1, ...
	//   many:1 (4 eng, 1 req) -> E0->R0, E1->R0, E2->R0, E3->R0
	//   1:many (1 eng, 4 req) -> E0->[R0, R1, R2, R3]
	if engineKeys != nil {
		mappings := engineToRequestMapping(engineKeys, requestKeys)
		for ek, rks := range mappings {
			m.requestKeys.Add(ek, rks)
		}
	}

	// Store requestKey -> PodCache mappings for all request keys.
	for _, requestKey := range requestKeys {
		keyStr := requestKey.String()
		podCache, found := m.data.Get(keyStr)
		if !found {
			podCache = &CostPodCache{key: keyStr}
		}

		for _, entry := range entries {
			podCache.Add(entry)
		}

		// Calculate the actual cost for this cache entry
		cost := podCache.CalculateByteSize(keyStr)
		m.data.Set(keyStr, podCache, cost)
		m.addKeyIndex(keyStr)
		traceLogger.Info("added pods to key", "requestKey", requestKey, "pods", entries, "cost-bytes", cost)
	}
	m.data.Wait()
	return nil
}

func (m *CostAwareMemoryIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no keys provided for lookup")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CostAwareMemoryIndex.Lookup")

	podsPerKey := make(map[BlockHash][]PodEntry)
	highestHitIdx := 0

	for idx, key := range requestKeys {
		keyStr := key.String()
		if pods, found := m.data.Get(keyStr); found { //nolint:nestif // TODO: can this be optimized?
			if pods == nil || pods.Len() == 0 {
				traceLogger.Info("no pods found for key, cutting search", "key", key)
				return podsPerKey, nil // early stop since prefix-chain breaks here
			}

			highestHitIdx = idx

			if podIdentifierSet.Len() == 0 {
				// If no pod identifiers are provided, return all pods
				pods.cache.Range(func(k, value interface{}) bool {
					if pod, ok := k.(PodEntry); ok {
						podsPerKey[key] = append(podsPerKey[key], pod)
					}
					return true
				})
			} else {
				// Filter pods based on the provided pod identifiers
				pods.cache.Range(func(k, value interface{}) bool {
					if pod, ok := k.(PodEntry); ok {
						if podIdentifierSet.Has(pod.PodIdentifier) {
							podsPerKey[key] = append(podsPerKey[key], pod)
						}
					}
					return true
				})
			}
		} else {
			traceLogger.Info("key not found in index", "key", key)
		}
	}

	traceLogger.Info("lookup completed", "highest-hit-index", highestHitIdx,
		"pods-per-key", podsPerKeyPrintHelper(podsPerKey))

	return podsPerKey, nil
}

// Evict removes a key and its associated pod entries from the index backend.
// keyType indicates whether the key is an EngineKey (requires engine→request lookup)
// or a RequestKey (used directly for speculative entries without engineKey mapping).
func (m *CostAwareMemoryIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(entries) == 0 {
		return fmt.Errorf("no entries provided for eviction from index")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CostAwareMemoryIndex.Evict")

	switch keyType {
	case EngineKey:
		rks, found := m.requestKeys.Get(key)
		if !found {
			traceLogger.Info("engineKey not found in mapping, nothing to evict", "engineKey", key)
			return nil
		}
		for _, rk := range rks {
			m.evictPodsFromRequestKey(rk, key, entries, traceLogger)
		}
		allEmpty := true
		for _, rk := range rks {
			if pc, found := m.data.Get(rk.String()); found && pc != nil && pc.Len() > 0 {
				allEmpty = false
				break
			}
		}
		if allEmpty {
			m.requestKeys.Remove(key)
		}
		m.data.Wait()
		return nil
	case RequestKey:
		m.evictPodsFromRequestKey(key, EmptyBlockHash, entries, traceLogger)
		m.data.Wait()
		return nil
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}
}

// evictPodsFromRequestKey removes the given pod entries from a single request key's cache.
// If the cache becomes empty, the request key is removed from the index.
func (m *CostAwareMemoryIndex) evictPodsFromRequestKey(
	requestKey, engineKey BlockHash, entries []PodEntry, traceLogger logr.Logger,
) {
	keyStr := requestKey.String()
	podCache, found := m.data.Get(keyStr)
	if !found || podCache == nil {
		traceLogger.Info("requestKey not found in index, nothing to evict", "requestKey", requestKey, "engineKey", engineKey)
		return
	}

	podCacheLenBefore := podCache.Len()

	for _, entry := range entries {
		podCache.Delete(entry)
	}

	if podCache.Len() == 0 {
		m.data.Del(keyStr)
		m.removeKeyIndex(keyStr)
		traceLogger.Info("removed requestKey from index as no pods remain", "requestKey", requestKey)
	} else if podCacheLenBefore != podCache.Len() {
		m.data.Set(keyStr, podCache, podCache.CalculateByteSize(keyStr))
		traceLogger.Info("evicted pods from key", "requestKey", requestKey, "engineKey", engineKey, "pods", entries)
	}
}

// Clear removes every entry for the pod from the index, across all device tiers.
// It is O(N) over the index but runs off the Lookup/Add hot path at a coarse
// cadence. The scan is chunked: it snapshots the request keys, then processes them
// in fixed-size chunks, each taking mu only briefly, so a Clear never blocks
// Lookup (which takes mu.RLock) for the whole pass. The trade is atomicity — a
// concurrent Lookup may see the pod cleared from some keys but not yet from
// others; that is acceptable since the pod's cache is cold post-reset and a stale
// hit only costs a cache miss.
//
// The engineKey->requestKey mapping (requestKeys) is intentionally left untouched:
// it is LRU-bounded, self-heals when the pod re-Adds the same prefixes, and any
// stale mapping resolves to an emptied request key that correctly breaks the
// prefix chain in Lookup. Reverse-pruning it would need an O(M) scan for no
// correctness gain.
func (m *CostAwareMemoryIndex) Clear(ctx context.Context, podIdentifier string) error {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CostAwareMemoryIndex.Clear")

	keys := m.snapshotKeyIndex()

	const clearChunkSize = 1024
	for start := 0; start < len(keys); start += clearChunkSize {
		end := min(start+clearChunkSize, len(keys))
		m.clearChunk(podIdentifier, keys[start:end])
	}

	m.data.Wait()
	traceLogger.Info("cleared pod from index", "pod", podIdentifier, "scanned", len(keys))
	return nil
}

// clearChunk removes the pod's entries from one chunk of request keys under a
// single mu hold, bounding how long Clear blocks the Lookup/Add path.
func (m *CostAwareMemoryIndex) clearChunk(podIdentifier string, keys []string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, keyStr := range keys {
		podCache, found := m.data.Get(keyStr)
		if !found || podCache == nil {
			m.removeKeyIndex(keyStr) // ristretto dropped it under us; drop the stale key
			continue
		}

		// Collect-then-delete: sync.Map.Range tolerates deletes by f, but collecting
		// first keeps the deletion explicit and the iteration simple.
		var matched []PodEntry
		podCache.cache.Range(func(k, _ any) bool {
			if entry, ok := k.(PodEntry); ok && entry.PodIdentifier == podIdentifier {
				matched = append(matched, entry)
			}
			return true
		})
		if len(matched) == 0 {
			continue
		}

		lenBefore := podCache.Len()
		for _, entry := range matched {
			podCache.Delete(entry)
		}

		switch {
		case podCache.Len() == 0:
			m.data.Del(keyStr)
			m.removeKeyIndex(keyStr)
		case podCache.Len() != lenBefore:
			m.data.Set(keyStr, podCache, podCache.CalculateByteSize(keyStr))
		}
	}
}

// GetRequestKey returns the last request key (highest index in the chain) associated with the given engineKey.
// Returns an error if the engineKey is not mapped (e.g., evicted earlier).
func (m *CostAwareMemoryIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	rks, found := m.requestKeys.Get(engineKey)
	if !found || len(rks) == 0 {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}
	return rks[len(rks)-1], nil
}
