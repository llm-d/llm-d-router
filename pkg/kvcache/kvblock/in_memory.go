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
	"strings"
	"sync"

	"github.com/go-logr/logr"
	lru "github.com/hashicorp/golang-lru/v2"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/collections"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

const (
	defaultInMemoryIndexSize = 1e8 // TODO: change to memory-size based configuration
	defaultPodsPerKey        = 10  // number of pods per key
)

// InMemoryIndexConfig holds the configuration for the InMemoryIndex.
type InMemoryIndexConfig struct {
	// Size is the maximum number of keys that can be stored in the index.
	Size int `json:"size"`
	// PodCacheSize is the maximum number of pod entries per key.
	PodCacheSize int `json:"podCacheSize"`
}

// DefaultInMemoryIndexConfig returns a default configuration for the InMemoryIndex.
func DefaultInMemoryIndexConfig() *InMemoryIndexConfig {
	return &InMemoryIndexConfig{
		Size:         defaultInMemoryIndexSize,
		PodCacheSize: defaultPodsPerKey,
	}
}

// NewInMemoryIndex creates a new InMemoryIndex instance.
func NewInMemoryIndex(cfg *InMemoryIndexConfig) (*InMemoryIndex, error) {
	if cfg == nil {
		cfg = DefaultInMemoryIndexConfig()
	}
	// Apply the default for an omitted field so partial configs (e.g. only
	// size set) work correctly.
	podCacheSize := cfg.PodCacheSize
	if podCacheSize <= 0 {
		podCacheSize = defaultPodsPerKey
	}

	cache, err := lru.New[BlockHash, *PodCache](cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory index: %w", err)
	}

	engineToRequestKeys, err := lru.New[BlockHash, []BlockHash](cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory engine key map: %w", err)
	}

	return &InMemoryIndex{
		data:                cache,
		engineToRequestKeys: engineToRequestKeys,
		podCacheSize:        podCacheSize,
		pods:                newInterner(),
		tiers:               newInterner(),
	}, nil
}

// InMemoryIndex is an in-memory implementation of the Index interface.
type InMemoryIndex struct {
	// mu protects engine-key-level check-and-act operations (Evict's allEmpty
	// check + mapping removal vs Add's pod entry insertion) to prevent TOCTOU races.
	mu sync.Mutex
	// data holds the mapping of requestKeys to sets of pod identifiers.
	data *lru.Cache[BlockHash, *PodCache]
	// engineToRequestKeys holds the mapping of engineKeys to requestKeys.
	engineToRequestKeys *lru.Cache[BlockHash, []BlockHash]
	// podCacheSize is the maximum number of pod entries per key.
	podCacheSize int
	// pods and tiers intern PodIdentifier and DeviceTier strings to dense
	// indices so ScoredLookup accumulates per-pod state in slices instead of
	// string-keyed maps. Both grow with the distinct values seen over the
	// process lifetime and are never compacted; entries are a few tens of
	// bytes each.
	pods  *interner
	tiers *interner
}

var _ Index = &InMemoryIndex{}

// PodCache holds one request key's pod entries in LRU order (least recent
// first), bounded by the index's podCacheSize. At per-key pod counts the
// linear slice operations are cheaper than a map-backed LRU, and iteration
// allocates nothing.
type PodCache struct {
	// mu protects entries.
	mu sync.Mutex
	// entries is ordered least recently added first.
	entries []podRecord
	// capacity bounds len(entries); adding beyond it evicts the least
	// recent entry.
	capacity int
}

// podRecord pairs a PodEntry with interned identifiers so ScoredLookup can
// accumulate per-pod state without hashing strings.
type podRecord struct {
	entry PodEntry
	// podIdx interns entry.PodIdentifier.
	podIdx uint32
	// weightTierIdx interns entry.DeviceTier, keying scoring weights.
	weightTierIdx uint32
	// statTierIdx interns the tier reported in per-tier match counts:
	// SpeculativeTier for speculative entries, else entry.DeviceTier.
	statTierIdx uint32
}

// addAll inserts records with LRU semantics: an existing entry refreshes its
// recency, a new entry appends, and overflow evicts the least recent entry.
func (pc *PodCache) addAll(recs []podRecord) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	for _, rec := range recs {
		found := false
		for i := range pc.entries {
			if pc.entries[i].entry == rec.entry {
				copy(pc.entries[i:], pc.entries[i+1:])
				pc.entries[len(pc.entries)-1] = rec
				found = true
				break
			}
		}
		if found {
			continue
		}
		if pc.capacity > 0 && len(pc.entries) >= pc.capacity {
			copy(pc.entries, pc.entries[1:])
			pc.entries[len(pc.entries)-1] = rec
			continue
		}
		pc.entries = append(pc.entries, rec)
	}
}

// removeAll deletes the given entries and reports whether the cache is empty
// afterwards.
func (pc *PodCache) removeAll(entries []PodEntry) (empty bool) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	for _, entry := range entries {
		for i := range pc.entries {
			if pc.entries[i].entry == entry {
				pc.entries = append(pc.entries[:i], pc.entries[i+1:]...)
				break
			}
		}
	}
	return len(pc.entries) == 0
}

// size returns the number of entries.
func (pc *PodCache) size() int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return len(pc.entries)
}

// filteredEntries returns the entries whose PodIdentifier is in allowed (all
// entries when allowed is empty) plus the unfiltered entry count.
func (pc *PodCache) filteredEntries(allowed sets.Set[string]) (filtered []PodEntry, total int) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	total = len(pc.entries)
	if total == 0 {
		return nil, 0
	}
	if allowed.Len() == 0 {
		filtered = make([]PodEntry, 0, total)
		for i := range pc.entries {
			filtered = append(filtered, pc.entries[i].entry)
		}
		return filtered, total
	}
	for i := range pc.entries {
		if allowed.Has(pc.entries[i].entry.PodIdentifier) {
			filtered = append(filtered, pc.entries[i].entry)
		}
	}
	return filtered, total
}

// matching returns the entries belonging to podIdentifier.
func (pc *PodCache) matching(podIdentifier string) []PodEntry {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	var matched []PodEntry
	for i := range pc.entries {
		if pc.entries[i].entry.PodIdentifier == podIdentifier {
			matched = append(matched, pc.entries[i].entry)
		}
	}
	return matched
}

// internRecord builds the interned representation of a PodEntry.
func (m *InMemoryIndex) internRecord(entry PodEntry) podRecord {
	statTier := entry.DeviceTier
	if entry.Speculative {
		statTier = SpeculativeTier
	}
	return podRecord{
		entry:         entry,
		podIdx:        m.pods.intern(entry.PodIdentifier),
		weightTierIdx: m.tiers.intern(entry.DeviceTier),
		statTierIdx:   m.tiers.intern(statTier),
	}
}

// Lookup receives a list of requestKeys and a set of pod identifiers,
// and retrieves the filtered pods associated with those keys.
// The filtering is done based on the pod identifiers provided.
// If the podIdentifierSet is empty, all pods are returned.
//
// It returns:
// 1. A map where the keys are those in (1) and the values are pod-identifiers.
// 2. An error if any occurred during the operation.
func (m *InMemoryIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no requestKeys provided for lookup")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Lookup")

	podsPerKey := make(map[BlockHash][]PodEntry)
	highestHitIdx := 0

	for idx, requestKey := range requestKeys {
		if idx&cancellationCheckMask == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		pods, found := m.data.Get(requestKey)
		if !found {
			if traceLogger.Enabled() {
				traceLogger.Info("key not found in index", "key", requestKey)
			}
			continue
		}
		var filtered []PodEntry
		total := 0
		if pods != nil {
			filtered, total = pods.filteredEntries(podIdentifierSet)
		}
		if total == 0 {
			if traceLogger.Enabled() {
				traceLogger.Info("no pods found for key, cutting search", "key", requestKey)
			}
			return podsPerKey, nil // early stop since prefix-chain breaks here
		}

		highestHitIdx = idx

		if len(filtered) > 0 {
			podsPerKey[requestKey] = filtered
		}
	}

	if traceLogger.Enabled() {
		traceLogger.Info("lookup completed", "highest-hit-index", highestHitIdx,
			"pods-per-key", podsPerKeyPrintHelper(podsPerKey))
	}

	return podsPerKey, nil
}

// Add adds a set of engineKeys/requestKeys and their associated pod entries to the index backend.
// If engineKeys is nil, only requestKey -> PodEntry mappings are created (no engineKey -> requestKey mapping).
// This is used for speculative entries where engine keys are not yet known.
// When engineKeys is non-nil, the mapping type is inferred from the ratio of array lengths.
func (m *InMemoryIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Add")

	// Build engine->request mappings when engine keys are provided.
	// The ratio of array lengths determines the mapping type:
	//   equal  (4 eng, 4 req) -> 1:1   E0->R0, E1->R1, ...
	//   many:1 (4 eng, 1 req) -> E0->R0, E1->R0, E2->R0, E3->R0
	//   1:many (1 eng, 4 req) -> E0->[R0, R1, R2, R3]
	if engineKeys != nil {
		mappings := engineToRequestMapping(engineKeys, requestKeys)
		for ek, rks := range mappings {
			m.engineToRequestKeys.Add(ek, rks)
		}
	}

	// Intern once per call; the same entries apply to every request key.
	records := make([]podRecord, len(entries))
	for i, entry := range entries {
		records[i] = m.internRecord(entry)
	}

	// Store requestKey -> PodCache mappings for all request keys.
	// Hold m.mu to prevent Evict from checking emptiness and removing the
	// engine→request mapping while we are inserting pod entries.
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, requestKey := range requestKeys {
		podCache, found := m.data.Get(requestKey)
		if !found {
			newPodCache := &PodCache{capacity: m.podCacheSize}

			// Try to add, but use existing if another thread added it first
			// This is a bounded retry (1) - not perfectly safe but for practical use-cases and scenarios
			// this should be sufficient
			contains, _ := m.data.ContainsOrAdd(requestKey, newPodCache)
			if contains {
				podCache, found = m.data.Get(requestKey)
				if !found { // Extremely irregular workload pattern - key evicted
					m.data.Add(requestKey, newPodCache)
					podCache = newPodCache
				}
			} else {
				// We successfully added our cache
				podCache = newPodCache
			}
		}

		podCache.addAll(records)

		if traceLogger.Enabled() {
			traceLogger.Info("added pods to key", "requestKey", requestKey, "pods", entries)
		}
	}

	return nil
}

// Evict removes a key and its associated pod entries from the index backend.
// keyType indicates whether the key is an EngineKey (requires engine→request lookup)
// or a RequestKey (used directly for speculative entries without engineKey mapping).
func (m *InMemoryIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	if len(entries) == 0 {
		return fmt.Errorf("no entries provided for eviction from index")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Evict")

	switch keyType {
	case EngineKey:
		rks, found := m.engineToRequestKeys.Get(key)
		if !found {
			traceLogger.Info("engineKey not found in mapping, nothing to evict", "engineKey", key)
			return nil
		}

		for _, rk := range rks {
			m.evictPodsFromRequestKey(rk, key, entries, traceLogger)
		}

		m.mu.Lock()
		allEmpty := true
		for _, rk := range rks {
			if pc, found := m.data.Get(rk); found && pc != nil && pc.size() > 0 {
				allEmpty = false
				break
			}
		}
		if allEmpty {
			m.engineToRequestKeys.Remove(key)
		}
		m.mu.Unlock()
		return nil
	case RequestKey:
		m.evictPodsFromRequestKey(key, EmptyBlockHash, entries, traceLogger)
		return nil
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}
}

// evictPodsFromRequestKey removes the given pod entries from a single request key's cache.
// If the cache becomes empty, the request key is removed from the index.
func (m *InMemoryIndex) evictPodsFromRequestKey(requestKey, engineKey BlockHash, entries []PodEntry, traceLogger logr.Logger) {
	podCache, found := m.data.Get(requestKey)
	if !found || podCache == nil {
		traceLogger.Info("requestKey not found in index, nothing to evict", "requestKey", requestKey, "engineKey", engineKey)
		return
	}

	isEmpty := podCache.removeAll(entries)

	traceLogger.Info("evicted pods from key", "requestKey", requestKey, "engineKey", engineKey, "pods", entries)

	if !isEmpty {
		return
	}

	// Remove key from main cache if empty.
	// Re-fetch and hold the lock through removal to prevent racing with Add.
	currentCache, stillExists := m.data.Get(requestKey)
	if !stillExists || currentCache == nil {
		return
	}

	currentCache.mu.Lock()
	if len(currentCache.entries) == 0 {
		m.data.Remove(requestKey)
		traceLogger.Info("removed requestKey from index as no pods remain", "requestKey", requestKey)
	}
	currentCache.mu.Unlock()
}

// Clear removes every entry for the pod from the index, across all device tiers.
// O(N) over the index, but Clear is rare and off the Lookup/Add hot path. Reuses
// evictPodsFromRequestKey for race-safe removal, and holds no global lock — only
// each PodCache's mu, briefly — so it does not stall Lookup.
//
// The engineKey->requestKey mapping (engineToRequestKeys) is intentionally left
// untouched: it is LRU-bounded, self-heals when the pod re-Adds the same prefixes,
// and any stale mapping resolves to an emptied request key that correctly breaks
// the prefix chain in Lookup.
func (m *InMemoryIndex) Clear(ctx context.Context, podIdentifier string) error {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Clear")

	for idx, requestKey := range m.data.Keys() {
		if idx&cancellationCheckMask == 0 && ctx.Err() != nil {
			return ctx.Err()
		}
		// Peek so a clear does not promote LRU recency on keys it scans.
		podCache, found := m.data.Peek(requestKey)
		if !found || podCache == nil {
			continue
		}

		matched := podCache.matching(podIdentifier)

		if len(matched) > 0 {
			m.evictPodsFromRequestKey(requestKey, EmptyBlockHash, matched, traceLogger)
		}
	}

	traceLogger.Info("cleared pod from index", "pod", podIdentifier)
	return nil
}

// GetRequestKey returns the last request key (highest index in the chain) associated with the given engineKey.
// This is what Pool uses for parent hash resolution.
// Returns an error if the engineKey mapping is missing (e.g., already evicted).
func (m *InMemoryIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	rks, found := m.engineToRequestKeys.Get(engineKey)
	if !found || len(rks) == 0 {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}
	return rks[len(rks)-1], nil
}

// podsPerKeyPrintHelper formats a map of keys to pod names for printing.
func podsPerKeyPrintHelper(ks map[BlockHash][]PodEntry) string {
	var b strings.Builder
	for k, v := range ks {
		fmt.Fprintf(&b, "%s: %v\n", k.String(), collections.SliceMap(v, func(pod PodEntry) string {
			return pod.String()
		}))
	}
	return b.String()
}
