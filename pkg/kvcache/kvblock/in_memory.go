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
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/collections"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

const (
	// defaultInMemoryIndexSize caps the index by entry count, not memory: the
	// underlying LRU evicts on count and has no notion of byte cost. To size
	// Size against available memory, estimate per-entry cost as roughly
	// PodCacheSize pod entries times the PodIdentifier and DeviceTier string
	// lengths, plus map/LRU bookkeeping overhead, and divide the memory budget
	// by that. CostAwareMemoryIndex tracks actual byte cost per entry and
	// evicts against a configured memory budget directly; prefer it when the
	// workload's per-entry size is hard to predict up front.
	defaultInMemoryIndexSize = 1e8
	defaultPodsPerKey        = 10 // number of pods per key
)

// InMemoryIndexConfig holds the configuration for the InMemoryIndex.
type InMemoryIndexConfig struct {
	// Size is the maximum number of keys that can be stored in the index. It
	// bounds entry count, not memory; see defaultInMemoryIndexSize for sizing
	// it against available memory.
	Size int `json:"size"`
	// PodCacheSize is the maximum number of pod entries per key.
	// A non-positive value selects defaultPodsPerKey.
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
	podCacheSize := cfg.PodCacheSize
	if podCacheSize <= 0 {
		podCacheSize = defaultPodsPerKey
	}

	m := &InMemoryIndex{
		engineToRequestKeys: newEngineKeyMap(cfg.Size),
		podCacheSize:        podCacheSize,
		pods:                newInterner(maxInternedPods),
		tiers:               newInterner(maxInternedTiers),
	}
	cache, err := newLRUStore(cfg.Size, m.onKeyEvicted)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory index: %w", err)
	}
	m.data = cache
	return m, nil
}

// cancellationCheckMask paces context-cancellation checks in loops over
// request keys: positions where idx&mask == 0 poll ctx.Err().
const cancellationCheckMask = 255

// maxInternedPods and maxInternedTiers cap the distinct pod identifiers and
// device tiers an index assigns ordinals to over its lifetime.
const (
	maxInternedPods  = 1 << 20
	maxInternedTiers = 1 << 12
)

// errIndexCardinality reports an Add whose entries would exceed a cap.
var errIndexCardinality = errors.New("index cardinality limit reached")

// InMemoryIndex is an in-memory implementation of the Index interface.
type InMemoryIndex struct {
	// mu orders Add's engine key mapping writes and pod entry inserts against the
	// removal of an emptied request key and of an engine key mapping whose request
	// keys are all empty. It also guards the lock-free engineToRequestKeys
	// structure, so Add (also when engineKeys is nil), Evict, and GetRequestKey
	// take it. Lookup does not take it.
	mu sync.Mutex
	// data holds the mapping of requestKeys to sets of pod identifiers.
	data *lruStore
	// engineToRequestKeys holds the mapping of engineKeys to requestKeys.
	// It is lock-free; m.mu guards it.
	engineToRequestKeys *engineKeyMap
	// podCacheSize is the maximum number of pod entries per key.
	podCacheSize int
	// pods and tiers assign the ordinals EntryRef carries. Neither is
	// reclaimed when entries leave the index: pod identifiers are endpoint
	// address:port values, bounded by the pod network's address space, and
	// tiers are the engine-reported names. Each is capped, and an Add past
	// a cap fails rather than admitting an entry that cannot be matched.
	pods  *interner
	tiers *interner
	// entries is the live entry count across every resident key, and
	// truncated the cumulative per-key pod-cap drops. Both feed the
	// residency gauges and Stats.
	entries   atomic.Int64
	truncated atomic.Int64
}

var _ StatsProvider = &InMemoryIndex{}

// onKeyEvicted accounts a main-store LRU eviction. Remove fires the same
// callback for keys the backend removes deliberately, but those are emptied
// first, so a non-empty cache here is a capacity eviction.
func (m *InMemoryIndex) onKeyEvicted(pc *PodCache) {
	if pc == nil {
		return
	}
	m.entries.Add(-int64(pc.size()))
	if pc.size() > 0 {
		metrics.IndexEvictions.WithLabelValues("capacity").Add(1)
	}
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
	entries []EntryRef
	// capacity bounds len(entries); adding beyond it evicts the least
	// recent entry.
	capacity int
}

// addAll inserts records with LRU semantics: an existing entry refreshes its
// recency, a new entry appends, and overflow evicts the least recent entry.
// It returns the entry-count change and the count of pods dropped by the
// capacity bound.
func (pc *PodCache) addAll(recs []EntryRef) (delta, truncated int) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	for _, rec := range recs {
		found := false
		for i := range pc.entries {
			if pc.entries[i].PodEntry == rec.PodEntry {
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
			truncated++
			continue
		}
		pc.entries = append(pc.entries, rec)
		delta++
	}
	return delta, truncated
}

// removeAll deletes the given entries and reports how many it removed and
// whether the cache is empty afterwards.
func (pc *PodCache) removeAll(entries []PodEntry) (removed int, empty bool) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	for _, entry := range entries {
		for i := range pc.entries {
			if pc.entries[i].PodEntry == entry {
				pc.entries = append(pc.entries[:i], pc.entries[i+1:]...)
				removed++
				break
			}
		}
	}
	return removed, len(pc.entries) == 0
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
			filtered = append(filtered, pc.entries[i].PodEntry)
		}
		return filtered, total
	}
	for i := range pc.entries {
		if allowed.Has(pc.entries[i].PodIdentifier) {
			filtered = append(filtered, pc.entries[i].PodEntry)
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
		if pc.entries[i].PodIdentifier == podIdentifier {
			matched = append(matched, pc.entries[i].PodEntry)
		}
	}
	return matched
}

// internRecords pairs each entry with its pod and tier ordinals. A batch is
// assigned all or nothing: when its new pods or tiers would exceed a cap, no
// ordinal is consumed and the error names the cap.
func (m *InMemoryIndex) internRecords(entries []PodEntry) ([]EntryRef, error) {
	m.pods.mu.Lock()
	defer m.pods.mu.Unlock()
	m.tiers.mu.Lock()
	defer m.tiers.mu.Unlock()

	if !m.pods.fitsLocked(func(yield func(string)) {
		for i := range entries {
			yield(entries[i].PodIdentifier)
		}
	}) {
		return nil, fmt.Errorf("%w: %d pod identifiers", errIndexCardinality, maxInternedPods)
	}
	if !m.tiers.fitsLocked(func(yield func(string)) {
		for i := range entries {
			yield(entries[i].DeviceTier)
		}
	}) {
		return nil, fmt.Errorf("%w: %d device tiers", errIndexCardinality, maxInternedTiers)
	}

	records := make([]EntryRef, len(entries))
	for i, entry := range entries {
		records[i] = EntryRef{
			PodEntry:    entry,
			PodOrdinal:  m.pods.internLocked(entry.PodIdentifier),
			TierOrdinal: m.tiers.internLocked(entry.DeviceTier),
		}
	}
	return records, nil
}

// Lookup receives a list of requestKeys and a set of pod identifiers,
// and retrieves the filtered pods associated with those keys.
// The filtering is done based on the pod identifiers provided.
// If the podIdentifierSet is empty, all pods are returned.
//
// It returns:
// 1. A map where the keys are those in (1) and the values are pod-identifiers.
// 2. An error if any occurred during the operation.
//
// For non-empty requestKeys, Lookup uses WalkKeys' cancellation checkpoints
// and recency-promotion rules. It retains Lookup's empty-input error.
func (m *InMemoryIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no requestKeys provided for lookup")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Lookup")

	podsPerKey := make(map[BlockHash][]PodEntry)
	highestHitIdx := 0
	visited := 0
	// Every exit, cancellation included, refreshes what was read.
	defer func() { m.data.Promote(requestKeys[:visited]) }()

	for idx, requestKey := range requestKeys {
		if idx&cancellationCheckMask == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		pods, found := m.data.Peek(requestKey)
		if !found {
			if traceLogger.Enabled() {
				traceLogger.Info("no pods found for key, cutting search", "key", requestKey)
			}
			break // early stop since prefix-chain breaks here
		}
		var filtered []PodEntry
		if pods != nil {
			filtered, _ = pods.filteredEntries(podIdentifierSet)
		}
		visited = idx + 1
		if len(filtered) == 0 {
			if traceLogger.Enabled() {
				traceLogger.Info("no pods found for key, cutting search", "key", requestKey)
			}
			break // early stop since prefix-chain breaks here
		}

		highestHitIdx = idx
		podsPerKey[requestKey] = filtered
	}

	if err := ctx.Err(); err != nil {
		return nil, err
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

	// Intern once per call, before anything is written: a rejected batch
	// leaves no mapping and no ordinal behind. The same records apply to
	// every request key.
	records, err := m.internRecords(entries)
	if err != nil {
		return err
	}

	// Hold m.mu across the engine->request mappings and the pod entry inserts,
	// also when engineKeys is nil. Evict and Clear re-check emptiness under m.mu
	// before removing a request key or an engine key mapping, so they cannot
	// remove one that this call has written but whose entries it has not
	// inserted yet.
	m.mu.Lock()
	defer m.mu.Unlock()

	// Build engine->request mappings when engine keys are provided.
	// The ratio of array lengths determines the mapping type:
	//   equal  (4 eng, 4 req) -> 1:1   E0->R0, E1->R1, ...
	//   many:1 (4 eng, 1 req) -> E0->R0, E1->R0, E2->R0, E3->R0
	//   1:many (1 eng, 4 req) -> E0->[R0, R1, R2, R3]
	if engineKeys != nil {
		mappings := engineToRequestMapping(engineKeys, requestKeys)
		for ek, rks := range mappings {
			m.engineToRequestKeys.set(ek, rks)
		}
	}

	// Store requestKey -> PodCache mappings for all request keys.

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

		delta, truncated := podCache.addAll(records)
		m.entries.Add(int64(delta))
		if truncated > 0 {
			m.truncated.Add(int64(truncated))
			metrics.IndexTruncatedEntries.Add(float64(truncated))
		}

		if traceLogger.Enabled() {
			traceLogger.Info("added pods to key", "requestKey", requestKey, "pods", entries)
		}
	}

	m.reportResidency()
	return nil
}

// Evict removes keys and their associated pod entries from the index backend.
// keyType indicates whether the keys are EngineKeys (each resolved independently
// via the engine→request mapping) or RequestKeys (used directly).
// A non-nil error means the index is unchanged.
func (m *InMemoryIndex) Evict(ctx context.Context, keyType KeyType, keys []BlockHash, entries []PodEntry) error {
	if len(keys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for eviction from index")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Evict")

	switch keyType {
	case EngineKey:
		// Resolve every engine key's request keys under one lock hold, then
		// evict the pod entries. The evictions release the lock per key: they
		// take it themselves to remove an emptied request key.
		groups := make([][]BlockHash, len(keys))
		m.mu.Lock()
		for i, key := range keys {
			rks, found := m.engineToRequestKeys.peek(key)
			if !found {
				traceLogger.Info("engineKey not found in mapping, nothing to evict", "engineKey", key)
				continue
			}
			groups[i] = rks
		}
		m.mu.Unlock()
		for i, key := range keys {
			for _, rk := range groups[i] {
				m.evictPodsFromRequestKey(rk, key, entries, traceLogger)
			}
		}

		// One lock hold covers every engine key's allEmpty check and mapping
		// removal, keeping the Add-vs-Evict TOCTOU protection while batching.
		m.mu.Lock()
		for i, key := range keys {
			if groups[i] == nil {
				continue
			}
			allEmpty := true
			for _, rk := range groups[i] {
				if pc, found := m.data.Peek(rk); found && pc != nil && pc.size() > 0 {
					allEmpty = false
					break
				}
			}
			if allEmpty {
				m.engineToRequestKeys.remove(key)
			}
		}
		m.mu.Unlock()
		m.reportResidency()
		return nil
	case RequestKey:
		for _, key := range keys {
			m.evictPodsFromRequestKey(key, EmptyBlockHash, entries, traceLogger)
		}
		m.reportResidency()
		return nil
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}
}

// evictPodsFromRequestKey removes the given pod entries from a single request key's cache.
// If the cache becomes empty, the request key is removed from the index.
func (m *InMemoryIndex) evictPodsFromRequestKey(requestKey, engineKey BlockHash, entries []PodEntry, traceLogger logr.Logger) {
	podCache, found := m.data.Peek(requestKey)
	if !found || podCache == nil {
		traceLogger.Info("requestKey not found in index, nothing to evict", "requestKey", requestKey, "engineKey", engineKey)
		return
	}

	removedEntries, isEmpty := podCache.removeAll(entries)
	m.entries.Add(-int64(removedEntries))

	traceLogger.Info("evicted pods from key", "requestKey", requestKey, "engineKey", engineKey, "pods", entries)

	if !isEmpty {
		return
	}

	m.mu.Lock()
	removed := m.removeIfEmptyLocked(requestKey, "empty")
	m.mu.Unlock()
	if removed && traceLogger.Enabled() {
		traceLogger.Info("removed requestKey from index as no pods remain", "requestKey", requestKey)
	}
}

// removeIfEmptyLocked removes requestKey from the index if its cache holds no
// entries, and reports whether it did, counting the removal under cause. The
// caller holds m.mu, so an Add that already fetched this cache has finished
// its write before the check.
func (m *InMemoryIndex) removeIfEmptyLocked(requestKey BlockHash, cause string) bool {
	podCache, found := m.data.Peek(requestKey)
	if !found || podCache == nil {
		return false
	}

	podCache.mu.Lock()
	empty := len(podCache.entries) == 0
	podCache.mu.Unlock()
	if !empty {
		return false
	}
	// podCache.mu is released before the removal because data.Remove fires
	// the eviction callback, which reads the cache's size under that lock.
	// The cache cannot gain entries in between: Add holds m.mu across its
	// inserts, and the caller holds m.mu here.
	m.data.Remove(requestKey)
	metrics.IndexEvictions.WithLabelValues(cause).Add(1)
	return true
}

// Clear removes every entry for the pod from the index, across all device tiers.
// O(N) over the index for a pod it has held, and O(1) for a pod it never held,
// such as one whose first snapshot is about to load. It removes the request keys
// it empties under one m.mu hold per chunk of keys, re-checking each key's
// emptiness under the lock. Lookup does not take m.mu, so Clear does not stall Lookup.
//
// Context cancellation does not interrupt Clear.
//
// The engineKey->requestKey mapping (engineToRequestKeys) is intentionally left
// untouched: it is LRU-bounded, self-heals when the pod re-Adds the same prefixes,
// and any stale mapping resolves to an emptied request key that correctly breaks
// the prefix chain in Lookup.
func (m *InMemoryIndex) Clear(ctx context.Context, podIdentifier string) error {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Clear")

	start := time.Now()

	// A pod is interned by its first Add and never released.
	m.pods.mu.Lock()
	_, interned := m.pods.ids[podIdentifier]
	m.pods.mu.Unlock()
	if !interned {
		return nil
	}

	var emptied []BlockHash
	for _, requestKey := range m.data.Keys() {
		// Peek so a clear does not promote LRU recency on keys it scans.
		podCache, found := m.data.Peek(requestKey)
		if !found || podCache == nil {
			continue
		}

		matched := podCache.matching(podIdentifier)
		if len(matched) == 0 {
			continue
		}
		removed, empty := podCache.removeAll(matched)
		m.entries.Add(-int64(removed))
		if empty {
			emptied = append(emptied, requestKey)
		}
	}

	// One m.mu hold per chunk bounds how long Clear blocks Add.
	const clearChunkSize = 1024
	for chunk := range slices.Chunk(emptied, clearChunkSize) {
		m.mu.Lock()
		for _, requestKey := range chunk {
			m.removeIfEmptyLocked(requestKey, "clear")
		}
		m.mu.Unlock()
	}

	metrics.IndexClearDuration.Observe(time.Since(start).Seconds())
	m.reportResidency()
	traceLogger.Info("cleared pod from index", "pod", podIdentifier)
	return nil
}

// GetRequestKey returns the last request key (highest index in the chain) associated with the given engineKey.
// This is what Pool uses for parent hash resolution.
// Returns an error if the engineKey mapping is missing (e.g., already evicted).
func (m *InMemoryIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	m.mu.Lock()
	rk, found := m.engineToRequestKeys.get(engineKey)
	m.mu.Unlock()
	if !found {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}
	return rk, nil
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
