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

	"github.com/go-logr/logr"
	lru "github.com/hashicorp/golang-lru/v2"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/collections"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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

	cache, err := newLRUStore(cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory index: %w", err)
	}

	index := &InMemoryIndex{
		data: cache, podCacheSize: podCacheSize,
		pods: newInterner(maxInternedPods), tiers: newInterner(maxInternedTiers),
		owners:        make(map[BlockHash]map[PodEntry]sets.Set[BlockHash]),
		engineEntries: make(map[BlockHash]map[BlockHash]sets.Set[PodEntry]),
		constituents:  make(map[BlockHash]sets.Set[BlockHash]),
	}
	cache.onEvict = index.forgetRequestOwners
	engineToRequestKeys, err := lru.NewWithEvict[BlockHash, []BlockHash](cfg.Size,
		func(engineKey BlockHash, _ []BlockHash) { index.releaseEngine(engineKey) })
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory engine key map: %w", err)
	}
	index.engineToRequestKeys = engineToRequestKeys
	return index, nil
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
	// mu protects engine-key-level check-and-act operations (Evict's allEmpty
	// check + mapping removal vs Add's pod entry insertion) to prevent TOCTOU races.
	mu sync.Mutex
	// owners and engineEntries track distinct engine references to resident entries.
	owners        map[BlockHash]map[PodEntry]sets.Set[BlockHash]
	engineEntries map[BlockHash]map[BlockHash]sets.Set[PodEntry]
	// constituents groups smaller engine blocks that jointly back one canonical key.
	constituents map[BlockHash]sets.Set[BlockHash]
	// data holds the mapping of requestKeys to sets of pod identifiers.
	data *lruStore
	// engineToRequestKeys holds the mapping of engineKeys to requestKeys.
	engineToRequestKeys *lru.Cache[BlockHash, []BlockHash]
	// podCacheSize is the maximum number of pod entries per key.
	podCacheSize int
	// pods and tiers assign the ordinals EntryRef carries. Neither is
	// reclaimed when entries leave the index: pod identifiers are endpoint
	// address:port values, bounded by the pod network's address space, and
	// tiers are the engine-reported names. Each is capped, and an Add past
	// a cap fails rather than admitting an entry that cannot be matched.
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
	entries []EntryRef
	// capacity bounds len(entries); adding beyond it evicts the least
	// recent entry.
	capacity int
}

// addAll inserts records with LRU semantics: an existing entry refreshes its
// recency, a new entry appends, and overflow evicts the least recent entry.
func (pc *PodCache) addAll(recs []EntryRef) {
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
			if pc.entries[i].PodEntry == entry {
				pc.entries = append(pc.entries[:i], pc.entries[i+1:]...)
				break
			}
		}
	}
	return len(pc.entries) == 0
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

	// Intern once per call, before anything is written: a rejected batch
	// leaves no mapping and no ordinal behind. The same records apply to
	// every request key.
	records, err := m.internRecords(entries)
	if err != nil {
		return err
	}

	// Serialize mapping invalidation with residency insertion.
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.addLocked(ctx, engineKeys, requestKeys, records, nil)
}

// AddByEngineKeys holds the invalidation lock through mapping resolution and insertion.
func (m *InMemoryIndex) AddByEngineKeys(ctx context.Context, engineKeys []BlockHash, entries []PodEntry) (int, error) {
	if len(engineKeys) == 0 || len(entries) == 0 {
		return 0, nil
	}
	records, err := m.internRecords(entries)
	if err != nil {
		return 0, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	seen := sets.New[BlockHash]()
	var requestKeys []BlockHash
	pending := make(map[BlockHash]sets.Set[BlockHash])
	for _, key := range engineKeys {
		complete := true
		for peer := range m.constituents[key] {
			if !m.engineToRequestKeys.Contains(peer) {
				complete = false
				break
			}
		}
		if !complete {
			continue
		}
		mappedKeys, found := m.engineToRequestKeys.Get(key)
		if !found {
			continue
		}
		for _, requestKey := range mappedKeys {
			if pending[requestKey] == nil {
				pending[requestKey] = sets.New[BlockHash]()
			}
			pending[requestKey].Insert(key)
			if !seen.Has(requestKey) {
				seen.Insert(requestKey)
				requestKeys = append(requestKeys, requestKey)
			}
		}
	}
	if len(requestKeys) == 0 {
		return 0, nil
	}
	if err := m.addLocked(ctx, nil, requestKeys, records, pending); err != nil {
		return 0, err
	}
	return len(requestKeys), nil
}

func (m *InMemoryIndex) addLocked(ctx context.Context, engineKeys, requestKeys []BlockHash, records []EntryRef, pending map[BlockHash]sets.Set[BlockHash]) error {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Add")
	var unmappedRequestKeys map[BlockHash]struct{}

	// Build engine->request mappings when engine keys are provided.
	// The ratio of array lengths determines the mapping type:
	//   equal  (4 eng, 4 req) -> 1:1   E0->R0, E1->R1, ...
	//   many:1 (4 eng, 1 req) -> E0->R0, E1->R0, E2->R0, E3->R0
	//   1:many (1 eng, 4 req) -> E0->[R0, R1, R2, R3]
	if engineKeys != nil {
		pending = make(map[BlockHash]sets.Set[BlockHash])
		mappings := engineToRequestMapping(engineKeys, requestKeys)
		evicted := false
		for ek, rks := range mappings {
			if previous, found := m.engineToRequestKeys.Peek(ek); found && !slices.Equal(previous, rks) {
				m.releaseEngine(ek)
			}
			evicted = m.engineToRequestKeys.Add(ek, rks) || evicted
			for _, rk := range rks {
				if pending[rk] == nil {
					pending[rk] = sets.New[BlockHash]()
				}
				pending[rk].Insert(ek)
			}
		}
		if len(engineKeys) > len(requestKeys) {
			groups := make(map[BlockHash]sets.Set[BlockHash])
			for engine, keys := range mappings {
				if groups[keys[0]] == nil {
					groups[keys[0]] = sets.New[BlockHash]()
				}
				groups[keys[0]].Insert(engine)
			}
			for _, group := range groups {
				for engine := range group {
					if m.engineToRequestKeys.Contains(engine) {
						m.constituents[engine] = group
					}
				}
			}
		}
		// Do not reinsert residency whose mapping expired while admitting the batch.
		if evicted {
			for ek, rks := range mappings {
				if m.engineToRequestKeys.Contains(ek) {
					continue
				}
				if unmappedRequestKeys == nil {
					unmappedRequestKeys = make(map[BlockHash]struct{})
				}
				for _, rk := range rks {
					unmappedRequestKeys[rk] = struct{}{}
				}
			}
		}
	}

	// Store requestKey -> PodCache mappings for all request keys.
	for _, requestKey := range requestKeys {
		if _, unmapped := unmappedRequestKeys[requestKey]; unmapped {
			continue
		}
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
		for engine := range pending[requestKey] {
			if m.engineToRequestKeys.Contains(engine) {
				m.addOwners(engine, requestKey, records)
			}
		}
		m.pruneOwners(requestKey, podCache)

		if traceLogger.Enabled() {
			traceLogger.Info("added pods to key", "requestKey", requestKey, "pods", records)
		}
	}

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

	m.mu.Lock()
	defer m.mu.Unlock()
	switch keyType {
	case EngineKey:
		for _, key := range keys {
			requestKeys, found := m.engineToRequestKeys.Peek(key)
			if !found {
				continue
			}
			for _, rk := range requestKeys {
				var remove []PodEntry
				for _, entry := range entries {
					m.removeDependentOwners(key, rk, entry)
					if len(m.owners[rk][entry]) == 0 {
						remove = append(remove, entry)
					}
				}
				m.evictPodsFromRequestKey(rk, key, remove, traceLogger)
			}
			keepMapping := len(m.engineEntries[key]) > 0
			for _, rk := range requestKeys {
				if pc, found := m.data.Peek(rk); found {
					pc.mu.Lock()
					for _, record := range pc.entries {
						if len(m.owners[rk][record.PodEntry]) == 0 {
							keepMapping = true
						}
					}
					pc.mu.Unlock()
				}
			}
			if !keepMapping {
				m.engineToRequestKeys.Remove(key)
			}
		}
		return nil
	case RequestKey:
		for _, key := range keys {
			for _, entry := range entries {
				for engine := range m.owners[key][entry] {
					m.removeOwner(engine, key, entry)
				}
			}
			m.evictPodsFromRequestKey(key, EmptyBlockHash, entries, traceLogger)
		}
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

	isEmpty := podCache.removeAll(entries)

	traceLogger.Info("evicted pods from key", "requestKey", requestKey, "engineKey", engineKey, "pods", entries)

	if !isEmpty {
		return
	}

	// Remove key from main cache if empty.
	// Re-fetch and hold the lock through removal to prevent racing with Add.
	currentCache, stillExists := m.data.Peek(requestKey)
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
// O(N) over the index for a pod it has held, and O(1) for a pod it never held.
// Clear serializes ownership cleanup with Add and Evict; Lookup does not acquire
// the ownership lock.
//
// Context cancellation does not interrupt Clear.
//
// The engineKey->requestKey mapping (engineToRequestKeys) is intentionally left
// untouched: it is LRU-bounded, self-heals when the pod re-Adds the same prefixes,
// and any stale mapping resolves to an emptied request key that correctly breaks
// the prefix chain in Lookup.
func (m *InMemoryIndex) Clear(ctx context.Context, podIdentifier string) error {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Clear")

	// A pod is interned by its first Add and never released.
	m.pods.mu.Lock()
	_, interned := m.pods.ids[podIdentifier]
	m.pods.mu.Unlock()
	if !interned {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, requestKey := range m.data.Keys() {
		// Peek so a clear does not promote LRU recency on keys it scans.
		podCache, found := m.data.Peek(requestKey)
		if !found || podCache == nil {
			continue
		}

		matched := podCache.matching(podIdentifier)

		if len(matched) > 0 {
			for _, entry := range matched {
				for engine := range m.owners[requestKey][entry] {
					m.removeOwner(engine, requestKey, entry)
				}
			}
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

// addOwners records idempotent engine references. Callers hold mu.
func (m *InMemoryIndex) addOwners(engine, request BlockHash, records []EntryRef) {
	for _, record := range records {
		if record.Speculative {
			continue
		}
		entry := record.PodEntry
		if m.owners[request] == nil {
			m.owners[request] = make(map[PodEntry]sets.Set[BlockHash])
		}
		if m.owners[request][entry] == nil {
			m.owners[request][entry] = sets.New[BlockHash]()
		}
		m.owners[request][entry].Insert(engine)
		if m.engineEntries[engine] == nil {
			m.engineEntries[engine] = make(map[BlockHash]sets.Set[PodEntry])
		}
		if m.engineEntries[engine][request] == nil {
			m.engineEntries[engine][request] = sets.New[PodEntry]()
		}
		m.engineEntries[engine][request].Insert(entry)
	}
}

func (m *InMemoryIndex) removeOwner(engine, request BlockHash, entry PodEntry) {
	delete(m.owners[request][entry], engine)
	if len(m.owners[request][entry]) == 0 {
		delete(m.owners[request], entry)
	}
	if len(m.owners[request]) == 0 {
		delete(m.owners, request)
	}
	delete(m.engineEntries[engine][request], entry)
	if len(m.engineEntries[engine][request]) == 0 {
		delete(m.engineEntries[engine], request)
	}
	if len(m.engineEntries[engine]) == 0 {
		delete(m.engineEntries, engine)
	}
}

func (m *InMemoryIndex) forgetRequestOwners(request BlockHash) {
	for entry, engines := range m.owners[request] {
		for engine := range engines {
			m.removeOwner(engine, request, entry)
		}
	}
}

// releaseEngine removes only entries with no surviving engine owner.
func (m *InMemoryIndex) releaseEngine(engine BlockHash) {
	for request, entries := range m.engineEntries[engine] {
		var remove []PodEntry
		for entry := range entries {
			m.removeDependentOwners(engine, request, entry)
			if len(m.owners[request][entry]) == 0 {
				remove = append(remove, entry)
			}
		}
		if pc, found := m.data.Peek(request); found {
			if pc.removeAll(remove) {
				m.data.Remove(request)
			}
		}
	}
	if m.constituents[engine] != nil {
		delete(m.constituents, engine)
	}

}

// Per-key entry eviction must not leave ownership that can preserve a later insertion.
func (m *InMemoryIndex) pruneOwners(request BlockHash, pc *PodCache) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	for entry, engines := range m.owners[request] {
		present := false
		for _, record := range pc.entries {
			if record.PodEntry == entry {
				present = true
				break
			}
		}
		if !present {
			for engine := range engines {
				m.removeOwner(engine, request, entry)
			}
		}
	}
}

// Losing any constituent invalidates its canonical entry; siblings in a 1:many
// split remain independent owners.
func (m *InMemoryIndex) removeDependentOwners(engine, request BlockHash, entry PodEntry) {
	if group := m.constituents[engine]; group != nil {
		for peer := range group {
			m.removeOwner(peer, request, entry)
		}
	} else {
		m.removeOwner(engine, request, entry)
	}
}
