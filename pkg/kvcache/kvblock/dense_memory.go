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

package kvblock

import (
	"context"
	"fmt"
	"math/bits"
	"sort"
	"sync"
	"time"

	"github.com/dustin/go-humanize"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

// defaultDenseMemoryBudget bounds the index by resident bytes.
const defaultDenseMemoryBudget = "2GiB"

// maxDensePodOrdinals is the lifetime supply of pod ordinals. Ordinals are
// never reused, and clearing a pod consumes one on revival, so the supply is
// sized for years of clears at fleet scale.
const maxDensePodOrdinals = 1 << 20

// The sweep watermark sits below the budget so a sweep evicts past the
// trigger point and the next additions land inside the budget without
// retriggering immediately.
const (
	denseWatermarkNum = 7
	denseWatermarkDen = 8
	sweepSampleSize   = 128
	sweepMaxBursts    = 16
	// denseStampSlack is how long a key's last-visit stamp stays fresh, in
	// seconds. Reads restamp a key only past it, so the read path pays one
	// comparison per key and one batched write acquisition per slack window,
	// and the sweep's recency signal stays accurate to the slack even when
	// residency crosses the budget long after the last read.
	denseStampSlack = 5
)

// Per-structure residency estimates. A Go map's per-entry cost oscillates
// with the growth cycle (the density budget table); these are the mid-cycle
// values. The sweep is a safety valve against the budget, not the density
// gate, which measures HeapAlloc directly.
const (
	tableBytesPerKey    = 40
	engineBytesPerEntry = 32
	groupRowBytes       = 48
)

// denseSpecTier is the group-row tier value for speculative rows, which
// carry no device tier. It cannot collide with a real ordinal (the dense
// interner stops at 7).
const denseSpecTier uint8 = 0xff

// DenseMemoryIndexConfig holds the configuration for the DenseMemoryIndex.
type DenseMemoryIndexConfig struct {
	// Size is the memory budget for the index's resident structures, in the
	// human-readable encoding CostAwareMemoryIndexConfig uses ("2GiB",
	// "500MiB"). The table grows dynamically; crossing the budget triggers
	// sampled eviction down to a watermark below the budget.
	Size string `json:"size,omitempty"`
	// PodCacheSize is the maximum number of pods per key: 3 inline entry
	// words plus arena chunks of 3. Values above 15 are rejected; a value
	// that is not 3 plus a multiple of 3 wastes trailing chunk slots.
	// A non-positive value selects 15.
	PodCacheSize int `json:"podCacheSize,omitempty"`
}

// DefaultDenseMemoryIndexConfig returns a default configuration for the
// DenseMemoryIndex.
func DefaultDenseMemoryIndexConfig() *DenseMemoryIndexConfig {
	return &DenseMemoryIndexConfig{
		Size:         defaultDenseMemoryBudget,
		PodCacheSize: denseMaxPodsPerKey,
	}
}

// denseGroupKey identifies one group-announced copy of a (pod, tier,
// speculative) pair under a key.
type denseGroupKey struct {
	pod   uint32
	tier  uint8
	spec  bool
	group GroupID
}

// DenseMemoryIndex is a memory-bound Index implementation with a dense
// in-memory representation: one map[BlockHash]keySlot table whose value-type
// slots hold three inline entry words each, a shared overflow arena, and the
// engine-key map, all under one RWMutex. Clear is O(1) through a dead-pod
// bitmap whose ordinals revive on re-Add; residency is bounded by a byte
// budget enforced through sampled eviction.
type DenseMemoryIndex struct {
	// mu guards every mutable field below and the internals of
	// engineToRequestKeys, which is itself lock-free. The interners' own
	// locks are nested inside mu, never the other way around.
	mu                  sync.RWMutex
	table               map[BlockHash]keySlot
	arena               denseArena
	engineToRequestKeys *engineKeyMap
	// groups holds the group side table: one row per live group-announced
	// copy, keyed per request key. A row exists while the copy is stored;
	// the entry word's tier bit survives until the last row for its (pod,
	// tier, speculative) drains, so grouped and ungrouped copies of one pod
	// and tier coexist exactly as full-PodEntry equality keeps them apart.
	groups    map[BlockHash]map[denseGroupKey]struct{}
	groupRows int
	pods      *interner
	tiers     *interner
	dead      []uint64
	budget    int64
	watermark int64
	podCap    int
	truncated int64
}

var _ Index = &DenseMemoryIndex{}
var _ KeyWalker = &DenseMemoryIndex{}
var _ OrdinalResolver = &DenseMemoryIndex{}
var _ StatsProvider = &DenseMemoryIndex{}

// NewDenseMemoryIndex creates a new DenseMemoryIndex instance.
func NewDenseMemoryIndex(cfg *DenseMemoryIndexConfig) (*DenseMemoryIndex, error) {
	if cfg == nil {
		cfg = DefaultDenseMemoryIndexConfig()
	}
	budget, err := humanize.ParseBytes(cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize dense memory index: %w", err)
	}
	if budget <= 0 {
		return nil, fmt.Errorf("failed to initialize dense memory index: size %q is not positive", cfg.Size)
	}
	podCap := cfg.PodCacheSize
	if podCap <= 0 {
		podCap = denseMaxPodsPerKey
	}
	if podCap > denseMaxPodsPerKey {
		return nil, fmt.Errorf("failed to initialize dense memory index: podCacheSize %d exceeds %d",
			podCap, denseMaxPodsPerKey)
	}
	return &DenseMemoryIndex{
		table:               make(map[BlockHash]keySlot),
		engineToRequestKeys: newEngineKeyMap(0), // the byte budget governs
		groups:              make(map[BlockHash]map[denseGroupKey]struct{}),
		pods:                newInterner(maxDensePodOrdinals),
		tiers:               newInterner(maxDenseTierOrdinal + 1),
		budget:              int64(budget),                                         //#nosec G115 -- ParseBytes is bounded by int64
		watermark:           int64(budget) / denseWatermarkDen * denseWatermarkNum, //#nosec G115 -- bounded by budget
		podCap:              podCap,
	}, nil
}

// PodOrdinal returns the ordinal assigned to the pod identifier, and whether
// the index has interned it.
func (d *DenseMemoryIndex) PodOrdinal(name string) (uint32, bool) {
	return d.pods.ordinalOf(name)
}

// PodName returns the pod identifier assigned to ordinal, and whether one is.
func (d *DenseMemoryIndex) PodName(ordinal uint32) (string, bool) {
	return d.pods.nameOf(ordinal)
}

// TierOrdinal returns the ordinal assigned to the device tier, and whether
// the index has interned it.
func (d *DenseMemoryIndex) TierOrdinal(name string) (uint32, bool) {
	return d.tiers.ordinalOf(name)
}

// TierName returns the device tier assigned to ordinal, and whether one is.
func (d *DenseMemoryIndex) TierName(ordinal uint32) (string, bool) {
	return d.tiers.nameOf(ordinal)
}

// Stats returns a snapshot of the index's residency and occupancy.
func (d *DenseMemoryIndex) Stats() MemoryIndexStats {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.statsLocked()
}

func (d *DenseMemoryIndex) statsLocked() MemoryIndexStats {
	return MemoryIndexStats{
		ResidentKeys:     len(d.table),
		ResidentBytes:    d.residencyLocked(),
		EngineMappings:   d.engineToRequestKeys.len(),
		GroupRows:        d.groupRows,
		TruncatedEntries: d.truncated,
	}
}

// residencyLocked estimates the resident bytes of every structure. Called
// with mu held.
func (d *DenseMemoryIndex) residencyLocked() int64 {
	bytes := int64(len(d.table)) * tableBytesPerKey
	bytes += int64(len(d.arena.words)) * 4
	bytes += int64(d.engineToRequestKeys.len()) * engineBytesPerEntry
	bytes += int64(len(d.engineToRequestKeys.arena)) * 16
	bytes += int64(d.groupRows) * groupRowBytes
	return bytes
}

// isDead reports whether the pod ordinal was cleared. Called with mu held.
func (d *DenseMemoryIndex) isDead(pod uint32) bool {
	i, j := int(pod/64), int(pod%64)
	return i < len(d.dead) && d.dead[i]&(1<<j) != 0
}

func (d *DenseMemoryIndex) setDead(pod uint32) {
	i := int(pod / 64)
	for len(d.dead) <= i {
		d.dead = append(d.dead, 0)
	}
	d.dead[i] |= 1 << (pod % 64)
}

// tick returns the current stamp: wall-clock seconds, so a stamp's age is
// directly comparable against denseStampSlack. Ties within a second are
// expected; the sweep selects its quarter by count, not by stamp value.
func (d *DenseMemoryIndex) tick() uint32 {
	return uint32(time.Now().Unix()) //#nosec G115 -- unix seconds fit uint32 past 2106
}

// internBatchLocked pairs each entry with its pod and tier ordinals, reviving
// a pod name whose current ordinal is dead. The batch is assigned all or
// nothing: when its new ordinals would exceed a cap, nothing is interned and
// the error names the cap. Called with mu held for writing.
func (d *DenseMemoryIndex) internBatchLocked(entries []PodEntry) ([]EntryRef, error) {
	d.pods.mu.Lock()
	defer d.pods.mu.Unlock()
	d.tiers.mu.Lock()
	defer d.tiers.mu.Unlock()

	pendingPods := map[string]struct{}{}
	pendingTiers := map[string]struct{}{}
	for _, e := range entries {
		if id, ok := d.pods.ids[e.PodIdentifier]; !ok || d.isDead(id) {
			pendingPods[e.PodIdentifier] = struct{}{}
		}
		if !e.Speculative {
			if _, ok := d.tiers.ids[e.DeviceTier]; !ok {
				pendingTiers[e.DeviceTier] = struct{}{}
			}
		}
	}
	if len(d.pods.names)+len(pendingPods) > maxDensePodOrdinals {
		return nil, fmt.Errorf("%w: %d pod ordinals", errIndexCardinality, maxDensePodOrdinals)
	}
	if len(d.tiers.names)+len(pendingTiers) > maxDenseTierOrdinal+1 {
		return nil, fmt.Errorf("%w: %d device tiers", errIndexCardinality, maxDenseTierOrdinal+1)
	}

	records := make([]EntryRef, len(entries))
	for i, e := range entries {
		revive := false
		if id, ok := d.pods.ids[e.PodIdentifier]; ok && d.isDead(id) {
			revive = true
		}
		pod := d.pods.assignLocked(e.PodIdentifier, revive)
		var tier uint32
		if !e.Speculative {
			tier = d.tiers.assignLocked(e.DeviceTier, false)
		}
		records[i] = EntryRef{PodEntry: e, PodOrdinal: pod, TierOrdinal: tier}
	}
	return records, nil
}

// entryWordFor packs the record's storage word. Speculative entries carry no
// tier bits.
func entryWordFor(ref *EntryRef) uint32 {
	var tiers uint32
	if !ref.Speculative {
		tiers = 1 << ref.TierOrdinal
	}
	return packEntry(ref.PodOrdinal, tiers, ref.Speculative, ref.HasGroup)
}

// Add stores requestKey -> pod entries and (optionally) engineKey ->
// requestKey mappings, mirroring InMemoryIndex.Add's contract. A rejected
// batch leaves no mapping and no ordinal behind.
func (d *DenseMemoryIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.DenseMemoryIndex.Add")

	d.mu.Lock()
	records, err := d.internBatchLocked(entries)
	if err != nil {
		d.mu.Unlock()
		return err
	}
	stamp := d.tick()
	d.mu.Unlock()

	// Each key's slot updates under its own acquisition: the RWMutex blocks
	// new readers behind a waiting writer, so a whole-batch hold would make
	// every concurrent match wait out the batch. No cross-key invariant spans
	// the loop, so releasing between keys is safe.
	for _, requestKey := range requestKeys {
		d.mu.Lock()
		slot := d.table[requestKey]
		// Dead words occupy the cap until the sweep's sampled pass reclaims
		// them; strip them before inserting at the cap so the head-drop
		// cannot evict a live pod's word while a dead one survives.
		if slotLen(&slot, &d.arena) == d.podCap {
			d.stripDeadWordsLocked(requestKey, slot)
			slot = d.table[requestKey]
		}
		for i := range records {
			ref := &records[i]
			if dropped := slotSetEntry(&slot, &d.arena, d.podCap, entryWordFor(ref)); dropped != 0 {
				d.truncated++
				metrics.IndexTruncatedEntries.Add(1)
				d.purgeWordRowsLocked(requestKey, dropped)
			}
			if ref.HasGroup {
				d.groupAddLocked(requestKey, ref)
			}
		}
		slot.stamp = stamp
		d.table[requestKey] = slot
		d.mu.Unlock()
		if traceLogger.Enabled() {
			traceLogger.Info("added pods to key", "requestKey", requestKey, "pods", entries)
		}
	}

	// The engine mappings land after the entries, under their own hold. An
	// engine-key evict that runs between the two either misses the mapping,
	// so its removal no-ops and this Add's set lands last, or sees it with
	// the entries already in place, so the evict's all-empty check cannot
	// drop a mapping whose entries have not landed yet.
	if engineKeys != nil {
		d.mu.Lock()
		for ek, rks := range engineToRequestMapping(engineKeys, requestKeys) {
			d.engineToRequestKeys.set(ek, rks)
		}
		d.mu.Unlock()
	}

	d.maybeSweep()
	return nil
}

// groupAddLocked records a group-announced copy. A re-announcement of a live
// row keeps one row, matching the replace-on-equal insert the non-dense
// backends apply to identical entries.
func (d *DenseMemoryIndex) groupAddLocked(key BlockHash, ref *EntryRef) {
	rows := d.groups[key]
	if rows == nil {
		rows = make(map[denseGroupKey]struct{})
		d.groups[key] = rows
	}
	gk := denseGroupKey{pod: ref.PodOrdinal, tier: denseRowTier(ref), spec: ref.Speculative, group: ref.GroupIdx}
	if _, ok := rows[gk]; !ok {
		rows[gk] = struct{}{}
		d.groupRows++
	}
}

// denseRowTier is the group-row tier value for the record; speculative rows
// carry the sentinel.
func denseRowTier(ref *EntryRef) uint8 {
	if ref.Speculative {
		return denseSpecTier
	}
	return uint8(ref.TierOrdinal) //#nosec G115 -- tier ordinals are capped at maxDenseTierOrdinal
}

// purgeWordRowsLocked removes the group rows a dropped word owned, keeping
// the side table in step with the entry list: rows exist exactly while the
// word carrying their tier bit does. Called with mu held for writing.
func (d *DenseMemoryIndex) purgeWordRowsLocked(key BlockHash, w uint32) {
	if w&entryGroupedBit == 0 {
		return
	}
	rows := d.groups[key]
	if rows == nil {
		return
	}
	pod, spec := entryPod(w), w&entrySpecBit != 0
	for gk := range rows {
		if gk.pod != pod || gk.spec != spec {
			continue
		}
		if !spec && w&(1<<gk.tier) == 0 {
			continue
		}
		delete(rows, gk)
		d.groupRows--
	}
	if len(rows) == 0 {
		delete(d.groups, key)
	}
}

// groupRemoveLocked deletes the row and reports whether the (pod, tier,
// speculative) pair it named has no rows left, which releases the entry
// word's bit.
func (d *DenseMemoryIndex) groupRemoveLocked(key BlockHash, gk denseGroupKey) (existed, drained bool) {
	rows := d.groups[key]
	if rows == nil {
		return false, false
	}
	if _, ok := rows[gk]; !ok {
		return false, false
	}
	delete(rows, gk)
	d.groupRows--
	if len(rows) == 0 {
		delete(d.groups, key)
	}
	for other := range rows {
		if other.pod == gk.pod && other.tier == gk.tier && other.spec == gk.spec {
			return true, false
		}
	}
	return true, true
}

// Evict removes keys and their associated pod entries from the index.
// A non-nil error means the index is unchanged.
func (d *DenseMemoryIndex) Evict(ctx context.Context, keyType KeyType, keys []BlockHash, entries []PodEntry) error {
	if len(keys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for eviction from index")
	}
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.DenseMemoryIndex.Evict")

	d.mu.Lock()
	defer d.mu.Unlock()

	switch keyType {
	case EngineKey:
		// One lock hold covers the resolution, the entry removal, and the
		// all-empty mapping removal, closing the Add-versus-Evict race.
		chains := make([][]BlockHash, len(keys))
		for i, key := range keys {
			rks, found := d.engineToRequestKeys.peek(key)
			if !found {
				traceLogger.Info("engineKey not found in mapping, nothing to evict", "engineKey", key)
				continue
			}
			chains[i] = rks
			for _, rk := range rks {
				d.evictFromKeyLocked(rk, entries)
			}
		}
		for i, key := range keys {
			if chains[i] == nil {
				continue
			}
			allEmpty := true
			for _, rk := range chains[i] {
				if slot, ok := d.table[rk]; ok && slotLen(&slot, &d.arena) > 0 {
					allEmpty = false
					break
				}
			}
			if allEmpty {
				d.engineToRequestKeys.remove(key)
			}
		}
		return nil
	case RequestKey:
		for _, key := range keys {
			d.evictFromKeyLocked(key, entries)
		}
		return nil
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}
}

// evictFromKeyLocked removes the matching entries from one request key,
// deleting the key when its last word clears. Called with mu held for
// writing.
func (d *DenseMemoryIndex) evictFromKeyLocked(requestKey BlockHash, entries []PodEntry) {
	slot, ok := d.table[requestKey]
	if !ok {
		return
	}
	changed := false
	for _, e := range entries {
		pod, ok := d.pods.ordinalOf(e.PodIdentifier)
		if !ok {
			continue
		}
		if e.HasGroup {
			rowTier := denseSpecTier
			var tiers uint32
			if !e.Speculative {
				tier, found := d.tiers.ordinalOf(e.DeviceTier)
				if !found {
					continue
				}
				rowTier = uint8(tier) //#nosec G115 -- tier ordinals are capped at maxDenseTierOrdinal
				tiers = 1 << tier
			}
			gk := denseGroupKey{pod: pod, tier: rowTier, spec: e.Speculative, group: e.GroupIdx}
			existed, drained := d.groupRemoveLocked(requestKey, gk)
			if !existed {
				continue
			}
			changed = true
			if drained {
				slotClearEntry(&slot, &d.arena, packEntry(pod, tiers, e.Speculative, true))
			}
			continue
		}
		if e.Speculative {
			if slotClearEntry(&slot, &d.arena, packEntry(pod, 0, true, false)) {
				changed = true
			}
			continue
		}
		tier, found := d.tiers.ordinalOf(e.DeviceTier)
		if !found {
			continue
		}
		if slotClearEntry(&slot, &d.arena, packEntry(pod, 1<<tier, false, false)) {
			changed = true
		}
	}
	if changed {
		if slotLen(&slot, &d.arena) == 0 {
			d.deleteKeyLocked(requestKey, slot, "empty")
		} else {
			d.table[requestKey] = slot
		}
	}
}

// deleteKeyLocked frees a key's arena chunks and removes it from the table.
func (d *DenseMemoryIndex) deleteKeyLocked(key BlockHash, slot keySlot, cause string) {
	for h := slot.chain; h != 0; {
		next := d.arena.words[(int(h)-1)*denseChunkWords+denseChunkLink]
		d.arena.freeChunk(h)
		h = next
	}
	delete(d.table, key)
	if rows := d.groups[key]; rows != nil {
		d.groupRows -= len(rows)
		delete(d.groups, key)
	}
	metrics.IndexEvictions.WithLabelValues(cause).Add(1)
}

// Clear removes every entry for the pod from the index: the pod's current
// ordinal is marked dead and reads filter it, so Clear is O(1). The dead
// ordinal's slots are freed by the sampled eviction pass; a re-Add revives
// the name under a fresh ordinal.
func (d *DenseMemoryIndex) Clear(ctx context.Context, podIdentifier string) error {
	start := time.Now()
	d.mu.Lock()
	if pod, ok := d.pods.ordinalOf(podIdentifier); ok {
		d.setDead(pod)
	}
	d.mu.Unlock()
	metrics.IndexClearDuration.Observe(time.Since(start).Seconds())
	log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.DenseMemoryIndex.Clear").
		Info("cleared pod from index", "pod", podIdentifier)
	return nil
}

// GetRequestKey returns the last request key associated with the given
// engineKey, the highest-index request key the engine key covers. It takes
// the write lock: the engine map's get refreshes the slot's recency stamp,
// which mutates the map, so concurrent lookups cannot share a read hold.
func (d *DenseMemoryIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	d.mu.Lock()
	rk, found := d.engineToRequestKeys.get(engineKey)
	d.mu.Unlock()
	if !found {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}
	return rk, nil
}

// Lookup receives a list of requestKeys and a set of pod identifiers, and
// retrieves the filtered pods associated with those keys, stopping at the
// first missing or filtered-empty key like InMemoryIndex.Lookup. Names are
// resolved through the interners while the table read lock is held; the
// interner write path is always nested inside the table write lock, so the
// read-read nesting cannot deadlock.
func (d *DenseMemoryIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no requestKeys provided for lookup")
	}
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.DenseMemoryIndex.Lookup")

	podsPerKey := make(map[BlockHash][]PodEntry)
	visited := 0
	stale := false
	fresh := d.tick() - denseStampSlack
	defer func() { d.stampPrefix(requestKeys[:visited], stale) }()

	for idx, requestKey := range requestKeys {
		if idx&cancellationCheckMask == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		d.mu.RLock()
		slot, found := d.table[requestKey]
		var filtered []PodEntry
		if found {
			if slot.stamp < fresh {
				stale = true
			}
			filtered = d.materializeLocked(requestKey, &slot, podIdentifierSet)
		}
		d.mu.RUnlock()
		if !found {
			break
		}
		visited = idx + 1
		if len(filtered) == 0 {
			break
		}
		podsPerKey[requestKey] = filtered
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if traceLogger.Enabled() {
		traceLogger.Info("lookup completed", "highest-hit-index", visited-1, "pods-per-key", podsPerKeyPrintHelper(podsPerKey))
	}
	return podsPerKey, nil
}

// materializeLocked decodes a key's entries into PodEntries, skipping dead
// pods and entries outside the filter. Called with mu held for reading.
func (d *DenseMemoryIndex) materializeLocked(key BlockHash, slot *keySlot, filter sets.Set[string]) []PodEntry {
	var out []PodEntry
	forEachSlotWord(slot, &d.arena, func(w uint32) bool {
		pod := entryPod(w)
		if d.isDead(pod) {
			return true
		}
		podName, ok := d.pods.nameOf(pod)
		if !ok {
			return true
		}
		if filter.Len() > 0 && !filter.Has(podName) {
			return true
		}
		spec := w&entrySpecBit != 0
		if w&entryGroupedBit != 0 {
			for gk := range d.groups[key] {
				if gk.pod != pod || gk.spec != spec {
					continue
				}
				out = append(out, PodEntry{
					PodIdentifier: podName,
					DeviceTier:    d.rowTierNameLocked(gk.tier),
					Speculative:   gk.spec,
					HasGroup:      true,
					GroupIdx:      gk.group,
				})
			}
			return true
		}
		if spec {
			out = append(out, PodEntry{PodIdentifier: podName, Speculative: true})
			return true
		}
		for mask := entryTiers(w); mask != 0; {
			b := mask & -mask
			mask ^= b
			tierName, _ := d.tiers.nameOf(uint32(bits.TrailingZeros32(b))) //#nosec G115 -- returns 0-31
			out = append(out, PodEntry{PodIdentifier: podName, DeviceTier: tierName})
		}
		return true
	})
	return out
}

func (d *DenseMemoryIndex) rowTierNameLocked(tier uint8) string {
	if tier == denseSpecTier {
		return ""
	}
	name, _ := d.tiers.nameOf(uint32(tier))
	return name
}

// denseWalkState is the reusable copy-out buffer for walks: the per-key
// found flags and entry ranges of the current batch, and the decoded entries.
type denseWalkState struct {
	found   []bool
	bounds  [][2]int
	entries []WalkEntry
}

var walkStatePool = sync.Pool{New: func() any { return &denseWalkState{} }}

// WalkKeys implements KeyWalker. Batches of keys are copied out under one
// read acquisition, decoding the packed words into WalkEntries (dead pods
// filtered, one entry per pod and tier pair), and the visits run after the
// lock is released, so no lock is held across a visit callback. The visited
// prefix's stamps update under one write acquisition when the walk ends.
func (d *DenseMemoryIndex) WalkKeys(ctx context.Context, requestKeys []BlockHash,
	visit func(pos int, found bool, entries []WalkEntry) bool,
) error {
	ws := walkStatePool.Get().(*denseWalkState)
	defer func() {
		ws.entries = ws.entries[:0]
		walkStatePool.Put(ws)
	}()
	visited := 0
	stale := false
	fresh := d.tick() - denseStampSlack
	defer func() { d.stampPrefix(requestKeys[:visited], stale) }()

	batch := walkBatchMin
	for start := 0; start < len(requestKeys); {
		end := min(start+batch, len(requestKeys))
		keys := requestKeys[start:end]

		ws.found = ws.found[:0]
		ws.bounds = ws.bounds[:0]
		ws.entries = ws.entries[:0]
		d.mu.RLock()
		for _, key := range keys {
			b0 := len(ws.entries)
			slot, found := d.table[key]
			if found {
				if slot.stamp < fresh {
					stale = true
				}
				d.decodeLocked(&slot, ws, b0)
			}
			ws.found = append(ws.found, found)
			ws.bounds = append(ws.bounds, [2]int{b0, len(ws.entries)})
		}
		d.mu.RUnlock()

		for i := range keys {
			pos := start + i
			if pos&cancellationCheckMask == 0 && ctx.Err() != nil {
				return ctx.Err()
			}
			if ws.found[i] {
				visited = pos + 1
			}
			if !visit(pos, ws.found[i], ws.entries[ws.bounds[i][0]:ws.bounds[i][1]]) {
				return ctx.Err()
			}
		}
		start = end
		batch = min(2*batch, walkBatchMax)
	}
	return ctx.Err()
}

// decodeLocked appends the slot's walk entries to ws.entries, skipping dead
// pods and folding duplicate (pod, tier, speculative) triples that grouped
// and ungrouped words of one pod can both carry. The fold is scoped to the
// key's own entries, which start at ws.entries[from]. Called with mu held
// for reading.
func (d *DenseMemoryIndex) decodeLocked(slot *keySlot, ws *denseWalkState, from int) {
	forEachSlotWord(slot, &d.arena, func(w uint32) bool {
		pod := entryPod(w)
		if d.isDead(pod) {
			return true
		}
		emit := func(e WalkEntry) {
			for i := from; i < len(ws.entries); i++ {
				if ws.entries[i] == e {
					return
				}
			}
			ws.entries = append(ws.entries, e)
		}
		if w&entrySpecBit != 0 {
			// The consumer ignores a speculative entry's tier ordinal; the
			// flag is the identity.
			emit(WalkEntry{PodOrdinal: pod, Speculative: true})
			return true
		}
		for mask := entryTiers(w); mask != 0; {
			b := mask & -mask
			mask ^= b
			emit(WalkEntry{PodOrdinal: pod, TierOrdinal: uint32(bits.TrailingZeros32(b))}) //#nosec G115 -- returns 0-31
		}
		return true
	})
}

// stampPrefix refreshes the stamps of keys whose last visit is older than
// denseStampSlack, under one write acquisition, skipping keys absent by
// then. stale is the caller's copy-out finding that at least one visited
// key qualifies; when it does not, the read path never takes a write
// acquisition. The stamps feed only the byte-budget sweep's recency
// ordering; Add refreshes the stamp of every key it touches.
func (d *DenseMemoryIndex) stampPrefix(keys []BlockHash, stale bool) {
	if len(keys) == 0 || !stale {
		return
	}
	stamp := d.tick()
	fresh := stamp - denseStampSlack
	d.mu.Lock()
	for _, key := range keys {
		if slot, ok := d.table[key]; ok && slot.stamp < fresh {
			slot.stamp = stamp
			d.table[key] = slot
		}
	}
	d.mu.Unlock()
}

// maybeSweep runs the sampled eviction when residency crosses the budget.
// Each burst samples and evicts under its own write acquisition, so no burst
// holds the whole table for long.
func (d *DenseMemoryIndex) maybeSweep() {
	d.mu.RLock()
	over := d.residencyLocked() > d.budget
	d.mu.RUnlock()
	if !over {
		return
	}
	for range sweepMaxBursts {
		d.mu.Lock()
		if d.residencyLocked() <= d.watermark {
			d.updateGaugesLocked()
			d.mu.Unlock()
			return
		}
		d.sweepBurstLocked()
		d.mu.Unlock()
	}
}

// sweepBurstLocked evicts from a sample of the table: every sampled slot
// whose entries are all dead, plus the oldest quarter by stamp, and the same
// from the engine map when it holds a quarter or more of residency. Called
// with mu held for writing.
func (d *DenseMemoryIndex) sweepBurstLocked() {
	type victim struct {
		key     BlockHash
		slot    keySlot
		allDead bool
		deleted bool
	}
	var victims []victim
	n := 0
	for key, slot := range d.table {
		allDead := true
		forEachSlotWord(&slot, &d.arena, func(w uint32) bool {
			if !d.isDead(entryPod(w)) {
				allDead = false
				return false
			}
			return true
		})
		victims = append(victims, victim{key: key, slot: slot, allDead: allDead})
		n++
		if n >= sweepSampleSize {
			break
		}
	}
	live := make([]int, 0, len(victims))
	for i := range victims {
		if !victims[i].allDead {
			live = append(live, i)
		}
	}
	if len(live) > 0 {
		// Select the oldest quarter by count, not by stamp value: wall-second
		// stamps tie heavily, and a value threshold would take every tied
		// victim at the boundary, not a quarter of the sample.
		sort.SliceStable(live, func(i, j int) bool { return victims[live[i]].slot.stamp < victims[live[j]].slot.stamp })
		for _, i := range live[:oldestQuarter(len(live))] {
			d.deleteKeyLocked(victims[i].key, victims[i].slot, "budget")
			victims[i].deleted = true
		}
	}
	for i := range victims {
		if victims[i].deleted {
			continue
		}
		if victims[i].allDead {
			d.deleteKeyLocked(victims[i].key, victims[i].slot, "dead")
			continue
		}
		// A mixed slot keeps its dead words filtered at read time; strip
		// them here so the sampled pass reclaims their storage.
		d.stripDeadWordsLocked(victims[i].key, victims[i].slot)
	}

	engineBytes := int64(d.engineToRequestKeys.len()) * engineBytesPerEntry
	if engineBytes*4 > d.residencyLocked() {
		d.sweepEngineLocked()
	}
}

// stripDeadWordsLocked removes the dead pods' words from a sampled key's
// slot, deleting the key when nothing live remains. Called with mu held for
// writing.
func (d *DenseMemoryIndex) stripDeadWordsLocked(key BlockHash, slot keySlot) {
	n := slotLen(&slot, &d.arena)
	for i := 0; i < n; {
		w := slotWord(&slot, &d.arena, i)
		if d.isDead(entryPod(w)) {
			d.purgeWordRowsLocked(key, w)
			slotRemoveAt(&slot, &d.arena, i, n)
			n--
			continue
		}
		i++
	}
	if n == 0 {
		d.deleteKeyLocked(key, slot, "dead")
		return
	}
	d.table[key] = slot
}

// sweepEngineLocked evicts the oldest quarter of an engine-map sample.
// Called with mu held for writing.
func (d *DenseMemoryIndex) sweepEngineLocked() {
	type mapping struct {
		key   BlockHash
		stamp uint32
	}
	var mappings []mapping
	n := 0
	for ek, slot := range d.engineToRequestKeys.slots {
		mappings = append(mappings, mapping{key: ek, stamp: slot.stamp})
		n++
		if n >= sweepSampleSize {
			break
		}
	}
	if len(mappings) == 0 {
		return
	}
	// Select the oldest quarter by count, not by stamp value: wall-second
	// stamps tie heavily, and a value threshold would take every tied
	// mapping at the boundary, not a quarter of the sample.
	sort.SliceStable(mappings, func(i, j int) bool { return mappings[i].stamp < mappings[j].stamp })
	for _, m := range mappings[:oldestQuarter(len(mappings))] {
		d.engineToRequestKeys.remove(m.key)
		metrics.IndexEvictions.WithLabelValues("budget").Add(1)
	}
}

// oldestQuarter returns the count of the oldest quarter of a sample of n,
// rounding up so a sample always yields at least one victim.
func oldestQuarter(n int) int {
	return (n + 3) / 4
}

// updateGaugesLocked refreshes the residency gauges. Called with mu held.
func (d *DenseMemoryIndex) updateGaugesLocked() {
	stats := d.statsLocked()
	metrics.IndexResidentKeys.Set(float64(stats.ResidentKeys))
	metrics.IndexResidentBytes.Set(float64(stats.ResidentBytes))
}
