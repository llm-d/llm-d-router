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
	"sync/atomic"
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

// denseShardCount is the number of independent write domains the table is
// split into. Request keys hash uniformly over the shards, so each shard
// carries an equal share of the keys and its own lock; writers to different
// shards proceed in parallel.
const denseShardCount = 16

// The sweep watermark sits below the budget so a sweep evicts past the
// trigger point and the next additions land inside the budget without
// retriggering immediately.
const (
	denseWatermarkNum = 7
	denseWatermarkDen = 8
	sweepSampleSize   = 128
	sweepMaxBursts    = 16
)

// Per-structure residency estimates. The table's per-key cost is the slot
// array at the load bound (32 B/slot at 7/10 load); like a Go map's, the
// real array oscillates with the doubling growth cycle, and the engine
// entry's 48 B is that cycle's average for its 24 B slots, measured at the
// 1:1 geometry. The sweep is a safety valve against the budget, not the
// density gate, which measures HeapAlloc directly.
const (
	tableBytesPerKey    = 46
	engineBytesPerEntry = 48
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
	// SampleModulus is the power-of-two modulus the index samples request
	// keys at: 0 or 1 keeps every key, 16 keeps roughly one key in 16. The
	// engine-key map is not sampled; it keeps every pair. Match results
	// keep block granularity with a resolution of up to modulus-1 blocks.
	SampleModulus int `json:"sampleModulus,omitempty"`
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

// denseShard is one independent share of the table: its own lock, slot
// array, arena, group rows, and sweep cursor. A request key belongs to
// exactly one shard, chosen by its hash.
type denseShard struct {
	mu          sync.RWMutex
	tbl         *denseTable
	arena       denseArena
	sweepCursor int
	// groups holds the group side table: one row per live group-announced
	// copy, keyed per request key. A row exists while the copy is stored;
	// the entry word's tier bit survives until the last row for its (pod,
	// tier, speculative) drains, so grouped and ungrouped copies of one pod
	// and tier coexist exactly as full-PodEntry equality keeps them apart.
	groups    map[BlockHash]map[denseGroupKey]struct{}
	groupRows int
	truncated int64
}

// DenseMemoryIndex is a memory-bound Index implementation with a dense
// in-memory representation: the request-key table is split into
// power-of-two shards, each an open-addressed slot array whose value-type
// slots hold three inline entry words each plus a shared overflow arena.
// The engine-key map, the pod and tier interners, and the dead-pod bitmap
// are shared; the lock order is shard locks in index order, then the engine
// map's lock, then the interners' locks. Slot addressability lets reads
// record recency evidence with one atomic OR and no write acquisition.
// Clear is O(1) through the dead-pod bitmap whose ordinals revive on
// re-Add; residency is bounded by a byte budget enforced through sampled
// eviction, each shard against its share of the budget.
type DenseMemoryIndex struct {
	shards    []denseShard
	shardMask int
	// engineMu guards engineToRequestKeys, which is itself lock-free.
	engineMu            sync.Mutex
	engineToRequestKeys *engineKeyMap
	pods                *interner
	tiers               *interner
	// dead is the cleared-pod bitmap, one bit per pod ordinal, accessed
	// with atomics. It is pre-sized to the ordinal supply, so bits are
	// only ever set, never grown into. The pool dispatches one pod's
	// events to one worker, so an Add cannot race the Clear of its own
	// pod. deadEpoch counts Clears, retiring the walk's decoded-entry
	// reuse when the dead set changes mid-batch.
	dead          []uint64
	deadEpoch     atomic.Uint64
	budget        int64
	watermark     int64
	podCap        int
	sampleModulus int
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
	if !validSampleModulus(cfg.SampleModulus) {
		return nil, fmt.Errorf("failed to initialize dense memory index: sampleModulus must be 0, 1, or a power of two, got %d",
			cfg.SampleModulus)
	}
	d := &DenseMemoryIndex{
		shards:              make([]denseShard, denseShardCount),
		shardMask:           denseShardCount - 1,
		engineToRequestKeys: newEngineKeyMap(0), // the byte budget governs
		pods:                newInterner(maxDensePodOrdinals),
		tiers:               newInterner(maxDenseTierOrdinal + 1),
		dead:                make([]uint64, maxDensePodOrdinals/64),
		budget:              int64(budget),                                         //#nosec G115 -- ParseBytes is bounded by int64
		watermark:           int64(budget) / denseWatermarkDen * denseWatermarkNum, //#nosec G115 -- bounded by budget
		podCap:              podCap,
		sampleModulus:       cfg.SampleModulus,
	}
	for i := range d.shards {
		s := &d.shards[i]
		s.tbl = newDenseTable(tableInitialSlots / denseShardCount)
		s.groups = make(map[BlockHash]map[denseGroupKey]struct{})
	}
	return d, nil
}

// shardOf returns the shard a request key belongs to. It reuses the table's
// hash: the shard mask and the per-shard table mask are independent.
func (d *DenseMemoryIndex) shardOf(key BlockHash) *denseShard {
	return &d.shards[int(tableHash(key))&d.shardMask] //#nosec G115 -- the hash is masked to the shard count
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
	var stats MemoryIndexStats
	for i := range d.shards {
		s := &d.shards[i]
		s.mu.RLock()
		stats.ResidentKeys += s.tbl.used
		stats.ResidentBytes += s.residencyLocked()
		stats.GroupRows += s.groupRows
		stats.TruncatedEntries += s.truncated
		s.mu.RUnlock()
	}
	stats.EngineMappings = d.engineToRequestKeys.len()
	return stats
}

// residencyLocked estimates the shard's resident bytes. Called with the
// shard's mu held.
func (s *denseShard) residencyLocked() int64 {
	bytes := int64(s.tbl.used) * tableBytesPerKey
	bytes += int64(len(s.arena.words)) * 4
	bytes += int64(s.groupRows) * groupRowBytes
	return bytes
}

// engineResidency estimates the engine map's resident bytes.
func (d *DenseMemoryIndex) engineResidency() int64 {
	return int64(d.engineToRequestKeys.len())*engineBytesPerEntry +
		int64(len(d.engineToRequestKeys.arena))*16
}

// isDead reports whether the pod ordinal was cleared.
func (d *DenseMemoryIndex) isDead(pod uint32) bool {
	i, j := int(pod/64), int(pod%64)
	return atomic.LoadUint64(&d.dead[i])&(1<<j) != 0
}

func (d *DenseMemoryIndex) setDead(pod uint32) {
	i, j := int(pod/64), int(pod%64)
	atomic.OrUint64(&d.dead[i], 1<<j)
}

// tick returns the current stamp: wall-clock seconds. Ties within a second
// are expected; the sweep selects its quarter by count, not by stamp value.
func (d *DenseMemoryIndex) tick() uint32 {
	return uint32(time.Now().Unix()) //#nosec G115 -- unix seconds fit uint32 past 2106
}

// internBatch pairs each entry with its pod and tier ordinals, reviving a
// pod name whose current ordinal is dead. The batch is assigned all or
// nothing: when its new ordinals would exceed a cap, nothing is interned and
// the error names the cap. The interners' own locks serialize the batch; no
// shard lock is held, so interning runs in parallel with any shard.
func (d *DenseMemoryIndex) internBatch(entries []PodEntry) ([]EntryRef, error) {
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

	records, err := d.internBatch(entries)
	if err != nil {
		return err
	}
	stamp := d.tick()

	// Each key's slot updates under its own shard's acquisition: no
	// cross-key invariant spans the loop, so releasing between keys is
	// safe, and distinct keys' shards proceed in parallel.
	sampledKeys := requestKeys
	if d.sampleModulus > 1 {
		sampledKeys = filterSampledKeys(requestKeys, d.sampleModulus)
	}
	touched := 0
	for _, requestKey := range sampledKeys {
		si := int(tableHash(requestKey)) & d.shardMask //#nosec G115 -- masked to the shard count
		s := &d.shards[si]
		s.mu.Lock()
		// Dead words occupy the cap until the sweep's sampled pass reclaims
		// them; strip them before inserting at the cap so the head-drop
		// cannot evict a live pod's word while a dead one survives.
		if i, ok := s.tbl.find(requestKey); ok && slotLen(&s.tbl.slots[i].slot, &s.arena) == d.podCap {
			d.stripDeadWordsLocked(s, requestKey)
		}
		i := s.tbl.slotForInsert(requestKey)
		slot := &s.tbl.slots[i].slot
		for r := range records {
			ref := &records[r]
			if dropped := slotSetEntry(slot, &s.arena, d.podCap, entryWordFor(ref)); dropped != 0 {
				s.truncated++
				metrics.IndexTruncatedEntries.Add(1)
				d.purgeWordRowsLocked(s, requestKey, dropped)
			}
			if ref.HasGroup {
				d.groupAddLocked(s, requestKey, ref)
			}
		}
		slot.stamp = stamp
		s.mu.Unlock()
		touched |= 1 << si
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
		d.engineMu.Lock()
		d.setEngineMappingsLocked(engineKeys, requestKeys)
		d.engineMu.Unlock()
	}

	for si := range d.shards {
		if touched&(1<<si) != 0 {
			d.maybeSweepShard(si)
		}
	}
	return nil
}

// setEngineMappingsLocked stores the proportional engine-key mappings without
// building the intermediate map: distribution position i maps to engine key
// i*len(engineKeys)/n and request key i*len(requestKeys)/n, both monotone in
// i, so each engine key's request keys are one contiguous subslice of
// requestKeys and the walk allocates nothing. Engine keys within one Add are
// distinct (per-event block hashes are chained); a repeated engine key would
// keep only its last run. Called with engineMu held.
func (d *DenseMemoryIndex) setEngineMappingsLocked(engineKeys, requestKeys []BlockHash) {
	e, r := len(engineKeys), len(requestKeys)
	if e == 0 || r == 0 {
		return
	}
	n := max(e, r)
	for prev := 0; prev < n; {
		pe := prev * e / n
		next := prev + 1
		for next < n && next*e/n == pe {
			next++
		}
		lo := prev * r / n
		d.engineToRequestKeys.set(engineKeys[pe], requestKeys[lo:lo+next-prev])
		prev = next
	}
}

// groupAddLocked records a group-announced copy. A re-announcement of a live
// row keeps one row, matching the replace-on-equal insert the non-dense
// backends apply to identical entries. Called with the shard's mu held for
// writing.
func (d *DenseMemoryIndex) groupAddLocked(s *denseShard, key BlockHash, ref *EntryRef) {
	rows := s.groups[key]
	if rows == nil {
		rows = make(map[denseGroupKey]struct{})
		s.groups[key] = rows
	}
	gk := denseGroupKey{pod: ref.PodOrdinal, tier: denseRowTier(ref), spec: ref.Speculative, group: ref.GroupIdx}
	if _, ok := rows[gk]; !ok {
		rows[gk] = struct{}{}
		s.groupRows++
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
// word carrying their tier bit does. Called with the shard's mu held for
// writing.
func (d *DenseMemoryIndex) purgeWordRowsLocked(s *denseShard, key BlockHash, w uint32) {
	if w&entryGroupedBit == 0 {
		return
	}
	rows := s.groups[key]
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
		s.groupRows--
	}
	if len(rows) == 0 {
		delete(s.groups, key)
	}
}

// groupRemoveLocked deletes the row and reports whether the (pod, tier,
// speculative) pair it named has no rows left, which releases the entry
// word's bit. Called with the shard's mu held for writing.
func (d *DenseMemoryIndex) groupRemoveLocked(s *denseShard, key BlockHash, gk denseGroupKey) (existed, drained bool) {
	rows := s.groups[key]
	if rows == nil {
		return false, false
	}
	if _, ok := rows[gk]; !ok {
		return false, false
	}
	delete(rows, gk)
	s.groupRows--
	if len(rows) == 0 {
		delete(s.groups, key)
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

	switch keyType {
	case EngineKey:
		// Phase one removes the entries under each request key's own shard
		// lock; phase two re-examines the chains under every shard lock
		// and the engine lock together, so the mapping-removal decision is
		// atomic against Adds, which insert entries before setting the
		// mapping.
		d.engineMu.Lock()
		chains := make([][]BlockHash, len(keys))
		for i, key := range keys {
			rks, found := d.engineToRequestKeys.peek(key)
			if !found {
				traceLogger.Info("engineKey not found in mapping, nothing to evict", "engineKey", key)
				continue
			}
			chains[i] = rks
		}
		d.engineMu.Unlock()
		for i := range keys {
			for _, rk := range chains[i] {
				s := d.shardOf(rk)
				s.mu.Lock()
				d.evictFromKeyLocked(s, rk, entries)
				s.mu.Unlock()
			}
		}
		for si := range d.shards {
			d.shards[si].mu.Lock()
		}
		d.engineMu.Lock()
		for _, key := range keys {
			rks, found := d.engineToRequestKeys.peek(key)
			if !found {
				continue
			}
			removable := true
			for _, rk := range rks {
				if !SampledKey(rk, d.sampleModulus) {
					// An unsampled request key never has a slot, so its
					// emptiness cannot be checked. The mapping stays for
					// pods that still hold the block; the engine sweep
					// reclaims it under capacity pressure.
					removable = false
					break
				}
				if j, ok := d.shardOf(rk).tbl.find(rk); ok && slotLen(&d.shardOf(rk).tbl.slots[j].slot, &d.shardOf(rk).arena) > 0 {
					removable = false
					break
				}
			}
			if removable {
				d.engineToRequestKeys.remove(key)
			}
		}
		d.engineMu.Unlock()
		for i := len(d.shards) - 1; i >= 0; i-- {
			d.shards[i].mu.Unlock()
		}
		return nil
	case RequestKey:
		for _, key := range keys {
			s := d.shardOf(key)
			s.mu.Lock()
			d.evictFromKeyLocked(s, key, entries)
			s.mu.Unlock()
		}
		return nil
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}
}

// evictFromKeyLocked removes the matching entries from one request key,
// deleting the key when its last word clears. Called with the shard's mu
// held for writing.
func (d *DenseMemoryIndex) evictFromKeyLocked(s *denseShard, requestKey BlockHash, entries []PodEntry) {
	i, ok := s.tbl.find(requestKey)
	if !ok {
		return
	}
	slot := &s.tbl.slots[i].slot
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
			existed, drained := d.groupRemoveLocked(s, requestKey, gk)
			if !existed {
				continue
			}
			changed = true
			if drained {
				slotClearEntry(slot, &s.arena, packEntry(pod, tiers, e.Speculative, true))
			}
			continue
		}
		if e.Speculative {
			if slotClearEntry(slot, &s.arena, packEntry(pod, 0, true, false)) {
				changed = true
			}
			continue
		}
		tier, found := d.tiers.ordinalOf(e.DeviceTier)
		if !found {
			continue
		}
		if slotClearEntry(slot, &s.arena, packEntry(pod, 1<<tier, false, false)) {
			changed = true
		}
	}
	if changed && slotLen(slot, &s.arena) == 0 {
		d.deleteKeyLocked(s, i, "empty")
	}
}

// deleteKeyLocked frees a key's arena chunks and removes it from the shard's
// table. Called with the shard's mu held for writing.
func (d *DenseMemoryIndex) deleteKeyLocked(s *denseShard, i int, cause string) {
	key := s.tbl.slots[i].key
	for h := s.tbl.slots[i].slot.chain; h != 0; {
		next := s.arena.words[(int(h)-1)*denseChunkWords+denseChunkLink]
		s.arena.freeChunk(h)
		h = next
	}
	s.tbl.deleteAt(i)
	if rows := s.groups[key]; rows != nil {
		s.groupRows -= len(rows)
		delete(s.groups, key)
	}
	metrics.IndexEvictions.WithLabelValues(cause).Add(1)
}

// Clear removes every entry for the pod from the index: the pod's current
// ordinal is marked dead and reads filter it, so Clear is O(1). The dead
// ordinal's slots are freed by the sampled eviction pass; a re-Add revives
// the name under a fresh ordinal.
func (d *DenseMemoryIndex) Clear(ctx context.Context, podIdentifier string) error {
	start := time.Now()
	if pod, ok := d.pods.ordinalOf(podIdentifier); ok {
		// Every shard's write lock, so no decode is mid-word when the bit
		// lands; the epoch bump retires the walk's decoded-entry reuse
		// across a batch's per-shard acquisitions.
		for si := range d.shards {
			d.shards[si].mu.Lock()
		}
		d.setDead(pod)
		d.deadEpoch.Add(1)
		for si := len(d.shards) - 1; si >= 0; si-- {
			d.shards[si].mu.Unlock()
		}
	}
	metrics.IndexClearDuration.Observe(time.Since(start).Seconds())
	log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.DenseMemoryIndex.Clear").
		Info("cleared pod from index", "pod", podIdentifier)
	return nil
}

// GetRequestKey returns the last request key associated with the given
// engineKey, the highest-index request key the engine key covers. It takes
// the engine map's lock: the map's get refreshes the slot's recency stamp,
// which mutates the map, so concurrent lookups cannot share it lock-free.
func (d *DenseMemoryIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	d.engineMu.Lock()
	rk, found := d.engineToRequestKeys.get(engineKey)
	d.engineMu.Unlock()
	if !found {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}
	return rk, nil
}

// Lookup receives a list of requestKeys and a set of pod identifiers, and
// retrieves the filtered pods associated with those keys, stopping at the
// first missing or filtered-empty key like InMemoryIndex.Lookup. Each key is
// read under its own shard's read lock, held only for that key; the prefix
// chain's stop-at-first-missing order is unaffected because a key lives in
// exactly one shard.
func (d *DenseMemoryIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no requestKeys provided for lookup")
	}
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.DenseMemoryIndex.Lookup")

	podsPerKey := make(map[BlockHash][]PodEntry)
	visited := 0

	for idx, requestKey := range requestKeys {
		if idx&cancellationCheckMask == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		s := d.shardOf(requestKey)
		s.mu.RLock()
		var filtered []PodEntry
		found := false
		if i, ok := s.tbl.find(requestKey); ok {
			found = true
			d.markRead(&s.tbl.slots[i])
			filtered = d.materializeLocked(s, requestKey, &s.tbl.slots[i].slot, podIdentifierSet)
		}
		s.mu.RUnlock()
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
// pods and entries outside the filter. Called with the shard's mu held for
// reading.
func (d *DenseMemoryIndex) materializeLocked(s *denseShard, key BlockHash, slot *keySlot, filter sets.Set[string]) []PodEntry {
	var out []PodEntry
	forEachSlotWord(slot, &s.arena, func(w uint32) bool {
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
			for gk := range s.groups[key] {
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
// runWords holds the last decoded key's entry words and runBounds its decoded
// entry range, so a key whose words repeat the last one reuses that range
// instead of decoding again. pos groups the batch's key positions by shard
// for the per-shard read acquisitions.
type denseWalkState struct {
	found     []bool
	bounds    [][2]int
	entries   []WalkEntry
	runWords  []uint32
	runBounds [2]int
	runEpoch  uint64
	pos       [denseShardCount][]int
}

var walkStatePool = sync.Pool{New: func() any { return &denseWalkState{} }}

// WalkKeys implements KeyWalker. Batches of keys are copied out under one
// read acquisition per shard the batch touches, decoding the packed words
// into WalkEntries (dead pods filtered, one entry per pod and tier pair),
// and the visits run after the locks are released, so no lock is held
// across a visit callback. A key whose words repeat the previously decoded
// key's reuses that key's decoded entries instead of decoding again; the
// comparison holds within one shard's pass. Each found key's slot records
// read evidence with one atomic OR; the read path takes no write
// acquisition.
func (d *DenseMemoryIndex) WalkKeys(ctx context.Context, requestKeys []BlockHash,
	visit func(pos int, found bool, entries []WalkEntry) bool,
) error {
	ws := walkStatePool.Get().(*denseWalkState)
	defer func() {
		ws.entries = ws.entries[:0]
		walkStatePool.Put(ws)
	}()

	batch := walkBatchMin
	for start := 0; start < len(requestKeys); {
		end := min(start+batch, len(requestKeys))
		keys := requestKeys[start:end]

		ws.found = ws.found[:0]
		ws.bounds = ws.bounds[:0]
		ws.entries = ws.entries[:0]
		for si := range ws.pos {
			ws.pos[si] = ws.pos[si][:0]
		}
		for i, key := range keys {
			ws.found = append(ws.found, false)
			ws.bounds = append(ws.bounds, [2]int{0, 0})
			si := int(tableHash(key)) & d.shardMask //#nosec G115 -- masked to the shard count
			ws.pos[si] = append(ws.pos[si], i)
		}
		runValid := false
		for si := range d.shards {
			if len(ws.pos[si]) == 0 {
				continue
			}
			s := &d.shards[si]
			s.mu.RLock()
			for _, i := range ws.pos[si] {
				key := keys[i]
				b0 := len(ws.entries)
				found := false
				if j, ok := s.tbl.find(key); ok {
					found = true
					d.markRead(&s.tbl.slots[j])
					slot := &s.tbl.slots[j].slot
					if runValid && d.deadEpoch.Load() == ws.runEpoch && d.matchRunWords(s, slot, ws) {
						ws.found[i] = found
						ws.bounds[i] = ws.runBounds
						continue
					}
					ws.runWords = ws.runWords[:0]
					d.decodeLocked(s, slot, ws, b0)
					ws.runBounds = [2]int{b0, len(ws.entries)}
					ws.runEpoch = d.deadEpoch.Load()
					runValid = true
				}
				ws.found[i] = found
				ws.bounds[i] = [2]int{b0, len(ws.entries)}
			}
			s.mu.RUnlock()
		}

		for i := range keys {
			pos := start + i
			if pos&cancellationCheckMask == 0 && ctx.Err() != nil {
				return ctx.Err()
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

// matchRunWords reports whether slot's entry words are exactly ws.runWords,
// the words of the last decoded key, in which case that key's decoded
// entries are also this key's. Equal word streams decode equally: the dead
// set cannot change under the read acquisition the caller holds. Called
// with the shard's mu held for reading.
func (d *DenseMemoryIndex) matchRunWords(s *denseShard, slot *keySlot, ws *denseWalkState) bool {
	n := 0
	equal := true
	forEachSlotWord(slot, &s.arena, func(w uint32) bool {
		if n >= len(ws.runWords) || ws.runWords[n] != w {
			equal = false
			return false
		}
		n++
		return true
	})
	return equal && n == len(ws.runWords)
}

// decodeLocked appends the slot's walk entries to ws.entries, skipping dead
// pods and folding duplicate (pod, tier, speculative) triples that grouped
// and ungrouped words of one pod can both carry. The fold is scoped to the
// key's own entries, which start at ws.entries[from]. Duplicate triples
// require a grouped word (same-identity words merge at insert), so the
// dedup scan runs only once the key has shown one. The slot's words are
// recorded in ws.runWords for the next key's run comparison. Called with
// the shard's mu held for reading.
func (d *DenseMemoryIndex) decodeLocked(s *denseShard, slot *keySlot, ws *denseWalkState, from int) {
	grouped := false
	forEachSlotWord(slot, &s.arena, func(w uint32) bool {
		ws.runWords = append(ws.runWords, w)
		pod := entryPod(w)
		if d.isDead(pod) {
			return true
		}
		if w&entryGroupedBit != 0 {
			grouped = true
		}
		emit := func(e WalkEntry) {
			if grouped {
				for i := from; i < len(ws.entries); i++ {
					if ws.entries[i] == e {
						return
					}
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

// markRead records read evidence in a slot, no write acquisition. Readers
// hold the shard's read lock and the sweep holds the shard's write lock, so
// no read can land mid-sweep: the bit means exactly "read since the sweep
// last sampled this slot", and the sweep clears it when it spends the
// evidence. A slot already marked skips the atomic OR, so a repeatedly
// walked key pays one atomic load instead of a read-modify-write.
func (d *DenseMemoryIndex) markRead(s *tableSlot) {
	if atomic.LoadUint32(&s.readBits) == 0 {
		atomic.OrUint32(&s.readBits, 1)
	}
}

// maybeSweepShard runs the sampled eviction for one shard when its share of
// residency crosses its share of the budget. Each burst samples and evicts
// under the shard's own write acquisition, so no burst holds the shard for
// long. The engine map sweeps when its residency crosses a quarter of the
// whole budget, matching the single-table heuristic at steady state.
func (d *DenseMemoryIndex) maybeSweepShard(si int) {
	s := &d.shards[si]
	shardBudget := d.budget / int64(len(d.shards))
	shardWatermark := d.watermark / int64(len(d.shards))
	s.mu.RLock()
	over := s.residencyLocked() > shardBudget
	s.mu.RUnlock()
	if !over {
		return
	}
	for range sweepMaxBursts {
		s.mu.Lock()
		if s.residencyLocked() <= shardWatermark {
			d.updateGauges(s)
			s.mu.Unlock()
			return
		}
		d.sweepBurstLocked(s)
		s.mu.Unlock()
	}
}

// sweepBurstLocked evicts from a sample of the shard's table: every sampled
// slot whose entries are all dead, plus the oldest quarter by stamp, and
// the same from the engine map when it holds a quarter or more of the
// budget. The sample is the occupied slots from the sweep cursor onward, up
// to sweepSampleSize keys or one full loop over the table. Called with the
// shard's mu held for writing.
func (d *DenseMemoryIndex) sweepBurstLocked(s *denseShard) {
	type victim struct {
		key     BlockHash
		stamp   uint32
		allDead bool
	}
	var victims []victim
	stamp := d.tick()
	tableLen := len(s.tbl.slots)
	scanned := 0
	for len(victims) < sweepSampleSize && scanned < tableLen {
		i := (s.sweepCursor + scanned) % tableLen
		scanned++
		if s.tbl.empty(i) {
			continue
		}
		sl := &s.tbl.slots[i]
		if atomic.LoadUint32(&sl.readBits) != 0 {
			// Read since the last burst that sampled this slot: young
			// again, and the evidence is spent. Any set bit counts: the
			// read lock excludes this burst, so every bit still standing
			// was set before it began.
			sl.slot.stamp = stamp
			atomic.StoreUint32(&sl.readBits, 0)
		}
		allDead := true
		forEachSlotWord(&sl.slot, &s.arena, func(w uint32) bool {
			if !d.isDead(entryPod(w)) {
				allDead = false
				return false
			}
			return true
		})
		victims = append(victims, victim{key: sl.key, stamp: sl.slot.stamp, allDead: allDead})
	}
	s.sweepCursor = (s.sweepCursor + scanned) % tableLen

	live := make([]victim, 0, len(victims))
	for i := range victims {
		if !victims[i].allDead {
			live = append(live, victims[i])
		}
	}
	if len(live) > 0 {
		// Select the oldest quarter by count, not by stamp value: wall-second
		// stamps tie heavily, and a value threshold would take every tied
		// victim at the boundary, not a quarter of the sample.
		sort.SliceStable(live, func(i, j int) bool { return live[i].stamp < live[j].stamp })
		for _, v := range live[:oldestQuarter(len(live))] {
			d.deleteFoundKeyLocked(s, v.key, "budget")
		}
	}
	for i := range victims {
		if victims[i].allDead {
			d.deleteFoundKeyLocked(s, victims[i].key, "dead")
			continue
		}
		// A mixed slot keeps its dead words filtered at read time; strip
		// them here so the sampled pass reclaims their storage.
		d.stripDeadWordsLocked(s, victims[i].key)
	}

	if d.engineResidency()*4 > d.budget {
		d.sweepEngineLocked()
	}
}

// deleteFoundKeyLocked removes the key if it is still resident. The sweep
// acts on sampled keys after other victims were deleted, and a deletion
// shifts the probe cluster, so the slot is located by key again here.
// Called with the shard's mu held for writing.
func (d *DenseMemoryIndex) deleteFoundKeyLocked(s *denseShard, key BlockHash, cause string) {
	if i, ok := s.tbl.find(key); ok {
		d.deleteKeyLocked(s, i, cause)
	}
}

// stripDeadWordsLocked removes the dead pods' words from a sampled key's
// slot, deleting the key when nothing live remains. Called with the shard's
// mu held for writing.
func (d *DenseMemoryIndex) stripDeadWordsLocked(s *denseShard, key BlockHash) {
	i, ok := s.tbl.find(key)
	if !ok {
		return
	}
	slot := &s.tbl.slots[i].slot
	n := slotLen(slot, &s.arena)
	for j := 0; j < n; {
		w := slotWord(slot, &s.arena, j)
		if d.isDead(entryPod(w)) {
			d.purgeWordRowsLocked(s, key, w)
			slotRemoveAt(slot, &s.arena, j, n)
			n--
			continue
		}
		j++
	}
	if n == 0 {
		d.deleteKeyLocked(s, i, "dead")
	}
}

// sweepEngineLocked evicts the oldest quarter of an engine-map sample.
// Called with engineMu held.
func (d *DenseMemoryIndex) sweepEngineLocked() {
	mappings := d.engineToRequestKeys.sampleMappings(sweepSampleSize)
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

// updateGaugesLocked refreshes the residency gauges. The calling shard is
// held for writing; the others are read under their own locks. Called with
// the held shard's mu held for writing.
func (d *DenseMemoryIndex) updateGauges(held *denseShard) {
	keys := held.tbl.used
	bytes := held.residencyLocked()
	rows := held.groupRows
	truncated := held.truncated
	for i := range d.shards {
		s := &d.shards[i]
		if s == held {
			continue
		}
		s.mu.RLock()
		keys += s.tbl.used
		bytes += s.residencyLocked()
		rows += s.groupRows
		truncated += s.truncated
		s.mu.RUnlock()
	}
	metrics.IndexResidentKeys.Set(float64(keys))
	metrics.IndexResidentBytes.Set(float64(bytes + d.engineResidency()))
}
