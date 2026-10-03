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

package kvcache

import (
	"context"
	"math"
	"sync"
	"unsafe"

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

// SpeculativeTier is the tier name under which speculative entries count in
// PodMatch.BlocksByTier. Speculative entries carry no engine-reported device
// tier; an entry whose device tier is reported under this name counts in the
// same chain.
const SpeculativeTier = "speculative"

// speculativeTierWeight scores speculative entries when the speculative tier
// has no configured weight.
const speculativeTierWeight = 1.0

// unknownTierWeight scores blocks held in a tier without a configured weight.
const unknownTierWeight = 0.0

// matchCancellationMask paces context-cancellation checks over key
// positions: positions where pos&mask == 0 poll ctx.Err().
const matchCancellationMask = 255

// PodMatch is one pod's prefix match for a key sequence. All values cover
// the contiguous chain of keys the pod holds, counted from the first key.
type PodMatch struct {
	// WeightedScore sums, per block of the chain, the highest device-tier
	// weight among the pod's entries for that block. Tiers without a
	// configured weight count unknownTierWeight; speculative entries count
	// speculativeTierWeight unless the speculative tier is configured.
	WeightedScore float64
	// MatchedBlocks is the chain length in blocks, regardless of tier.
	MatchedBlocks int
	// ConfirmedBlocks is the chain length in blocks counting only keys the
	// pod holds in an engine-reported device tier; the tier may change from
	// block to block. Speculative entries end the chain.
	ConfirmedBlocks int
	// BlocksByTier is the per-tier chain length: a tier counts a block only
	// while the pod holds every previous block in that same tier.
	// Speculative entries count under SpeculativeTier. Never nil.
	BlocksByTier map[string]int
}

// MatchBlockKeys runs the prefix matcher over keys for the pods in podFilter
// (every pod when empty) and returns one PodMatch per pod that holds the
// first key. Empty keys match nothing. The matcher walks the index when the
// backend is a kvblock.KeyWalker and otherwise materializes Lookup; both
// feed the same accumulator.
func (k *Indexer) MatchBlockKeys(ctx context.Context, keys []kvblock.BlockHash,
	podFilter sets.Set[string],
) (map[string]PodMatch, error) {
	if len(keys) == 0 {
		return map[string]PodMatch{}, nil
	}

	tracer := tracing.Tracer(TracerScope)
	ctx, span := tracer.Start(ctx, "match_block_keys",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()
	span.SetAttributes(
		semconv.LLMDKVCachePrefixMatchKeyCount(len(keys)),
		semconv.LLMDKVCachePrefixMatchPodFilterCount(podFilter.Len()),
		semconv.LLMDKVCachePrefixMatchWalked(k.keyWalker != nil),
	)

	var matches map[string]PodMatch
	var err error
	if k.keyWalker != nil {
		matches, err = matchWalk(ctx, k.keyWalker, keys, k.tierWeights, podFilter)
	} else {
		matches, err = matchLookup(ctx, k.kvBlockIndex, keys, k.tierWeights, podFilter)
	}
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}

	blocksFound := maxMatchedBlocks(matches)
	if k.recordHits {
		metrics.MaxPodHitCount.Add(float64(blocksFound))
		metrics.LookupHits.Add(float64(blocksFound))
	}
	span.SetAttributes(
		semconv.LLMDKVCachePrefixMatchPodsMatched(len(matches)),
		semconv.LLMDKVCachePrefixMatchLongestChain(blocksFound),
	)
	return matches, nil
}

// maxMatchedBlocks returns the longest chain among matches.
func maxMatchedBlocks(matches map[string]PodMatch) int {
	longest := 0
	for _, m := range matches {
		if m.MatchedBlocks > longest {
			longest = m.MatchedBlocks
		}
	}
	return longest
}

// matchWalk feeds the accumulator from an ordered index walk. The walk ends
// at the first key without entries or once no chain is alive.
func matchWalk(ctx context.Context, walker kvblock.KeyWalker, keys []kvblock.BlockHash,
	weights map[string]float64, filter sets.Set[string],
) (map[string]PodMatch, error) {
	acc := acquireAccumulator(weights, filter)
	defer releaseAccumulator(acc)

	err := walker.WalkKeys(ctx, keys, func(_ int, found bool, entries []kvblock.EntryRef) bool {
		return found && len(entries) > 0 && acc.key(entries)
	})
	if err != nil {
		return nil, err
	}
	return acc.result(), nil
}

// matchLookup feeds the accumulator from a materialized Lookup result, for
// backends without the walk capability.
func matchLookup(ctx context.Context, index kvblock.Index, keys []kvblock.BlockHash,
	weights map[string]float64, filter sets.Set[string],
) (map[string]PodMatch, error) {
	keyToPods, err := index.Lookup(ctx, keys, filter)
	if err != nil {
		return nil, err
	}
	return matchMaterialized(ctx, keys, keyToPods, weights, filter)
}

// stringIdentical reports whether two strings have identical backing data and length.
func stringIdentical(a, b string) bool {
	return unsafe.StringData(a) == unsafe.StringData(b) && len(a) == len(b)
}

func fastHash(s string) uint32 {
	n := len(s)
	if n < 4 {
		if n == 0 {
			return 0
		}
		h := uint32(s[0])
		for i := 1; i < n; i++ {
			h = h*31 + uint32(s[i])
		}
		return h
	}
	return uint32(s[0])*50625 + uint32(s[n/2])*1351 + uint32(s[n-2])*31 + uint32(s[n-1])
}

func matchMaterialized(ctx context.Context, keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	weights map[string]float64, filter sets.Set[string],
) (map[string]PodMatch, error) {
	acc := acquireAccumulator(weights, filter)
	defer releaseAccumulator(acc)

	for pos, key := range keys {
		if pos&matchCancellationMask == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		entries := keyToPods[key]
		if len(entries) == 0 {
			break
		}
		if !acc.keyPods(entries) {
			break
		}
	}
	// Cancellation is sampled at checkpoints along the keys and once more at
	// completion, so a cancelled request never reports a match.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return acc.result(), nil
}

func (a *prefixAccumulator) podOrdinal(name string) uint32 {
	if id, ok := a.podsMap[name]; ok {
		return id
	}
	id := clampOrdinal(len(a.podsMap))
	a.podsMap[name] = id
	return id
}

func (a *prefixAccumulator) tierOrdinal(name string) uint32 {
	if id, ok := a.tiersMap[name]; ok {
		return id
	}
	id := clampOrdinal(len(a.tiersMap))
	a.tiersMap[name] = id
	return id
}

func (a *prefixAccumulator) cachedTierOrdinal(name string) uint32 {
	for i := 0; i < len(a.tierCache); i++ {
		entry := &a.tierCache[i]
		if stringIdentical(entry.name, name) || entry.name == name {
			return entry.ord
		}
		if entry.name == "" {
			entry.name = name
			entry.ord = a.tierOrdinal(name)
			return entry.ord
		}
	}
	return a.tierOrdinal(name)
}

// clampOrdinal bounds a table size below the math.MaxUint32 sentinel that
// speculativeTierOrdinal reserves, so a table that somehow grew that large
// cannot collide with it.
func clampOrdinal(n int) uint32 {

	if n > math.MaxUint32-1 {
		n = math.MaxUint32 - 1
	}
	return uint32(n)
}

// speculativeTierOrdinal keys the speculative per-tier chain. Feeders assign
// tier ordinals from zero, so the top of the range never collides.
const speculativeTierOrdinal = math.MaxUint32

// slotTable is an open-addressed map from pod ordinal to request-local slot.
// It is sized by the first key's entry count, so request state scales with
// the live candidates rather than with every ordinal an index ever assigned.
// It packs (ordinal << 32) | (slot_index + 1) into a single uint64 bucket
// to minimize cache lines and avoid struct/alignment overhead.
type slotTable struct {
	buckets []uint64
}

func (t *slotTable) reset(numEntries int) {
	size := 2
	for size < numEntries*2 {
		size <<= 1
	}
	if cap(t.buckets) < size {
		t.buckets = make([]uint64, size)
		return
	}
	t.buckets = t.buckets[:size]
	clear(t.buckets)
}

func (t *slotTable) lookup(ordinal uint32) (int32, bool) {
	buckets := t.buckets
	if len(buckets) == 0 {
		return 0, false
	}
	mask := uint32(len(buckets) - 1)
	i := (ordinal * 2654435761) & mask
	for {
		b := buckets[i]
		if b == 0 {
			return 0, false
		}
		if uint32(b>>32) == ordinal {
			return int32(uint32(b) - 1), true
		}
		i = (i + 1) & mask
	}
}

func (t *slotTable) insert(ordinal uint32, slot int32) {
	buckets := t.buckets
	mask := uint32(len(buckets) - 1)
	i := (ordinal * 2654435761) & mask
	for buckets[i] != 0 {
		i = (i + 1) & mask
	}
	buckets[i] = (uint64(ordinal) << 32) | uint64(slot+1)
}

// tierChain tracks one tier's contiguous prefix for a candidate pod.
type tierChain struct {
	ordinal uint32
	name    string
	count   int
	// seen is the key stamp of the last key where the pod held this tier.
	seen  uint32
	alive bool
}

// tierWeight is one tier's resolved weight, keyed by tier ordinal.
type tierWeight struct {
	ordinal uint32
	weight  float64
}

// matchSlot is one candidate pod's accumulated state.
type matchSlot struct {
	// --- Hot loop state (fits entirely within a single 64-byte cache line) ---
	score          float64 // 8 bytes
	weight         float64 // 8 bytes
	matched        int     // 8 bytes
	confirmed      int     // 8 bytes
	tier0Count     int     // 8 bytes
	confirmedSeen  uint32  // 4 bytes
	tier0Seen      uint32  // 4 bytes
	seen           uint32  // 4 bytes
	tier0Ordinal   uint32  // 4 bytes
	confirmedAlive bool    // 1 byte
	tier0Alive     bool    // 1 byte
	hasTier0       bool    // 1 byte
	isMultiTier    bool    // 1 byte

	// --- Cold fields (moved to secondary cache line) ---
	pod       string
	tier0Name string
	tiers     []tierChain
	_         [7]byte // Padding to align to exactly 128 bytes
}

type posCacheEntry struct {
	podName       string
	tierName      string
	speculative   bool
	podOrd        uint32
	tierOrd       uint32
	slot          int32
	weight        float64
	confirmed     bool
	isSingleTier0 bool
}

type posCacheEntryKey struct {
	podAndRawTier  uint64
	weight         float64
	slot           int32
	tierOrd        uint32
	isSingleTier0  bool
	confirmed      bool
	refSpeculative bool
	_              [5]byte
}

type podCacheEntry struct {
	name string
	ord  uint32
}

type tierCacheEntry struct {
	name string
	ord  uint32
}

// prefixAccumulator folds an ordered walk over request keys into per-pod
// prefix matches. It is the single implementation of the matching rules:
// candidates are the pods holding the first key, each chain ends at the
// first key its pod does not hold, duplicate entries for a pod at one key
// take the highest weight, and every tier tracks its own contiguous prefix.
//
// Feeders present each key's entries through key, in key order, and stop at
// the first key without entries or once key reports no live chain. Ordinals
// only need to be stable within one accumulation; they key request-local
// tables and never size state, so sparse or large values cost nothing.
type prefixAccumulator struct {
	weights   map[string]float64
	filter    sets.Set[string]
	hasFilter bool

	table    slotTable
	slots    []matchSlot
	active   []int32
	keyStamp uint32
	first    bool

	// weightCache holds the weight of every tier seen in this accumulation,
	// scanned linearly: requests see a handful of tiers.
	weightCache []tierWeight

	podsMap  map[string]uint32
	tiersMap map[string]uint32
	refsBuf  []kvblock.EntryRef

	posCache               []posCacheEntryKey
	mru                    uint64
	weightCacheDirect      [64]float64
	weightCacheSet         uint64
	speculativeTierMask    uint64
	speculativeTierChecked uint64
	speculativeWeight      float64
	speculativeWeightSet   bool

	posCache256 [256]posCacheEntry
	podCache    [2048]podCacheEntry
	tierCache   [4]tierCacheEntry

	mruName      string
	mruOrd       uint32
	hasMru       bool
	lastTierName string
	lastTierOrd  uint32
	hasLastTier  bool

	singularCache     map[string][]map[string]int
	lastSingularName  string
	lastSingularCount int
	lastSingularMap   map[string]int
}

var accumulatorPool = sync.Pool{New: func() any {
	slots := make([]matchSlot, 256)
	for i := range slots {
		slots[i].tiers = make([]tierChain, 0, 4)
	}
	return &prefixAccumulator{
		table:         slotTable{buckets: make([]uint64, 512)},
		slots:         slots[:0],
		active:        make([]int32, 0, 256),
		weightCache:   make([]tierWeight, 0, 8),
		podsMap:       make(map[string]uint32, 128),
		tiersMap:      make(map[string]uint32, 16),
		refsBuf:       make([]kvblock.EntryRef, 0, 512),
		posCache:      make([]posCacheEntryKey, 0, 512),
		singularCache: make(map[string][]map[string]int, 8),
	}
}}

func acquireAccumulator(weights map[string]float64, filter sets.Set[string]) *prefixAccumulator {
	a := accumulatorPool.Get().(*prefixAccumulator)
	a.weights, a.filter = weights, filter
	a.hasFilter = filter.Len() > 0
	a.slots = a.slots[:0]
	a.active = a.active[:0]
	a.keyStamp = 0
	a.first = true
	a.weightCache = a.weightCache[:0]
	if a.podsMap == nil {
		a.podsMap = make(map[string]uint32, 128)
	} else {
		clear(a.podsMap)
	}
	if a.tiersMap == nil {
		a.tiersMap = make(map[string]uint32, 16)
	} else {
		clear(a.tiersMap)
	}
	clear(a.refsBuf)
	a.refsBuf = a.refsBuf[:0]
	a.mru = ^uint64(0)
	a.weightCacheSet = 0
	a.speculativeTierMask = 0
	a.speculativeTierChecked = 0
	a.speculativeWeightSet = false

	a.hasMru = false
	a.hasLastTier = false
	a.mruName = ""
	a.lastTierName = ""
	clear(a.podCache[:])
	clear(a.posCache256[:])
	clear(a.tierCache[:])
	if a.singularCache == nil {
		a.singularCache = make(map[string][]map[string]int, 8)
	}
	a.lastSingularName = ""
	a.lastSingularCount = 0
	a.lastSingularMap = nil
	return a
}

func releaseAccumulator(a *prefixAccumulator) {
	a.weights, a.filter = nil, nil
	for i := 0; i < len(a.slots); i++ {
		a.slots[i].pod = ""
	}
	accumulatorPool.Put(a)
}

// keyFirst is the cold path for folding the very first key's entries.
func (a *prefixAccumulator) keyFirst(entries []kvblock.EntryRef) bool {
	a.table.reset(len(entries))
	if cap(a.posCache) < len(entries) {
		a.posCache = make([]posCacheEntryKey, len(entries))
	} else {
		a.posCache = a.posCache[:len(entries)]
	}
	for i := range a.posCache {
		a.posCache[i].podAndRawTier = ^uint64(0)
	}

	weightCacheDirect := &a.weightCacheDirect
	weightCacheSet := a.weightCacheSet

	for i := range entries {
		ref := &entries[i]
		if i > 0 {
			prev := &entries[i-1]
			if ref.PodOrdinal == prev.PodOrdinal && ref.TierOrdinal == prev.TierOrdinal &&
				ref.Speculative == prev.Speculative {
				continue
			}
		}

		s, ok := a.table.lookup(ref.PodOrdinal)
		if !ok {
			if a.hasFilter && !a.filter.Has(ref.PodIdentifier) {
				continue
			}
			s = a.newSlot(ref.PodIdentifier)
			a.table.insert(ref.PodOrdinal, s)
		}

		slot := &a.slots[s]

		tier, tierOrdinal := ref.DeviceTier, ref.TierOrdinal
		isSpeculative := ref.Speculative
		if !isSpeculative {
			if ref.TierOrdinal < 64 {
				if (a.speculativeTierChecked & (1 << ref.TierOrdinal)) != 0 {
					isSpeculative = (a.speculativeTierMask & (1 << ref.TierOrdinal)) != 0
				} else {
					isSpeculative = ref.DeviceTier == SpeculativeTier
					a.speculativeTierChecked |= (1 << ref.TierOrdinal)
					if isSpeculative {
						a.speculativeTierMask |= (1 << ref.TierOrdinal)
					}
				}
			} else {
				isSpeculative = ref.DeviceTier == SpeculativeTier
			}
		}
		if isSpeculative {
			tier, tierOrdinal = SpeculativeTier, speculativeTierOrdinal
		} else {
			slot.confirmedSeen = a.keyStamp
		}

		var w float64
		switch {
		case tierOrdinal < 64:
			if (weightCacheSet & (1 << tierOrdinal)) != 0 {
				w = weightCacheDirect[tierOrdinal]
			} else {
				w = a.weightOf(tier, tierOrdinal)
				weightCacheSet = a.weightCacheSet
			}
		case tierOrdinal == speculativeTierOrdinal && a.speculativeWeightSet:
			w = a.speculativeWeight
		default:
			w = a.weightOf(tier, tierOrdinal)
		}

		if slot.seen != a.keyStamp {
			slot.seen = a.keyStamp
			slot.weight = w
		} else if w > slot.weight {
			slot.weight = w
		}

		// stampTier inline in keyFirst
		stamped := false
		if !slot.isMultiTier {
			if slot.hasTier0 {
				if slot.tier0Ordinal == tierOrdinal {
					slot.tier0Seen = a.keyStamp
					stamped = true
				} else {
					// Fallback to multi-tier
					slot.isMultiTier = true
					slot.tiers = append(slot.tiers, tierChain{
						ordinal: slot.tier0Ordinal,
						name:    slot.tier0Name,
						count:   slot.tier0Count,
						seen:    slot.tier0Seen,
						alive:   slot.tier0Alive,
					})
					slot.tiers = append(slot.tiers, tierChain{
						ordinal: tierOrdinal,
						name:    tier,
						seen:    a.keyStamp,
						alive:   true,
					})
					stamped = true
				}
			} else {
				slot.hasTier0 = true
				slot.tier0Ordinal = tierOrdinal
				slot.tier0Name = tier
				slot.tier0Seen = a.keyStamp
				slot.tier0Alive = true
				stamped = true
			}
		} else {
			// Multi-tier path
			for t := range slot.tiers {
				if slot.tiers[t].ordinal == tierOrdinal {
					slot.tiers[t].seen = a.keyStamp
					stamped = true
					break
				}
			}
			if !stamped {
				slot.tiers = append(slot.tiers, tierChain{
					ordinal: tierOrdinal,
					name:    tier,
					seen:    a.keyStamp,
					alive:   true,
				})
			}
		}

		if i < len(a.posCache) {
			a.posCache[i] = posCacheEntryKey{
				podAndRawTier:  uint64(ref.PodOrdinal) | (uint64(ref.TierOrdinal) << 32),
				slot:           s,
				weight:         w,
				confirmed:      tierOrdinal != speculativeTierOrdinal,
				tierOrd:        tierOrdinal,
				refSpeculative: ref.Speculative,
				isSingleTier0:  !slot.isMultiTier && slot.hasTier0 && slot.tier0Ordinal == tierOrdinal && tierOrdinal != speculativeTierOrdinal,
			}
		}
	}
	a.weightCacheSet = weightCacheSet
	return a.endKey()
}

// key folds one key's entries into the chains and reports whether any chain
// is still alive. entries is borrowed for the duration of the call.
func (a *prefixAccumulator) key(entries []kvblock.EntryRef) bool {
	a.keyStamp++
	if a.first {
		return a.keyFirst(entries)
	}

	posCache := a.posCache
	slots := a.slots
	keyStamp := a.keyStamp
	expectedSeen := (uint64(keyStamp) << 32) | uint64(keyStamp)
	mru := a.mru
	weightCacheSet := a.weightCacheSet
	weightCacheDirect := &a.weightCacheDirect

	for i := range entries {
		ref := &entries[i]

		if i < len(posCache) {
			entry := &posCache[i]
			if *(*uint64)(unsafe.Pointer(&ref.PodOrdinal)) == entry.podAndRawTier && entry.refSpeculative == ref.Speculative {
				s := entry.slot
				slot := &slots[s]
				if slot.seen < keyStamp-1 {
					continue
				}
				w := entry.weight
				if slot.seen != keyStamp {
					slot.seen = keyStamp
					slot.weight = w
				} else if w > slot.weight {
					slot.weight = w
				}
				if entry.isSingleTier0 && !slot.isMultiTier {
					*(*uint64)(unsafe.Pointer(&slot.confirmedSeen)) = expectedSeen
				} else {
					if entry.confirmed {
						slot.confirmedSeen = keyStamp
					}
					// stampTier inline
					if !slot.isMultiTier {
						if slot.tier0Ordinal == entry.tierOrd {
							slot.tier0Seen = keyStamp
						}
					} else {
						if slot.tiers[0].ordinal == entry.tierOrd {
							slot.tiers[0].seen = keyStamp
						} else {
							for t := 1; t < len(slot.tiers); t++ {
								if slot.tiers[t].ordinal == entry.tierOrd {
									slot.tiers[t].seen = keyStamp
									break
								}
							}
						}
					}
				}
				continue
			}
		}

		if i > 0 {
			prev := &entries[i-1]
			if ref.PodOrdinal == prev.PodOrdinal && ref.TierOrdinal == prev.TierOrdinal &&
				ref.Speculative == prev.Speculative {
				continue
			}
		}

		var s int32
		var ok bool
		if uint32(mru>>32) == ref.PodOrdinal {
			s = int32(mru)
			ok = true
		} else {
			s, ok = a.table.lookup(ref.PodOrdinal)
			if ok {
				mru = (uint64(ref.PodOrdinal) << 32) | uint64(uint32(s))
			}
		}

		if !ok {
			continue // the first key fixes the candidate set
		}
		slot := &slots[s]
		if slot.seen < keyStamp-1 {
			continue // entries for dead slots should be pruned early
		}

		tier, tierOrdinal := ref.DeviceTier, ref.TierOrdinal
		isSpeculative := ref.Speculative
		if !isSpeculative {
			if tierOrdinal < 64 {
				if (a.speculativeTierChecked & (1 << tierOrdinal)) != 0 {
					isSpeculative = (a.speculativeTierMask & (1 << tierOrdinal)) != 0
				} else {
					isSpeculative = ref.DeviceTier == SpeculativeTier
					a.speculativeTierChecked |= (1 << tierOrdinal)
					if isSpeculative {
						a.speculativeTierMask |= (1 << tierOrdinal)
					}
				}
			} else {
				isSpeculative = ref.DeviceTier == SpeculativeTier
			}
		}
		if isSpeculative {
			tier, tierOrdinal = SpeculativeTier, speculativeTierOrdinal
		} else {
			slot.confirmedSeen = keyStamp
		}

		var w float64
		switch {
		case tierOrdinal < 64:
			if (weightCacheSet & (1 << tierOrdinal)) != 0 {
				w = weightCacheDirect[tierOrdinal]
			} else {
				w = a.weightOf(tier, tierOrdinal)
				weightCacheSet = a.weightCacheSet
			}
		case tierOrdinal == speculativeTierOrdinal && a.speculativeWeightSet:
			w = a.speculativeWeight
		default:
			w = a.weightOf(tier, tierOrdinal)
		}

		if slot.seen != keyStamp {
			slot.seen = keyStamp
			slot.weight = w
		} else if w > slot.weight {
			slot.weight = w
		}

		// stampTier inline
		if !slot.isMultiTier {
			if slot.tier0Ordinal == tierOrdinal {
				slot.tier0Seen = keyStamp
			}
		} else {
			if slot.tiers[0].ordinal == tierOrdinal {
				slot.tiers[0].seen = keyStamp
			} else {
				for t := 1; t < len(slot.tiers); t++ {
					if slot.tiers[t].ordinal == tierOrdinal {
						slot.tiers[t].seen = keyStamp
						break
					}
				}
			}
		}

		if i < len(posCache) {
			posCache[i] = posCacheEntryKey{
				podAndRawTier:  uint64(ref.PodOrdinal) | (uint64(ref.TierOrdinal) << 32),
				slot:           s,
				weight:         w,
				confirmed:      tierOrdinal != speculativeTierOrdinal,
				tierOrd:        tierOrdinal,
				refSpeculative: ref.Speculative,
				isSingleTier0:  !slot.isMultiTier && slot.hasTier0 && slot.tier0Ordinal == tierOrdinal && tierOrdinal != speculativeTierOrdinal,
			}
		}
	}
	a.mru = mru
	a.weightCacheSet = weightCacheSet
	return a.endKey()
}

// endKey closes the current key and reports whether any chain is still
// alive.
func (a *prefixAccumulator) endKey() bool {
	keyStamp := a.keyStamp
	slots := a.slots
	expectedSeen := (uint64(keyStamp) << 32) | uint64(keyStamp)
	if a.first {
		a.first = false
		nSlots := len(slots)
		if cap(a.active) < nSlots {
			a.active = make([]int32, nSlots)
		} else {
			a.active = a.active[:nSlots]
		}
		for i := range slots {
			s := &slots[i]
			s.matched, s.score = 1, s.weight
			if s.confirmedSeen == keyStamp {
				s.confirmed, s.confirmedAlive = 1, true
			}
			if !s.isMultiTier {
				if s.hasTier0 {
					s.tier0Count = 1
				}
			} else {
				for t := range s.tiers {
					s.tiers[t].count = 1
				}
			}
			a.active[i] = int32(i)
		}
		return len(a.active) > 0
	}

	active := a.active
	n := len(active)
	idx := 0

	if len(active) == len(slots) {
		for ; idx < n; idx++ {
			s := &slots[idx]
			if s.seen != keyStamp {
				break
			}
			s.matched++
			s.score += s.weight
			if !s.isMultiTier {
				if s.confirmedAlive && s.tier0Alive && *(*uint64)(unsafe.Pointer(&s.confirmedSeen)) == expectedSeen {
					s.confirmed++
					s.tier0Count++
				} else {
					if s.confirmedAlive {
						if s.confirmedSeen == keyStamp {
							s.confirmed++
						} else {
							s.confirmedAlive = false
						}
					}
					if s.tier0Alive {
						if s.tier0Seen == keyStamp {
							s.tier0Count++
						} else {
							s.tier0Alive = false
						}
					}
				}
			} else {
				if s.confirmedAlive && s.confirmedSeen == keyStamp {
					s.confirmed++
				} else {
					s.confirmedAlive = false
				}
				for t := range s.tiers {
					tc := &s.tiers[t]
					if tc.alive {
						if tc.seen == keyStamp {
							tc.count++
						} else {
							tc.alive = false
						}
					}
				}
			}
		}
	} else {
		for ; idx < n; idx++ {
			i := active[idx]
			s := &slots[i]
			if s.seen != keyStamp {
				break
			}
			s.matched++
			s.score += s.weight
			if !s.isMultiTier {
				if s.confirmedAlive && s.tier0Alive && *(*uint64)(unsafe.Pointer(&s.confirmedSeen)) == expectedSeen {
					s.confirmed++
					s.tier0Count++
				} else {
					if s.confirmedAlive {
						if s.confirmedSeen == keyStamp {
							s.confirmed++
						} else {
							s.confirmedAlive = false
						}
					}
					if s.tier0Alive {
						if s.tier0Seen == keyStamp {
							s.tier0Count++
						} else {
							s.tier0Alive = false
						}
					}
				}
			} else {
				if s.confirmedAlive && s.confirmedSeen == keyStamp {
					s.confirmed++
				} else {
					s.confirmedAlive = false
				}
				for t := range s.tiers {
					tc := &s.tiers[t]
					if tc.alive {
						if tc.seen == keyStamp {
							tc.count++
						} else {
							tc.alive = false
						}
					}
				}
			}
		}
	}

	if idx < n {
		keepCount := idx
		for j := idx + 1; j < n; j++ {
			i := active[j]
			s := &slots[i]
			if s.seen != keyStamp {
				continue
			}
			s.matched++
			s.score += s.weight
			if !s.isMultiTier {
				if s.confirmedAlive && s.tier0Alive && *(*uint64)(unsafe.Pointer(&s.confirmedSeen)) == expectedSeen {
					s.confirmed++
					s.tier0Count++
				} else {
					if s.confirmedAlive {
						if s.confirmedSeen == keyStamp {
							s.confirmed++
						} else {
							s.confirmedAlive = false
						}
					}
					if s.tier0Alive {
						if s.tier0Seen == keyStamp {
							s.tier0Count++
						} else {
							s.tier0Alive = false
						}
					}
				}
			} else {
				if s.confirmedAlive && s.confirmedSeen == keyStamp {
					s.confirmed++
				} else {
					s.confirmedAlive = false
				}
				for t := range s.tiers {
					tc := &s.tiers[t]
					if tc.alive {
						if tc.seen == keyStamp {
							tc.count++
						} else {
							tc.alive = false
						}
					}
				}
			}
			active[keepCount] = i
			keepCount++
		}
		a.active = active[:keepCount]
	}
	return len(a.active) > 0
}

// result materializes the accumulated matches.
func (a *prefixAccumulator) result() map[string]PodMatch {
	out := make(map[string]PodMatch, len(a.slots))
	for i := range a.slots {
		s := &a.slots[i]
		var byTier map[string]int
		if !s.isMultiTier {
			if s.hasTier0 {
				if (stringIdentical(s.tier0Name, a.lastSingularName) || s.tier0Name == a.lastSingularName) && s.tier0Count == a.lastSingularCount {
					byTier = a.lastSingularMap
				} else {
					slice := a.singularCache[s.tier0Name]
					if s.tier0Count < len(slice) && slice[s.tier0Count] != nil {
						byTier = slice[s.tier0Count]
					} else {
						if s.tier0Count >= len(slice) {
							newSlice := make([]map[string]int, s.tier0Count+64)
							copy(newSlice, slice)
							slice = newSlice
							a.singularCache[s.tier0Name] = slice
						}
						m := map[string]int{s.tier0Name: s.tier0Count}
						slice[s.tier0Count] = m
						byTier = m
					}
					a.lastSingularName = s.tier0Name
					a.lastSingularCount = s.tier0Count
					a.lastSingularMap = byTier
				}
			}
		} else {
			n := len(s.tiers)
			byTier = make(map[string]int, n)
			for j := 0; j < n; j++ {
				tc := &s.tiers[j]
				byTier[tc.name] = tc.count
			}
		}
		out[s.pod] = PodMatch{WeightedScore: s.score, MatchedBlocks: s.matched, ConfirmedBlocks: s.confirmed, BlocksByTier: byTier}
	}
	return out
}

func (a *prefixAccumulator) keyPodsFirst(entries []kvblock.PodEntry) bool {
	a.table.reset(len(entries))

	weightCacheDirect := &a.weightCacheDirect
	weightCacheSet := a.weightCacheSet

	for i := range entries {
		e := &entries[i]
		if i > 0 {
			prev := &entries[i-1]
			if (stringIdentical(prev.PodIdentifier, e.PodIdentifier) || prev.PodIdentifier == e.PodIdentifier) &&
				(stringIdentical(prev.DeviceTier, e.DeviceTier) || prev.DeviceTier == e.DeviceTier) &&
				prev.Speculative == e.Speculative {
				continue
			}
		}

		// Resolve pod ordinal
		var podOrd uint32
		h := fastHash(e.PodIdentifier)
		idx := h & 2047
		if stringIdentical(a.podCache[idx].name, e.PodIdentifier) || a.podCache[idx].name == e.PodIdentifier {
			podOrd = a.podCache[idx].ord
		} else {
			podOrd = a.podOrdinal(e.PodIdentifier)
			a.podCache[idx].name = e.PodIdentifier
			a.podCache[idx].ord = podOrd
		}

		a.mruName = e.PodIdentifier
		a.mruOrd = podOrd
		a.hasMru = true

		s, ok := a.table.lookup(podOrd)
		if !ok {
			if a.hasFilter && !a.filter.Has(e.PodIdentifier) {
				continue
			}
			s = a.newSlot(e.PodIdentifier)
			a.table.insert(podOrd, s)
		}

		// Resolve tier ordinal and tier name
		var tierOrd uint32
		var tier string
		if e.Speculative || e.DeviceTier == SpeculativeTier {
			tier = SpeculativeTier
			tierOrd = speculativeTierOrdinal
		} else {
			tier = e.DeviceTier
			if a.hasLastTier && (stringIdentical(a.lastTierName, e.DeviceTier) || a.lastTierName == e.DeviceTier) {
				tierOrd = a.lastTierOrd
			} else {
				tierOrd = a.cachedTierOrdinal(e.DeviceTier)
				a.lastTierName = e.DeviceTier
				a.lastTierOrd = tierOrd
				a.hasLastTier = true
			}
		}

		slot := &a.slots[s]
		if tierOrd != speculativeTierOrdinal {
			slot.confirmedSeen = a.keyStamp
		}

		var w float64
		switch {
		case tierOrd < 64:
			if (weightCacheSet & (1 << tierOrd)) != 0 {
				w = weightCacheDirect[tierOrd]
			} else {
				w = a.weightOf(tier, tierOrd)
				weightCacheSet = a.weightCacheSet
			}
		case tierOrd == speculativeTierOrdinal && a.speculativeWeightSet:
			w = a.speculativeWeight
		default:
			w = a.weightOf(tier, tierOrd)
		}

		if slot.seen != a.keyStamp {
			slot.seen = a.keyStamp
			slot.weight = w
		} else if w > slot.weight {
			slot.weight = w
		}

		// stampTier inline in keyPodsFirst
		stamped := false
		if !slot.isMultiTier {
			if slot.hasTier0 {
				if slot.tier0Ordinal == tierOrd {
					slot.tier0Seen = a.keyStamp
					stamped = true
				} else {
					// Fallback to multi-tier
					slot.isMultiTier = true
					slot.tiers = append(slot.tiers, tierChain{
						ordinal: slot.tier0Ordinal,
						name:    slot.tier0Name,
						count:   slot.tier0Count,
						seen:    slot.tier0Seen,
						alive:   slot.tier0Alive,
					})
					slot.tiers = append(slot.tiers, tierChain{
						ordinal: tierOrd,
						name:    tier,
						seen:    a.keyStamp,
						alive:   true,
					})
					stamped = true
				}
			} else {
				slot.hasTier0 = true
				slot.tier0Ordinal = tierOrd
				slot.tier0Name = tier
				slot.tier0Seen = a.keyStamp
				slot.tier0Alive = true
				stamped = true
			}
		} else {
			// Multi-tier path
			for t := range slot.tiers {
				if slot.tiers[t].ordinal == tierOrd {
					slot.tiers[t].seen = a.keyStamp
					stamped = true
					break
				}
			}
			if !stamped {
				slot.tiers = append(slot.tiers, tierChain{
					ordinal: tierOrd,
					name:    tier,
					seen:    a.keyStamp,
					alive:   true,
				})
			}
		}

		if i < 256 {
			a.posCache256[i] = posCacheEntry{
				podName:       e.PodIdentifier,
				tierName:      e.DeviceTier,
				speculative:   e.Speculative,
				podOrd:        podOrd,
				tierOrd:       tierOrd,
				slot:          s,
				weight:        w,
				confirmed:     tierOrd != speculativeTierOrdinal,
				isSingleTier0: !slot.isMultiTier && slot.hasTier0 && slot.tier0Ordinal == tierOrd && tierOrd != speculativeTierOrdinal,
			}
		}
	}
	a.weightCacheSet = weightCacheSet
	return a.endKey()
}

func (a *prefixAccumulator) keyPods(entries []kvblock.PodEntry) bool {
	a.keyStamp++
	if a.first {
		return a.keyPodsFirst(entries)
	}

	slots := a.slots
	keyStamp := a.keyStamp
	expectedSeen := (uint64(keyStamp) << 32) | uint64(keyStamp)
	weightCacheSet := a.weightCacheSet
	weightCacheDirect := &a.weightCacheDirect

	for i := range entries {
		e := &entries[i]

		var s int32
		var podOrd uint32
		var tierOrd uint32
		var tier string
		var ok bool

		if i < 256 {
			entry := &a.posCache256[i]
			if (stringIdentical(entry.podName, e.PodIdentifier) || entry.podName == e.PodIdentifier) &&
				(stringIdentical(entry.tierName, e.DeviceTier) || entry.tierName == e.DeviceTier) &&
				entry.speculative == e.Speculative {
				s := entry.slot
				slot := &slots[s]
				if slot.seen < keyStamp-1 {
					continue
				}
				w := entry.weight
				if slot.seen != keyStamp {
					slot.seen = keyStamp
					slot.weight = w
				} else if w > slot.weight {
					slot.weight = w
				}
				if entry.isSingleTier0 && !slot.isMultiTier {
					*(*uint64)(unsafe.Pointer(&slot.confirmedSeen)) = expectedSeen
				} else {
					if entry.confirmed {
						slot.confirmedSeen = keyStamp
					}
					// stampTier inline
					if !slot.isMultiTier {
						if slot.tier0Ordinal == entry.tierOrd {
							slot.tier0Seen = keyStamp
						}
					} else {
						if slot.tiers[0].ordinal == entry.tierOrd {
							slot.tiers[0].seen = keyStamp
						} else {
							for t := 1; t < len(slot.tiers); t++ {
								if slot.tiers[t].ordinal == entry.tierOrd {
									slot.tiers[t].seen = keyStamp
									break
								}
							}
						}
					}
				}
				continue
			}
		}

		if i > 0 {
			prev := &entries[i-1]
			if (stringIdentical(prev.PodIdentifier, e.PodIdentifier) || prev.PodIdentifier == e.PodIdentifier) &&
				(stringIdentical(prev.DeviceTier, e.DeviceTier) || prev.DeviceTier == e.DeviceTier) &&
				prev.Speculative == e.Speculative {
				continue
			}
		}

		var found bool
		if a.hasMru && (stringIdentical(a.mruName, e.PodIdentifier) || a.mruName == e.PodIdentifier) {
			podOrd = a.mruOrd
			found = true
		} else {
			h := fastHash(e.PodIdentifier)
			idx := h & 2047
			if stringIdentical(a.podCache[idx].name, e.PodIdentifier) || a.podCache[idx].name == e.PodIdentifier {
				podOrd = a.podCache[idx].ord
				found = true
			} else {
				podOrd, found = a.podsMap[e.PodIdentifier]
				if found {
					a.podCache[idx].name = e.PodIdentifier
					a.podCache[idx].ord = podOrd
				}
			}
			if found {
				a.mruName = e.PodIdentifier
				a.mruOrd = podOrd
				a.hasMru = true
			}
		}

		if !found {
			continue
		}

		s, ok = a.table.lookup(podOrd)
		if !ok {
			continue
		}

		if e.Speculative || e.DeviceTier == SpeculativeTier {
			tier = SpeculativeTier
			tierOrd = speculativeTierOrdinal
		} else {
			tier = e.DeviceTier
			if a.hasLastTier && (stringIdentical(a.lastTierName, e.DeviceTier) || a.lastTierName == e.DeviceTier) {
				tierOrd = a.lastTierOrd
			} else {
				tierOrd = a.cachedTierOrdinal(e.DeviceTier)
				a.lastTierName = e.DeviceTier
				a.lastTierOrd = tierOrd
				a.hasLastTier = true
			}
		}

		slot := &slots[s]
		if slot.seen < keyStamp-1 {
			continue
		}

		if tierOrd != speculativeTierOrdinal {
			slot.confirmedSeen = keyStamp
		}

		var w float64
		switch {
		case tierOrd < 64:
			if (weightCacheSet & (1 << tierOrd)) != 0 {
				w = weightCacheDirect[tierOrd]
			} else {
				w = a.weightOf(tier, tierOrd)
				weightCacheSet = a.weightCacheSet
			}
		case tierOrd == speculativeTierOrdinal && a.speculativeWeightSet:
			w = a.speculativeWeight
		default:
			w = a.weightOf(tier, tierOrd)
		}

		if slot.seen != keyStamp {
			slot.seen = keyStamp
			slot.weight = w
		} else if w > slot.weight {
			slot.weight = w
		}

		if !slot.isMultiTier {
			if slot.tier0Ordinal == tierOrd {
				slot.tier0Seen = keyStamp
			}
		} else {
			if slot.tiers[0].ordinal == tierOrd {
				slot.tiers[0].seen = keyStamp
			} else {
				for t := 1; t < len(slot.tiers); t++ {
					if slot.tiers[t].ordinal == tierOrd {
						slot.tiers[t].seen = keyStamp
						break
					}
				}
			}
		}

		if i < 256 {
			a.posCache256[i] = posCacheEntry{
				podName:       e.PodIdentifier,
				tierName:      e.DeviceTier,
				speculative:   e.Speculative,
				podOrd:        podOrd,
				tierOrd:       tierOrd,
				slot:          s,
				weight:        w,
				confirmed:     tierOrd != speculativeTierOrdinal,
				isSingleTier0: !slot.isMultiTier && slot.hasTier0 && slot.tier0Ordinal == tierOrd && tierOrd != speculativeTierOrdinal,
			}
		}
	}
	a.weightCacheSet = weightCacheSet
	return a.endKey()
}

// newSlot appends a candidate, reusing a pooled slot's tier storage when one
// is available.
func (a *prefixAccumulator) newSlot(pod string) int32 {
	n := len(a.slots)
	if n < cap(a.slots) {
		a.slots = a.slots[:n+1]
		s := &a.slots[n]
		s.pod = pod
		s.matched = 0
		s.score = 0
		s.seen = 0
		s.weight = 0
		s.tiers = s.tiers[:0]
		s.confirmed = 0
		s.confirmedSeen = 0
		s.confirmedAlive = false
		s.hasTier0 = false
		s.isMultiTier = false
		s.tier0Ordinal = 0
		s.tier0Seen = 0
		s.tier0Count = 0
		s.tier0Alive = false
		s.tier0Name = ""
	} else {
		a.slots = append(a.slots, matchSlot{
			pod:   pod,
			tiers: make([]tierChain, 0, 4),
		})
	}
	// Candidate pods per request stay far below the int32 range; clamp
	// defensively so a pathological request cannot wrap the index negative.
	slot := n
	if slot > math.MaxInt32 {
		slot = math.MaxInt32
	}
	return int32(slot)
}

// weightOf resolves a tier's weight, caching by ordinal so the configured
// map is consulted once per tier per accumulation.
func (a *prefixAccumulator) weightOf(tier string, ordinal uint32) float64 {
	if ordinal < 64 {
		if (a.weightCacheSet & (1 << ordinal)) != 0 {
			return a.weightCacheDirect[ordinal]
		}
		var isSpeculative bool
		if (a.speculativeTierChecked & (1 << ordinal)) != 0 {
			isSpeculative = (a.speculativeTierMask & (1 << ordinal)) != 0
		} else {
			isSpeculative = tier == SpeculativeTier
			a.speculativeTierChecked |= (1 << ordinal)
			if isSpeculative {
				a.speculativeTierMask |= (1 << ordinal)
			}
		}
		w := unknownTierWeight
		if isSpeculative {
			w = speculativeTierWeight
		} else if configured, ok := a.weights[tier]; ok {
			w = configured
		}
		a.weightCacheDirect[ordinal] = w
		a.weightCacheSet |= (1 << ordinal)
		return w
	}
	if ordinal == speculativeTierOrdinal {
		if a.speculativeWeightSet {
			return a.speculativeWeight
		}
		w := speculativeTierWeight
		if configured, ok := a.weights[tier]; ok {
			w = configured
		}
		a.speculativeWeight = w
		a.speculativeWeightSet = true
		return w
	}
	for i := range a.weightCache {
		if a.weightCache[i].ordinal == ordinal {
			return a.weightCache[i].weight
		}
	}
	w := unknownTierWeight
	if tier == SpeculativeTier {
		w = speculativeTierWeight
	}
	if configured, ok := a.weights[tier]; ok {
		w = configured
	}
	a.weightCache = append(a.weightCache, tierWeight{ordinal: ordinal, weight: w})
	return w
}
