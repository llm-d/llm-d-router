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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

// SpeculativeTier is the tier name under which speculative entries count in
// PodMatch.BlocksByTier. Speculative entries carry no engine-reported device
// tier; an entry whose device tier is reported under this name counts in the
// same chain.
const SpeculativeTier = "speculative"

// defaultTierWeight scores blocks held in a tier without a configured weight.
const defaultTierWeight = 1.0

// matchCancellationMask paces context-cancellation checks over key
// positions: positions where pos&mask == 0 poll ctx.Err().
const matchCancellationMask = 255

// PodMatch is one pod's prefix match for a key sequence. All values cover
// the contiguous chain of keys the pod holds, counted from the first key.
type PodMatch struct {
	// WeightedScore sums, per block of the chain, the highest device-tier
	// weight among the pod's entries for that block; tiers without a
	// configured weight count defaultTierWeight.
	WeightedScore float64
	// MatchedBlocks is the chain length in blocks, regardless of tier.
	MatchedBlocks int
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
		attribute.Int("llm_d.kv_cache.prefix_match.key_count", len(keys)),
		attribute.Int("llm_d.kv_cache.prefix_match.pod_filter_count", podFilter.Len()),
		attribute.Bool("llm_d.kv_cache.prefix_match.walked", k.keyWalker != nil),
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
		attribute.Int("llm_d.kv_cache.prefix_match.pods_matched", len(matches)),
		attribute.Int("llm_d.kv_cache.prefix_match.longest_chain", blocksFound),
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

// matchMaterialized feeds the accumulator from a Lookup result, walking keys
// in order and stopping at the first key without entries. Pod and tier
// ordinals are assigned per call, since materialized entries carry none.
func matchMaterialized(ctx context.Context, keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	weights map[string]float64, filter sets.Set[string],
) (map[string]PodMatch, error) {
	acc := acquireAccumulator(weights, filter)
	defer releaseAccumulator(acc)

	pods, tiers := ordinalTable{}, ordinalTable{}
	var refs []kvblock.EntryRef
	for pos, key := range keys {
		if pos&matchCancellationMask == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		entries := keyToPods[key]
		if len(entries) == 0 {
			break
		}
		refs = refs[:0]
		for _, e := range entries {
			refs = append(refs, kvblock.EntryRef{
				PodEntry:    e,
				PodOrdinal:  pods.of(e.PodIdentifier),
				TierOrdinal: tiers.of(e.DeviceTier),
			})
		}
		if !acc.key(refs) {
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

// ordinalTable assigns dense ordinals to names in first-seen order.
type ordinalTable map[string]uint32

func (t ordinalTable) of(name string) uint32 {
	if id, ok := t[name]; ok {
		return id
	}
	id := uint32(len(t))
	t[name] = id
	return id
}

// speculativeTierOrdinal keys the speculative per-tier chain. Feeders assign
// tier ordinals from zero, so the top of the range never collides.
const speculativeTierOrdinal = math.MaxUint32

// slotRef maps one pod ordinal to a request-local slot.
type slotRef struct {
	ordinal uint32
	slot    uint32 // slot index plus one; zero marks an empty bucket
}

// slotTable is an open-addressed map from pod ordinal to request-local slot.
// It is sized by the first key's entry count, so request state scales with
// the live candidates rather than with every ordinal an index ever assigned.
type slotTable struct {
	buckets []slotRef
}

func (t *slotTable) reset(numEntries int) {
	size := 2
	for size < numEntries*2 {
		size <<= 1
	}
	if cap(t.buckets) < size {
		t.buckets = make([]slotRef, size)
		return
	}
	t.buckets = t.buckets[:size]
	clear(t.buckets)
}

func (t *slotTable) lookup(ordinal uint32) (int32, bool) {
	mask := uint32(len(t.buckets) - 1)
	i := ordinal * 2654435761 & mask
	for {
		b := t.buckets[i]
		if b.slot == 0 {
			return 0, false
		}
		if b.ordinal == ordinal {
			return int32(b.slot - 1), true
		}
		i = (i + 1) & mask
	}
}

func (t *slotTable) insert(ordinal uint32, slot int32) {
	mask := uint32(len(t.buckets) - 1)
	i := ordinal * 2654435761 & mask
	for t.buckets[i].slot != 0 {
		i = (i + 1) & mask
	}
	t.buckets[i] = slotRef{ordinal: ordinal, slot: uint32(slot) + 1}
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
	pod     string
	matched int
	score   float64
	// seen is the key stamp of the last key holding this pod; weight is the
	// highest tier weight among its entries at that key.
	seen   uint32
	weight float64
	tiers  []tierChain
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
	weights map[string]float64
	filter  sets.Set[string]

	table    slotTable
	slots    []matchSlot
	active   []int32
	keyStamp uint32
	first    bool

	// weightCache holds the weight of every tier seen in this accumulation,
	// scanned linearly: requests see a handful of tiers.
	weightCache []tierWeight
}

var accumulatorPool = sync.Pool{New: func() any { return &prefixAccumulator{} }}

func acquireAccumulator(weights map[string]float64, filter sets.Set[string]) *prefixAccumulator {
	a, _ := accumulatorPool.Get().(*prefixAccumulator)
	a.weights, a.filter = weights, filter
	a.slots = a.slots[:0]
	a.active = a.active[:0]
	a.keyStamp = 0
	a.first = true
	a.weightCache = a.weightCache[:0]
	return a
}

func releaseAccumulator(a *prefixAccumulator) {
	a.weights, a.filter = nil, nil
	accumulatorPool.Put(a)
}

// key folds one key's entries into the chains and reports whether any chain
// is still alive. entries is borrowed for the duration of the call.
func (a *prefixAccumulator) key(entries []kvblock.EntryRef) bool {
	a.keyStamp++
	if a.first {
		a.table.reset(len(entries))
	}

	var prev *kvblock.EntryRef
	for i := range entries {
		ref := &entries[i]
		// Another rank of an endpoint just folded at this key adds nothing
		// to its chains.
		if prev != nil && ref.PodOrdinal == prev.PodOrdinal && ref.TierOrdinal == prev.TierOrdinal &&
			ref.Speculative == prev.Speculative {
			continue
		}
		prev = ref

		s, ok := a.table.lookup(ref.PodOrdinal)
		if !ok {
			if !a.first || (a.filter.Len() > 0 && !a.filter.Has(ref.PodIdentifier)) {
				continue // the first key fixes the candidate set
			}
			s = a.newSlot(ref.PodIdentifier)
			a.table.insert(ref.PodOrdinal, s)
		}
		slot := &a.slots[s]

		w := a.weightOf(ref.DeviceTier, ref.TierOrdinal)
		switch {
		case slot.seen != a.keyStamp:
			slot.seen = a.keyStamp
			slot.weight = w
		case w > slot.weight:
			slot.weight = w
		}

		tier, tierOrdinal := ref.DeviceTier, ref.TierOrdinal
		if ref.Speculative || ref.DeviceTier == SpeculativeTier {
			tier, tierOrdinal = SpeculativeTier, speculativeTierOrdinal
		}
		if !a.stampTier(slot, tierOrdinal) && a.first {
			slot.tiers = append(slot.tiers, tierChain{ordinal: tierOrdinal, name: tier, seen: a.keyStamp, alive: true})
		}
	}
	return a.endKey()
}

// stampTier marks tier as held at the current key and reports whether the
// slot tracks that tier.
func (a *prefixAccumulator) stampTier(slot *matchSlot, tierOrdinal uint32) bool {
	for i := range slot.tiers {
		if slot.tiers[i].ordinal == tierOrdinal {
			slot.tiers[i].seen = a.keyStamp
			return true
		}
	}
	return false
}

// endKey closes the current key and reports whether any chain is still
// alive.
func (a *prefixAccumulator) endKey() bool {
	if a.first {
		a.first = false
		for i := range a.slots {
			s := &a.slots[i]
			s.matched, s.score = 1, s.weight
			for t := range s.tiers {
				s.tiers[t].count = 1
			}
			a.active = append(a.active, int32(i))
		}
		return len(a.active) > 0
	}

	keep := a.active[:0]
	for _, i := range a.active {
		s := &a.slots[i]
		if s.seen != a.keyStamp {
			continue // the chain ends at the first key the pod does not hold
		}
		s.matched++
		s.score += s.weight
		for t := range s.tiers {
			tc := &s.tiers[t]
			switch {
			case !tc.alive:
			case tc.seen == a.keyStamp:
				tc.count++
			default:
				tc.alive = false
			}
		}
		keep = append(keep, i)
	}
	a.active = keep
	return len(a.active) > 0
}

// result materializes the accumulated matches.
func (a *prefixAccumulator) result() map[string]PodMatch {
	out := make(map[string]PodMatch, len(a.slots))
	for i := range a.slots {
		s := &a.slots[i]
		byTier := make(map[string]int, len(s.tiers))
		for _, tc := range s.tiers {
			byTier[tc.name] = tc.count
		}
		out[s.pod] = PodMatch{WeightedScore: s.score, MatchedBlocks: s.matched, BlocksByTier: byTier}
	}
	return out
}

// newSlot appends a candidate, reusing a pooled slot's tier storage when one
// is available.
func (a *prefixAccumulator) newSlot(pod string) int32 {
	n := len(a.slots)
	if n < cap(a.slots) {
		a.slots = a.slots[:n+1]
		s := &a.slots[n]
		*s = matchSlot{pod: pod, tiers: s.tiers[:0]}
	} else {
		a.slots = append(a.slots, matchSlot{pod: pod})
	}
	return int32(n)
}

// weightOf resolves a tier's weight, caching by ordinal so the configured
// map is consulted once per tier per accumulation.
func (a *prefixAccumulator) weightOf(tier string, ordinal uint32) float64 {
	for i := range a.weightCache {
		if a.weightCache[i].ordinal == ordinal {
			return a.weightCache[i].weight
		}
	}
	w := defaultTierWeight
	if configured, ok := a.weights[tier]; ok {
		w = configured
	}
	a.weightCache = append(a.weightCache, tierWeight{ordinal: ordinal, weight: w})
	return w
}
