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
	// while the pod holds every previous block in that same tier, and only in
	// complete engine-retrieval units (see PodEntry.RetrievalSpan).
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

// matchMaterialized feeds the accumulator from a Lookup result, walking keys
// in order and stopping at the first key without entries. Pod and tier
// ordinals are assigned per call, since materialized entries carry none.
func matchMaterialized(ctx context.Context, keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	weights map[string]float64, filter sets.Set[string],
) (map[string]PodMatch, error) {
	acc := acquireAccumulator(weights, filter)
	defer releaseAccumulator(acc)

	refs := acc.refs[:0]
	defer func() {
		clear(refs[:cap(refs)])
		acc.refs = refs[:0]
	}()

	for pos, key := range keys {
		if pos&matchCancellationMask == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		entries := keyToPods[key]
		if len(entries) == 0 {
			break
		}
		// Keys along a cached prefix usually hold the same entries in the same
		// order. Reuse matching prefix ordinals in place, and fold a full
		// repeat directly into pendingRun.
		same := !acc.first && len(entries) == len(refs)
		j := 0
		for ; j < len(entries) && j < len(refs); j++ {
			e, r := &entries[j], &refs[j].PodEntry
			if e.PodIdentifier != r.PodIdentifier || e.DeviceTier != r.DeviceTier || e.Speculative != r.Speculative || e.RetrievalSpan != r.RetrievalSpan {
				same = false
				break
			}
		}
		// Identical span-1 keys can share one increment. A multi-span unit has
		// to visit creditKey on every key, including repeats of the same entries.
		if same && !acc.sawMulti && !podEntriesHaveMultiSpan(entries) {
			acc.pendingRun++
			continue
		}
		if same {
			if !acc.key(refs) {
				break
			}
			continue
		}
		refs = refs[:j]
		for ; j < len(entries); j++ {
			e := entries[j]
			refs = append(refs, kvblock.EntryRef{
				PodEntry:    e,
				PodOrdinal:  acc.podOrdinal(e.PodIdentifier),
				TierOrdinal: acc.tierOrdinal(e.DeviceTier),
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
	id := clampOrdinal(len(t))
	t[name] = id
	return id
}

// tierNameOrdinal pairs a device-tier name with its request-local ordinal.
type tierNameOrdinal struct {
	name    string
	ordinal uint32
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

// slotRef maps one pod ordinal to a request-local slot.
type slotRef struct {
	ordinal uint32
	slot    int32 // slot index plus one; zero marks an empty bucket
}

// slotTable is an open-addressed map from pod ordinal to request-local slot.
// It is sized by the first key's entry count, so request state scales with
// the live candidates rather than with every ordinal an index ever assigned.
type slotTable struct {
	buckets []slotRef
	mask    uint32
}

func (t *slotTable) reset(numEntries int) {
	// Capping growth at 1<<31 keeps size-1 well within uint32 range, so the
	// mask conversion below never truncates.
	size := 2
	for size < numEntries*2 && size < 1<<31 {
		size <<= 1
	}
	if cap(t.buckets) < size {
		t.buckets = make([]slotRef, size)
	} else {
		t.buckets = t.buckets[:size]
		clear(t.buckets)
	}
	if size-1 > math.MaxUint32 {
		size = math.MaxUint32 + 1
	}
	t.mask = uint32(size - 1)
}

func (t *slotTable) lookup(ordinal uint32) (int32, bool) {
	i := ordinal * 2654435761 & t.mask
	for {
		b := t.buckets[i]
		if b.slot == 0 {
			return 0, false
		}
		if b.ordinal == ordinal {
			return b.slot - 1, true
		}
		i = (i + 1) & t.mask
	}
}

func (t *slotTable) insert(ordinal uint32, slot int32) {
	i := ordinal * 2654435761 & t.mask
	for t.buckets[i].slot != 0 {
		i = (i + 1) & t.mask
	}
	t.buckets[i] = slotRef{ordinal: ordinal, slot: slot + 1}
}

// tierChain tracks one tier's contiguous prefix for a candidate pod.
// Retrieval-unit progress lives on unitTrack: BlocksByTier stays broken
// when this tier misses a key, even if a later unit of the same tier completes.
type tierChain struct {
	ordinal uint32
	name    string
	count   int
	// seen is the key stamp of the last key where the pod held this tier.
	seen  uint32
	alive bool
	// seenSpan is the retrieval span observed at the current key.
	seenSpan int
}

// unitTrack follows one tier's retrieval units for the whole prefix.
// A tier that first appears after the opening key, or that starts again
// after a gap, still completes here. That does not extend BlocksByTier.
type unitTrack struct {
	ordinal  uint32
	seen     uint32
	seenSpan int
	pending  int
	unitSpan int
}

// deferredKey is one multi-span-only key whose weight is waiting for a
// retrieval unit to complete. covers lists the in-progress units that still
// include the key; the weight is dropped once every cover aborts.
type deferredKey struct {
	weight float64
	covers []uint32
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
	alive  bool
	// confirmed tracks the chain of keys held in a non-speculative tier;
	// confirmedSeen is the key stamp of the last key holding one.
	confirmed      int
	confirmedSeen  uint32
	confirmedAlive bool
	tiers          []tierChain
	tiersBuf       [2]tierChain
	// units tracks retrieval-unit progress for every multi-span tier, including
	// tiers that are absent from tiers because they were not on the first key.
	units    []unitTrack
	unitsBuf [2]unitTrack
	// uncredited holds per-key weights for keys the pod holds that have not
	// yet been credited because every entry at those keys belonged to an
	// incomplete multi-key retrieval unit.
	uncredited []deferredKey
	// keyHoldSpan1 / keyHoldMulti record how the pod holds the current key.
	// MatchedBlocks credits span-1 holds immediately; multi-span-only holds
	// defer credit until a retrieval unit completes.
	keyHoldSpan1 bool
	keyHoldMulti bool
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
	// sawMulti is set once any entry in this accumulation belongs to a
	// multi-key retrieval unit. Until then endKey keeps the span-1 path.
	sawMulti bool

	// prev holds the entry signatures of the last fully folded key; pendingRun
	// counts consecutive matching keys whose slot increments are deferred.
	prev       []entrySig
	pendingRun int

	// weightCache holds the weight of every tier seen in this accumulation,
	// scanned linearly: requests see a handful of tiers.
	weightCache []tierWeight

	// Materialized-lookup scratch state, allocated lazily when matchMaterialized
	// runs so walked calls allocate no extra state.
	pods         ordinalTable
	tierOrdinals []tierNameOrdinal
	refs         []kvblock.EntryRef
}

var accumulatorPool = sync.Pool{New: func() any { return &prefixAccumulator{} }}

func acquireAccumulator(weights map[string]float64, filter sets.Set[string]) *prefixAccumulator {
	a, _ := accumulatorPool.Get().(*prefixAccumulator)
	a.weights, a.filter = weights, filter
	a.slots = a.slots[:0]
	a.active = a.active[:0]
	a.keyStamp = 0
	a.first = true
	a.sawMulti = false
	a.weightCache = a.weightCache[:0]
	a.prev = a.prev[:0]
	a.pendingRun = 0
	if a.pods != nil {
		clear(a.pods)
	}
	a.tierOrdinals = a.tierOrdinals[:0]
	return a
}

func releaseAccumulator(a *prefixAccumulator) {
	a.weights, a.filter = nil, nil
	for i := range a.slots {
		a.slots[i].pod = ""
		clear(a.slots[i].tiers)
		clear(a.slots[i].units)
		for j := range a.slots[i].uncredited {
			a.slots[i].uncredited[j].covers = nil
		}
		a.slots[i].uncredited = a.slots[i].uncredited[:0]
	}
	if a.pods != nil {
		clear(a.pods)
	}
	clear(a.tierOrdinals)
	a.tierOrdinals = a.tierOrdinals[:0]
	accumulatorPool.Put(a)
}

func (a *prefixAccumulator) podOrdinal(name string) uint32 {
	if a.pods == nil {
		a.pods = make(ordinalTable, 64)
	}
	return a.pods.of(name)
}

func (a *prefixAccumulator) tierOrdinal(name string) uint32 {
	for i := range a.tierOrdinals {
		if a.tierOrdinals[i].name == name {
			return a.tierOrdinals[i].ordinal
		}
	}
	id := clampOrdinal(len(a.tierOrdinals))
	a.tierOrdinals = append(a.tierOrdinals, tierNameOrdinal{name: name, ordinal: id})
	return id
}

// flushRun applies deferred increments from consecutive identical keys.
// Only used on the span-1 path; multi-span keys never enter pendingRun.
func (a *prefixAccumulator) flushRun() {
	if a.pendingRun == 0 {
		return
	}
	n := a.pendingRun
	a.pendingRun = 0
	fn := float64(n)
	for _, i := range a.active {
		s := &a.slots[i]
		s.matched += n
		s.score += fn * s.weight
		if s.confirmedAlive {
			s.confirmed += n
		}
		for t := range s.tiers {
			if s.tiers[t].alive {
				s.tiers[t].count += n
			}
		}
	}
}

// key folds one key's entries into the chains and reports whether any chain
// is still alive. entries is borrowed for the duration of the call.
func (a *prefixAccumulator) key(entries []kvblock.EntryRef) bool {
	// When entries match the previous key, every live slot holds this key in
	// the same tiers; batch the increments until the run ends. Skip once any
	// multi-canonical retrieval unit is in play: those keys must go through
	// creditKey so incomplete spans are not scored.
	if !a.first && !a.sawMulti && !entriesHaveMultiSpan(entries) && a.samePrev(entries) {
		a.pendingRun++
		return len(a.active) > 0
	}
	a.flushRun()
	a.keyStamp++
	if a.first {
		a.table.reset(len(entries))
	}
	if cap(a.prev) < len(entries) {
		a.prev = make([]entrySig, len(entries))
	} else {
		a.prev = a.prev[:len(entries)]
	}

	hasFilter := a.filter.Len() > 0
	var prevSig entrySig
	for i := range entries {
		ref := &entries[i]
		sig := sigOf(ref)
		a.prev[i] = sig
		// Another rank of an endpoint just folded at this key adds nothing
		// to its chains.
		if i > 0 && sig == prevSig {
			continue
		}
		prevSig = sig

		s, ok := a.table.lookup(ref.PodOrdinal)
		if !ok {
			if !a.first || (hasFilter && !a.filter.Has(ref.PodIdentifier)) {
				continue // the first key fixes the candidate set
			}
			s = a.newSlot(ref.PodIdentifier)
			a.table.insert(ref.PodOrdinal, s)
		}
		slot := &a.slots[s]
		if !a.first && !slot.alive {
			continue
		}

		tier, tierOrdinal := ref.DeviceTier, ref.TierOrdinal
		if ref.Speculative || ref.DeviceTier == SpeculativeTier {
			tier, tierOrdinal = SpeculativeTier, speculativeTierOrdinal
		} else {
			slot.confirmedSeen = a.keyStamp
		}

		w := a.weightOf(tier, tierOrdinal)
		switch {
		case slot.seen != a.keyStamp:
			slot.seen = a.keyStamp
			slot.weight = w
			slot.keyHoldSpan1 = false
			slot.keyHoldMulti = false
		case w > slot.weight:
			slot.weight = w
		}

		span := ref.RetrievalSpan
		if span < 1 {
			span = 1
		}
		if span == 1 {
			slot.keyHoldSpan1 = true
		} else {
			a.sawMulti = true
			slot.keyHoldMulti = true
			a.noteUnit(slot, tierOrdinal, span)
		}
		if !a.stampTier(slot, tierOrdinal, span) && a.first {
			slot.tiers = append(slot.tiers, tierChain{
				ordinal:  tierOrdinal,
				name:     tier,
				seen:     a.keyStamp,
				alive:    true,
				seenSpan: span,
			})
		}
	}
	return a.endKey()
}

// entrySig packs the fields of an EntryRef that affect matching into one
// 64-bit word so slice comparisons avoid padded-struct equality helpers.
// Speculative entries map to speculativeTierOrdinal, which clampOrdinal and
// the index interner reserve above normal tier ordinals.
type entrySig uint64

func sigOf(ref *kvblock.EntryRef) entrySig {
	tier := ref.TierOrdinal
	if ref.Speculative {
		tier = speculativeTierOrdinal
	}
	return entrySig(ref.PodOrdinal)<<32 | entrySig(tier)
}

func (a *prefixAccumulator) samePrev(entries []kvblock.EntryRef) bool {
	if len(entries) != len(a.prev) {
		return false
	}
	for i := range entries {
		if sigOf(&entries[i]) != a.prev[i] {
			return false
		}
	}
	return true
}

func entriesHaveMultiSpan(entries []kvblock.EntryRef) bool {
	for i := range entries {
		if entries[i].RetrievalSpan > 1 {
			return true
		}
	}
	return false
}

func podEntriesHaveMultiSpan(entries []kvblock.PodEntry) bool {
	for i := range entries {
		if entries[i].RetrievalSpan > 1 {
			return true
		}
	}
	return false
}

// noteUnit records that a multi-span tier is held at the current key.
// The tier is tracked even when it is not part of the first-key prefix.
func (a *prefixAccumulator) noteUnit(slot *matchSlot, ordinal uint32, span int) {
	for i := range slot.units {
		if slot.units[i].ordinal == ordinal {
			slot.units[i].seen = a.keyStamp
			slot.units[i].seenSpan = span
			return
		}
	}
	slot.units = append(slot.units, unitTrack{ordinal: ordinal, seen: a.keyStamp, seenSpan: span})
}

// stampTier marks tier as held at the current key with the given retrieval
// span and reports whether the slot tracks that tier.
func (a *prefixAccumulator) stampTier(slot *matchSlot, tierOrdinal uint32, span int) bool {
	for i := range slot.tiers {
		if slot.tiers[i].ordinal == tierOrdinal {
			slot.tiers[i].seen = a.keyStamp
			slot.tiers[i].seenSpan = span
			return true
		}
	}
	return false
}

// endKey closes the current key and reports whether any chain is still
// alive. Requests whose blocks are all independently retrievable stay on
// endKeyIndependent; the first multi-key retrieval unit switches the rest of
// the accumulation onto creditKey.
func (a *prefixAccumulator) endKey() bool {
	if !a.sawMulti {
		return a.endKeyIndependent()
	}
	if a.first {
		a.first = false
		for i := range a.slots {
			s := &a.slots[i]
			s.alive = true
			a.creditKey(s, true)
			a.active = append(a.active, int32(i))
		}
		return len(a.active) > 0
	}

	keep := a.active[:0]
	for _, i := range a.active {
		s := &a.slots[i]
		if s.seen != a.keyStamp {
			s.alive = false
			continue // the chain ends at the first key the pod does not hold
		}
		a.creditKey(s, false)
		keep = append(keep, i)
	}
	a.active = keep
	return len(a.active) > 0
}

// endKeyIndependent is the span-1 scoring path: every held key counts.
func (a *prefixAccumulator) endKeyIndependent() bool {
	if a.first {
		a.first = false
		for i := range a.slots {
			s := &a.slots[i]
			s.alive = true
			s.matched, s.score = 1, s.weight
			if s.confirmedSeen == a.keyStamp {
				s.confirmed, s.confirmedAlive = 1, true
			}
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
			s.alive = false
			continue // the chain ends at the first key the pod does not hold
		}
		s.matched++
		s.score += s.weight
		switch {
		case !s.confirmedAlive:
		case s.confirmedSeen == a.keyStamp:
			s.confirmed++
		default:
			s.confirmedAlive = false
		}
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

// creditKey applies retrieval-span rules for the current key.
//
// MatchedBlocks / WeightedScore credit a key when the pod holds it through at
// least one span-1 entry, or when a multi-key retrieval unit completes.
// Holding a key only through an incomplete multi-key unit defers that credit.
// A unit completes even when its tier was missing from the first key or broke
// on an earlier key. An aborted unit drops the deferred keys that no other
// in-progress unit still covers.
//
// BlocksByTier still requires a contiguous per-tier chain among tiers present
// on the first key: a tier that missed an earlier key stays broken.
func (a *prefixAccumulator) creditKey(s *matchSlot, first bool) {
	a.creditBlocksByTier(s, first)

	var openBuf [4]uint32
	var doneBuf [4]finishedUnit
	var abortBuf [4]uint32
	open := openBuf[:0]
	done := doneBuf[:0]
	aborted := abortBuf[:0]

	for u := range s.units {
		ut := &s.units[u]
		if ut.seen != a.keyStamp {
			if ut.pending > 0 {
				aborted = append(aborted, ut.ordinal)
				ut.pending = 0
				ut.unitSpan = 0
			}
			continue
		}
		span := ut.seenSpan
		if span < 1 {
			span = 1
		}
		if ut.pending == 0 {
			ut.unitSpan = span
		}
		ut.pending++
		if ut.pending == ut.unitSpan {
			done = append(done, finishedUnit{ordinal: ut.ordinal, blocks: ut.unitSpan})
			ut.pending = 0
			ut.unitSpan = 0
			continue
		}
		open = append(open, ut.ordinal)
	}
	// Forget aborted covers before crediting a unit that finished on this
	// key, so a finished unit can still claim a key the aborted one shared.
	for _, ordinal := range aborted {
		discardUnit(s, ordinal)
	}
	for _, fin := range done {
		noteTierBlocks(s, fin.ordinal, fin.blocks)
		creditUnit(s, fin.ordinal)
	}

	// ConfirmedBlocks must stay a continuous non-speculative prefix. Update
	// confirmedAlive for this key before any credit path so a completed
	// multi-span unit cannot resurrect a chain that already broke.
	keyConfirmed := s.confirmedSeen == a.keyStamp
	if first {
		s.confirmedAlive = keyConfirmed
	} else if !keyConfirmed {
		s.confirmedAlive = false
	}

	switch {
	case s.keyHoldSpan1:
		s.matched++
		s.score += s.weight
		if keyConfirmed && s.confirmedAlive {
			s.confirmed++
		}
		if len(done) > 0 {
			if s.confirmedAlive {
				s.confirmed = s.matched
			}
		} else if !unitPending(s) {
			// No retrieval unit is in progress. Leftover weights belong to
			// units that already aborted; len(uncredited) is not itself a
			// reason to keep them.
			s.uncredited = s.uncredited[:0]
		}
	case len(done) > 0:
		s.matched++
		s.score += s.weight
		if s.confirmedAlive {
			s.confirmed = s.matched
		}
	default:
		if s.keyHoldMulti && len(open) > 0 {
			covers := make([]uint32, len(open))
			copy(covers, open)
			s.uncredited = append(s.uncredited, deferredKey{weight: s.weight, covers: covers})
		}
	}
}

// creditBlocksByTier updates contiguous per-tier counts. Multi-span counts are
// added later, and only for a unit that completes while the chain is still alive.
func (a *prefixAccumulator) creditBlocksByTier(s *matchSlot, first bool) {
	for t := range s.tiers {
		tc := &s.tiers[t]
		heldThisKey := tc.seen == a.keyStamp
		if !heldThisKey {
			if !first && tc.alive {
				tc.alive = false
			}
			continue
		}
		span := tc.seenSpan
		if span < 1 {
			span = 1
		}
		if span != 1 {
			if first {
				tc.alive = true
			}
			continue
		}
		if first {
			tc.count = 1
			tc.alive = true
		} else if tc.alive {
			tc.count++
		}
	}
}

// finishedUnit is one retrieval unit that reached its span on this key.
type finishedUnit struct {
	ordinal uint32
	blocks  int
}

// noteTierBlocks adds a completed unit to BlocksByTier when that tier's
// contiguous prefix is still alive. Tiers that started late, or that already
// broke, are left unchanged.
func noteTierBlocks(s *matchSlot, ordinal uint32, blocks int) {
	if blocks <= 0 {
		return
	}
	for i := range s.tiers {
		if s.tiers[i].ordinal == ordinal && s.tiers[i].alive {
			s.tiers[i].count += blocks
			return
		}
	}
}

// unitPending reports whether any retrieval unit is still accumulating.
// Deferred weights are not a signal: an aborted unit leaves weights behind
// until they are discarded.
func unitPending(s *matchSlot) bool {
	for i := range s.units {
		if s.units[i].pending > 0 {
			return true
		}
	}
	return false
}

// discardUnit drops one aborted unit. A deferred key is removed only when no
// other in-progress unit still covers it.
func discardUnit(s *matchSlot, ordinal uint32) {
	kept := s.uncredited[:0]
	for i := range s.uncredited {
		covers := removeOrdinal(s.uncredited[i].covers, ordinal)
		if len(covers) == 0 {
			continue
		}
		s.uncredited[i].covers = covers
		kept = append(kept, s.uncredited[i])
	}
	s.uncredited = kept
}

// creditUnit scores every deferred key still covered by a unit that just
// finished, then removes those keys so a later unit cannot score them again.
func creditUnit(s *matchSlot, ordinal uint32) {
	kept := s.uncredited[:0]
	for i := range s.uncredited {
		if !containsOrdinal(s.uncredited[i].covers, ordinal) {
			kept = append(kept, s.uncredited[i])
			continue
		}
		s.score += s.uncredited[i].weight
		s.matched++
	}
	s.uncredited = kept
}

func containsOrdinal(ids []uint32, ordinal uint32) bool {
	for _, id := range ids {
		if id == ordinal {
			return true
		}
	}
	return false
}

func removeOrdinal(ids []uint32, ordinal uint32) []uint32 {
	n := 0
	for _, id := range ids {
		if id != ordinal {
			ids[n] = id
			n++
		}
	}
	return ids[:n]
}

// result materializes the accumulated matches.
func (a *prefixAccumulator) result() map[string]PodMatch {
	a.flushRun()
	out := make(map[string]PodMatch, len(a.slots))
	for i := range a.slots {
		s := &a.slots[i]
		byTier := make(map[string]int, len(s.tiers))
		for _, tc := range s.tiers {
			byTier[tc.name] = tc.count
		}
		out[s.pod] = PodMatch{WeightedScore: s.score, MatchedBlocks: s.matched, ConfirmedBlocks: s.confirmed, BlocksByTier: byTier}
	}
	return out
}

// newSlot appends a candidate, backing its tiers slice with inline storage
// unless a pooled slot already holds a larger backing array.
func (a *prefixAccumulator) newSlot(pod string) int32 {
	n := len(a.slots)
	if n < cap(a.slots) {
		a.slots = a.slots[:n+1]
	} else {
		a.slots = append(a.slots, matchSlot{})
	}
	s := &a.slots[n]
	var tiers []tierChain
	if cap(s.tiers) > len(s.tiersBuf) {
		tiers = s.tiers[:0]
	}
	var units []unitTrack
	if cap(s.units) > len(s.unitsBuf) {
		units = s.units[:0]
	}
	uncredited := s.uncredited[:0]
	*s = matchSlot{pod: pod, uncredited: uncredited}
	if tiers != nil {
		s.tiers = tiers
	} else {
		s.tiers = s.tiersBuf[:0]
	}
	if units != nil {
		s.units = units
	} else {
		s.units = s.unitsBuf[:0]
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
