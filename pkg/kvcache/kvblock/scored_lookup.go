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
	"errors"
	"fmt"
	"math/bits"
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
)

// SpeculativeTier is the tier name under which speculative entries count in
// PodMatchStats.BlocksByTier. Speculative entries carry no engine-reported
// device tier, so they are reported under this sentinel instead.
const SpeculativeTier = "speculative"

// cancellationCheckMask paces context-cancellation checks in loops over
// request keys: positions where idx&mask == 0 poll ctx.Err().
const cancellationCheckMask = 255

// ErrScoredLookupUnsupported is returned by decorating indexes whose wrapped
// backend does not implement ScoredLookupIndex. Callers fall back to
// Lookup plus external scoring.
var ErrScoredLookupUnsupported = errors.New("scored lookup not supported by index backend")

// PodMatchStats aggregates one pod's prefix-match results from a scored
// lookup. All values cover the contiguous chain of request keys the pod
// holds, counted from the first key.
type PodMatchStats struct {
	// WeightedScore sums, per block of the chain, the highest device-tier
	// weight among the pod's entries for that block; tiers without a
	// configured weight count 1.0.
	WeightedScore float64
	// MatchedBlocks is the chain length in blocks, regardless of tier.
	MatchedBlocks int
	// BlocksByTier is the per-tier chain length: a tier counts a block only
	// while the pod holds every previous block in that same tier.
	// Speculative entries count under SpeculativeTier. Never nil.
	BlocksByTier map[string]int
}

// ScoredLookupIndex is an optional Index capability that fuses Lookup and
// longest-consecutive-prefix scoring into one pass, without materializing the
// per-key pod entry map. tierWeights maps device tier names to scoring
// weights; unlisted tiers weigh 1.0. The result holds one entry per pod that
// holds the first request key (after podIdentifierSet filtering, when
// non-empty).
type ScoredLookupIndex interface {
	ScoredLookup(ctx context.Context, requestKeys []BlockHash,
		podIdentifierSet sets.Set[string], tierWeights map[string]float64) (map[string]PodMatchStats, error)
}

// interner assigns dense uint32 indices to strings. Indices are stable for
// the lifetime of the interner and never reused.
type interner struct {
	mu    sync.RWMutex
	ids   map[string]uint32
	names []string
}

func newInterner() *interner {
	return &interner{ids: make(map[string]uint32)}
}

// intern returns the index for s, assigning the next free index on first use.
func (in *interner) intern(s string) uint32 {
	in.mu.RLock()
	id, ok := in.ids[s]
	in.mu.RUnlock()
	if ok {
		return id
	}
	in.mu.Lock()
	defer in.mu.Unlock()
	if id, ok := in.ids[s]; ok {
		return id
	}
	id = uint32(len(in.names))
	in.ids[s] = id
	in.names = append(in.names, s)
	return id
}

// lookup returns the index for s without assigning one.
func (in *interner) lookup(s string) (uint32, bool) {
	in.mu.RLock()
	defer in.mu.RUnlock()
	id, ok := in.ids[s]
	return id, ok
}

// snapshot returns the interned names indexed by id. The backing array of a
// returned snapshot is never mutated, only appended to under the lock.
func (in *interner) snapshot() []string {
	in.mu.RLock()
	defer in.mu.RUnlock()
	return in.names
}

// scoredScratch holds the interned-pod-indexed tables a ScoredLookup call
// needs, reused through a pool so steady-state per-request cost does not grow
// with the number of pods ever interned. A cold scratch (pool miss, e.g.
// after a GC cycle) still allocates ~12 bytes per interned pod. Validity is
// generation-stamped: an entry is meaningful only when its stamp equals the
// current call's gen, so reuse needs no zeroing.
type scoredScratch struct {
	gen uint32
	// slotRef maps an interned pod id to its request-local slot: the
	// validity generation in the high 32 bits, the slot in the low 32.
	slotRef []uint64
	// allowGen stamps pods that pass the call's pod-identifier filter.
	allowGen []uint32
}

var scoredScratchPool = sync.Pool{New: func() any { return &scoredScratch{} }}

// nextGen advances the scratch generation, resizing the tables to nPods and
// clearing them on generation wrap-around so stale stamps can never match.
func (sc *scoredScratch) nextGen(nPods int) uint32 {
	if len(sc.slotRef) < nPods {
		sc.slotRef = make([]uint64, nPods)
		sc.allowGen = make([]uint32, nPods)
		sc.gen = 0
	}
	if sc.gen == ^uint32(0) {
		clear(sc.slotRef)
		clear(sc.allowGen)
		sc.gen = 0
	}
	sc.gen++
	return sc.gen
}

// slotCursor is one candidate's per-key working state.
type slotCursor struct {
	// seen stamps presence at the current key.
	seen uint32
	// weight is the highest tier weight at the current key.
	weight float64
	// tiers is the tier bitmask at the current key.
	tiers uint64
}

// slotChain is one candidate's accumulated chain state.
type slotChain struct {
	podIdx uint32
	// matched is the contiguous matched-block count.
	matched int
	// score is the accumulated weighted score.
	score float64
	// aliveTiers is the bitmask of tiers still contiguously held.
	aliveTiers uint64
}

// maxTierMaskBits bounds the per-tier chain bitmasks. Lookups against an
// index that has interned more tiers report ErrScoredLookupUnsupported so
// callers run the legacy path instead of silently dropping tiers.
const maxTierMaskBits = 64

// ScoredLookup walks requestKeys in order and accumulates per-pod prefix
// scores directly from the interned pod records. Candidate pods (those
// holding the first key) get request-local slots, so working state is sized
// by candidates - bounded by the per-key pod cache - not by every pod ever
// interned. The walk stops at the first key with no qualifying pod, matching
// the composition of Lookup and the longest-prefix scorer.
func (m *InMemoryIndex) ScoredLookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string], tierWeights map[string]float64,
) (map[string]PodMatchStats, error) {
	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no requestKeys provided for lookup")
	}

	podNames := m.pods.snapshot()
	nPods := len(podNames)
	if nPods == 0 {
		return map[string]PodMatchStats{}, nil
	}

	sc, _ := scoredScratchPool.Get().(*scoredScratch)
	defer scoredScratchPool.Put(sc)

	// The tier view is snapshotted per attempt. A record referencing a tier
	// interned after the snapshot would score with a default weight and drop
	// out of the per-tier counts, so a walk that observes interner growth at
	// its end retries under the fresh view instead of paying a staleness
	// check in the hot loop. Tier count only grows and the sentinel applies
	// past maxTierMaskBits, so retries are bounded.
retry:
	tierNames := m.tiers.snapshot()
	nTiers := len(tierNames)
	if nTiers > maxTierMaskBits {
		return nil, ErrScoredLookupUnsupported
	}

	weightByTier := make([]float64, nTiers)
	for i := range weightByTier {
		weightByTier[i] = 1.0
	}
	for name, w := range tierWeights {
		if idx, ok := m.tiers.lookup(name); ok && int(idx) < nTiers {
			weightByTier[idx] = w
		}
	}

	gen := sc.nextGen(nPods)

	filtered := podIdentifierSet.Len() > 0
	if filtered {
		for id := range podIdentifierSet {
			if idx, ok := m.pods.lookup(id); ok && int(idx) < nPods {
				sc.allowGen[idx] = gen
			}
		}
	}

	// Per-slot working state for the candidate pods, in slot order.
	var (
		cur        []slotCursor // per-key state
		chains     []slotChain  // accumulated chain state
		tierCounts []int        // flat [slot*nTiers + tier] chain counters
		active     []int32      // slots still in the any-tier chain
		keyStamp   uint32
	)
	genRef := uint64(gen) << 32

	for idx, key := range requestKeys {
		if idx&cancellationCheckMask == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		pc, found := m.data.Get(key)
		if !found || pc == nil {
			break
		}
		keyStamp++
		firstKey := idx == 0
		pc.mu.Lock()
		for i := range pc.entries {
			rec := &pc.entries[i]
			if int(rec.podIdx) >= nPods {
				continue // interned after this lookup's snapshot
			}
			if filtered && sc.allowGen[rec.podIdx] != gen {
				continue
			}
			ref := sc.slotRef[rec.podIdx]
			var s int32
			switch {
			case ref>>32 == uint64(gen):
				s = int32(uint32(ref))
			case firstKey:
				// The first key defines the candidate set: assign slots.
				s = int32(len(chains))
				sc.slotRef[rec.podIdx] = genRef | uint64(uint32(s))
				chains = append(chains, slotChain{podIdx: rec.podIdx})
				cur = append(cur, slotCursor{})
				tierCounts = append(tierCounts, make([]int, nTiers)...)
			default:
				continue // not a candidate: absent from the first key
			}
			w := 1.0
			if int(rec.weightTierIdx) < nTiers {
				w = weightByTier[rec.weightTierIdx]
			}
			var tierBit uint64
			if rec.statTierIdx < maxTierMaskBits {
				tierBit = 1 << rec.statTierIdx
			}
			c := &cur[s]
			if c.seen != keyStamp {
				c.seen = keyStamp
				c.weight = w
				c.tiers = tierBit
			} else {
				if w > c.weight {
					c.weight = w
				}
				c.tiers |= tierBit
			}
		}
		pc.mu.Unlock()

		if firstKey {
			if len(chains) == 0 {
				break
			}
			active = make([]int32, len(chains))
			for s := range chains {
				active[s] = int32(s)
				chains[s].score = cur[s].weight
				chains[s].matched = 1
				chains[s].aliveTiers = cur[s].tiers
				countAliveTiers(tierCounts, int32(s), chains[s].aliveTiers, nTiers)
			}
			continue
		}

		keep := active[:0]
		for _, s := range active {
			if cur[s].seen != keyStamp {
				continue // pod missing at this key: chain breaks
			}
			chains[s].score += cur[s].weight
			chains[s].matched++
			chains[s].aliveTiers &= cur[s].tiers
			countAliveTiers(tierCounts, s, chains[s].aliveTiers, nTiers)
			keep = append(keep, s)
		}
		active = keep
		if len(active) == 0 {
			break
		}
	}

	if len(m.tiers.snapshot()) != nTiers {
		// A tier was interned mid-walk; its records were scored under the
		// stale view. Rewalk under the fresh one.
		goto retry
	}

	{
		result := make(map[string]PodMatchStats, len(chains))
		for s := range chains {
			byTier := make(map[string]int)
			for t := 0; t < nTiers; t++ {
				if c := tierCounts[s*nTiers+t]; c > 0 {
					byTier[tierNames[t]] = c
				}
			}
			result[podNames[chains[s].podIdx]] = PodMatchStats{
				WeightedScore: chains[s].score,
				MatchedBlocks: chains[s].matched,
				BlocksByTier:  byTier,
			}
		}
		return result, nil
	}
}

// countAliveTiers increments the per-tier chain counters for every tier still
// alive in mask.
func countAliveTiers(tierCounts []int, slot int32, mask uint64, nTiers int) {
	for mask != 0 {
		t := bits.TrailingZeros64(mask)
		mask &^= 1 << t
		if t < nTiers {
			tierCounts[int(slot)*nTiers+t]++
		}
	}
}
