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

// The entry word is 32 bits, one word per pod and group flavor a key holds:
//
//	bits  0-7   tier mask, one bit per interner tier ordinal 0-7
//	bit   8     speculative, the word holds the pod's speculative copies
//	bit   9     grouped, the word's copies are group-announced and their
//	            reference counts live in the side table
//	bits 10-11  unused
//	bits 12-31  pod ordinal
//
// A speculative word carries no tier bits: the in-tree producer emits
// speculative entries without a device tier, so bit 8 is a complete identity
// for them. A word with no tier bits and no speculative bit is an empty slot
// (the zero word); pod ordinal zero is a valid ordinal and never marks one.
const (
	entryTierMask   uint32 = 0xff
	entrySpecBit    uint32 = 1 << 8
	entryGroupedBit uint32 = 1 << 9
	entryPodShift          = 12
)

// maxDenseTierOrdinal is the highest tier ordinal an entry word can carry;
// the dense interner rejects an Add that would assign an ordinal past it.
const maxDenseTierOrdinal = 7

func entryPod(w uint32) uint32 { return w >> entryPodShift }

func entryTiers(w uint32) uint32 { return w & entryTierMask }

func packEntry(pod, tiers uint32, speculative, grouped bool) uint32 {
	w := pod<<entryPodShift | tiers
	if speculative {
		w |= entrySpecBit
	}
	if grouped {
		w |= entryGroupedBit
	}
	return w
}

// entryEmpty reports whether w is an unused slot.
func entryEmpty(w uint32) bool { return w&(entryTierMask|entrySpecBit) == 0 }

// entrySameIdentity reports whether w and v describe the same pod and group
// flavor, so their tier bits merge into one word.
func entrySameIdentity(w, v uint32) bool {
	return w&(entrySpecBit|entryGroupedBit|^entryTierMask) == v&(entrySpecBit|entryGroupedBit|^entryTierMask)
}

// keySlot is one request key's map value: three inline entry words, a handle
// into the shared overflow arena for the rest, and the key's last-visit stamp
// for sampled eviction. The slot is a map value with no pointers: it is
// copied out, mutated, and stored back under the table lock.
type keySlot struct {
	entries [3]uint32
	// chain addresses the first arena chunk; 0 ends the list.
	chain uint32
	stamp uint32
}

const (
	// denseInlineEntries is the inline word count per slot.
	denseInlineEntries = 3
	// denseChunkEntries is the entry count per arena chunk; the chunk's
	// last word is the link to the next chunk.
	denseChunkEntries  = 3
	denseChunkLink     = denseChunkEntries
	denseChunkWords    = denseChunkEntries + 1
	denseMaxPodsPerKey = denseInlineEntries + 4*denseChunkEntries
)

// denseArena is the shared overflow storage for keySlot entry lists. A chunk
// is four words: three entries and a link to the next chunk. Chunks are
// addressed by 1-based handle (handle h is words[(h-1)*denseChunkWords]).
// Freed chunks are recycled through the free list.
type denseArena struct {
	words []uint32
	free  []uint32
}

func (a *denseArena) allocChunk() uint32 {
	if n := len(a.free); n > 0 {
		h := a.free[n-1]
		a.free = a.free[:n-1]
		i := int(h-1) * denseChunkWords
		for j := range denseChunkWords {
			a.words[i+j] = 0
		}
		return h
	}
	i := len(a.words)
	for range denseChunkWords {
		a.words = append(a.words, 0)
	}
	return uint32(i/denseChunkWords + 1) //#nosec G115 -- bounded by addressable memory well below 2^32 chunks
}

func (a *denseArena) freeChunk(h uint32) {
	a.free = append(a.free, h)
}

// slotLen returns the slot's entry count, inline plus arena. The list is
// compact: empty words only trail the last real one.
func slotLen(slot *keySlot, arena *denseArena) int {
	n := 0
	for _, w := range slot.entries {
		if entryEmpty(w) {
			break
		}
		n++
	}
	for h := slot.chain; h != 0; {
		base := int(h-1) * denseChunkWords
		for j := range denseChunkEntries {
			if !entryEmpty(arena.words[base+j]) {
				n++
			}
		}
		h = arena.words[base+denseChunkLink]
	}
	return n
}

// slotWord returns the i-th entry word.
func slotWord(slot *keySlot, arena *denseArena, i int) uint32 {
	if i < denseInlineEntries {
		return slot.entries[i]
	}
	j := i - denseInlineEntries
	h := slot.chain
	for j >= denseChunkEntries {
		h = arena.words[(h-1)*denseChunkWords+denseChunkLink]
		j -= denseChunkEntries
	}
	return arena.words[int(h-1)*denseChunkWords+j]
}

// slotSetWord overwrites the i-th entry word. The word must exist.
func slotSetWord(slot *keySlot, arena *denseArena, i int, w uint32) {
	if i < denseInlineEntries {
		slot.entries[i] = w
		return
	}
	j := i - denseInlineEntries
	h := slot.chain
	for j >= denseChunkEntries {
		h = arena.words[(h-1)*denseChunkWords+denseChunkLink]
		j -= denseChunkEntries
	}
	arena.words[int(h-1)*denseChunkWords+j] = w
}

// forEachSlotWord calls fn for every entry word in order; a false return
// stops the walk.
func forEachSlotWord(slot *keySlot, arena *denseArena, fn func(w uint32) bool) {
	for _, w := range slot.entries {
		if entryEmpty(w) {
			return
		}
		if !fn(w) {
			return
		}
	}
	for h := slot.chain; h != 0; {
		base := int(h-1) * denseChunkWords
		for j := range denseChunkEntries {
			w := arena.words[base+j]
			if entryEmpty(w) {
				continue
			}
			if !fn(w) {
				return
			}
		}
		h = arena.words[base+denseChunkLink]
	}
}

// slotAppend appends w at the tail of the entry list. The list must not be
// full.
func slotAppend(slot *keySlot, arena *denseArena, w uint32) {
	for i := range slot.entries {
		if entryEmpty(slot.entries[i]) {
			slot.entries[i] = w
			return
		}
	}
	// Walk to the last chunk; allocate one when the chain is full.
	h := slot.chain
	if h == 0 {
		h = arena.allocChunk()
		slot.chain = h
		arena.words[int(h-1)*denseChunkWords] = w
		return
	}
	for {
		next := arena.words[(h-1)*denseChunkWords+denseChunkLink]
		if next == 0 {
			break
		}
		h = next
	}
	base := int(h-1) * denseChunkWords
	for j := range denseChunkEntries {
		if entryEmpty(arena.words[base+j]) {
			arena.words[base+j] = w
			return
		}
	}
	next := arena.allocChunk()
	arena.words[base+denseChunkLink] = next
	arena.words[int(next-1)*denseChunkWords] = w
}

// slotRemoveAt removes the i-th entry word, shifting the tail down. The
// trailing chunk is freed when it empties.
func slotRemoveAt(slot *keySlot, arena *denseArena, i, n int) {
	for k := i; k < n-1; k++ {
		slotSetWord(slot, arena, k, slotWord(slot, arena, k+1))
	}
	slotSetWord(slot, arena, n-1, 0)
	// A partially filled chunk is only ever the last one; free it once it
	// empties, unlinking it from its predecessor.
	if n-1 >= denseInlineEntries && (n-1-denseInlineEntries)%denseChunkEntries == 0 {
		prev := uint32(0)
		h := slot.chain
		for arena.words[(h-1)*denseChunkWords+denseChunkLink] != 0 {
			prev, h = h, arena.words[(h-1)*denseChunkWords+denseChunkLink]
		}
		if prev == 0 {
			slot.chain = 0
		} else {
			arena.words[(prev-1)*denseChunkWords+denseChunkLink] = 0
		}
		arena.freeChunk(h)
	}
}

// slotSetEntry folds w into the entry list: an existing word with the same
// identity merges the tier bits and rotates to the tail (the recency rule,
// reproducing the per-key add order of the non-dense backends); a new word
// appends at the tail. When the list holds max words, the head word, the
// least-recently-announced pod, is dropped. It returns the dropped word, or
// zero when none was, so the caller can purge any state the word owns.
func slotSetEntry(slot *keySlot, arena *denseArena, max int, w uint32) uint32 {
	n := slotLen(slot, arena)
	for i := 0; i < n; i++ {
		v := slotWord(slot, arena, i)
		if entrySameIdentity(v, w) {
			slotSetWord(slot, arena, i, v|w)
			if i != n-1 {
				slotRemoveAt(slot, arena, i, n)
				slotAppend(slot, arena, v|w)
			}
			return 0
		}
	}
	var dropped uint32
	if n == max {
		dropped = slotWord(slot, arena, 0)
		slotRemoveAt(slot, arena, 0, n)
	}
	slotAppend(slot, arena, w)
	return dropped
}

// slotClearEntry clears the payload bits w names on the matching word and
// removes the word when its payload empties. It reports whether the slot
// changed, so callers store the mutated slot back even on a partial clear.
func slotClearEntry(slot *keySlot, arena *denseArena, w uint32) bool {
	n := slotLen(slot, arena)
	for i := 0; i < n; i++ {
		v := slotWord(slot, arena, i)
		if entrySameIdentity(v, w) {
			// Clear only the payload bits w names, never the identity bits.
			cleared := v &^ (w & (entryTierMask | entrySpecBit))
			if entryEmpty(cleared) {
				slotRemoveAt(slot, arena, i, n)
			} else {
				slotSetWord(slot, arena, i, cleared)
			}
			return true
		}
	}
	return false
}
