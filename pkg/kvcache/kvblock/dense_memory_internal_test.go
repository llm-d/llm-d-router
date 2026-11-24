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
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

func TestEntryWordRoundTrip(t *testing.T) {
	for _, pod := range []uint32{0, 1, 7, 0xfff, 0xfffff} {
		for _, tiers := range []uint32{1, 1 << 7, 0xff} {
			for _, spec := range []bool{false, true} {
				for _, grouped := range []bool{false, true} {
					w := packEntry(pod, tiers, spec, grouped)
					assert.Equal(t, pod, entryPod(w), "pod %d", pod)
					assert.Equal(t, tiers, entryTiers(w), "tiers %b", tiers)
					assert.Equal(t, spec, w&entrySpecBit != 0)
					assert.Equal(t, grouped, w&entryGroupedBit != 0)
					assert.False(t, entryEmpty(w), "a word with tiers or spec is never empty")
				}
			}
		}
	}
	assert.True(t, entryEmpty(0))
	assert.True(t, entryEmpty(packEntry(5, 0, false, true)),
		"a grouped word with no tiers and no spec bit is an empty slot")

	// Identity compares pod, spec, and grouped bits, never tier bits.
	assert.True(t, entrySameIdentity(packEntry(5, 1, false, false), packEntry(5, 2, false, false)))
	assert.False(t, entrySameIdentity(packEntry(5, 1, false, false), packEntry(6, 1, false, false)))
	assert.False(t, entrySameIdentity(packEntry(5, 0, true, false), packEntry(5, 1, false, false)))
	assert.False(t, entrySameIdentity(packEntry(5, 1, false, false), packEntry(5, 1, false, true)))
}

func TestArenaChunkRecycle(t *testing.T) {
	var a denseArena
	h1 := a.allocChunk()
	h2 := a.allocChunk()
	assert.NotEqual(t, h1, h2)
	assert.Len(t, a.words, 2*denseChunkWords)

	a.freeChunk(h1)
	h3 := a.allocChunk()
	assert.Equal(t, h1, h3, "a freed chunk is recycled from the free list")
	assert.Len(t, a.words, 2*denseChunkWords, "recycling does not grow the arena")
	for _, w := range a.words {
		assert.Equal(t, uint32(0), w, "a recycled chunk is zeroed")
	}
}

// TestSlotListOps exercises the slot list across the inline-arena boundary:
// append, in-order read, remove with tail shift, and chunk freeing.
func TestSlotListOps(t *testing.T) {
	var arena denseArena
	var slot keySlot

	words := make([]uint32, denseMaxPodsPerKey)
	for i := range words {
		words[i] = packEntry(uint32(i+1), 1, false, false)
	}
	for i, w := range words {
		slotAppend(&slot, &arena, w)
		require.Equal(t, i+1, slotLen(&slot, &arena), "length after %d appends", i+1)
	}
	// Four chunks of three plus three inline.
	require.Equal(t, 4, len(arena.words)/denseChunkWords)
	for i, w := range words {
		assert.Equal(t, w, slotWord(&slot, &arena, i), "word %d reads back in order", i)
	}

	// Removing a middle word shifts the tail down.
	n := slotLen(&slot, &arena)
	slotRemoveAt(&slot, &arena, 1, n)
	assert.Equal(t, words[0], slotWord(&slot, &arena, 0))
	assert.Equal(t, words[2], slotWord(&slot, &arena, 1))
	assert.Equal(t, words[denseMaxPodsPerKey-1], slotWord(&slot, &arena, denseMaxPodsPerKey-2))
	assert.Equal(t, denseMaxPodsPerKey-1, slotLen(&slot, &arena))

	// Removing the tail word frees the emptied trailing chunk.
	chunks := len(arena.words) / denseChunkWords
	for n := slotLen(&slot, &arena); n > denseInlineEntries; n-- {
		slotRemoveAt(&slot, &arena, n-1, n)
	}
	assert.Len(t, arena.free, chunks, "each emptied chunk returns to the free list")
	assert.Equal(t, denseInlineEntries, slotLen(&slot, &arena))
	assert.Equal(t, uint32(0), slot.chain, "the chain empties with the last chunk")

	// forEachSlotWord visits every word in order and honors early stop.
	var seen []uint32
	forEachSlotWord(&slot, &arena, func(w uint32) bool {
		seen = append(seen, w)
		return true
	})
	// The index-1 removal shifted pods 3 and 4 into the inline words.
	assert.Equal(t, []uint32{words[0], words[2], words[3]}, seen)
	seen = nil
	forEachSlotWord(&slot, &arena, func(w uint32) bool {
		seen = append(seen, w)
		return false
	})
	assert.Len(t, seen, 1, "a false return stops the walk")
}

// TestSlotSetEntryMergeAndRotate pins the recency rule: a merge rotates the
// matching word to the tail, and a full list drops the head.
func TestSlotSetEntryMergeAndRotate(t *testing.T) {
	var arena denseArena
	var slot keySlot

	for i := 1; i <= 4; i++ {
		slotSetEntry(&slot, &arena, denseMaxPodsPerKey, packEntry(uint32(i), 1, false, false))
	}
	// Merging pod 1's second tier rotates it behind pod 4.
	assert.Equal(t, uint32(0), slotSetEntry(&slot, &arena, denseMaxPodsPerKey, packEntry(1, 2, false, false)))
	n := slotLen(&slot, &arena)
	assert.Equal(t, 4, n)
	assert.Equal(t, uint32(2), entryPod(slotWord(&slot, &arena, 0)), "the merge rotates pod 1 behind pod 4")
	assert.Equal(t, uint32(3), entryPod(slotWord(&slot, &arena, n-1-2)))
	assert.Equal(t, uint32(4), entryPod(slotWord(&slot, &arena, n-2)))
	assert.Equal(t, uint32(1), entryPod(slotWord(&slot, &arena, n-1)))
	assert.Equal(t, uint32(3), entryTiers(slotWord(&slot, &arena, n-1)), "the merged word carries both tiers")

	// Filling past the cap drops the head, the least-recently-announced pod.
	var dropped uint32
	for i := 5; i <= denseMaxPodsPerKey+1; i++ {
		if w := slotSetEntry(&slot, &arena, denseMaxPodsPerKey, packEntry(uint32(i), 1, false, false)); w != 0 {
			dropped = w
		}
	}
	assert.Equal(t, uint32(2), entryPod(dropped), "the dropped word is the head")
	assert.Equal(t, denseMaxPodsPerKey, slotLen(&slot, &arena))
	// The merge left pod 2 at the head; the truncation drops it.
	assert.Equal(t, uint32(3), entryPod(slotWord(&slot, &arena, 0)), "the head word was dropped")

	// A configured cap below the maximum truncates at the configured value.
	var smallArena denseArena
	var smallSlot keySlot
	for i := 1; i <= 4; i++ {
		slotSetEntry(&smallSlot, &smallArena, 3, packEntry(uint32(i), 1, false, false))
	}
	assert.Equal(t, 3, slotLen(&smallSlot, &smallArena))
	assert.Equal(t, uint32(2), entryPod(slotWord(&smallSlot, &smallArena, 0)), "the cap drops the head")
}

// TestSlotClearEntryBits pins partial clears: clearing named bits keeps the
// word, clearing the last payload removes it.
func TestSlotClearEntryBits(t *testing.T) {
	var arena denseArena
	var slot keySlot

	w := packEntry(7, 1|2, false, false)
	slotSetEntry(&slot, &arena, denseMaxPodsPerKey, w)
	slotSetEntry(&slot, &arena, denseMaxPodsPerKey, packEntry(8, 1, false, false))

	// Clear one tier bit: the word survives with the other.
	assert.True(t, slotClearEntry(&slot, &arena, packEntry(7, 1, false, false)),
		"a partial clear still changes the slot")
	assert.Equal(t, uint32(2), entryTiers(slotWord(&slot, &arena, 0)))

	// Clear the last tier bit: the word is removed.
	assert.True(t, slotClearEntry(&slot, &arena, packEntry(7, 2, false, false)))
	assert.Equal(t, 1, slotLen(&slot, &arena))
	assert.Equal(t, uint32(8), entryPod(slotWord(&slot, &arena, 0)))

	// Clearing an absent identity is a no-op.
	assert.False(t, slotClearEntry(&slot, &arena, packEntry(9, 1, false, false)))
	assert.Equal(t, 1, slotLen(&slot, &arena))
}

// TestDenseIndexRejectsTierCardinalityPastTheCap pins the whole-batch
// rejection: an Add that would assign a 9th distinct device tier fails,
// stores nothing, and consumes no tier ordinal.
func TestDenseIndexRejectsTierCardinalityPastTheCap(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)

	var entries []PodEntry
	for tier := 0; tier <= maxDenseTierOrdinal+1; tier++ {
		entries = append(entries, PodEntry{PodIdentifier: "pod-a", DeviceTier: fmt.Sprintf("tier-%d", tier)})
	}
	err = index.Add(ctx, nil, []BlockHash{1}, entries)
	require.ErrorIs(t, err, errIndexCardinality)

	hits, err := index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, hits[1], "a rejected batch must store nothing")

	// The first eight tiers fit afterwards; the ninth still fails.
	var eight []PodEntry
	for tier := 0; tier <= maxDenseTierOrdinal; tier++ {
		eight = append(eight, PodEntry{PodIdentifier: "pod-a", DeviceTier: fmt.Sprintf("tier-%d", tier)})
	}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, eight))
	err = index.Add(ctx, nil, []BlockHash{2}, []PodEntry{{PodIdentifier: "pod-b", DeviceTier: "tier-8"}})
	require.ErrorIs(t, err, errIndexCardinality)
}

// TestDenseIndexPodCapTruncation pins the default per-key pod cap: the 16th
// and later pods of a key drop the least-recently-announced pod, and the
// truncation counter reports each drop.
func TestDenseIndexPodCapTruncation(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)

	const totalPods = 20
	var entries []PodEntry
	for p := 0; p < totalPods; p++ {
		entries = append(entries, PodEntry{PodIdentifier: fmt.Sprintf("pod-%02d", p), DeviceTier: "gpu"})
	}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, entries))

	hits, err := index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	require.Len(t, hits[1], denseMaxPodsPerKey)
	for p := 0; p < totalPods-denseMaxPodsPerKey; p++ {
		assert.NotContains(t, hits[1], PodEntry{PodIdentifier: fmt.Sprintf("pod-%02d", p), DeviceTier: "gpu"},
			"the least-recently-announced pods are the ones dropped")
	}
	assert.EqualValues(t, totalPods-denseMaxPodsPerKey, index.Stats().TruncatedEntries)
}

// TestDenseIndexConfiguredPodCapTruncation pins a PodCacheSize below the
// maximum.
func TestDenseIndexConfiguredPodCapTruncation(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB", PodCacheSize: 5})
	require.NoError(t, err)

	var entries []PodEntry
	for p := 0; p < 8; p++ {
		entries = append(entries, PodEntry{PodIdentifier: fmt.Sprintf("pod-%02d", p), DeviceTier: "gpu"})
	}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, entries))

	hits, err := index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	require.Len(t, hits[1], 5)
	assert.NotContains(t, hits[1], PodEntry{PodIdentifier: "pod-00", DeviceTier: "gpu"})
	assert.Contains(t, hits[1], PodEntry{PodIdentifier: "pod-07", DeviceTier: "gpu"})
	assert.EqualValues(t, 3, index.Stats().TruncatedEntries)
}

// TestDenseIndexSweepFreesDeadSlots pins the sweep's dead-slot paths: after a
// pod is cleared, a sweep under budget pressure removes its words, deleting
// keys that hold nothing else. The table stays within one sweep sample so
// every key is visited deterministically.
func TestDenseIndexSweepFreesDeadSlots(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	// 120 keys of two inline words each sit above the 4KiB budget.
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "4KiB"})
	require.NoError(t, err)

	var entries []PodEntry
	for _, pod := range []string{"pod-a", "pod-b"} {
		entries = append(entries, PodEntry{PodIdentifier: pod, DeviceTier: "gpu"})
	}
	for k := 1; k <= 120; k++ {
		require.NoError(t, index.Add(ctx, nil, []BlockHash{BlockHash(k)}, entries))
	}
	require.NoError(t, index.Clear(ctx, "pod-a"))

	// Key 120 is the newest, so it survives the budget deletions that ran
	// during the adds.
	hits, err := index.Lookup(ctx, []BlockHash{120}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, hits[120], 1, "the cleared pod's entries are filtered immediately")

	// Push residency over the budget and trigger the sweep.
	for k := 1000; k <= 1015; k++ {
		require.NoError(t, index.Add(ctx, nil, []BlockHash{BlockHash(k)}, []PodEntry{{PodIdentifier: "pod-c", DeviceTier: "gpu"}}))
	}

	index.mu.RLock()
	var deadWords []uint32
	survivors := 0
	for i := range index.tbl.slots {
		if index.tbl.empty(i) {
			continue
		}
		key := index.tbl.slots[i].key
		slot := index.tbl.slots[i].slot
		forEachSlotWord(&slot, &index.arena, func(w uint32) bool {
			if index.isDead(entryPod(w)) {
				deadWords = append(deadWords, uint32(key))
			}
			return true
		})
		if slotLen(&slot, &index.arena) > 0 {
			survivors++
		}
	}
	index.mu.RUnlock()
	assert.Empty(t, deadWords, "no live word carries a dead pod after the sweep")
	assert.Positive(t, survivors, "live entries survive the sweep")
}

// TestDenseIndexGetRequestKeyConcurrent races parallel parent resolutions.
// GetRequestKey refreshes the engine map's recency stamp, which mutates the
// map, so concurrent lookups must not share a read hold of the index lock.
func TestDenseIndexGetRequestKeyConcurrent(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)

	const mappings = 64
	engineKeys := make([]BlockHash, mappings)
	requestKeys := make([]BlockHash, mappings)
	pods := make([]PodEntry, mappings)
	for i := range mappings {
		engineKeys[i] = BlockHash(1000 + i)
		requestKeys[i] = BlockHash(2000 + i)
		pods[i] = PodEntry{PodIdentifier: fmt.Sprintf("pod-%02d", i), DeviceTier: "gpu"}
	}
	require.NoError(t, index.Add(ctx, engineKeys, requestKeys, pods))

	const workers = 8
	var wg sync.WaitGroup
	start := make(chan struct{})
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func(w int) {
			defer wg.Done()
			<-start
			for i := 0; i < 500; i++ {
				if _, err := index.GetRequestKey(ctx, engineKeys[(w+i)%mappings]); err != nil {
					t.Error(err)
					return
				}
			}
		}(w)
	}
	close(start)
	wg.Wait()
}

// TestDenseIndexEngineEvictKeepsMappingAddedConcurrently races one pod's
// engine-key eviction of a block against another pod's store of the same
// block, across the lock holds Add's entry loop releases. The surviving
// entry must stay reachable through its engine key. The racing store carries
// a large entry set so its first lock hold (interner plus mapping writes)
// runs long enough for the evict to queue behind it and land between the
// holds.
func TestDenseIndexEngineEvictKeepsMappingAddedConcurrently(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	const (
		engineKey  BlockHash = 9001
		requestKey BlockHash = 1
		trials               = 2000
		fillerPods           = 500
	)
	podA := PodEntry{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}
	storePods := []PodEntry{{PodIdentifier: "10.0.0.2:8000", DeviceTier: "gpu"}}
	for p := range fillerPods {
		storePods = append(storePods, PodEntry{PodIdentifier: fmt.Sprintf("10.1.%d.%d:8000", p>>8, p&0xff), DeviceTier: "cpu"})
	}

	for i := range trials {
		index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB", PodCacheSize: denseMaxPodsPerKey})
		require.NoError(t, err)
		require.NoError(t, index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{podA}))

		var wg sync.WaitGroup
		start := make(chan struct{})
		wg.Add(2)
		go func() {
			defer wg.Done()
			<-start
			_ = index.Evict(ctx, EngineKey, []BlockHash{engineKey}, []PodEntry{podA})
		}()
		go func() {
			defer wg.Done()
			<-start
			_ = index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, storePods)
		}()
		close(start)
		wg.Wait()

		got, err := index.GetRequestKey(ctx, engineKey)
		require.NoError(t, err, "trial %d: engine key mapping removed while pod B's entry remains", i)
		require.Equal(t, requestKey, got, "trial %d", i)
	}
}

// TestDenseIndexClearReAddAtWordCap pins the word cap against dead words: a
// cleared pod's word occupies the cap until the sweep reclaims it, so an
// insert at the cap must strip dead words first rather than drop the head,
// which can be a live pod's word.
func TestDenseIndexClearReAddAtWordCap(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB", PodCacheSize: 5})
	require.NoError(t, err)

	pod := func(p int) PodEntry {
		return PodEntry{PodIdentifier: fmt.Sprintf("pod-%02d", p), DeviceTier: "gpu"}
	}
	var entries []PodEntry
	for p := 0; p < 5; p++ {
		entries = append(entries, pod(p))
	}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, entries))
	require.NoError(t, index.Clear(ctx, "pod-02"))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, []PodEntry{pod(5)}))

	hits, err := index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, hits[1], 5, "the cap holds five live pods")
	assert.NotContains(t, hits[1], pod(2), "the cleared pod is filtered")
	for _, p := range []int{0, 1, 3, 4, 5} {
		assert.Contains(t, hits[1], pod(p), "live pod %d evicted while a dead word survived", p)
	}
}

// TestDenseIndexSweepReadEvidence pins the sweep's read evidence across a
// burst that does not sample the read key: a read is honored by whichever
// burst next samples the slot, whatever bit it would have set. The first
// burst below stops before the read key's slot (its sample fills at 128
// victims, the key sits at the 160th occupied slot), the second starts at
// it. Between the two, the key is read once.
func TestDenseIndexSweepReadEvidence(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	// A budget far above residency keeps Add's sweeps off; the test drives
	// bursts directly.
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "64MiB"})
	require.NoError(t, err)

	const numKeys = 200
	pod := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	for k := 1; k <= numKeys; k++ {
		require.NoError(t, index.Add(ctx, nil, []BlockHash{BlockHash(k)}, []PodEntry{pod}))
	}

	// Stamps a second below the sweep's refresh stamp, so an honored read is
	// the only way the key's stamp can reach the current wall second.
	base := uint32(time.Now().Unix())
	index.mu.Lock()
	var slots []int
	for i := range index.tbl.slots {
		if !index.tbl.empty(i) {
			slots = append(slots, i)
			index.tbl.slots[i].slot.stamp = base - 1000
		}
	}
	require.Greater(t, len(slots), 170, "the table must hold more keys than one burst samples")
	readKey := index.tbl.slots[slots[159]].key
	index.mu.Unlock()

	hits, err := index.Lookup(ctx, []BlockHash{readKey}, sets.Set[string]{})
	require.NoError(t, err)
	require.Contains(t, hits, readKey)

	index.mu.Lock()
	index.sweepCursor = 0
	index.sweepBurstLocked()
	i, ok := index.tbl.find(readKey)
	require.True(t, ok, "the first burst must not sample the read key")
	index.sweepCursor = i
	index.sweepBurstLocked()
	i, ok = index.tbl.find(readKey)
	require.True(t, ok, "the read key was evicted as cold despite the read between the bursts")
	require.GreaterOrEqual(t, index.tbl.slots[i].slot.stamp, base,
		"the read between the bursts did not refresh the key's stamp")
	index.mu.Unlock()
}
