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

package approximateprefix

import (
	"math/rand/v2"
	"slices"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestRecentWindowOwnsHashesAndSnapshots(t *testing.T) {
	w := newRecentWindow(4)
	hashes := []blockHash{1, 2}
	w.record("ep", hashes)
	hashes[0] = 99
	require.Equal(t, [][]blockHash{{1, 2}}, w.snapshot("ep"))

	snapshot := w.snapshot("ep")
	snapshot[0][0] = 88
	snapshot[0] = []blockHash{77}
	require.Equal(t, [][]blockHash{{1, 2}}, w.snapshot("ep"))

	snapshot = w.snapshot("ep")
	for i := range 20 {
		w.record("ep", []blockHash{blockHash(i + 3)})
	}
	w.removePrompt("ep", []blockHash{22})
	w.forget("ep")
	require.Equal(t, [][]blockHash{{1, 2}}, snapshot)
}

func TestRecentWindowRemoveDuplicatesAndContinue(t *testing.T) {
	w := newRecentWindow(6)
	w.record("ep", []blockHash{9, 9})
	w.record("ep", []blockHash{1, 2})
	w.record("ep", []blockHash{3})
	w.record("ep", []blockHash{1, 2})
	require.Equal(t, [][]blockHash{{1, 2}, {3}, {1, 2}}, w.snapshot("ep"))

	w.removePrompt("ep", []blockHash{1, 2})
	w.record("ep", []blockHash{4, 5, 6, 7, 8})
	require.Equal(t, [][]blockHash{{3}, {4, 5, 6, 7, 8}}, w.snapshot("ep"))
	w.record("ep", []blockHash{9})
	require.Equal(t, [][]blockHash{{4, 5, 6, 7, 8}, {9}}, w.snapshot("ep"))
	w.removePrompt("ep", []blockHash{4, 5, 6, 7, 8})
	w.removePrompt("ep", []blockHash{9})
	require.Nil(t, w.snapshot("ep"))
	w.record("ep", []blockHash{10, 11, 12, 13, 14, 15, 16})
	require.Equal(t, [][]blockHash{{10, 11, 12, 13, 14, 15}}, w.snapshot("ep"))
}

func TestRecentWindowReplaceAppendsNewestSlot(t *testing.T) {
	w := newRecentWindow(10)
	first := w.record("ep", []blockHash{1, 2})
	w.record("ep", []blockHash{3, 4})
	w.replace("ep", first, []blockHash{1, 2, 9})

	require.Equal(t, [][]blockHash{{3, 4}, {1, 2, 9}}, w.snapshot("ep"))
	set := w.sequenced("ep")
	require.Equal(t, uint64(1), set.SequenceBase(), "withdrawn oldest slot is trimmed")
	require.Equal(t, uint64(4), set.SequenceTip())
	require.Equal(t, 9, set.ElementSize(2), "one retraction of generation 0")
	require.Equal(t, 24, set.ElementSize(3), "one content slot for the replacement")
}

func TestRecentWindowReplaceIgnoresStaleEntries(t *testing.T) {
	w := newRecentWindow(10)
	w.record("ep", []blockHash{1, 2})
	other := w.record("ep", []blockHash{1, 2})

	// An equal prompt recorded since is a different entry.
	w.replace("ep", other, []blockHash{7})
	require.Equal(t, [][]blockHash{{1, 2}, {7}}, w.snapshot("ep"))
	w.replace("ep", other, []blockHash{8})
	require.Equal(t, [][]blockHash{{1, 2}, {7}}, w.snapshot("ep"), "a replaced entry is not replaced twice")

	// An entry withdrawn by removePrompt is not replaced.
	kept := w.record("ep", []blockHash{6})
	w.removePrompt("ep", []blockHash{6})
	w.replace("ep", kept, []blockHash{10})
	w.replace("ep", nil, []blockHash{10})
	w.replace("ep", other, nil)
	w.replace("unknown", other, []blockHash{10})
	require.Equal(t, [][]blockHash{{1, 2}, {7}}, w.snapshot("ep"))

	// An entry trimmed out of the window is not resurrected.
	small := newRecentWindow(3)
	old := small.record("ep", []blockHash{1, 2})
	small.record("ep", []blockHash{3, 4})
	small.replace("ep", old, []blockHash{9})
	require.Equal(t, [][]blockHash{{3, 4}}, small.snapshot("ep"))
}

func TestRecentWindowReplaceKeepsHandleAcrossCompaction(t *testing.T) {
	w := newRecentWindow(64)
	entries := make([][]blockHash, 0, 200)
	for i := range 200 {
		entries = append(entries, w.record("ep", []blockHash{blockHash(i)}))
	}
	last := entries[len(entries)-1]
	w.replace("ep", last, []blockHash{1000})
	snap := w.snapshot("ep")
	require.Equal(t, []blockHash{1000}, snap[len(snap)-1])
	require.NotContains(t, snap, []blockHash{199}, "the replaced prompt is withdrawn")
}

func TestRecentWindowReplaceStreamMatchesSnapshot(t *testing.T) {
	w := newRecentWindow(8)
	entries := make([][]blockHash, 0, 40)
	for i := range 40 {
		entries = append(entries, w.record("ep", []blockHash{blockHash(i), blockHash(i + 100)}))
		if i%3 == 0 {
			w.replace("ep", entries[i], []blockHash{blockHash(i), blockHash(i + 100), blockHash(i + 200)})
		}
		if i%5 == 0 {
			w.removePrompt("ep", []blockHash{blockHash(i), blockHash(i + 100)})
		}
	}

	set := w.sequenced("ep")
	elems := make([][]byte, 0, set.SequenceTip()-set.SequenceBase())
	for gen := set.SequenceBase(); gen < set.SequenceTip(); gen++ {
		elems = append(elems, set.AppendElement(nil, gen))
	}
	peer, err := (&attrprefix.PrefixHashSet{}).NewSequence(set.SequenceBase(), elems)
	require.NoError(t, err)

	var got [][]blockHash
	for _, prompt := range peer.(*attrprefix.PrefixHashSet).Prompts {
		if len(prompt) > 0 {
			got = append(got, uint64sToBlockHashes(prompt))
		}
	}
	require.Equal(t, w.snapshot("ep"), got, "a peer rebuilding from the stream sees the local window")
}

func TestRecentWindowMatchesReference(t *testing.T) {
	for _, budget := range []int{1, 2, 3, 7, 32} {
		t.Run(strconv.Itoa(budget), func(t *testing.T) {
			w := newRecentWindow(budget)
			want := make(map[string]*referenceWindow)
			type handle struct {
				entry []blockHash
				id    int
			}
			handles := make(map[string][]handle)
			rng := rand.New(rand.NewPCG(1, uint64(budget)))
			endpoints := []string{"ep-a", "ep-b", "ep-c"}
			for step := range 4000 {
				id := endpoints[rng.IntN(len(endpoints))]
				hashes := make([]blockHash, rng.IntN(2*budget+1))
				for i := range hashes {
					hashes[i] = blockHash(rng.Uint64N(4))
				}
				switch rng.IntN(10) {
				case 0:
					w.forget(id)
					delete(want, id)
					delete(handles, id)
				case 4, 5:
					// Replace a random earlier entry, which may be trimmed or withdrawn by now.
					if hs := handles[id]; len(hs) > 0 {
						h := hs[rng.IntN(len(hs))]
						w.replace(id, h.entry, hashes)
						referenceOf(want, id).replace(h.id, hashes, budget)
					}
				case 1, 2, 3:
					// Also withdraw known records, including duplicate sequences.
					if visible := want[id].visible(); len(visible) > 0 && rng.IntN(2) == 0 {
						hashes = slices.Clone(visible[rng.IntN(len(visible))])
					}
					w.removePrompt(id, hashes)
					referenceOf(want, id).remove(hashes, budget)
				default:
					entry := w.record(id, hashes)
					refID := referenceOf(want, id).record(hashes, budget)
					if len(entry) > 0 {
						handles[id] = append(handles[id], handle{entry: entry, id: refID})
					}
				}
				for _, endpoint := range endpoints {
					got := w.snapshot(endpoint)
					expected := want[endpoint].visible()
					if len(expected) == 0 {
						require.Nil(t, got, "step %d endpoint %s", step, endpoint)
					} else {
						require.Equal(t, expected, got, "step %d endpoint %s", step, endpoint)
					}
				}
			}
		})
	}
}

// referenceSlot is one element in referenceWindow. A drop slot is a retraction.
// Empty hashes are a withdrawn prompt.
type referenceSlot struct {
	hashes []blockHash
	drop   bool
	id     int
}

// referenceWindow models recentWindow independently: the same integer is the
// hash budget and the slot budget. A withdrawal keeps its slot and appends a
// retraction, and both count toward the slot budget.
type referenceWindow struct {
	slots  []referenceSlot
	total  int
	nextID int
}

func referenceOf(windows map[string]*referenceWindow, id string) *referenceWindow {
	w := windows[id]
	if w == nil {
		w = &referenceWindow{}
		windows[id] = w
	}
	return w
}

func (w *referenceWindow) visible() [][]blockHash {
	if w == nil {
		return nil
	}
	var out [][]blockHash
	for _, slot := range w.slots {
		if slot.drop || len(slot.hashes) == 0 {
			continue
		}
		out = append(out, slot.hashes)
	}
	return out
}

// record returns the id of the appended slot, or 0 for an empty prompt.
func (w *referenceWindow) record(hashes []blockHash, budget int) int {
	if len(hashes) == 0 {
		return 0
	}
	prompt := hashes
	if len(prompt) > budget {
		prompt = prompt[:budget]
	}
	cp := slices.Clone(prompt)
	w.nextID++
	id := w.nextID
	w.slots = append(w.slots, referenceSlot{hashes: cp, id: id})
	w.total += len(cp)
	w.trim(budget)
	return id
}

// replace withdraws the slot with the given id and appends a retraction and
// the new prompt. It is a no-op when the slot is gone or already withdrawn.
func (w *referenceWindow) replace(id int, hashes []blockHash, budget int) {
	if id == 0 || len(hashes) == 0 {
		return
	}
	for i := range w.slots {
		slot := &w.slots[i]
		if slot.drop || slot.id != id || len(slot.hashes) == 0 {
			continue
		}
		prompt := hashes
		if len(prompt) > budget {
			prompt = prompt[:budget]
		}
		cp := slices.Clone(prompt)
		w.total -= len(slot.hashes)
		slot.hashes = nil
		w.nextID++
		w.slots = append(w.slots, referenceSlot{drop: true}, referenceSlot{hashes: cp, id: w.nextID})
		w.total += len(cp)
		w.trim(budget)
		return
	}
}

func (w *referenceWindow) remove(hashes []blockHash, budget int) {
	if len(hashes) == 0 {
		return
	}
	prompt := hashes
	if len(prompt) > budget {
		prompt = prompt[:budget]
	}
	matched := false
	n := len(w.slots)
	for i := 0; i < n; i++ {
		slot := &w.slots[i]
		if slot.drop || !slices.Equal(slot.hashes, prompt) {
			continue
		}
		w.total -= len(slot.hashes)
		slot.hashes = nil
		w.slots = append(w.slots, referenceSlot{drop: true})
		matched = true
	}
	if !matched {
		return
	}
	w.trim(budget)
}

func (w *referenceWindow) trim(budget int) {
	for w.total > budget && len(w.slots) > 1 {
		w.dropFront()
	}
	for len(w.slots) > budget {
		w.dropFront()
	}
	for len(w.slots) > 0 && (w.slots[0].drop || len(w.slots[0].hashes) == 0) {
		w.dropFront()
	}
}

func (w *referenceWindow) dropFront() {
	if !w.slots[0].drop {
		w.total -= len(w.slots[0].hashes)
	}
	w.slots = w.slots[1:]
}

func TestRecentWindowConcurrentAccess(t *testing.T) {
	w := newRecentWindow(32)
	var wg sync.WaitGroup
	for worker := range 4 {
		wg.Go(func() {
			id := strconv.Itoa(worker)
			for i := range 200 {
				hashes := []blockHash{blockHash(i), blockHash(i + 1)}
				w.record(id, hashes)
				w.record("shared", hashes)
				w.snapshot("shared")
				w.removePrompt("shared", hashes)
				if i%7 == 0 {
					w.forget("shared")
				}
			}
		})
	}
	wg.Wait()
	for worker := range 4 {
		require.Equal(t, 16, len(w.snapshot(strconv.Itoa(worker))))
	}
	total := 0
	for _, p := range w.snapshot("shared") {
		total += len(p)
	}
	require.LessOrEqual(t, total, w.max)
}

func BenchmarkRecentWindowRecord(b *testing.B) {
	for _, promptHashes := range []int{1, 64} {
		for _, budget := range []int{256, 4096, defaultLRUCapacityPerServer} {
			b.Run("hashes="+strconv.Itoa(promptHashes)+"/budget="+strconv.Itoa(budget), func(b *testing.B) {
				w := newRecentWindow(budget)
				hashes := make([]blockHash, promptHashes)
				for i := range hashes {
					hashes[i] = blockHash(i)
				}
				for range budget / promptHashes {
					w.record("ep", hashes)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					w.record("ep", hashes)
				}
			})
		}
	}
}

// BenchmarkRecentWindowRecordThenReplace measures a request that is recorded at
// admission and replaced at completion, with a full window behind it.
func BenchmarkRecentWindowRecordThenReplace(b *testing.B) {
	for _, promptHashes := range []int{1, 64} {
		b.Run("hashes="+strconv.Itoa(promptHashes), func(b *testing.B) {
			w := newRecentWindow(defaultLRUCapacityPerServer)
			hashes := make([]blockHash, promptHashes)
			for i := range hashes {
				hashes[i] = blockHash(i)
			}
			for range defaultLRUCapacityPerServer / promptHashes {
				w.record("ep", hashes)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				w.replace("ep", w.record("ep", hashes), hashes)
			}
		})
	}
}

func BenchmarkRecentWindowRecordAndSnapshot(b *testing.B) {
	for _, budget := range []int{256, 4096} {
		b.Run(strconv.Itoa(budget), func(b *testing.B) {
			w := newRecentWindow(budget)
			hashes := []blockHash{1}
			for range budget {
				w.record("ep", hashes)
			}
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					w.record("ep", hashes)
					w.snapshot("ep")
				}
			})
		})
	}
}
