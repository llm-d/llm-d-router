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

func TestRecentWindowMatchesReference(t *testing.T) {
	for _, budget := range []int{1, 2, 3, 7, 32} {
		t.Run(strconv.Itoa(budget), func(t *testing.T) {
			w := newRecentWindow(budget)
			want := make(map[string]*referenceWindow)
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
				case 1, 2, 3:
					// Also withdraw known records, including duplicate sequences.
					if visible := want[id].visible(); len(visible) > 0 && rng.IntN(2) == 0 {
						hashes = slices.Clone(visible[rng.IntN(len(visible))])
					}
					w.removePrompt(id, hashes)
					referenceOf(want, id).remove(hashes, budget)
				default:
					w.record(id, hashes)
					referenceOf(want, id).record(hashes, budget)
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
}

// referenceWindow models recentWindow independently: the same integer is the
// hash budget and the slot budget. A withdrawal keeps its slot and appends a
// retraction, and both count toward the slot budget.
type referenceWindow struct {
	slots []referenceSlot
	total int
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

func (w *referenceWindow) record(hashes []blockHash, budget int) {
	if len(hashes) == 0 {
		return
	}
	prompt := hashes
	if len(prompt) > budget {
		prompt = prompt[:budget]
	}
	cp := slices.Clone(prompt)
	w.slots = append(w.slots, referenceSlot{hashes: cp})
	w.total += len(cp)
	w.trim(budget)
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
