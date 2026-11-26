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
			want := make(map[string][][]blockHash)
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
					if len(want[id]) > 0 && rng.IntN(2) == 0 {
						hashes = slices.Clone(want[id][rng.IntN(len(want[id]))])
					}
					w.removePrompt(id, hashes)
					prompt := hashes[:min(len(hashes), budget)]
					want[id] = slices.DeleteFunc(want[id], func(p []blockHash) bool {
						return slices.Equal(p, prompt)
					})
				default:
					w.record(id, hashes)
					want[id] = referenceRecentRecord(want[id], hashes, budget)
				}
				for _, endpoint := range endpoints {
					got := w.snapshot(endpoint)
					if len(want[endpoint]) == 0 {
						require.Nil(t, got, "step %d endpoint %s", step, endpoint)
					} else {
						require.Equal(t, want[endpoint], got, "step %d endpoint %s", step, endpoint)
					}
				}
			}
		})
	}
}

// referenceRecentRecord recomputes the budget from all records on every append.
func referenceRecentRecord(prompts [][]blockHash, hashes []blockHash, budget int) [][]blockHash {
	if len(hashes) == 0 {
		return prompts
	}
	prompts = append(prompts, slices.Clone(hashes[:min(len(hashes), budget)]))
	total := 0
	for _, p := range prompts {
		total += len(p)
	}
	for total > budget {
		total -= len(prompts[0])
		prompts = prompts[1:]
	}
	return slices.Clone(prompts)
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
