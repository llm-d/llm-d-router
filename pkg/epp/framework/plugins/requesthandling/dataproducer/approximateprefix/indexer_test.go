/*
Copyright 2025 The Kubernetes Authors.
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
	"context"
	"fmt"
	"maps"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIndexer_AddAndGet(t *testing.T) {
	pod := server{
		ServerID:       ServerID{Namespace: "default", Name: "server1"},
		NumOfGPUBlocks: 2,
	}
	i := newIndexer(context.Background(), 3, "test-name", "test-type").(*indexer) // Initialize with an lruSize greater than server.numOfGPUBlocks to verify server-defined limits take precedence.

	hash1 := blockHash(1)
	// Add an entry to the cache
	i.Add([]blockHash{hash1}, pod)

	// Retrieve the entry
	assert.Equal(t, 1, i.PodBlockCounts()[pod.ServerID], "Cache size should be 1 after adding an entry")
	servers := i.Get(hash1)
	assert.Contains(t, servers, pod.ServerID, "Cache should contain the added server")

	// Add another entry to the cache, the cache size should be incremented to 2.
	i.Add([]blockHash{blockHash(2)}, pod)
	assert.Equal(t, 2, i.PodBlockCounts()[pod.ServerID], "Cache size should  be 2 after adding an entry")

	// Add another entry to the cache, which should evict the first one due to max size.
	i.Add([]blockHash{blockHash(3)}, pod)
	assert.Equal(t, 2, i.PodBlockCounts()[pod.ServerID], "Cache size should still be 2 after adding an entry")

	servers = i.Get(blockHash(4))
	assert.Empty(t, servers, "Cache should not contain non-existent hash")

	// A batch larger than the capacity keeps its leading blocks: matching is
	// anchored at the first block.
	i.Add([]blockHash{blockHash(4), blockHash(5), blockHash(6)}, pod)

	assert.Equal(t, 2, i.PodBlockCounts()[pod.ServerID], "Cache size should stay at capacity after a batch add")
	assert.NotEmpty(t, i.Get(blockHash(4)), "head hashes should remain cached")
	assert.NotEmpty(t, i.Get(blockHash(5)), "head hashes should remain cached")
	assert.Empty(t, i.Get(blockHash(6)), "hash truncated off the batch tail must not be reported as cached")

	// Eviction pressure from a later batch strips the tail first: the head
	// anchors all matching for the prompt and stays cached longest.
	i.Add([]blockHash{blockHash(7)}, pod)
	assert.NotEmpty(t, i.Get(blockHash(4)), "head hash should survive tail-first eviction")
	assert.Empty(t, i.Get(blockHash(5)), "tail hash should be evicted before the head")
}

func TestIndexer_RemovePodAndEviction(t *testing.T) {
	const indexerSize = 10

	i := newIndexer(context.Background(), indexerSize, "test-name", "test-type").(*indexer)

	server1 := server{ServerID: ServerID{Namespace: "default", Name: "server1"}}
	server2 := server{ServerID: ServerID{Namespace: "default", Name: "server2"}}

	// Add indexerSize hashes to both servers
	hashes := make([]blockHash, 0, indexerSize)
	for j := range indexerSize {
		h := blockHash(j)
		hashes = append(hashes, h)
		i.Add([]blockHash{h}, server1)
		i.Add([]blockHash{h}, server2)
	}

	// Ensure all entries are added
	assert.Equal(t, indexerSize, i.PodBlockCounts()[server1.ServerID], "server1 should have 10 entries")
	assert.Equal(t, indexerSize, i.PodBlockCounts()[server2.ServerID], "server2 should have 10 entries")
	for _, h := range hashes {
		pods := i.Get(h)
		assert.Len(t, pods, 2, "Each hash should be associated with exactly 2 pods")
		assert.Contains(t, pods, server1.ServerID, "hash should be associated with server1")
		assert.Contains(t, pods, server2.ServerID, "hash should be associated with server2")
	}

	// Add indexerSize hash to server1 → should evict blockHash(0)
	evictedHash := blockHash(0)
	newHash := blockHash(indexerSize)
	i.Add([]blockHash{newHash}, server1)

	// server1 LRU should still be at max capacity
	assert.Equal(t, indexerSize, i.PodBlockCounts()[server1.ServerID], "server1 LRU should maintain max size")

	pods := i.Get(evictedHash)
	assert.NotContains(t, pods, server1.ServerID, "server1 should no longer hold hash 0")
	assert.Contains(t, pods, server2.ServerID, "server2 should still have hash 0")

	// Remove server2
	i.RemovePod(server2.ServerID)

	pods = i.Get(evictedHash)
	assert.Empty(t, pods, "hash 0 should have no pods after both eviction and removal")
	assert.ElementsMatch(t, []ServerID{server1.ServerID}, i.Pods(), "only server1 should remain tracked")
	for j := 1; j <= indexerSize; j++ {
		assert.Equal(t, podSet{server1.ServerID: {}}, i.Get(blockHash(j)), "hash %d should only be held by server1", j)
	}

	i.RemovePod(server1.ServerID)
	assert.Empty(t, i.Pods(), "no pods should remain after removing the last pod")
	assert.Empty(t, i.Get(newHash), "no hash should be held after removing the last pod")
}

// TestIndexer_ConcurrentAddRemovePod checks that a concurrent Add and RemovePod
// end as if they ran in some order: the pod is either untracked or holds the
// whole batch.
func TestIndexer_ConcurrentAddRemovePod(t *testing.T) {
	lruSize := 10
	hashes := []blockHash{1, 2, 3}
	for iter := range 100 {
		i := newIndexer(t.Context(), lruSize, "test-name", "test-type").(*indexer)
		pod := server{ServerID: ServerID{Namespace: "default", Name: "pod1"}}

		start := make(chan struct{})
		var wg sync.WaitGroup
		wg.Go(func() { <-start; i.Add(hashes, pod) })
		wg.Go(func() { <-start; i.RemovePod(pod.ServerID) })
		close(start)
		wg.Wait()

		got := i.MatchLongestPrefix(hashes, []ServerID{pod.ServerID})[0]
		if len(i.Pods()) == 0 {
			assert.Zero(t, got, "iter %d: untracked pod still matches", iter)
		} else {
			assert.Equal(t, len(hashes), got, "iter %d: tracked pod holds a partial batch", iter)
		}
	}
}

// TestIndexer_MatchLongestPrefixRunLengths checks every run length from 0 to
// the prompt length, across prompt lengths that cross the galloping and
// binary search boundaries.
func TestIndexer_MatchLongestPrefixRunLengths(t *testing.T) {
	for _, promptLen := range []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100} {
		hashes := make([]blockHash, promptLen)
		for k := range hashes {
			hashes[k] = blockHash(1000*promptLen + k)
		}
		i := newIndexer(t.Context(), 1000, "test-name", "test-type").(*indexer)
		candidates := make([]ServerID, promptLen+1)
		for run := 0; run <= promptLen; run++ {
			candidates[run] = ServerID{Namespace: "default", Name: fmt.Sprintf("run-%d", run)}
			i.Add(hashes[:run], server{ServerID: candidates[run]})
		}
		got := i.MatchLongestPrefix(hashes, candidates)
		for run := range candidates {
			assert.Equal(t, run, got[run], "prompt length %d", promptLen)
		}
	}
}

// TestIndexer_MatchConcurrentWithAddAndRemove runs queries while each pod is
// filled with ascending runs and then removed. Between removals a pod's run
// only grows, so a query that overlaps no removal must return a run some Add
// produced and never less than an earlier query saw. A reader that observes a
// partially inserted batch breaks one of the two.
func TestIndexer_MatchConcurrentWithAddAndRemove(t *testing.T) {
	const promptLen = 64
	hashes := make([]blockHash, promptLen)
	for k := range hashes {
		hashes[k] = blockHash(k + 1)
	}
	runs := []int{0, 1, 7, 16, 33, promptLen}
	i := newIndexer(t.Context(), 1000, "test-name", "test-type").(*indexer)
	candidates := make([]ServerID, 8)
	// gens[j] is odd while candidate j is being removed and advances by two
	// per removal.
	gens := make([]atomic.Int64, len(candidates))
	for j := range candidates {
		candidates[j] = ServerID{Namespace: "default", Name: fmt.Sprintf("pod-%d", j)}
	}

	var wg sync.WaitGroup
	for j, id := range candidates {
		wg.Go(func() {
			for range 40 {
				for _, run := range runs[1:] {
					i.Add(hashes[:run], server{ServerID: id})
				}
				gens[j].Add(1)
				i.RemovePod(id)
				gens[j].Add(1)
			}
		})
	}
	for range 4 {
		wg.Go(func() {
			lastGen := make([]int64, len(candidates))
			lastRun := make([]int, len(candidates))
			for range 2000 {
				before := make([]int64, len(candidates))
				for j := range gens {
					before[j] = gens[j].Load()
				}
				got := i.MatchLongestPrefix(hashes, candidates)
				for j, run := range got {
					if g := gens[j].Load(); g != before[j] || g%2 == 1 {
						continue
					}
					assert.Contains(t, runs, run)
					if before[j] == lastGen[j] {
						assert.GreaterOrEqual(t, run, lastRun[j], "candidate %d run decreased without a removal", j)
					}
					lastGen[j], lastRun[j] = before[j], run
				}
			}
		})
	}
	wg.Wait()
}

// podSet holds a set of pods that may have a specific prefix hash.
type podSet map[ServerID]struct{}

// Get returns the pods holding hash. It scans every pod and exists for tests.
func (i *indexer) Get(hash blockHash) podSet {
	i.mu.RLock()
	caches := maps.Clone(i.pods)
	i.mu.RUnlock()

	var res podSet
	for id, pc := range caches {
		pc.mu.RLock()
		ok := pc.lru.Contains(hash)
		pc.mu.RUnlock()
		if ok {
			if res == nil {
				res = make(podSet)
			}
			res[id] = struct{}{}
		}
	}
	return res
}
