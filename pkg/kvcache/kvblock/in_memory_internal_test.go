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
	"fmt"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestInMemoryIndexEvictKeepsEntryAddedConcurrently drives, for each public
// eviction entrypoint, the interleaving where Add has fetched a key's PodCache
// and the eviction empties that cache before Add writes to it. The test holds
// m.mu as Add does, so the eviction must not remove the key until the write lands.
func TestInMemoryIndexEvictKeepsEntryAddedConcurrently(t *testing.T) {
	const (
		engineKey  BlockHash = 101
		requestKey BlockHash = 1
	)
	podA := PodEntry{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}
	podB := PodEntry{PodIdentifier: "10.0.0.2:8000", DeviceTier: "gpu"}

	tests := []struct {
		name  string
		evict func(context.Context, *InMemoryIndex) error
	}{
		{
			name: "Evict by request key",
			evict: func(ctx context.Context, idx *InMemoryIndex) error {
				return idx.Evict(ctx, RequestKey, []BlockHash{requestKey}, []PodEntry{podA})
			},
		},
		{
			name: "Evict by engine key",
			evict: func(ctx context.Context, idx *InMemoryIndex) error {
				return idx.Evict(ctx, EngineKey, []BlockHash{engineKey}, []PodEntry{podA})
			},
		},
		{
			name: "Clear",
			evict: func(ctx context.Context, idx *InMemoryIndex) error {
				return idx.Clear(ctx, podA.PodIdentifier)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()
			idx, err := NewInMemoryIndex(nil)
			require.NoError(t, err)
			require.NoError(t, idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{podA}))

			recordsB, err := idx.internRecords([]PodEntry{podB})
			require.NoError(t, err)

			// Add's critical section: the key's PodCache has been fetched.
			idx.mu.Lock()
			podCache, found := idx.data.Get(requestKey)
			require.True(t, found)

			evicted := make(chan error, 1)
			go func() { evicted <- tt.evict(ctx, idx) }()

			require.Eventually(t, func() bool { return podCache.size() == 0 }, time.Second, time.Millisecond)
			// Give an eviction that does not wait for m.mu the chance to remove the key.
			var evictErr error
			done := false
			select {
			case evictErr = <-evicted:
				done = true
			case <-time.After(50 * time.Millisecond):
			}

			podCache.addAll(recordsB)
			idx.mu.Unlock()
			if !done {
				evictErr = <-evicted
			}
			require.NoError(t, evictErr)

			got, err := idx.Lookup(ctx, []BlockHash{requestKey}, nil)
			require.NoError(t, err)
			assert.Equal(t, []PodEntry{podB}, got[requestKey])
		})
	}
}

// TestInMemoryIndexEngineEvictKeepsMappingAddedConcurrently races one pod's
// engine-key eviction of a block against another pod's store of the same
// block. The surviving entry must stay reachable through its engine key, or
// later removals and child stores for that engine key cannot resolve it.
func TestInMemoryIndexEngineEvictKeepsMappingAddedConcurrently(t *testing.T) {
	ctx := t.Context()
	const (
		engineKey  BlockHash = 9001
		requestKey BlockHash = 1
		trials               = 20000
	)
	podA := PodEntry{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}
	podB := PodEntry{PodIdentifier: "10.0.0.2:8000", DeviceTier: "gpu"}

	for i := range trials {
		idx, err := NewInMemoryIndex(nil)
		require.NoError(t, err)
		require.NoError(t, idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{podA}))

		var wg sync.WaitGroup
		start := make(chan struct{})
		wg.Add(2)
		go func() {
			defer wg.Done()
			<-start
			_ = idx.Evict(ctx, EngineKey, []BlockHash{engineKey}, []PodEntry{podA})
		}()
		go func() {
			defer wg.Done()
			<-start
			_ = idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{podB})
		}()
		close(start)
		wg.Wait()

		got, err := idx.GetRequestKey(ctx, engineKey)
		require.NoError(t, err, "trial %d: engine key mapping removed while pod B's entry remains", i)
		require.Equal(t, requestKey, got, "trial %d", i)
	}
}

// TestInMemoryIndexConcurrentWritersMatchSequentialModel runs one writer per
// pod entry, each issuing random stores and removals over a small shared key
// space. A writer's ops are sequential, as a kvevents shard's are per pod, so
// the final state of its entry on a key is fixed by its own last op on that
// key however the writers interleave. The test checks every entry against
// that, and that every key held through an engine key still resolves.
func TestInMemoryIndexConcurrentWritersMatchSequentialModel(t *testing.T) {
	const (
		pods   = 4
		keys   = 3
		ops    = 40
		rounds = 1000
	)
	engineKey := func(k int) BlockHash { return BlockHash(1000 + k) }
	requestKey := func(k int) BlockHash { return BlockHash(1 + k) }

	type writer struct {
		entry PodEntry
		// engine writers store and remove by engine key like kvevents;
		// the others write speculative entries like PreRequest.
		engine bool
		// clear lets the writer clear its pod, which also removes the pod's
		// speculative entries, so it is only set when no speculative writer runs.
		clear   bool
		present [keys]bool
	}

	tests := []struct {
		name        string
		speculative bool
	}{
		{name: "engine and speculative writers", speculative: true},
		{name: "engine writers with clear"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			for r := range rounds {
				idx, err := NewInMemoryIndex(nil)
				require.NoError(t, err)

				var writers []*writer
				for p := range pods {
					id := fmt.Sprintf("10.0.0.%d:8000", p+1)
					writers = append(writers, &writer{
						entry: PodEntry{PodIdentifier: id, DeviceTier: "gpu"}, engine: true, clear: !tc.speculative,
					})
					if tc.speculative {
						writers = append(writers, &writer{entry: PodEntry{PodIdentifier: id, Speculative: true}})
					}
				}

				var wg sync.WaitGroup
				start := make(chan struct{})
				for i, w := range writers {
					wg.Add(1)
					go func() {
						defer wg.Done()
						rng := rand.New(rand.NewPCG(uint64(r), uint64(i))) //#nosec G404 -- test data
						entries := []PodEntry{w.entry}
						<-start
						for range ops {
							k := rng.IntN(keys)
							switch op := rng.IntN(4); {
							case op == 0 && w.engine:
								_ = idx.Add(ctx, []BlockHash{engineKey(k)}, []BlockHash{requestKey(k)}, entries)
								w.present[k] = true
							case op == 0:
								_ = idx.Add(ctx, nil, []BlockHash{requestKey(k)}, entries)
								w.present[k] = true
							case op == 1 && w.engine:
								_ = idx.Evict(ctx, EngineKey, []BlockHash{engineKey(k)}, entries)
								w.present[k] = false
							case op == 3 && w.clear:
								_ = idx.Clear(ctx, w.entry.PodIdentifier)
								w.present = [keys]bool{}
							default:
								_ = idx.Evict(ctx, RequestKey, []BlockHash{requestKey(k)}, entries)
								w.present[k] = false
							}
						}
					}()
				}
				close(start)
				wg.Wait()

				for k := range keys {
					// Lookup stops at the first missing key, so each key is looked up on its own.
					got, err := idx.Lookup(ctx, []BlockHash{requestKey(k)}, nil)
					require.NoError(t, err)
					heldByEngineWriter := false
					for _, w := range writers {
						require.Equal(t, w.present[k], slices.Contains(got[requestKey(k)], w.entry),
							"round %d: entry %v on key %d", r, w.entry, requestKey(k))
						heldByEngineWriter = heldByEngineWriter || (w.engine && w.present[k])
					}
					if heldByEngineWriter {
						_, err := idx.GetRequestKey(ctx, engineKey(k))
						require.NoError(t, err, "round %d: engine key %d does not resolve", r, engineKey(k))
					}
				}
			}
		})
	}
}
