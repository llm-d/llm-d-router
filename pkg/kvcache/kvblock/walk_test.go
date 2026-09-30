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

package kvblock_test

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// visit is one WalkKeys callback, with the borrowed entries copied out.
type visit struct {
	pos     int
	found   bool
	entries []EntryRef
}

func walkAll(t *testing.T, walker KeyWalker, keys []BlockHash) []visit {
	t.Helper()
	var visits []visit
	err := walker.WalkKeys(context.Background(), keys, func(pos int, found bool, entries []EntryRef) bool {
		visits = append(visits, visit{pos: pos, found: found, entries: append([]EntryRef(nil), entries...)})
		return true
	})
	require.NoError(t, err)
	return visits
}

func podEntries(refs []EntryRef) []PodEntry {
	out := make([]PodEntry, len(refs))
	for i, ref := range refs {
		out[i] = ref.PodEntry
	}
	return out
}

func TestWalkKeysVisitsEveryPositionInOrder(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)
	podA := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	podB := PodEntry{PodIdentifier: "pod-b", DeviceTier: "cpu"}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{10}, []PodEntry{podA}))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{30}, []PodEntry{podA, podB}))

	visits := walkAll(t, index, []BlockHash{10, 20, 30})

	require.Len(t, visits, 3)
	assert.Equal(t, 0, visits[0].pos)
	assert.True(t, visits[0].found)
	assert.Equal(t, []PodEntry{podA}, podEntries(visits[0].entries))
	assert.Equal(t, 1, visits[1].pos)
	assert.False(t, visits[1].found)
	assert.Empty(t, visits[1].entries)
	assert.Equal(t, 2, visits[2].pos)
	assert.True(t, visits[2].found)
	assert.Equal(t, []PodEntry{podA, podB}, podEntries(visits[2].entries))
}

// A key listed twice is visited at both positions.
func TestWalkKeysVisitsDuplicatePositions(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)
	podA := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{10}, []PodEntry{podA}))

	visits := walkAll(t, index, []BlockHash{10, 10})

	require.Len(t, visits, 2)
	for i, v := range visits {
		assert.Equal(t, i, v.pos)
		assert.True(t, v.found)
		assert.Equal(t, []PodEntry{podA}, podEntries(v.entries))
	}
}

func TestWalkKeysStopsWhenVisitReturnsFalse(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)
	keys := []BlockHash{1, 2, 3, 4, 5}
	require.NoError(t, index.Add(ctx, nil, keys, []PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}))

	visited := 0
	err = index.WalkKeys(ctx, keys, func(pos int, _ bool, _ []EntryRef) bool {
		visited++
		return pos < 2
	})
	require.NoError(t, err)
	assert.Equal(t, 3, visited)
}

func TestWalkKeysCancellation(t *testing.T) {
	baseCtx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)
	keys := make([]BlockHash, 1000)
	for i := range keys {
		keys[i] = BlockHash(i + 1)
	}
	require.NoError(t, index.Add(baseCtx, nil, keys, []PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}))

	t.Run("before the walk", func(t *testing.T) {
		ctx, cancel := context.WithCancel(baseCtx)
		cancel()
		visited := 0
		err := index.WalkKeys(ctx, keys, func(int, bool, []EntryRef) bool {
			visited++
			return true
		})
		require.ErrorIs(t, err, context.Canceled)
		assert.Zero(t, visited)
	})

	t.Run("during the walk", func(t *testing.T) {
		ctx, cancel := context.WithCancel(baseCtx)
		defer cancel()
		visited := 0
		err := index.WalkKeys(ctx, keys, func(int, bool, []EntryRef) bool {
			visited++
			cancel()
			return true
		})
		require.ErrorIs(t, err, context.Canceled)
		assert.Less(t, visited, len(keys))
	})

	// A walk shorter than the checkpoint interval still reports the
	// cancellation, whether it runs to the end or stops early.
	t.Run("short walk", func(t *testing.T) {
		for name, stopEarly := range map[string]bool{"runs to the end": false, "stops early": true} {
			t.Run(name, func(t *testing.T) {
				ctx, cancel := context.WithCancel(baseCtx)
				defer cancel()
				err := index.WalkKeys(ctx, keys[:2], func(int, bool, []EntryRef) bool {
					cancel()
					return !stopEarly
				})
				require.ErrorIs(t, err, context.Canceled)
			})
		}
	})
}

// A prefix that is only read stays resident: a walk refreshes the recency of
// every key it visits, so the next insertion under capacity pressure evicts
// an unread key instead.
func TestWalkKeysRefreshesIndexLRU(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 2, PodCacheSize: 1})
	require.NoError(t, err)

	entry := []PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{10}, entry))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{20}, entry))

	visits := walkAll(t, index, []BlockHash{10})
	require.True(t, visits[0].found)

	require.NoError(t, index.Add(ctx, nil, []BlockHash{30}, entry))

	visits = walkAll(t, index, []BlockHash{10, 20})
	assert.True(t, visits[0].found, "a walked key must survive the next insertion")
	assert.False(t, visits[1].found, "the unread key is the one evicted")
}

// Only the keys a walk visited are promoted: a walk that stops early leaves
// the unread tail of the prompt where it was in the LRU order.
func TestWalkKeysPromotesOnlyVisitedKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 3, PodCacheSize: 1})
	require.NoError(t, err)

	entry := []PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}
	for _, key := range []BlockHash{10, 20, 30} {
		require.NoError(t, index.Add(ctx, nil, []BlockHash{key}, entry))
	}

	err = index.WalkKeys(ctx, []BlockHash{10, 20, 30}, func(int, bool, []EntryRef) bool { return false })
	require.NoError(t, err)

	require.NoError(t, index.Add(ctx, nil, []BlockHash{40}, entry))

	visits := walkAll(t, index, []BlockHash{10, 20, 30})
	assert.True(t, visits[0].found, "the visited key was promoted")
	assert.False(t, visits[1].found, "the oldest unvisited key was evicted")
	assert.True(t, visits[2].found)
}

// A walk that ends on cancellation still refreshes the keys it read: the
// cancellation is noticed at the next checkpoint, and everything visited
// before it counts as read.
func TestWalkKeysPromotesVisitedKeysOnCancellation(t *testing.T) {
	const numKeys, checkpoint = 300, 256
	baseCtx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: numKeys, PodCacheSize: 1})
	require.NoError(t, err)
	entry := []PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}
	keys := make([]BlockHash, numKeys)
	for i := range keys {
		keys[i] = BlockHash(i + 1)
		require.NoError(t, index.Add(baseCtx, nil, keys[i:i+1], entry))
	}

	ctx, cancel := context.WithCancel(baseCtx)
	defer cancel()
	visited := 0
	err = index.WalkKeys(ctx, keys, func(int, bool, []EntryRef) bool {
		visited++
		cancel()
		return true
	})
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, checkpoint, visited)

	// The unvisited keys are now the oldest; inserting that many evicts
	// exactly them.
	for i := 0; i < numKeys-checkpoint; i++ {
		require.NoError(t, index.Add(baseCtx, nil, []BlockHash{BlockHash(numKeys + i + 1)}, entry))
	}
	visits := walkAll(t, index, []BlockHash{keys[0], keys[checkpoint-1], keys[checkpoint], keys[numKeys-1]})
	assert.True(t, visits[0].found, "first visited key was promoted")
	assert.True(t, visits[1].found, "last visited key was promoted")
	assert.False(t, visits[2].found, "first unvisited key was evicted")
	assert.False(t, visits[3].found, "last unvisited key was evicted")
}

// Lookup keeps the same retention contract as the walk.
func TestLookupRefreshesIndexLRU(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 2, PodCacheSize: 1})
	require.NoError(t, err)

	entry := []PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{10}, entry))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{20}, entry))

	pods, err := index.Lookup(ctx, []BlockHash{10}, nil)
	require.NoError(t, err)
	require.Contains(t, pods, BlockHash(10))

	require.NoError(t, index.Add(ctx, nil, []BlockHash{30}, entry))

	pods, err = index.Lookup(ctx, []BlockHash{10, 20}, nil)
	require.NoError(t, err)
	assert.Contains(t, pods, BlockHash(10), "a looked-up key must survive the next insertion")
	assert.NotContains(t, pods, BlockHash(20), "the unread key is the one evicted")
}

// Ordinals identify pods and tiers consistently across keys and entry order.
func TestWalkKeysOrdinalsAreStable(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)
	gpuA := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	cpuB := PodEntry{PodIdentifier: "pod-b", DeviceTier: "cpu"}
	specA := PodEntry{PodIdentifier: "pod-a", Speculative: true}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, []PodEntry{gpuA, cpuB}))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{2}, []PodEntry{cpuB, gpuA, specA}))

	visits := walkAll(t, index, []BlockHash{1, 2})
	byEntry := map[PodEntry]EntryRef{}
	for _, v := range visits[0].entries {
		byEntry[v.PodEntry] = v
	}
	for _, v := range visits[1].entries {
		if first, ok := byEntry[v.PodEntry]; ok {
			assert.Equal(t, first.PodOrdinal, v.PodOrdinal, "%v pod ordinal", v.PodEntry)
			assert.Equal(t, first.TierOrdinal, v.TierOrdinal, "%v tier ordinal", v.PodEntry)
		} else {
			byEntry[v.PodEntry] = v
		}
	}
	assert.NotEqual(t, byEntry[gpuA].PodOrdinal, byEntry[cpuB].PodOrdinal)
	assert.NotEqual(t, byEntry[gpuA].TierOrdinal, byEntry[cpuB].TierOrdinal)
	assert.Equal(t, byEntry[gpuA].PodOrdinal, byEntry[specA].PodOrdinal, "the same pod shares one ordinal across tiers")
	assert.True(t, byEntry[specA].Speculative)

	// Ordinals survive removal: a pod cleared or evicted and added again
	// keeps its own.
	require.NoError(t, index.Clear(ctx, "pod-a"))
	require.NoError(t, index.Evict(ctx, 1, RequestKey, []PodEntry{cpuB}))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{3}, []PodEntry{gpuA, cpuB}))
	for _, v := range walkAll(t, index, []BlockHash{3})[0].entries {
		assert.Equal(t, byEntry[v.PodEntry].PodOrdinal, v.PodOrdinal, "%v pod ordinal after removal", v.PodEntry)
		assert.Equal(t, byEntry[v.PodEntry].TierOrdinal, v.TierOrdinal, "%v tier ordinal after removal", v.PodEntry)
	}
}

// nonWalkerIndex is an Index without the walk capability.
type nonWalkerIndex struct{ Index }

func TestWalkCapabilityThroughDecorators(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)
	keys := []BlockHash{1, 2, 3}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1, 3}, []PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}))
	direct := walkAll(t, index, keys)

	for name, wrapped := range map[string]Index{
		"traced(instrumented)": NewTracedIndex(NewInstrumentedIndex(index)),
		"instrumented(traced)": NewInstrumentedIndex(NewTracedIndex(index)),
	} {
		t.Run(name, func(t *testing.T) {
			walker, ok := wrapped.(KeyWalker)
			require.True(t, ok, "decorators must keep the walk capability")
			assert.Equal(t, direct, walkAll(t, walker, keys))
		})
	}

	for name, wrapped := range map[string]Index{
		"traced":       NewTracedIndex(nonWalkerIndex{index}),
		"instrumented": NewInstrumentedIndex(nonWalkerIndex{index}),
	} {
		t.Run(name+" without capability", func(t *testing.T) {
			_, ok := wrapped.(KeyWalker)
			assert.False(t, ok, "decorators must not claim a capability the backend lacks")
		})
	}
}

// Concurrent walks and writes on one index must each see complete per-key
// entries.
func TestWalkKeysConcurrentReadersAndWriters(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 1 << 12, PodCacheSize: 16})
	require.NoError(t, err)

	const numKeys, numPods = 200, 4
	keys := make([]BlockHash, numKeys)
	for i := range keys {
		keys[i] = BlockHash(i + 1)
	}
	entries := make([]PodEntry, numPods)
	for p := range entries {
		entries[p] = PodEntry{PodIdentifier: fmt.Sprintf("pod-%d", p), DeviceTier: "gpu"}
	}
	require.NoError(t, index.Add(ctx, nil, keys, entries))

	stop := make(chan struct{})
	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			_ = index.Add(ctx, nil, keys, entries) // refreshes recency only
			churn := fmt.Sprintf("10.9.0.%d:8000", i%256)
			_ = index.Add(ctx, nil, []BlockHash{1 << 40}, []PodEntry{{PodIdentifier: churn, DeviceTier: "gpu"}})
			_ = index.Clear(ctx, churn)
		}
	}()

	const readers, iterations = 8, 40
	errCh := make(chan error, readers)
	var wg sync.WaitGroup
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range iterations {
				var visitErr error
				err := index.WalkKeys(ctx, keys, func(pos int, found bool, entries []EntryRef) bool {
					if !found || len(entries) != numPods {
						visitErr = fmt.Errorf("position %d: found=%v entries=%d, want %d", pos, found, len(entries), numPods)
						return false
					}
					return true
				})
				if err == nil {
					err = visitErr
				}
				if err != nil {
					errCh <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	<-writerDone
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}
