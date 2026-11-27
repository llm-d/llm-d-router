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
	"runtime"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// populateIndex fills an in-memory index with numKeys request keys, each held
// by numPods pods on the gpu tier. Returns the index and the ordered keys.
func populateIndex(b *testing.B, numKeys, numPods int) (*kvblock.InMemoryIndex, []kvblock.BlockHash) {
	b.Helper()
	idx, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{
		Size:         1 << 20,
		PodCacheSize: 128,
	})
	if err != nil {
		b.Fatal(err)
	}
	keys := make([]kvblock.BlockHash, numKeys)
	for i := range keys {
		keys[i] = kvblock.BlockHash(uint64(i) + 1)
	}
	entries := make([]kvblock.PodEntry, numPods)
	for p := range entries {
		entries[p] = kvblock.PodEntry{
			PodIdentifier: fmt.Sprintf("10.0.%d.%d:8000", p/256, p%256),
			DeviceTier:    "gpu",
		}
	}
	if err := idx.Add(context.Background(), nil, keys, entries); err != nil {
		b.Fatal(err)
	}
	return idx, keys
}

// BenchmarkInMemoryIndexLookup measures a full-hit lookup of the block keys a
// 240K-token prompt produces at block size 64 (3750 keys), across fleet sizes.
func BenchmarkInMemoryIndexLookup(b *testing.B) {
	const numKeys = 3750
	for _, numPods := range []int{8, 96} {
		for _, filtered := range []bool{false, true} {
			name := fmt.Sprintf("keys=3750/pods=%d/filtered=%v", numPods, filtered)
			b.Run(name, func(b *testing.B) {
				idx, keys := populateIndex(b, numKeys, numPods)
				podSet := sets.New[string]()
				if filtered {
					for p := 0; p < numPods/2; p++ {
						podSet.Insert(fmt.Sprintf("10.0.%d.%d:8000", p/256, p%256))
					}
				}
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					res, err := idx.Lookup(ctx, keys, podSet)
					if err != nil {
						b.Fatal(err)
					}
					if len(res) != numKeys {
						b.Fatalf("unexpected result size %d", len(res))
					}
				}
			})
		}
	}
}

// BenchmarkInMemoryIndexClearUnseenPod measures clearing a pod the index never
// held, as a subscriber does before it loads a pod's first snapshot.
func BenchmarkInMemoryIndexClearUnseenPod(b *testing.B) {
	idx, _ := populateIndex(b, 1<<20, 8)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := idx.Clear(ctx, "10.1.0.1:8000"); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkInMemoryIndexAdd measures ingesting one BlockStored event's worth
// of keys (64 blocks) for a single pod entry, over a warm index.
func BenchmarkInMemoryIndexAdd(b *testing.B) {
	idx, _ := populateIndex(b, 1<<16, 4)
	entries := []kvblock.PodEntry{{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}}
	engineKeys := make([]kvblock.BlockHash, 64)
	requestKeys := make([]kvblock.BlockHash, 64)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		base := uint64(i)*64 + 1
		for j := range engineKeys {
			engineKeys[j] = kvblock.BlockHash(base + uint64(j))
			requestKeys[j] = kvblock.BlockHash(base + uint64(j))
		}
		if err := idx.Add(ctx, engineKeys, requestKeys, entries); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkInMemoryIndexMixed measures lookup latency while concurrent
// writers ingest events, approximating the 96-rank firehose against
// per-request scoring reads.
func BenchmarkInMemoryIndexMixed(b *testing.B) {
	const numKeys = 3750
	idx, keys := populateIndex(b, numKeys, 16)
	ctx := context.Background()

	stop := make(chan struct{})
	defer close(stop)
	for w := 0; w < 4; w++ {
		go func(w int) {
			entries := []kvblock.PodEntry{{PodIdentifier: fmt.Sprintf("10.1.0.%d:8000", w), DeviceTier: "gpu"}}
			engineKeys := make([]kvblock.BlockHash, 64)
			requestKeys := make([]kvblock.BlockHash, 64)
			for i := uint64(0); ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				base := (uint64(w)<<32 | i*64) + 1 //#nosec G115 -- test data, w is a small worker index
				for j := range engineKeys {
					engineKeys[j] = kvblock.BlockHash(base + uint64(j))
					requestKeys[j] = kvblock.BlockHash(base + uint64(j))
				}
				_ = idx.Add(ctx, engineKeys, requestKeys, entries)
			}
		}(w)
	}

	podSet := sets.New[string]()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Lookup(ctx, keys, podSet); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkInMemoryIndexWalkKeys measures the ordered walk over the same
// workload as BenchmarkInMemoryIndexLookup, touching every entry.
func BenchmarkInMemoryIndexWalkKeys(b *testing.B) {
	const numKeys = 3750
	for _, numPods := range []int{8, 96} {
		b.Run(fmt.Sprintf("keys=3750/pods=%d", numPods), func(b *testing.B) {
			idx, keys := populateIndex(b, numKeys, numPods)
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				seen := 0
				err := idx.WalkKeys(ctx, keys, func(_ int, _ bool, entries []kvblock.EntryRef) bool {
					seen += len(entries)
					return true
				})
				if err != nil {
					b.Fatal(err)
				}
				if seen != numKeys*numPods {
					b.Fatalf("visited %d entries, want %d", seen, numKeys*numPods)
				}
			}
		})
	}
}

// Key patterns for engineKeyWorkload: spread multiplies by a 64-bit odd
// constant, sequential keeps the index, and sharedLowBits moves it into the
// high bits so every key has the same low 40 bits.
var (
	spreadKeys        = func(i uint64) kvblock.BlockHash { return kvblock.BlockHash(i * 0x9E3779B97F4A7C15) }
	sequentialKeys    = func(i uint64) kvblock.BlockHash { return kvblock.BlockHash(i) }
	sharedLowBitsKeys = func(i uint64) kvblock.BlockHash { return kvblock.BlockHash(i << 40) }
)

// engineKeyWorkload builds the engine and request keys for numTokens cached
// tokens at the given engine page size, split into BlockStored-sized events.
// Engine keys come from key; request keys from spreadKeys.
func engineKeyWorkload(numTokens, pageSize, eventTokens int, key func(uint64) kvblock.BlockHash) (engineKeys, requestKeys [][]kvblock.BlockHash) {
	const blockSizeTokens = 16
	for start := 0; start < numTokens; start += eventTokens {
		eks := make([]kvblock.BlockHash, eventTokens/pageSize)
		for i := range eks {
			eks[i] = key(uint64(start/pageSize + i + 1)) //#nosec G115 -- test data, small positive ints
		}
		rks := make([]kvblock.BlockHash, eventTokens/blockSizeTokens)
		for i := range rks {
			rks[i] = spreadKeys(uint64(start/blockSizeTokens+i+1)) ^ 1 //#nosec G115 -- test data, small positive ints
		}
		engineKeys = append(engineKeys, eks)
		requestKeys = append(requestKeys, rks)
	}
	return engineKeys, requestKeys
}

func liveHeap() uint64 {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ms.HeapAlloc
}

// BenchmarkInMemoryIndexEngineKeyMemory measures the retained heap of an
// index holding 1M cached tokens for one pod at canonical block size 16.
// Page size 1 maps 16 engine keys to each request key, page size 16 maps one,
// and page size 64 maps each engine key to four request keys. Page size 0
// adds no engine keys, so its B/token is the request-key baseline the other
// cases are compared against.
func BenchmarkInMemoryIndexEngineKeyMemory(b *testing.B) {
	const numTokens = 1 << 20
	const eventTokens = 1024
	entries := []kvblock.PodEntry{{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}}
	for _, pageSize := range []int{0, 1, 16, 64} {
		b.Run(fmt.Sprintf("page_size=%d", pageSize), func(b *testing.B) {
			engineKeys, requestKeys := engineKeyWorkload(numTokens, max(pageSize, 1), eventTokens, spreadKeys)
			if pageSize == 0 {
				clear(engineKeys)
			}
			ctx := context.Background()
			var retained uint64
			for b.Loop() {
				b.StopTimer()
				before := liveHeap()
				b.StartTimer()
				idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
				if err != nil {
					b.Fatal(err)
				}
				for i := range engineKeys {
					if err := idx.Add(ctx, engineKeys[i], requestKeys[i], entries); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				retained = liveHeap() - before
				runtime.KeepAlive(idx)
				b.StartTimer()
			}
			b.ReportMetric(float64(retained)/numTokens, "B/token")
		})
	}
}

// BenchmarkInMemoryIndexGetRequestKey measures parent resolution against an
// index holding 1M engine keys at page size 1, for engine keys that are
// spread, sequential, or share their low bits.
func BenchmarkInMemoryIndexGetRequestKey(b *testing.B) {
	const numTokens = 1 << 20
	patterns := []struct {
		name string
		key  func(uint64) kvblock.BlockHash
	}{
		{name: "spread", key: spreadKeys},
		{name: "sequential", key: sequentialKeys},
		{name: "shared_low_bits", key: sharedLowBitsKeys},
	}
	for _, p := range patterns {
		b.Run("keys="+p.name, func(b *testing.B) {
			engineKeys, requestKeys := engineKeyWorkload(numTokens, 1, 1024, p.key)
			idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
			if err != nil {
				b.Fatal(err)
			}
			ctx := context.Background()
			entries := []kvblock.PodEntry{{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}}
			var all []kvblock.BlockHash
			for i := range engineKeys {
				if err := idx.Add(ctx, engineKeys[i], requestKeys[i], entries); err != nil {
					b.Fatal(err)
				}
				all = append(all, engineKeys[i]...)
			}
			b.ReportAllocs()
			i := 0
			for b.Loop() {
				if _, err := idx.GetRequestKey(ctx, all[i%len(all)]); err != nil {
					b.Fatal(err)
				}
				i += 7919
			}
		})
	}
}
