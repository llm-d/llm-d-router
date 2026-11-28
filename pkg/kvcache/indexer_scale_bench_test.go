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

package kvcache_test

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock/workload"
)

// scaleBenchEnabled gates the long-running residency operating points; they
// run manually, not in CI's default path.
func scaleBenchEnabled() bool {
	return os.Getenv("KVINDEXER_SCALE_BENCH") == "1"
}

// warmMatcher populates an in-memory index from an agentic workload stream
// and returns a matcher over it plus the stream's longest match query.
func warmMatcher(b *testing.B, p workload.Profile) (*kvcache.Indexer, []kvblock.BlockHash) {
	b.Helper()
	idx, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 20, PodCacheSize: 256})
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	var query []kvblock.BlockHash
	g := workload.New(p)
	for {
		op, ok := g.Next()
		if !ok {
			break
		}
		switch op.Kind {
		case workload.AddOp:
			if err := idx.Add(ctx, op.EngineKeys, op.RequestKeys, op.Entries); err != nil {
				b.Fatal(err)
			}
		case workload.EvictOp:
			keys := op.RequestKeys
			if op.KeyType == kvblock.EngineKey {
				keys = op.EngineKeys
			}
			if err := idx.Evict(ctx, op.KeyType, keys, op.Entries); err != nil {
				b.Fatal(err)
			}
		case workload.ClearOp:
			if err := idx.Clear(ctx, op.Pod); err != nil {
				b.Fatal(err)
			}
		case workload.MatchOp:
			if len(op.Query) > len(query) {
				query = op.Query
			}
		}
	}
	return kvcache.NewIndexerForTest(&mockTokenProcessor{}, idx, kvcache.DefaultKVCacheBackendConfig()), query
}

// BenchmarkPrefixMatchDepths is the match-depth baseline: MatchBlockKeys
// latency and allocs at walk depths 64, 512, and 3750 (the block counts a
// 4K-, 32K-, and 240K-token prompt produces at block size 64), across
// fleet sizes, against a workload-warmed index.
func BenchmarkPrefixMatchDepths(b *testing.B) {
	for _, depth := range []int{64, 512, 3750} {
		for _, pods := range []int{8, 96} {
			b.Run(fmt.Sprintf("depth=%d/pods=%d", depth, pods), func(b *testing.B) {
				p := workload.Agentic(pods, 2, depth)
				indexer, query := warmMatcher(b, p)
				if len(query) < depth {
					b.Fatalf("longest query %d shorter than depth %d", len(query), depth)
				}
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					matches, err := indexer.MatchBlockKeys(ctx, query, nil)
					if err != nil {
						b.Fatal(err)
					}
					if len(matches) == 0 {
						b.Fatal("no matches over a warm index")
					}
				}
			})
		}
	}
}

// BenchmarkClearAtScale measures Clear over a pod the index held, on a
// million-key index. Every iteration pays the O(N) scan; the per-matching
// key removal is second-order at these densities.
func BenchmarkClearAtScale(b *testing.B) {
	if !scaleBenchEnabled() {
		b.Skip("set KVINDEXER_SCALE_BENCH=1 to run the operating-point benches")
	}
	const numKeys = 1 << 20
	idx, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 21, PodCacheSize: 64})
	if err != nil {
		b.Fatal(err)
	}
	populateScale(b, idx, numKeys, 8)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := idx.Clear(ctx, workload.PodName(i%8)); err != nil {
			b.Fatal(err)
		}
	}
}

// populateScale fills an index with numKeys request keys, each held by
// podsPerKey pods on the gpu tier, in Add batches of one event's worth.
func populateScale(b *testing.B, idx *kvblock.InMemoryIndex, numKeys, podsPerKey int) {
	b.Helper()
	ctx := context.Background()
	entries := make([]kvblock.PodEntry, podsPerKey)
	for p := range entries {
		entries[p] = kvblock.PodEntry{PodIdentifier: workload.PodName(p), DeviceTier: "gpu"}
	}
	const batchSize = 64
	keys := make([]kvblock.BlockHash, batchSize)
	for base := uint64(1); base <= uint64(numKeys); base += batchSize {
		for j := range keys {
			keys[j] = kvblock.BlockHash(base + uint64(j))
		}
		if err := idx.Add(ctx, keys, keys, entries); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkIndexerContentionMatrix is the write-mix baseline:
// MatchBlockKeys latency while concurrent writers ingest events, across
// read:write mixes, with parallel readers.
func BenchmarkIndexerContentionMatrix(b *testing.B) {
	const depth = 512
	for _, writers := range []int{0, 1, 4} {
		name := fmt.Sprintf("writers=%d", writers)
		if writers == 0 {
			name = "pure-read"
		}
		b.Run(name, func(b *testing.B) {
			indexer, query := warmMatcher(b, workload.Agentic(16, 2, depth))
			ctx := context.Background()

			stop := make(chan struct{})
			var wg sync.WaitGroup
			defer func() {
				close(stop)
				wg.Wait()
			}()
			for w := 0; w < writers; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()
					entries := []kvblock.PodEntry{{PodIdentifier: workload.PodName(100 + w), DeviceTier: "gpu"}}
					engineKeys := make([]kvblock.BlockHash, 64)
					requestKeys := make([]kvblock.BlockHash, 64)
					for i := uint64(0); ; i++ {
						select {
						case <-stop:
							return
						default:
						}
						base := (uint64(w)<<40 | i*64) + 1 //#nosec G115 -- test data, w is a small worker index
						for j := range engineKeys {
							engineKeys[j] = kvblock.BlockHash(base + uint64(j))
							requestKeys[j] = kvblock.BlockHash(base + uint64(j))
						}
						_ = indexer.KVBlockIndex().Add(ctx, engineKeys, requestKeys, entries)
					}
				}(w)
			}

			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					if _, err := indexer.MatchBlockKeys(ctx, query, nil); err != nil {
						b.Error(err)
						return
					}
				}
			})
		})
	}
}

// BenchmarkIndexerOperatingPoints reports steady-state bytes per key (live
// heap and RSS), GC pause totals, and cycle count at the 1M and 10M
// residency operating points, plus match latency at depth 3750. Runs only
// under KVINDEXER_SCALE_BENCH=1; the 10M point needs a machine that can
// hold the production backend's ~600 B/key layout.
func BenchmarkIndexerOperatingPoints(b *testing.B) {
	if !scaleBenchEnabled() {
		b.Skip("set KVINDEXER_SCALE_BENCH=1 to run the operating-point benches")
	}
	for _, numKeys := range []int{1 << 20, 10 << 20} {
		b.Run(fmt.Sprintf("keys=%d", numKeys), func(b *testing.B) {
			before := workload.ReadHeap()
			beforeRSS := workload.RSSBytes()
			idx, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: numKeys * 2, PodCacheSize: 64})
			if err != nil {
				b.Fatal(err)
			}
			populateScale(b, idx, numKeys, 2)
			indexer := kvcache.NewIndexerForTest(&mockTokenProcessor{}, idx, kvcache.DefaultKVCacheBackendConfig())
			after := workload.ReadHeap()
			afterRSS := workload.RSSBytes()

			keys := make([]kvblock.BlockHash, 3750)
			for i := range keys {
				keys[i] = kvblock.BlockHash(uint64(i) + 1)
			}
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := indexer.MatchBlockKeys(ctx, keys, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()

			fk := float64(numKeys)
			b.ReportMetric(float64(workload.Delta(after.HeapAlloc, before.HeapAlloc))/fk, "heap-B/key")
			b.ReportMetric(float64(workload.Delta(afterRSS, beforeRSS))/fk, "rss-B/key")
			pause := after.PauseTotal - before.PauseTotal
			if pause < 0 {
				pause = 0
			}
			b.ReportMetric(pause.Seconds(), "gc-pause-total-s")
			b.ReportMetric(float64(workload.Delta(uint64(after.NumGC), uint64(before.NumGC))), "gc-cycles")
		})
	}
}
