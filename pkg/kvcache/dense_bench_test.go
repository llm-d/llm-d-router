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
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock/workload"
)

// benchBackends pairs each backend's factory with its name for the
// comparative benches.
var benchBackends = []struct {
	name string
	new  func() (kvblock.Index, error)
}{
	{"in-memory", func() (kvblock.Index, error) {
		return kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 20, PodCacheSize: 256})
	}},
	{"dense", func() (kvblock.Index, error) {
		return kvblock.NewDenseMemoryIndex(&kvblock.DenseMemoryIndexConfig{Size: "1GiB"})
	}},
}

// warmMatcherBackend populates an index from an agentic workload stream and
// returns a matcher over it plus the stream's longest match query.
func warmMatcherBackend(b *testing.B, p workload.Profile, factory func() (kvblock.Index, error)) (*kvcache.Indexer, []kvblock.BlockHash) {
	b.Helper()
	idx, err := factory()
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

// BenchmarkPrefixMatchDepthsBackends is the dense-versus-in-memory match
// gate: MatchBlockKeys latency and allocs at walk depths 64, 512, and 3750
// across fleet sizes, over identically warmed indexes.
func BenchmarkPrefixMatchDepthsBackends(b *testing.B) {
	for _, depth := range []int{64, 512, 3750} {
		for _, pods := range []int{8, 96} {
			for _, backend := range benchBackends {
				b.Run(fmt.Sprintf("depth=%d/pods=%d/%s", depth, pods, backend.name), func(b *testing.B) {
					indexer, query := warmMatcherBackend(b, workload.Agentic(pods, 2, depth), backend.new)
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
}

// BenchmarkPrefixMatchInteractiveBackends is the interactive-generator
// non-regression gate: MatchBlockKeys latency over the interactive workload
// at conversational depths.
func BenchmarkPrefixMatchInteractiveBackends(b *testing.B) {
	for _, depth := range []int{16, 64} {
		for _, backend := range benchBackends {
			b.Run(fmt.Sprintf("depth=%d/%s", depth, backend.name), func(b *testing.B) {
				indexer, query := warmMatcherBackend(b, workload.Interactive(16, 1, depth), backend.new)
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

// BenchmarkIndexAddThroughput is the ingestion gate: request keys admitted
// per second in one-event Add batches, per backend.
func BenchmarkIndexAddThroughput(b *testing.B) {
	for _, backend := range benchBackends {
		b.Run(backend.name, func(b *testing.B) {
			idx, err := backend.new()
			if err != nil {
				b.Fatal(err)
			}
			ctx := context.Background()
			entries := []kvblock.PodEntry{
				{PodIdentifier: workload.PodName(0), DeviceTier: "gpu"},
				{PodIdentifier: workload.PodName(1), DeviceTier: "gpu"},
			}
			const batchSize = 64
			keys := make([]kvblock.BlockHash, batchSize)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				base := uint64(i)*batchSize + 1 //#nosec G115 -- test data bounded by b.N
				for j := range keys {
					keys[j] = kvblock.BlockHash(base + uint64(j))
				}
				if err := idx.Add(ctx, keys, keys, entries); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(b.N*batchSize)/b.Elapsed().Seconds(), "keys/s")
		})
	}
}

// BenchmarkContentionMatrixBackends is the dense-versus-in-memory
// read-under-write gate: MatchBlockKeys latency while concurrent writers
// ingest events, with parallel readers.
func BenchmarkContentionMatrixBackends(b *testing.B) {
	const depth = 512
	for _, writers := range []int{0, 1, 4} {
		name := fmt.Sprintf("writers=%d", writers)
		if writers == 0 {
			name = "pure-read"
		}
		for _, backend := range benchBackends {
			b.Run(name+"/"+backend.name, func(b *testing.B) {
				indexer, query := warmMatcherBackend(b, workload.Agentic(16, 2, depth), backend.new)
				ctx := context.Background()

				stop := make(chan struct{})
				defer close(stop)
				for w := 0; w < writers; w++ {
					go func(w int) {
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
}

// BenchmarkDenseOperatingPoints reports the dense backend's steady-state
// bytes per key (live heap and RSS), GC pause totals, and cycle count at the
// 1M and 10M residency operating points, plus match latency at depth 3750.
// Runs only under KVINDEXER_SCALE_BENCH=1; the stage gate is heap B/key at
// or under 144.
func BenchmarkDenseOperatingPoints(b *testing.B) {
	if !scaleBenchEnabled() {
		b.Skip("set KVINDEXER_SCALE_BENCH=1 to run the operating-point benches")
	}
	for _, numKeys := range []int{1 << 20, 10 << 20} {
		b.Run(fmt.Sprintf("keys=%d", numKeys), func(b *testing.B) {
			before := workload.ReadHeap()
			beforeRSS := workload.RSSBytes()
			// A byte budget well above the expected residency keeps the
			// sweep out of the measurement.
			idx, err := kvblock.NewDenseMemoryIndex(&kvblock.DenseMemoryIndexConfig{
				Size: fmt.Sprintf("%d", numKeys*256),
			})
			if err != nil {
				b.Fatal(err)
			}
			populateScaleIdx(b, idx, numKeys, 2)
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
			b.ReportMetric(float64(after.HeapAlloc-before.HeapAlloc)/fk, "heap-B/key")
			b.ReportMetric(float64(afterRSS-beforeRSS)/fk, "rss-B/key")
			b.ReportMetric((after.PauseTotal - before.PauseTotal).Seconds(), "gc-pause-total-s")
			b.ReportMetric(float64(after.NumGC-before.NumGC), "gc-cycles")
		})
	}
}

// populateScaleIdx fills an index with numKeys request keys, each held by
// podsPerKey pods on the gpu tier, in Add batches of one event's worth.
func populateScaleIdx(b *testing.B, idx kvblock.Index, numKeys, podsPerKey int) {
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
