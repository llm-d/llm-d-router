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

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// benchContext carries a discarding logger so index trace logging costs
// nothing and the controller-runtime unset-logger warning never fires.
func benchContext() context.Context {
	return log.IntoContext(context.Background(), logr.Discard())
}

// benchKeys is the block-key count of a 240K-token prompt at block size 64.
const benchKeys = 3750

// benchIndex fills an in-memory index with benchKeys request keys, each held
// by numPods pods on the gpu tier, behind the production decorator chain.
func benchIndex(b *testing.B, numPods int) (kvblock.Index, []kvblock.BlockHash) {
	b.Helper()
	inner, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 20, PodCacheSize: 128})
	if err != nil {
		b.Fatal(err)
	}
	keys := make([]kvblock.BlockHash, benchKeys)
	for i := range keys {
		keys[i] = kvblock.BlockHash(uint64(i) + 1)
	}
	entries := make([]kvblock.PodEntry, numPods)
	for p := range entries {
		entries[p] = kvblock.PodEntry{PodIdentifier: fmt.Sprintf("10.0.%d.%d:8000", p/256, p%256), DeviceTier: "gpu"}
	}
	if err := inner.Add(context.Background(), nil, keys, entries); err != nil {
		b.Fatal(err)
	}
	return kvblock.NewTracedIndex(kvblock.NewInstrumentedIndex(inner)), keys
}

func benchMatcher(b *testing.B, idx kvblock.Index) *kvcache.Indexer {
	b.Helper()
	return kvcache.NewIndexerForTest(&mockTokenProcessor{}, idx, kvcache.DefaultKVCacheBackendConfig())
}

func requireMatches(b *testing.B, matches map[string]kvcache.PodMatch, numPods int) {
	b.Helper()
	if len(matches) != numPods {
		b.Fatalf("matched %d pods, want %d", len(matches), numPods)
	}
	for pod, m := range matches {
		if m.MatchedBlocks != benchKeys {
			b.Fatalf("%s matched %d blocks, want %d", pod, m.MatchedBlocks, benchKeys)
		}
	}
}

// BenchmarkMatchBlockKeys measures the matcher over the block keys a
// 240K-token prompt produces, walking the index, across fleet sizes.
func BenchmarkMatchBlockKeys(b *testing.B) {
	for _, numPods := range []int{8, 96} {
		b.Run(fmt.Sprintf("keys=3750/pods=%d", numPods), func(b *testing.B) {
			idx, keys := benchIndex(b, numPods)
			indexer := benchMatcher(b, idx)
			ctx := benchContext()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matches, err := indexer.MatchBlockKeys(ctx, keys, nil)
				if err != nil {
					b.Fatal(err)
				}
				requireMatches(b, matches, numPods)
			}
		})
	}
}

// BenchmarkMatchBlockKeysMaterialized measures the fallback for backends
// without the walk capability: Lookup plus the same accumulator.
func BenchmarkMatchBlockKeysMaterialized(b *testing.B) {
	for _, numPods := range []int{8, 96} {
		b.Run(fmt.Sprintf("keys=3750/pods=%d", numPods), func(b *testing.B) {
			idx, keys := benchIndex(b, numPods)
			indexer := benchMatcher(b, nonWalkerIndex{idx})
			ctx := benchContext()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matches, err := indexer.MatchBlockKeys(ctx, keys, nil)
				if err != nil {
					b.Fatal(err)
				}
				requireMatches(b, matches, numPods)
			}
		})
	}
}

// BenchmarkMatchBlockKeysParallel measures the matcher under concurrent
// request traffic sharing one index.
func BenchmarkMatchBlockKeysParallel(b *testing.B) {
	for _, numPods := range []int{8, 96} {
		b.Run(fmt.Sprintf("keys=3750/pods=%d", numPods), func(b *testing.B) {
			idx, keys := benchIndex(b, numPods)
			indexer := benchMatcher(b, idx)
			ctx := benchContext()
			b.ReportAllocs()
			b.ResetTimer()
			// Error rather than Fatal: FailNow from a RunParallel worker
			// exits that goroutine and stalls the benchmark.
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					matches, err := indexer.MatchBlockKeys(ctx, keys, nil)
					if err != nil {
						b.Error(err)
						return
					}
					if len(matches) != numPods {
						b.Errorf("matched %d pods, want %d", len(matches), numPods)
						return
					}
				}
			})
		})
	}
}

// BenchmarkMatchBlockKeysWithWriters measures the matcher while eight
// writers ingest events into the same index, approximating per-request
// matching against the KV-event firehose.
func BenchmarkMatchBlockKeysWithWriters(b *testing.B) {
	const numPods, writers = 96, 8
	idx, keys := benchIndex(b, numPods)
	indexer := benchMatcher(b, idx)
	ctx := benchContext()

	stop := make(chan struct{})
	defer close(stop)
	for w := 0; w < writers; w++ {
		go func(w int) {
			entries := []kvblock.PodEntry{{PodIdentifier: fmt.Sprintf("10.1.0.%d:8000", w), DeviceTier: "gpu"}}
			engineKeys := make([]kvblock.BlockHash, 64)
			requestKeys := make([]kvblock.BlockHash, 64)
			// Writer key ranges sit above the request keys, so writes
			// contend with the walk without joining its matches.
			for i := uint64(0); ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				base := (uint64(w+1)<<32 | i*64) + 1
				for j := range engineKeys {
					engineKeys[j] = kvblock.BlockHash(base + uint64(j))
					requestKeys[j] = kvblock.BlockHash(base + uint64(j))
				}
				_ = idx.Add(ctx, engineKeys, requestKeys, entries)
			}
		}(w)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matches, err := indexer.MatchBlockKeys(ctx, keys, nil)
		if err != nil {
			b.Fatal(err)
		}
		if len(matches) != numPods {
			b.Fatalf("matched %d pods, want %d", len(matches), numPods)
		}
	}
}
