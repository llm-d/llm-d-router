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
				base := (uint64(w)<<32 | i*64) + 1
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

// BenchmarkScoredLookup measures the fused lookup-and-score walk over the
// same workload as BenchmarkInMemoryIndexLookup.
func BenchmarkScoredLookup(b *testing.B) {
	const numKeys = 3750
	tierWeights := map[string]float64{"gpu": 1.0, "cpu": 0.8}
	for _, numPods := range []int{8, 96} {
		b.Run(fmt.Sprintf("keys=3750/pods=%d", numPods), func(b *testing.B) {
			idx, keys := populateIndex(b, numKeys, numPods)
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				stats, err := idx.ScoredLookup(ctx, keys, nil, tierWeights)
				if err != nil {
					b.Fatal(err)
				}
				if len(stats) != numPods {
					b.Fatalf("unexpected stats size %d", len(stats))
				}
			}
		})
	}
}

// BenchmarkScoredLookupAfterChurn measures the fused walk after 10,000
// since-cleared pods have been interned, pinning that per-request cost tracks
// live candidates rather than interner history.
func BenchmarkScoredLookupAfterChurn(b *testing.B) {
	const numKeys = 3750
	tierWeights := map[string]float64{"gpu": 1.0, "cpu": 0.8}
	idx, keys := populateIndex(b, numKeys, 96)
	ctx := context.Background()
	churnKey := []kvblock.BlockHash{1 << 40}
	for i := 0; i < 10_000; i++ {
		pod := fmt.Sprintf("10.9.%d.%d:8000", i/256, i%256)
		if err := idx.Add(ctx, nil, churnKey, []kvblock.PodEntry{{PodIdentifier: pod, DeviceTier: "gpu"}}); err != nil {
			b.Fatal(err)
		}
		if err := idx.Clear(ctx, pod); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stats, err := idx.ScoredLookup(ctx, keys, nil, tierWeights)
		if err != nil {
			b.Fatal(err)
		}
		if len(stats) != 96 {
			b.Fatalf("unexpected stats size %d", len(stats))
		}
	}
}
