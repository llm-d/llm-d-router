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
)

// Kermit-shaped fixture: 1,034 blocks held by 40 endpoints, each endpoint
// carrying eight rank entries that share its PodIdentifier and differ only by
// GroupIdx. Every block therefore holds 320 entries that must collapse to 40
// matched pods.
const (
	kermitBlocks = 1034
	kermitPods   = 40
	kermitRanks  = 8
)

var kermitBackends = []*kvcache.KVCacheBackendConfig{{Name: "gpu", Weight: 1.0}, {Name: "cpu", Weight: 0.8}}

func kermitPodID(i int) string { return fmt.Sprintf("10.0.%d.%d:8200", i/256, i%256) }

// kermitIndexer builds the fixture behind the production decorator chain.
func kermitIndexer(tb testing.TB) (*kvcache.Indexer, []kvblock.BlockHash) {
	tb.Helper()
	inner, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 20, PodCacheSize: 512})
	if err != nil {
		tb.Fatal(err)
	}
	keys := make([]kvblock.BlockHash, kermitBlocks)
	for i := range keys {
		keys[i] = kvblock.BlockHash(i + 1)
	}
	entries := make([]kvblock.PodEntry, 0, kermitPods*kermitRanks)
	for p := range kermitPods {
		for r := range kermitRanks {
			entries = append(entries, kvblock.PodEntry{
				PodIdentifier: kermitPodID(p), DeviceTier: "gpu",
				HasGroup: true, GroupIdx: kvblock.GroupID(r),
			})
		}
	}
	if err := inner.Add(context.Background(), nil, keys, entries); err != nil {
		tb.Fatal(err)
	}
	wrapped := kvblock.NewTracedIndex(kvblock.NewInstrumentedIndex(inner))
	return kvcache.NewIndexerForTest(&mockTokenProcessor{}, wrapped, kermitBackends), keys
}

// validateKermit checks complete semantics outside any timed region.
func validateKermit(tb testing.TB, matches map[string]kvcache.PodMatch) {
	tb.Helper()
	if len(matches) != kermitPods {
		tb.Fatalf("got %d matched pods, want %d (rank entries must collapse per endpoint)", len(matches), kermitPods)
	}
	for p := range kermitPods {
		m, ok := matches[kermitPodID(p)]
		if !ok {
			tb.Fatalf("missing endpoint %s", kermitPodID(p))
		}
		if m.MatchedBlocks != kermitBlocks {
			tb.Fatalf("%s matched %d blocks, want %d", kermitPodID(p), m.MatchedBlocks, kermitBlocks)
		}
		if m.WeightedScore != float64(kermitBlocks) {
			tb.Fatalf("%s weighted score %v, want %d", kermitPodID(p), m.WeightedScore, kermitBlocks)
		}
		if m.BlocksByTier["gpu"] != kermitBlocks {
			tb.Fatalf("%s gpu tier count %d, want %d", kermitPodID(p), m.BlocksByTier["gpu"], kermitBlocks)
		}
	}
}

func TestKermitFixtureSemantics(t *testing.T) {
	indexer, keys := kermitIndexer(t)
	matches, err := indexer.MatchBlockKeys(context.Background(), keys, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateKermit(t, matches)
}

func BenchmarkKermitMatchBlockKeys(b *testing.B) {
	indexer, keys := kermitIndexer(b)
	ctx := context.Background()
	matches, err := indexer.MatchBlockKeys(ctx, keys, nil)
	if err != nil {
		b.Fatal(err)
	}
	validateKermit(b, matches)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := indexer.MatchBlockKeys(ctx, keys, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKermitMatchBlockKeysParallel(b *testing.B) {
	indexer, keys := kermitIndexer(b)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := indexer.MatchBlockKeys(ctx, keys, nil); err != nil {
				b.Error(err)
				return
			}
		}
	})
}
