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

// Multi-rank fixture: 1,034 blocks held by 40 endpoints, each endpoint
// carrying eight rank entries that share its PodIdentifier and differ only by
// GroupIdx. Every block therefore holds 320 entries that must collapse to 40
// matched pods.
const (
	multiRankBlocks = 1034
	multiRankPods   = 40
	multiRankRanks  = 8
)

var multiRankBackends = []*kvcache.KVCacheBackendConfig{{Name: "gpu", Weight: 1.0}, {Name: "cpu", Weight: 0.8}}

func multiRankPodID(i int) string { return fmt.Sprintf("10.0.%d.%d:8200", i/256, i%256) }

// multiRankIndexer builds the fixture behind the production decorator chain.
func multiRankIndexer(tb testing.TB) (*kvcache.Indexer, []kvblock.BlockHash) {
	tb.Helper()
	inner, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 20, PodCacheSize: 512})
	if err != nil {
		tb.Fatal(err)
	}
	keys := make([]kvblock.BlockHash, multiRankBlocks)
	for i := range keys {
		keys[i] = kvblock.BlockHash(i + 1)
	}
	entries := make([]kvblock.PodEntry, 0, multiRankPods*multiRankRanks)
	for p := range multiRankPods {
		for r := range multiRankRanks {
			entries = append(entries, kvblock.PodEntry{
				PodIdentifier: multiRankPodID(p), DeviceTier: "gpu",
				HasGroup: true, GroupIdx: kvblock.GroupID(r),
			})
		}
	}
	if err := inner.Add(context.Background(), nil, keys, entries); err != nil {
		tb.Fatal(err)
	}
	wrapped := kvblock.NewTracedIndex(kvblock.NewInstrumentedIndex(inner))
	return kvcache.NewIndexerForTest(&mockTokenProcessor{}, wrapped, multiRankBackends), keys
}

// validateMultiRank checks complete semantics outside any timed region.
func validateMultiRank(tb testing.TB, matches map[string]kvcache.PodMatch) {
	tb.Helper()
	if len(matches) != multiRankPods {
		tb.Fatalf("got %d matched pods, want %d (rank entries must collapse per endpoint)", len(matches), multiRankPods)
	}
	for p := range multiRankPods {
		m, ok := matches[multiRankPodID(p)]
		if !ok {
			tb.Fatalf("missing endpoint %s", multiRankPodID(p))
		}
		if m.MatchedBlocks != multiRankBlocks {
			tb.Fatalf("%s matched %d blocks, want %d", multiRankPodID(p), m.MatchedBlocks, multiRankBlocks)
		}
		if m.WeightedScore != float64(multiRankBlocks) {
			tb.Fatalf("%s weighted score %v, want %d", multiRankPodID(p), m.WeightedScore, multiRankBlocks)
		}
		if m.BlocksByTier["gpu"] != multiRankBlocks {
			tb.Fatalf("%s gpu tier count %d, want %d", multiRankPodID(p), m.BlocksByTier["gpu"], multiRankBlocks)
		}
	}
}

func TestMultiRankFixtureSemantics(t *testing.T) {
	indexer, keys := multiRankIndexer(t)
	matches, err := indexer.MatchBlockKeys(context.Background(), keys, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateMultiRank(t, matches)
}

func BenchmarkMatchBlockKeysMultiRank(b *testing.B) {
	indexer, keys := multiRankIndexer(b)
	ctx := context.Background()
	matches, err := indexer.MatchBlockKeys(ctx, keys, nil)
	if err != nil {
		b.Fatal(err)
	}
	validateMultiRank(b, matches)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := indexer.MatchBlockKeys(ctx, keys, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatchBlockKeysMultiRankParallel(b *testing.B) {
	indexer, keys := multiRankIndexer(b)
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
