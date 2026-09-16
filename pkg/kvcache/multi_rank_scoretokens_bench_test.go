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

// BenchmarkScoreTokensMultiRank measures the tokens-in scoring path on the
// multi-rank shape: 1,034 blocks held by 40 endpoints that each report eight
// rank entries sharing the endpoint identifier, so every block carries 320
// entries collapsing to 40 scored pods. Only APIs shared with main are used,
// so the same file measures both sides of the comparison.
func BenchmarkScoreTokensMultiRank(b *testing.B) {
	const (
		blocks    = 1034
		pods      = 40
		ranks     = 8
		blockSize = 64
	)
	ctx := log.IntoContext(context.Background(), logr.Discard())
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: blockSize})
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := kvcache.NewDefaultConfig()
	if err != nil {
		b.Fatal(err)
	}
	cfg.KVBlockIndexConfig.InMemoryConfig = &kvblock.InMemoryIndexConfig{Size: 1 << 20, PodCacheSize: 512}
	indexer, err := kvcache.NewKVCacheIndexer(ctx, cfg, tp)
	if err != nil {
		b.Fatal(err)
	}

	tokens := make([]uint32, blocks*blockSize)
	for i := range tokens {
		tokens[i] = uint32(i%50000 + 1)
	}
	keys, err := indexer.ComputeBlockKeysFromTokens(ctx, tokens, "bench-model", nil)
	if err != nil {
		b.Fatal(err)
	}
	entries := make([]kvblock.PodEntry, 0, pods*ranks)
	for p := 0; p < pods; p++ {
		for r := 0; r < ranks; r++ {
			entries = append(entries, kvblock.PodEntry{
				PodIdentifier: fmt.Sprintf("10.0.%d.%d:8200", p/256, p%256), DeviceTier: "gpu",
				HasGroup: true, GroupIdx: kvblock.GroupID(r),
			})
		}
	}
	if err := indexer.KVBlockIndex().Add(ctx, nil, keys, entries); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scores, err := indexer.ScoreTokens(ctx, tokens, "bench-model", nil, nil)
		if err != nil {
			b.Fatal(err)
		}
		if len(scores) != pods {
			b.Fatalf("scored %d pods, want %d", len(scores), pods)
		}
	}
}
