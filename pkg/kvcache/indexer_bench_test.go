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

// BenchmarkScoreTokens measures the full request-time scoring path — block-key
// derivation, index lookup, and longest-prefix scoring — for a 240K-token
// prompt at block size 64 against a fully warm index.
func BenchmarkScoreTokens(b *testing.B) {
	const (
		numTokens = 240_000
		blockSize = 64
	)
	for _, numPods := range []int{8, 96} {
		b.Run(fmt.Sprintf("tokens=240K/pods=%d", numPods), func(b *testing.B) {
			ctx := context.Background()
			tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: blockSize})
			if err != nil {
				b.Fatal(err)
			}
			cfg, err := kvcache.NewDefaultConfig()
			if err != nil {
				b.Fatal(err)
			}
			cfg.KVBlockIndexConfig.InMemoryConfig = &kvblock.InMemoryIndexConfig{
				Size:         1 << 20,
				PodCacheSize: 128,
			}
			indexer, err := kvcache.NewKVCacheIndexer(ctx, cfg, tp)
			if err != nil {
				b.Fatal(err)
			}

			tokens := make([]uint32, numTokens)
			for i := range tokens {
				tokens[i] = uint32(i%50000 + 1)
			}

			keys, err := indexer.ComputeBlockKeysFromTokens(ctx, tokens, "bench-model", nil)
			if err != nil {
				b.Fatal(err)
			}
			entries := make([]kvblock.PodEntry, numPods)
			for p := range entries {
				entries[p] = kvblock.PodEntry{
					PodIdentifier: fmt.Sprintf("10.0.%d.%d:8000", p/256, p%256),
					DeviceTier:    "gpu",
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
				if len(scores) != numPods {
					b.Fatalf("unexpected score count %d", len(scores))
				}
			}
		})
	}
}
