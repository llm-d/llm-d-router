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
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/test/utils/mooncake"
)

// BenchmarkMooncakeReplay replays the Mooncake trace against the in-memory
// index from concurrent goroutines, after loading every worker's resident
// blocks. QueryOnly matches each turn's keys. MixedOverload also applies the
// turn's stores and evictions, as KV events would. It reports block_ops/s over
// Dynamo's denominators (mooncake.Replay). Set mooncake.TraceEnv to run it.
func BenchmarkMooncakeReplay(b *testing.B) {
	r := mooncake.Load(b)
	ctx := benchContext()
	inner, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 23, PodCacheSize: mooncake.Workers})
	if err != nil {
		b.Fatal(err)
	}
	idx := kvblock.NewTracedIndex(kvblock.NewInstrumentedIndex(inner))
	matcher := benchMatcher(b, idx)

	workers := make([][]kvblock.PodEntry, mooncake.Workers)
	for w := range workers {
		workers[w] = []kvblock.PodEntry{{PodIdentifier: fmt.Sprintf("10.0.0.%d:8000", w), DeviceTier: "gpu"}}
	}
	type turn struct {
		query, stored, removed []kvblock.BlockHash
		entries                []kvblock.PodEntry
	}
	turns := make([]turn, len(r.Turns))
	for i, t := range r.Turns {
		turns[i] = turn{
			mooncake.Convert[kvblock.BlockHash](t.Query),
			mooncake.Convert[kvblock.BlockHash](t.Stored),
			mooncake.Convert[kvblock.BlockHash](t.Removed),
			workers[t.Worker],
		}
		if warm := mooncake.Convert[kvblock.BlockHash](t.Warm); len(warm) > 0 {
			if err := inner.Add(ctx, warm, warm, turns[i].entries); err != nil {
				b.Fatal(err)
			}
		}
	}

	for _, mode := range []string{"QueryOnly", "MixedOverload"} {
		b.Run(mode, func(b *testing.B) {
			blocks := r.QueryBlocks
			if mode == "MixedOverload" {
				blocks = r.EventBlocks
			}
			var goroutines atomic.Int64
			b.ReportAllocs()
			b.RunParallel(func(pb *testing.PB) {
				// Each goroutine walks the turns from its own offset.
				i := int(goroutines.Add(1)) * 9973
				for pb.Next() {
					i++
					t := &turns[i%len(turns)]
					if _, err := matcher.MatchBlockKeys(ctx, t.query, nil); err != nil {
						b.Error(err)
						return
					}
					if mode == "QueryOnly" {
						continue
					}
					if len(t.stored) > 0 {
						if err := idx.Add(ctx, t.stored, t.stored, t.entries); err != nil {
							b.Error(err)
							return
						}
					}
					if len(t.removed) > 0 {
						if err := idx.Evict(ctx, kvblock.EngineKey, t.removed, t.entries); err != nil {
							b.Error(err)
							return
						}
					}
				}
			})
			b.ReportMetric(blocks*float64(b.N)/b.Elapsed().Seconds(), "block_ops/s")
		})
	}
}
