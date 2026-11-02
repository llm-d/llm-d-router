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

package approximateprefix

import (
	"context"
	"fmt"
	"runtime"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/llm-d/llm-d-router/test/utils/mooncake"
)

// BenchmarkParallel runs concurrent queries, alone (read) or each followed by
// an Add of the same prompt to one pod (cycle), as a request does.
func BenchmarkParallel(b *testing.B) {
	for _, mode := range []string{"read", "cycle"} {
		for _, layout := range []string{"hotspot", "mixedDepth"} {
			for _, numPods := range []int{96, 256} {
				b.Run(fmt.Sprintf("%s/%s/pods=%d", mode, layout, numPods), func(b *testing.B) {
					p, err := newDataProducer(b.Context(), ApproxPrefixCachePluginType, defaultConfig, testHandle())
					if err != nil {
						b.Fatal(err)
					}
					pods, _ := benchPods(numPods)
					ids := make([]ServerID, numPods)
					for j, s := range pods {
						ids[j] = s.ServerID
					}
					hashes := benchHashes(p, benchLayouts[layout](p, pods))
					var next atomic.Int64
					b.ReportAllocs()
					b.RunParallel(func(pb *testing.PB) {
						for pb.Next() {
							p.indexerInst.MatchLongestPrefix(hashes, ids)
							if mode == "cycle" {
								p.indexerInst.Add(hashes, pods[next.Add(1)%int64(numPods)])
							}
						}
					})
				})
			}
		}
	}
}

// BenchmarkRetainedHeap fills every pod's LRU to capacity with prompts that
// share a system prompt and reports the heap the indexer retains.
func BenchmarkRetainedHeap(b *testing.B) {
	for _, numPods := range []int{16, 96} {
		b.Run(fmt.Sprintf("pods=%d", numPods), func(b *testing.B) {
			pods, _ := benchPods(numPods)
			hashes := make([]blockHash, benchPromptBlocks)
			for b.Loop() {
				var before, after runtime.MemStats
				runtime.GC()
				runtime.ReadMemStats(&before)
				ctx, cancel := context.WithCancel(b.Context())
				idx := newIndexer(ctx, defaultLRUCapacityPerServer, "bench", "bench")
				for i, s := range pods {
					for j := range defaultLRUCapacityPerServer / benchPromptBlocks {
						for k := range hashes {
							if k < benchSharedBlocks {
								hashes[k] = blockHash(1<<62 | uint64(k))
							} else {
								hashes[k] = blockHash(uint64(i)<<40 | uint64(j)<<20 | uint64(k))
							}
						}
						idx.Add(hashes, s)
					}
				}
				runtime.GC()
				runtime.ReadMemStats(&after)
				b.ReportMetric(float64(int64(after.HeapAlloc)-int64(before.HeapAlloc))/(1<<20), "MiB-retained")
				runtime.KeepAlive(idx)
				cancel()
			}
		})
	}
}

// BenchmarkMooncakeReplay replays the Mooncake trace against the indexer from
// concurrent goroutines, after adding every turn's query to its worker in
// trace order. QueryOnly matches each turn against all workers. MixedOverload
// also adds the query to the turn's worker, as the plugin does after
// scheduling. It reports block_ops/s over Dynamo's denominators
// (mooncake.Replay). Set mooncake.TraceEnv to run it.
func BenchmarkMooncakeReplay(b *testing.B) {
	r := mooncake.Load(b)
	idx := newIndexer(b.Context(), mooncake.WorkerBlocks, "bench", "bench")
	pods := make([]server, mooncake.Workers)
	ids := make([]ServerID, mooncake.Workers)
	for w := range pods {
		ids[w] = ServerID{Namespace: "default", Name: "pod-" + strconv.Itoa(w)}
		pods[w] = server{ServerID: ids[w], NumOfGPUBlocks: mooncake.WorkerBlocks}
	}

	type turn struct {
		query []blockHash
		pod   server
	}
	turns := make([]turn, len(r.Turns))
	for i, t := range r.Turns {
		turns[i] = turn{mooncake.Convert[blockHash](t.Query), pods[t.Worker]}
		idx.Add(turns[i].query, turns[i].pod)
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
					idx.MatchLongestPrefix(t.query, ids)
					if mode == "MixedOverload" {
						idx.Add(t.query, t.pod)
					}
				}
			})
			b.ReportMetric(blocks*float64(b.N)/b.Elapsed().Seconds(), "block_ops/s")
		})
	}
}
