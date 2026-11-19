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

package kvevents //nolint:testpackage // benchmarks use the unexported eventDedupFilter

import (
	"fmt"
	"runtime"
	"runtime/metrics"
	"testing"
)

const (
	dedupBenchPods  = 16
	dedupBenchBatch = 64
)

// dedupBenchScopes spreads entries over dedupBenchPods pods with a GPU and a
// CPU tier each, the layout an engine fleet with an offload tier produces.
func dedupBenchScopes() []blockScope {
	scopes := make([]blockScope, 0, 2*dedupBenchPods)
	for p := range dedupBenchPods {
		for _, tier := range []string{"gpu", "cpu"} {
			scopes = append(scopes, testScope(fmt.Sprintf("10.0.0.%d:8000", p), tier, 0, noDataParallelRank))
		}
	}
	return scopes
}

// dedupBenchHash spreads sequential ids over the hash space like engine block
// hashes.
func dedupBenchHash(i uint64) uint64 {
	i ^= i >> 33
	i *= 0xff51afd7ed558ccd
	i ^= i >> 33
	return i
}

func fillDedupFilter(f *eventDedupFilter, scopes []blockScope, numEntries int) {
	hashes := make([]uint64, dedupBenchBatch)
	for base := 0; base < numEntries; base += dedupBenchBatch {
		for j := range hashes {
			hashes[j] = dedupBenchHash(uint64(base + j))
		}
		f.trackStore(scopes[(base/dedupBenchBatch)%len(scopes)], hashes)
	}
}

// BenchmarkEventDedupFilterHeldBytes reports the heap the filter retains per
// outstanding block hash, the allocations it made to get there, and the
// GC CPU time of one full collection with the filter live.
func BenchmarkEventDedupFilterHeldBytes(b *testing.B) {
	scopes := dedupBenchScopes()
	for _, numEntries := range []int{1 << 20, 10 << 20} {
		b.Run(fmt.Sprintf("entries=%dM", numEntries>>20), func(b *testing.B) {
			for b.Loop() {
				var before, after runtime.MemStats
				runtime.GC()
				runtime.ReadMemStats(&before)
				baseGC := gcCPUSeconds()
				f := newEventDedupFilter()
				fillDedupFilter(f, scopes, numEntries)
				runtime.GC()
				runtime.ReadMemStats(&after)
				liveGC := gcCPUSeconds()
				b.ReportMetric(float64(int64(after.HeapAlloc)-int64(before.HeapAlloc))/float64(numEntries), "B/entry")
				b.ReportMetric(float64(after.Mallocs-before.Mallocs)/float64(numEntries), "allocs/entry")
				b.ReportMetric((liveGC-baseGC)*1e3, "gc-cpu-ms")
				runtime.KeepAlive(f)
			}
		})
	}
}

// gcCPUSeconds runs one full collection and returns the GC CPU time it took.
func gcCPUSeconds() float64 {
	sample := []metrics.Sample{{Name: "/cpu/classes/gc/total:cpu-seconds"}}
	metrics.Read(sample)
	start := sample[0].Value.Float64()
	runtime.GC()
	metrics.Read(sample)
	return sample[0].Value.Float64() - start
}

// BenchmarkEventDedupFilterChurn measures the ingest cost at steady state: each
// op stores one batch of new hashes and removes the oldest batch of the same
// scope, holding the filter at 1M entries.
func BenchmarkEventDedupFilterChurn(b *testing.B) {
	const numEntries = 1 << 20
	scopes := dedupBenchScopes()
	f := newEventDedupFilter()
	fillDedupFilter(f, scopes, numEntries)
	stored := make([]uint64, dedupBenchBatch)
	removed := make([]uint64, dedupBenchBatch)
	next := numEntries
	b.ReportAllocs()
	for b.Loop() {
		scope := scopes[(next/dedupBenchBatch)%len(scopes)]
		for j := range stored {
			stored[j] = dedupBenchHash(uint64(next + j))
			removed[j] = dedupBenchHash(uint64(next - numEntries + j))
		}
		f.trackStore(scope, stored)
		if got, _ := f.filterRemove(scope, removed); len(got) != dedupBenchBatch {
			b.Fatalf("forwarded %d of %d removes", len(got), dedupBenchBatch)
		}
		next += dedupBenchBatch
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*2*dedupBenchBatch), "ns/hash")
}
