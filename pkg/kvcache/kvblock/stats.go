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

package kvblock

import (
	"unsafe"

	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

// MemoryIndexStats is a snapshot of a memory backend's residency.
type MemoryIndexStats struct {
	// ResidentKeys is the number of request keys held.
	ResidentKeys int
	// ResidentBytes is the backend's estimate of its resident structures.
	ResidentBytes int64
	// EngineMappings is the engine-key map's mapping count.
	EngineMappings int
	// TruncatedEntries is the cumulative count of pods dropped by the
	// per-key pod cap.
	TruncatedEntries int64
}

// Per-structure residency estimates for the in-memory backend: one lruStore
// map entry plus a PodCache allocation per key, one EntryRef per allocated
// entry slot (PodCache entries double on growth and never shrink, so the
// estimate follows slot capacity, not entry count), and one mapping with its
// request-key slice in the engine-key map. The estimates size the
// resident-bytes gauge, which informs capacity sizing; measured heap density,
// not this gauge, is the memory gate.
const (
	inMemoryBytesPerKey    = 177
	inMemoryBytesPerEngine = 140
)

// inMemoryBytesPerSlot is the per-slot cost of a PodCache entry: EntryRef
// itself, sized from the type so the estimate cannot drift from it.
var inMemoryBytesPerSlot = int64(unsafe.Sizeof(EntryRef{})) //#nosec G103 -- sizing a resident struct, no pointer arithmetic

// The eviction-cause children are resolved once: WithLabelValues runs a label
// lookup per call, and the capacity path runs it while the store's write lock
// is held.
var (
	evictionsCapacity = metrics.IndexEvictions.WithLabelValues("capacity")
	evictionsClear    = metrics.IndexEvictions.WithLabelValues("clear")
	evictionsEmpty    = metrics.IndexEvictions.WithLabelValues("empty")
)

// Stats returns a snapshot of the index's residency.
func (m *InMemoryIndex) Stats() MemoryIndexStats {
	return MemoryIndexStats{
		ResidentKeys:     m.data.Len(),
		ResidentBytes:    m.residentBytes(),
		EngineMappings:   m.engineToRequestKeys.len(),
		TruncatedEntries: m.truncated.Load(),
	}
}

// residentBytes estimates the resident bytes from the key, entry-slot, and
// engine-mapping counts.
func (m *InMemoryIndex) residentBytes() int64 {
	bytes := int64(m.data.Len()) * inMemoryBytesPerKey
	bytes += m.entrySlots.Load() * inMemoryBytesPerSlot
	bytes += int64(m.engineToRequestKeys.len()) * inMemoryBytesPerEngine
	return bytes
}

// reportResidency refreshes the residency gauges at the end of a mutating
// call. The gauges inform capacity sizing, so lagging a concurrent mutation
// by one call is fine.
func (m *InMemoryIndex) reportResidency() {
	metrics.IndexResidentKeys.Set(float64(m.data.Len()))
	metrics.IndexResidentBytes.Set(float64(m.residentBytes()))
}
