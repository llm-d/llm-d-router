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

package preciseprefixcache

import (
	"fmt"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// lookupResult builds a lookup result where every block key is held by every
// endpoint, the worst case for a scan over the result.
func lookupResult(blocks, endpoints int) map[kvblock.BlockHash][]kvblock.PodEntry {
	identifiers := make([]string, endpoints)
	for e := range endpoints {
		identifiers[e] = endpointIdentifier(fmt.Sprintf("10.0.%d.%d", e/256, e%256), "8000")
	}

	keyToPods := make(map[kvblock.BlockHash][]kvblock.PodEntry, blocks)
	for b := range blocks {
		entries := make([]kvblock.PodEntry, 0, endpoints)
		for e := range endpoints {
			entries = append(entries, kvblock.PodEntry{
				PodIdentifier: identifiers[e],
				DeviceTier:    "gpu",
			})
		}
		keyToPods[kvblock.BlockHash(b+1)] = entries
	}
	return keyToPods
}

// BenchmarkRecordConfirmedEndpoints guards the property that recording costs
// scale with the number of candidate endpoints and not with prompt length: the
// block counts vary by a factor of 60 here, the endpoint counts do not.
func BenchmarkRecordConfirmedEndpoints(b *testing.B) {
	shapes := []struct {
		name              string
		blocks, endpoints int
	}{
		{"32blocks_4endpoints", 32, 4},
		{"250blocks_4endpoints", 250, 4},
		{"250blocks_8endpoints", 250, 8},
		{"2000blocks_8endpoints", 2000, 8},
	}

	for _, shape := range shapes {
		keyToPods := lookupResult(shape.blocks, shape.endpoints)
		b.Run(shape.name, func(b *testing.B) {
			p := &Producer{healthMonitor: NewKVEventsHealthMonitor(true)}
			b.ReportAllocs()
			for range b.N {
				// The set is per request, as in produceFromBlockKeys.
				p.recordConfirmedEndpoints(keyToPods,
					make(map[string]struct{}, shape.endpoints), shape.endpoints)
			}
		})
	}
}

// BenchmarkRecordConfirmedEndpointsParallel covers concurrent requests, which
// record against the same endpoints and so contend on the same per-endpoint
// locks.
func BenchmarkRecordConfirmedEndpointsParallel(b *testing.B) {
	const endpoints = 8
	keyToPods := lookupResult(250, endpoints)
	p := &Producer{healthMonitor: NewKVEventsHealthMonitor(true)}

	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p.recordConfirmedEndpoints(keyToPods,
				make(map[string]struct{}, endpoints), endpoints)
		}
	})
}
