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

package kvblock //nolint:testpackage // validates private ownership cleanup

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEngineOwnershipLifecycle(t *testing.T) {
	for _, action := range []string{"replace", "request-evict", "pod-evict", "clear", "tier", "scope", "batch"} {
		t.Run(action, func(t *testing.T) {
			ctx := t.Context()
			size, pods := 8, 4
			if action == "request-evict" {
				size = 2
			}
			if action == "pod-evict" {
				pods = 1
			}
			idx, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: size, PodCacheSize: pods})
			require.NoError(t, err)
			gpu := PodEntry{PodIdentifier: "pod1", DeviceTier: "gpu", HasGroup: true, GroupIdx: 0}
			cpu := gpu
			cpu.DeviceTier = "cpu"
			group := gpu
			group.GroupIdx = 1
			add := func(engine BlockHash, keys []BlockHash, entries ...PodEntry) {
				require.NoError(t, idx.Add(ctx, []BlockHash{engine}, keys, entries))
			}
			add(101, []BlockHash{11}, gpu)
			add(102, []BlockHash{11}, gpu)
			switch action {
			case "replace":
				add(101, []BlockHash{21}, gpu)
				require.NoError(t, idx.Evict(ctx, EngineKey, []BlockHash{102}, []PodEntry{gpu}))
				hits, err := idx.Lookup(ctx, []BlockHash{11}, nil)
				require.NoError(t, err)
				require.Empty(t, hits)
			case "request-evict":
				add(103, []BlockHash{12, 13}, gpu)
				require.NotContains(t, idx.owners, BlockHash(11))
				// Both existing mappings point at an evicted request key. Re-admit only one owner.
				add(102, []BlockHash{11}, gpu)
				require.NoError(t, idx.Evict(ctx, EngineKey, []BlockHash{102}, []PodEntry{gpu}))
				hits, err := idx.Lookup(ctx, []BlockHash{11}, nil)
				require.NoError(t, err)
				require.Empty(t, hits)
			case "pod-evict":
				add(103, []BlockHash{11}, cpu)
				require.NotContains(t, idx.owners[11], gpu)
				add(102, []BlockHash{11}, gpu)
				require.NoError(t, idx.Evict(ctx, EngineKey, []BlockHash{102}, []PodEntry{gpu}))
				hits, err := idx.Lookup(ctx, []BlockHash{11}, nil)
				require.NoError(t, err)
				require.Empty(t, hits)
			case "clear":
				require.NoError(t, idx.Clear(ctx, "pod1"))
				require.Empty(t, idx.owners)
				require.Empty(t, idx.engineEntries)
			case "tier":
				n, err := idx.AddByEngineKeys(ctx, []BlockHash{101, 102}, []PodEntry{cpu})
				require.NoError(t, err)
				require.Equal(t, 1, n)
				require.NoError(t, idx.Evict(ctx, EngineKey, []BlockHash{101}, []PodEntry{cpu}))
				hits, err := idx.Lookup(ctx, []BlockHash{11}, nil)
				require.NoError(t, err)
				require.Len(t, hits[11], 2)
				require.NoError(t, idx.Evict(ctx, EngineKey, []BlockHash{102}, []PodEntry{cpu}))
				hits, err = idx.Lookup(ctx, []BlockHash{11}, nil)
				require.NoError(t, err)
				require.Equal(t, []PodEntry{gpu}, hits[11])
			case "scope":
				add(102, []BlockHash{11}, cpu, group)
				require.NoError(t, idx.Evict(ctx, EngineKey, []BlockHash{101, 102}, []PodEntry{gpu}))
				hits, err := idx.Lookup(ctx, []BlockHash{11}, nil)
				require.NoError(t, err)
				require.ElementsMatch(t, []PodEntry{cpu, group}, hits[11])
			case "batch":
				require.NoError(t, idx.Add(ctx, []BlockHash{101}, []BlockHash{11, 12, 13, 14}, []PodEntry{gpu}))
				// All newly inserted records retain ownership despite request-LRU churn.
				for _, key := range []BlockHash{11, 12, 13, 14} {
					require.Contains(t, idx.owners[key][gpu], BlockHash(101))
				}
			}
			for request, entries := range idx.owners {
				pc, found := idx.data.Peek(request)
				require.True(t, found)
				for entry, engines := range entries {
					require.NotEmpty(t, engines)
					live := false
					for _, record := range pc.entries {
						if record.PodEntry == entry {
							live = true
						}
					}
					require.True(t, live)
					for engine := range engines {
						require.Contains(t, idx.engineEntries[engine][request], entry)
						require.True(t, idx.engineToRequestKeys.Contains(engine))
					}
				}
			}
			for engine, requests := range idx.engineEntries {
				for request, entries := range requests {
					for entry := range entries {
						require.Contains(t, idx.owners[request][entry], engine)
					}
				}
			}
		})
	}
}

func TestEngineOwnershipCapacityScope(t *testing.T) {
	ctx := t.Context()
	idx, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 3, PodCacheSize: 8})
	require.NoError(t, err)
	old := PodEntry{PodIdentifier: "old-pod", DeviceTier: "gpu", HasGroup: true, GroupIdx: 0}
	cpu := old
	cpu.DeviceTier = "cpu"
	other := old
	other.PodIdentifier = "live-pod"
	group := old
	group.GroupIdx = 1
	require.NoError(t, idx.Add(ctx, []BlockHash{1}, []BlockHash{11}, []PodEntry{old, cpu}))
	require.NoError(t, idx.Add(ctx, []BlockHash{2}, []BlockHash{11}, []PodEntry{other, group}))
	for _, engine := range []BlockHash{3, 4} {
		require.NoError(t, idx.Add(ctx, []BlockHash{engine}, []BlockHash{900}, []PodEntry{old}))
	}
	hits, err := idx.Lookup(ctx, []BlockHash{11}, nil)
	require.NoError(t, err)
	require.ElementsMatch(t, []PodEntry{other, group}, hits[11])
}

func TestEngineOwnershipConstituentTierUpdate(t *testing.T) {
	ctx := t.Context()
	idx, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 3, PodCacheSize: 4})
	require.NoError(t, err)
	gpu := []PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}
	cpu := []PodEntry{{PodIdentifier: "pod1", DeviceTier: "cpu"}}
	require.NoError(t, idx.Add(ctx, []BlockHash{1, 2}, []BlockHash{11}, gpu))
	require.NoError(t, idx.Evict(ctx, EngineKey, []BlockHash{1}, gpu))
	hits, err := idx.Lookup(ctx, []BlockHash{11}, nil)
	require.NoError(t, err)
	require.Empty(t, hits)
	count, err := idx.AddByEngineKeys(ctx, []BlockHash{2}, cpu)
	require.NoError(t, err)
	require.Zero(t, count)
}

func TestEngineOwnershipOversizedTierUpdate(t *testing.T) {
	idx, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 1, PodCacheSize: 4})
	require.NoError(t, err)
	ctx := t.Context()
	require.NoError(t, idx.Add(ctx, []BlockHash{1, 2}, []BlockHash{11}, []PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}))
	count, err := idx.AddByEngineKeys(ctx, []BlockHash{1, 2}, []PodEntry{{PodIdentifier: "pod1", DeviceTier: "cpu"}})
	require.NoError(t, err)
	require.Zero(t, count)
	require.LessOrEqual(t, len(idx.constituents), idx.engineToRequestKeys.Len())
}
