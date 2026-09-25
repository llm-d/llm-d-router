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

package kvevents

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/require"
)

func TestSnapshotOffloadPreservesEveryCanonicalBlock(t *testing.T) {
	ctx := context.Background()
	pool, index, tokens := newTestPool(t, 4)
	pool.strict = true
	pool.ownsEntries = true
	prompt := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	require.NoError(t, pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101}, Tokens: prompt, BlockSize: 8, DeviceTier: "GPU"},
		&BlockStoredEvent{BlockHashes: []uint64{101}, BlockSize: 8, DeviceTier: "CPU"},
		&AllBlocksClearedEvent{},
	}}, "pod", "model"))
	keys, err := tokens.TokensToKVBlockKeys(0, prompt, "model", nil)
	require.NoError(t, err)
	entries, err := index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	for _, key := range keys {
		require.Equal(t, []kvblock.PodEntry{{PodIdentifier: "pod", DeviceTier: "cpu"}}, entries[key])
	}
}

func TestSnapshotGenerationCleanupPreservesOtherPublishers(t *testing.T) {
	ctx := context.Background()
	poolA, index, tokens := newTestPool(t, 4)
	poolB, err := NewPool(DefaultConfig(), index, tokens, nil)
	require.NoError(t, err)
	poolA.strict = true
	poolA.ownsEntries = true
	poolB.strict = true
	poolB.ownsEntries = true
	events := &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101}, Tokens: []uint32{1, 2, 3, 4}, BlockSize: 4, DeviceTier: "GPU"},
	}}
	require.NoError(t, poolA.processEventBatch(ctx, events, "generation-a", "model"))
	require.NoError(t, poolB.processEventBatch(ctx, events, "generation-b", "model"))
	require.NoError(t, poolA.clearSnapshotGeneration(ctx))

	keys, err := tokens.TokensToKVBlockKeys(0, []uint32{1, 2, 3, 4}, "model", nil)
	require.NoError(t, err)
	entries, err := index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	require.Equal(t, []kvblock.PodEntry{{PodIdentifier: "generation-b", DeviceTier: "gpu"}}, entries[keys[0]])
}

func TestSnapshotEngineMetadataIsGenerationLocal(t *testing.T) {
	ctx := context.Background()
	poolA, index, tokens := newTestPool(t, 4)
	poolB, err := NewPool(DefaultConfig(), index, tokens, nil)
	require.NoError(t, err)
	poolA.strict = true
	poolA.ownsEntries = true
	poolB.strict = true
	poolB.ownsEntries = true
	promptA := []uint32{1, 2, 3, 4}
	promptB := []uint32{5, 6, 7, 8}
	require.NoError(t, poolA.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101}, Tokens: promptA, BlockSize: 4, DeviceTier: "GPU"},
	}}, "generation-a", "model"))
	require.NoError(t, poolB.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101}, Tokens: promptB, BlockSize: 4, DeviceTier: "GPU"},
	}}, "generation-b", "model"))

	require.NoError(t, poolA.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{&AllBlocksClearedEvent{}}}, "generation-a", "model"))
	keysA, err := tokens.TokensToKVBlockKeys(0, promptA, "model", nil)
	require.NoError(t, err)
	keysB, err := tokens.TokensToKVBlockKeys(0, promptB, "model", nil)
	require.NoError(t, err)
	entries, err := index.Lookup(ctx, append(keysA, keysB...), nil)
	require.NoError(t, err)
	require.Empty(t, entries[keysA[0]])
	require.Equal(t, []kvblock.PodEntry{{PodIdentifier: "generation-b", DeviceTier: "gpu"}}, entries[keysB[0]])

	err = poolB.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{102}, ParentHash: 101, Tokens: []uint32{9, 10, 11, 12}, BlockSize: 4, DeviceTier: "GPU"},
	}}, "generation-b", "model")
	require.NoError(t, err, "publisher B has its own mapping for the colliding hash")
	err = poolA.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{103}, ParentHash: 101, Tokens: []uint32{9, 10, 11, 12}, BlockSize: 4, DeviceTier: "GPU"},
	}}, "generation-a", "model")
	require.NoError(t, err, "publisher A retains its local metadata after GPU clear")

	require.NoError(t, poolA.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{999}, Tokens: []uint32{13, 14, 15, 16}, BlockSize: 4, DeviceTier: "GPU"},
	}}, "generation-a", "model"))
	err = poolB.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{104}, ParentHash: 999, Tokens: []uint32{9, 10, 11, 12}, BlockSize: 4, DeviceTier: "GPU"},
	}}, "generation-b", "model")
	require.ErrorContains(t, err, "snapshot engine key not found")
}

func TestSnapshotGenerationCleanupAfterOffloadAndReset(t *testing.T) {
	ctx := context.Background()
	pool, index, tokens := newTestPool(t, 4)
	pool.strict = true
	pool.ownsEntries = true
	prompt := []uint32{1, 2, 3, 4}
	require.NoError(t, pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101}, Tokens: prompt, BlockSize: 4, DeviceTier: "GPU"},
		&BlockStoredEvent{BlockHashes: []uint64{101}, BlockSize: 4, DeviceTier: "CPU"},
		&AllBlocksClearedEvent{},
	}}, "generation", "model"))

	keys, err := tokens.TokensToKVBlockKeys(0, prompt, "model", nil)
	require.NoError(t, err)
	entries, err := index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	require.Equal(t, []kvblock.PodEntry{{PodIdentifier: "generation", DeviceTier: "cpu"}}, entries[keys[0]])

	require.NoError(t, pool.clearSnapshotGeneration(ctx))
	entries, err = index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	require.Empty(t, entries[keys[0]])
	require.Empty(t, pool.snapshotEntries)
}

func TestStrictSnapshotRejectsStoreWithoutEngineHashesBeforeIndexing(t *testing.T) {
	ctx := context.Background()
	pool, index, tokens := newTestPool(t, 4)
	pool.strict = true
	pool.ownsEntries = true
	prompt := []uint32{1, 2, 3, 4}
	err := pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{Tokens: prompt, BlockSize: 4, DeviceTier: "GPU"},
	}}, "generation", "model")
	require.ErrorContains(t, err, "snapshot engine mapping requires engine and request keys")

	keys, err := tokens.TokensToKVBlockKeys(0, prompt, "model", nil)
	require.NoError(t, err)
	entries, err := index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	require.Empty(t, entries[keys[0]])
	require.Empty(t, pool.snapshotEntries)
}

func TestLiveOnlyGenerationCleanupPreservesOtherPublishers(t *testing.T) {
	ctx := context.Background()
	_, index, tokens := newTestPool(t, 4)
	liveOnly, err := newGenerationPool(index, tokens, nil)
	require.NoError(t, err)
	snapshot, err := newGenerationPool(index, tokens, nil)
	require.NoError(t, err)
	snapshot.strict = true
	events := &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101}, Tokens: []uint32{1, 2, 3, 4}, BlockSize: 4, DeviceTier: "GPU"},
		&BlockStoredEvent{BlockHashes: []uint64{102}, ParentHash: 101, Tokens: []uint32{5, 6, 7, 8}, BlockSize: 4, DeviceTier: "GPU"},
	}}
	require.NoError(t, liveOnly.processEventBatch(ctx, events, "generation-live", "model"))
	require.NoError(t, snapshot.processEventBatch(ctx, events, "generation-snapshot", "model"))
	require.NoError(t, liveOnly.clearSnapshotGeneration(ctx))

	keys, err := tokens.TokensToKVBlockKeys(0, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, "model", nil)
	require.NoError(t, err)
	entries, err := index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	for _, key := range keys {
		require.Equal(t, []kvblock.PodEntry{{PodIdentifier: "generation-snapshot", DeviceTier: "gpu"}}, entries[key])
	}
}

func TestStagedSnapshotMatchesDirectApplication(t *testing.T) {
	ctx := context.Background()
	events := &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101, 102}, Tokens: []uint32{1, 2, 3, 4, 5, 6, 7, 8}, BlockSize: 4, DeviceTier: "GPU"},
		&BlockStoredEvent{BlockHashes: []uint64{103}, ParentHash: 102, Tokens: []uint32{9, 10, 11, 12}, BlockSize: 4, DeviceTier: "GPU"},
		&BlockStoredEvent{BlockHashes: []uint64{101, 102}, BlockSize: 4, DeviceTier: "CPU"},
		&BlockRemovedEvent{BlockHashes: []uint64{102}, DeviceTier: "GPU"},
		&BlockStoredEvent{BlockHashes: []uint64{201}, Tokens: []uint32{21, 22, 23, 24}, BlockSize: 4, DeviceTier: "GPU"},
		&AllBlocksClearedEvent{},
		&BlockStoredEvent{BlockHashes: []uint64{301}, Tokens: []uint32{31, 32, 33, 34}, BlockSize: 4, DeviceTier: "GPU"},
		&BlockRemovedEvent{BlockHashes: []uint64{101}, DeviceTier: "CPU"},
	}}
	direct, directIndex, _ := newTestPool(t, 4)
	direct.strict, direct.ownsEntries = true, true
	require.NoError(t, direct.processEventBatch(ctx, events, "generation", "model"))

	staged, stagedIndex, _ := newTestPool(t, 4)
	staged.strict, staged.ownsEntries, staged.staged = true, true, true
	require.NoError(t, staged.processEventBatch(ctx, events, "generation", "model"))
	require.Equal(t, direct.snapshotEntries, staged.snapshotEntries)
	for owned := range staged.snapshotEntries {
		entries, err := stagedIndex.Lookup(ctx, []kvblock.BlockHash{owned.key}, nil)
		require.NoError(t, err)
		require.Empty(t, entries[owned.key], "staged entries reached the shared index before publish")
	}

	require.NoError(t, staged.publishStaged(ctx))
	require.NotEmpty(t, direct.snapshotEntries)
	for owned := range direct.snapshotEntries {
		want, err := directIndex.Lookup(ctx, []kvblock.BlockHash{owned.key}, nil)
		require.NoError(t, err)
		got, err := stagedIndex.Lookup(ctx, []kvblock.BlockHash{owned.key}, nil)
		require.NoError(t, err)
		require.ElementsMatch(t, want[owned.key], got[owned.key])
	}

	// After publish the pool writes through, and cleanup removes everything.
	require.NoError(t, staged.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{401}, Tokens: []uint32{41, 42, 43, 44}, BlockSize: 4, DeviceTier: "GPU"},
	}}, "generation", "model"))
	require.NoError(t, staged.clearSnapshotGeneration(ctx))
	for owned := range direct.snapshotEntries {
		got, err := stagedIndex.Lookup(ctx, []kvblock.BlockHash{owned.key}, nil)
		require.NoError(t, err)
		require.Empty(t, got[owned.key])
	}
}

func TestCompactSnapshotStateKeepsLiveBookkeeping(t *testing.T) {
	ctx := context.Background()
	pool, index, tokens := newTestPool(t, 4)
	pool.strict = true
	pool.ownsEntries = true
	// A replay stores every retained source event, then removes the excess.
	replay := &EventBatch{}
	prompts := make(map[uint64][]uint32)
	for i := range uint64(64) {
		prompt := []uint32{uint32(4 * i), uint32(4*i + 1), uint32(4*i + 2), uint32(4*i + 3)}
		prompts[100+i] = prompt
		replay.Events = append(replay.Events,
			&BlockStoredEvent{BlockHashes: []uint64{100 + i}, Tokens: prompt, BlockSize: 4, DeviceTier: "GPU"})
	}
	for i := range uint64(60) {
		replay.Events = append(replay.Events, &BlockRemovedEvent{BlockHashes: []uint64{100 + i}, DeviceTier: "GPU"})
	}
	require.NoError(t, pool.processEventBatch(ctx, replay, "generation", "model"))
	entries := make(map[snapshotOwnedEntry]struct{}, len(pool.snapshotEntries))
	for owned := range pool.snapshotEntries {
		entries[owned] = struct{}{}
	}
	refs := make(map[string]map[dedupKey]int)
	for pod, bucket := range pool.dedup.refs {
		refs[pod] = make(map[dedupKey]int)
		for key, count := range bucket {
			refs[pod][key] = count
		}
	}

	pool.compactSnapshotState()
	require.Equal(t, entries, pool.snapshotEntries)
	require.Equal(t, refs, pool.dedup.refs)
	require.Len(t, pool.snapshotEntries, 4)
	require.Len(t, pool.dedup.refs["generation"], 4)

	// Live events after compaction keep updating the rebuilt bookkeeping.
	require.NoError(t, pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockRemovedEvent{BlockHashes: []uint64{160}, DeviceTier: "GPU"},
	}}, "generation", "model"))
	keys, err := tokens.TokensToKVBlockKeys(0, prompts[160], "model", nil)
	require.NoError(t, err)
	found, err := index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	require.Empty(t, found[keys[0]])
	require.Len(t, pool.snapshotEntries, 3)
	require.Len(t, pool.dedup.refs["generation"], 3)
}
