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
	poolB.strict = true
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
	poolB.strict = true
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
