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

package kvevents //nolint:testpackage // tests use unexported processEventBatch

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

const (
	windowTestPod   = "pod-hybrid"
	windowTestModel = "test-model"
)

// newWindowTestPool is newTestPool with IndexSlidingWindowGroups set.
func newWindowTestPool(t *testing.T) (*Pool, kvblock.Index, kvblock.TokenProcessor) {
	t.Helper()
	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 16, HashSeed: "test"})
	require.NoError(t, err)
	cfg := DefaultConfig()
	cfg.IndexSlidingWindowGroups = true
	pool, err := NewPool(cfg, idx, tp, nil)
	require.NoError(t, err)
	require.NotNil(t, pool.WindowIndex())
	return pool, idx, tp
}

// groupStore is one HMA group's store of 16-token blocks.
func groupStore(group int, kind KVCacheSpecKind, window *int, tokens []uint32, engineKeys []uint64, parent uint64) *BlockStoredEvent {
	return &BlockStoredEvent{
		BlockHashes: engineKeys, Tokens: tokens, ParentHash: parent, BlockSize: 16,
		GroupIdx: &group, KVCacheSpecKind: kind, KVCacheSpecSlidingWindowSize: window,
	}
}

func process(ctx context.Context, pool *Pool, events ...GenericEvent) {
	pool.processEventBatch(ctx, &EventBatch{Events: events}, windowTestPod, windowTestModel)
}

// requireGroups checks that every key holds exactly the pod's entries for the
// given groups in index.
func requireGroups(ctx context.Context, t *testing.T, index kvblock.Index, keys []kvblock.BlockHash, want ...kvblock.GroupID) {
	t.Helper()
	for _, key := range keys {
		result, err := index.Lookup(ctx, []kvblock.BlockHash{key}, nil)
		require.NoError(t, err)
		var got []kvblock.GroupID
		for _, entry := range result[key] {
			require.True(t, entry.HasGroup)
			require.Equal(t, windowTestPod, entry.PodIdentifier)
			got = append(got, entry.GroupIdx)
		}
		require.ElementsMatch(t, want, got, "key %v", key)
	}
}

func TestWindowIndex_OffByDefault(t *testing.T) {
	pool, _, _ := newTestPool(t, 16)
	assert.Nil(t, pool.WindowIndex())
}

// A hybrid pod's sliding-window group goes to the window index; the prefix
// index holds the full-attention group only, as with the option off.
func TestWindowIndex_SlidingWindowStoreAndRemove(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newWindowTestPool(t)
	window := 128
	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 3000)

	process(ctx, pool,
		groupStore(0, KVCacheSpecKindFullAttention, nil, tokens, engineKeys, 0),
		groupStore(1, KVCacheSpecKindSlidingWindow, &window, tokens, engineKeys, 0))

	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
	require.NoError(t, err)
	require.Len(t, keys, 4)
	requireGroups(ctx, t, idx, keys, 0)
	requireGroups(ctx, t, pool.WindowIndex(), keys, 1)

	group := 1
	process(ctx, pool, &BlockRemovedEvent{BlockHashes: engineKeys, GroupIdx: &group})
	requireGroups(ctx, t, idx, keys, 0)
	requireGroups(ctx, t, pool.WindowIndex(), keys)
}

// A sliding-window store without a known window size is skipped.
func TestWindowIndex_UnknownWindowSkipped(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, tp := newWindowTestPool(t)
	tokens := makeTokens(64)
	metric := metrics.KVEventStoresSkipped.WithLabelValues(string(KVCacheSpecKindSlidingWindow), "unknown_sliding_window")
	before := counterValue(t, metric)

	process(ctx, pool, groupStore(1, KVCacheSpecKindSlidingWindow, nil, tokens, makeEngineKeys(4, 3100), 0))

	assert.Equal(t, 1.0, counterValue(t, metric)-before)
	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, pool.WindowIndex(), keys)
}

// Clearing a pod clears its window index entries too.
func TestWindowIndex_ClearedWithPod(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, tp := newWindowTestPool(t)
	window := 128
	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 3300)

	process(ctx, pool,
		groupStore(0, KVCacheSpecKindFullAttention, nil, tokens, engineKeys, 0),
		groupStore(1, KVCacheSpecKindSlidingWindow, &window, tokens, engineKeys, 0))
	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, pool.WindowIndex(), keys, 1)

	process(ctx, pool, &AllBlocksClearedEvent{})
	requireGroups(ctx, t, pool.WindowIndex(), keys)
}

// A store that leaves out the window size keeps the size already learned for
// its group: the group's removals still evict from the window index, and the
// store itself is indexed there.
func TestWindowIndex_StoreWithoutWindowSize(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newWindowTestPool(t)
	window := 128
	tokens := makeTokens(128)
	first, second := makeEngineKeys(4, 3400), makeEngineKeys(4, 3404)

	process(ctx, pool,
		groupStore(0, KVCacheSpecKindFullAttention, nil, tokens[:64], first, 0),
		groupStore(1, KVCacheSpecKindSlidingWindow, &window, tokens[:64], first, 0))
	firstKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens[:64], windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, pool.WindowIndex(), firstKeys, 1)

	group := 1
	process(ctx, pool,
		groupStore(0, KVCacheSpecKindFullAttention, nil, tokens[64:], second, first[3]),
		groupStore(1, KVCacheSpecKindSlidingWindow, nil, tokens[64:], second, first[3]),
		&BlockRemovedEvent{BlockHashes: first, GroupIdx: &group})

	secondKeys, err := tp.TokensToKVBlockKeys(firstKeys[3], tokens[64:], windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, pool.WindowIndex(), firstKeys)
	requireGroups(ctx, t, pool.WindowIndex(), secondKeys, 1)
	requireGroups(ctx, t, idx, append(firstKeys, secondKeys...), 0)
}

// Removals of a sliding-window group go to the window index by the group's
// kind, even when the catalog has no window size for it.
func TestWindowIndex_RemovalRoutedByKind(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, tp := newWindowTestPool(t)
	window := 128
	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 3500)

	process(ctx, pool,
		groupStore(0, KVCacheSpecKindFullAttention, nil, tokens, engineKeys, 0),
		groupStore(1, KVCacheSpecKindSlidingWindow, &window, tokens, engineKeys, 0))
	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, pool.WindowIndex(), keys, 1)

	pool.GroupCatalog().Learn(windowTestPod, 1, kvblock.GroupMetadata{
		Kind: string(KVCacheSpecKindSlidingWindow), BlockSize: 16,
	})
	group := 1
	process(ctx, pool, &BlockRemovedEvent{BlockHashes: engineKeys, GroupIdx: &group})
	requireGroups(ctx, t, pool.WindowIndex(), keys)
}

// vLLM publishes a step's sliding-window store before its full-attention
// store, and the sliding-window store holds only the blocks inside the window:
// for a new prompt, one hash over the whole span with no parent; after a NIXL
// transfer or a CPU restore, the last block alone.
func TestWindowIndex_StoreBeforeFullAttention(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	window := 16
	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 3600)
	for _, tc := range []struct {
		name   string
		tokens []uint32
		parent uint64
	}{
		{"new prompt", tokens, 0},
		{"last block", tokens[48:], engineKeys[2]},
		{"last block without parent hash", tokens[48:], 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool, idx, tp := newWindowTestPool(t)
			process(ctx, pool,
				groupStore(0, KVCacheSpecKindSlidingWindow, &window, tc.tokens, engineKeys[3:], tc.parent),
				groupStore(1, KVCacheSpecKindFullAttention, nil, tokens, engineKeys, 0))

			keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
			require.NoError(t, err)
			requireGroups(ctx, t, idx, keys, 1)
			requireGroups(ctx, t, pool.WindowIndex(), keys[:3])
			requireGroups(ctx, t, pool.WindowIndex(), keys[3:], 0)
		})
	}
}

// A sliding-window removal later in the batch applies after the store.
func TestWindowIndex_RemovalLaterInBatch(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, tp := newWindowTestPool(t)
	window := 16
	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 3700)
	group := 0

	process(ctx, pool,
		groupStore(0, KVCacheSpecKindSlidingWindow, &window, tokens[48:], engineKeys[3:], engineKeys[2]),
		groupStore(1, KVCacheSpecKindFullAttention, nil, tokens, engineKeys, 0),
		&BlockRemovedEvent{BlockHashes: engineKeys[3:], GroupIdx: &group})

	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, pool.WindowIndex(), keys)
}

// An AllBlocksCleared later in the batch drops the sliding-window store before
// it.
func TestWindowIndex_ClearLaterInBatch(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newWindowTestPool(t)
	window := 16
	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 3800)

	process(ctx, pool,
		groupStore(0, KVCacheSpecKindSlidingWindow, &window, tokens[48:], engineKeys[3:], engineKeys[2]),
		&AllBlocksClearedEvent{},
		groupStore(1, KVCacheSpecKindFullAttention, nil, tokens, engineKeys, 0))

	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, idx, keys, 1)
	requireGroups(ctx, t, pool.WindowIndex(), keys)
}

// A sliding-window store whose blocks the prefix index does not map is skipped.
func TestWindowIndex_UnresolvedStoreSkipped(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, tp := newWindowTestPool(t)
	window := 16
	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 3900)
	metric := metrics.KVEventStoresSkipped.WithLabelValues(string(KVCacheSpecKindSlidingWindow), "unresolved_window_blocks")
	before := counterValue(t, metric)

	process(ctx, pool, groupStore(0, KVCacheSpecKindSlidingWindow, &window, tokens[48:], engineKeys[3:], engineKeys[2]))

	assert.Equal(t, 1.0, counterValue(t, metric)-before)
	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, pool.WindowIndex(), keys)
}

func TestWindowIndex_ConfiguredSize(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 16, HashSeed: "test"})
	require.NoError(t, err)
	cfg := DefaultConfig()
	cfg.IndexSlidingWindowGroups = true
	cfg.WindowIndexConfig = &kvblock.InMemoryIndexConfig{Size: 2}
	pool, err := NewPool(cfg, idx, tp, nil)
	require.NoError(t, err)
	window := 128
	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 4000)

	process(ctx, pool,
		groupStore(0, KVCacheSpecKindFullAttention, nil, tokens, engineKeys, 0),
		groupStore(1, KVCacheSpecKindSlidingWindow, &window, tokens, engineKeys, 0))

	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, windowTestModel, nil)
	require.NoError(t, err)
	requireGroups(ctx, t, idx, keys, 0)
	requireGroups(ctx, t, pool.WindowIndex(), keys[:2])
	requireGroups(ctx, t, pool.WindowIndex(), keys[2:], 1)
}
