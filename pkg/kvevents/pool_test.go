package kvevents //nolint:testpackage // tests use unexported processEventBatch

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// newTestPool creates a Pool with real InMemoryIndex and
// ChunkedTokenDatabase. blockSize (blockSizeTokens) is the canonical block size used by the
// TokenProcessor; engine block sizes are derived per-event from the ratio of
// tokens to engine keys.
func newTestPool(t *testing.T, blockSize int) (
	*Pool, kvblock.Index, kvblock.TokenProcessor,
) {
	t.Helper()

	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: blockSize,
		HashSeed:        "test",
	})
	require.NoError(t, err)

	cfg := DefaultConfig()
	pool := NewPool(cfg, idx, tp, nil)
	return pool, idx, tp
}

// makeTokens creates a token slice [1, 2, ..., n].
func makeTokens(n int) []uint32 {
	tokens := make([]uint32, n)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i is small
	}
	return tokens
}

// makeEngineKeys creates engine key slice [base, base+1, ..., base+n-1].
func makeEngineKeys(n int, base uint64) []uint64 {
	keys := make([]uint64, n)
	for i := range keys {
		keys[i] = base + uint64(i) // #nosec G115 -- test data, i is small
	}
	return keys
}

// TestCanonicalWritePath_FallbackLegacy verifies that when BlockSize equals
// the engine block size, the pool takes the 1:1 path: engine keys are passed
// directly to Index.Add with 1:1 mapping.
func TestCanonicalWritePath_FallbackLegacy(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, _ := newTestPool(t, 16) // BlockSize == engine block size -> 1:1 path

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 500) // 4 keys, engine block size 16

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-legacy", "test-model")

	// Verify engine->request mapping exists in the Index (legacy 1:1 path)
	for _, ek := range engineKeys {
		reqKey, err := idx.GetRequestKey(ctx, kvblock.BlockHash(ek))
		require.NoError(t, err, "engine key %d should be resolvable via index", ek)
		assert.NotEqual(t, kvblock.EmptyBlockHash, reqKey)
	}
}

// TestCanonicalWritePath_ManyToOne verifies the many:1 mapping when engine block size (16)
// is smaller than canonical (64): 4 engine keys map to 1 canonical request key.
func TestCanonicalWritePath_ManyToOne(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 64)

	// 128 tokens, 8 engine keys -> engine block size 16
	// canonical block size = 64 -> 2 full canonical keys
	// 4 engine keys per canonical key (many:1)
	tokens := makeTokens(128)
	engineKeys := makeEngineKeys(8, 100)

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-a", "test-model")

	// Compute expected canonical keys independently
	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 2)

	// Verify both canonical keys are in the index with pod-a
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "canonical key should have exactly one pod")
		assert.Equal(t, "pod-a", result[ck][0].PodIdentifier)
	}

	// Verify engine keys are resolvable via the Index
	// Engine keys 0-3 should resolve to canonical[0], 4-7 to canonical[1]
	for i, ek := range engineKeys {
		reqKey, err := idx.GetRequestKey(ctx, kvblock.BlockHash(ek))
		require.NoError(t, err, "engine key %d should be in index", ek)
		expectedCanonical := canonicalKeys[i/4]
		assert.Equal(t, expectedCanonical, reqKey,
			"engine key %d should resolve to canonical key %d", i, i/4)
	}
}

// TestCanonicalWritePath_OneToMany verifies the 1:many mapping when engine block size (128)
// is larger than canonical (64): 1 engine key maps to 2 canonical request keys.
func TestCanonicalWritePath_OneToMany(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 64)

	// 256 tokens, 2 engine keys -> engine block size 128
	// canonical block size = 64 -> 4 full canonical keys
	// Each engine key covers two canonical keys (1:many)
	tokens := makeTokens(256)
	engineKeys := makeEngineKeys(2, 200)

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-b", "test-model")

	// Compute expected canonical keys independently
	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// Verify all 4 canonical keys are in the index with pod-b
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "canonical key should have exactly one pod")
		assert.Equal(t, "pod-b", result[ck][0].PodIdentifier)
	}

	// Verify engine key 0 resolves to canonical[1] (last of its mapped keys)
	reqKey0, err := idx.GetRequestKey(ctx, kvblock.BlockHash(engineKeys[0]))
	require.NoError(t, err)
	assert.Equal(t, canonicalKeys[1], reqKey0, "engine key 0 should resolve to its last mapped canonical key")

	// Verify engine key 1 resolves to canonical[3]
	reqKey1, err := idx.GetRequestKey(ctx, kvblock.BlockHash(engineKeys[1]))
	require.NoError(t, err)
	assert.Equal(t, canonicalKeys[3], reqKey1, "engine key 1 should resolve to its last mapped canonical key")

	// Verify evicting engine key 0 removes canonical keys 0 and 1
	removeBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: []uint64{engineKeys[0]},
			},
		},
	}
	pool.processEventBatch(ctx, removeBatch, "pod-b", "test-model")

	for _, ck := range canonicalKeys[:2] {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Empty(t, result[ck], "canonical key mapped to evicted engine key should be gone")
	}

	// Canonical keys 2 and 3 should still be present
	for _, ck := range canonicalKeys[2:] {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Len(t, result[ck], 1, "canonical keys mapped to non-evicted engine key should remain")
	}
}

// TestCanonicalEviction_Eager verifies eager eviction: removing one engine key evicts its
// mapped canonical key from the index while leaving unrelated canonical keys intact.
func TestCanonicalEviction_Eager(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 64)

	// 128 tokens, 8 engine keys -> engine block size 16
	// canonical block size = 64 -> 2 full canonical keys
	// 4 engine keys per canonical key (many:1)
	tokens := makeTokens(128)
	engineKeys := makeEngineKeys(8, 100)

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-a", "test-model")

	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 2)

	// Evict engineKey 0 which maps to canonical key 0
	removeBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: []uint64{engineKeys[0]},
			},
		},
	}
	pool.processEventBatch(ctx, removeBatch, "pod-a", "test-model")

	// Verify canonical key 0 is evicted
	result0, err := idx.Lookup(ctx, []kvblock.BlockHash{canonicalKeys[0]}, nil)
	require.NoError(t, err)
	assert.Empty(t, result0[canonicalKeys[0]], "canonical key 0 should be evicted after engine key 0 removal")

	// Verify canonical key 1 still present
	result1, err := idx.Lookup(ctx, []kvblock.BlockHash{canonicalKeys[1]}, nil)
	require.NoError(t, err)
	assert.Len(t, result1[canonicalKeys[1]], 1, "canonical key 1 should still have pod-a")

	// Verify engine key 0 is no longer resolvable
	_, err = idx.GetRequestKey(ctx, kvblock.BlockHash(engineKeys[0]))
	assert.Error(t, err, "evicted engine key should not be resolvable")
}

// TestCanonicalWritePath_CrossEngineScoring verifies that two engines with different block sizes
// (16 and 32) storing the same tokens produce identical canonical keys, so both pods appear in lookups.
func TestCanonicalWritePath_CrossEngineScoring(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 64)

	tokens := makeTokens(128)

	// Engine A: block size 16, 8 engine keys
	batchA := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: makeEngineKeys(8, 100),
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batchA, "pod-a", "test-model")

	// Engine B: block size 32, 4 engine keys
	batchB := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: makeEngineKeys(4, 200),
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batchB, "pod-b", "test-model")

	// Both produce the same 2 canonical keys
	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 2)

	// Both pods should appear under each canonical key
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		pods := result[ck]
		require.Len(t, pods, 2, "both pods should be present")

		podIDs := map[string]bool{}
		for _, p := range pods {
			podIDs[p.PodIdentifier] = true
		}
		assert.True(t, podIDs["pod-a"], "pod-a should be present")
		assert.True(t, podIDs["pod-b"], "pod-b should be present")
	}
}

// TestCanonicalEviction_UnknownEngineKey verifies that evicting an engine key not in the
// Index is a no-op — no panic, no error.
func TestCanonicalEviction_UnknownEngineKey(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, _ := newTestPool(t, 64)

	// Evict an engine key that was never stored
	removeBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: []uint64{999999},
			},
		},
	}

	// Should not panic or error, just skip
	assert.NotPanics(t, func() {
		pool.processEventBatch(ctx, removeBatch, "pod-x", "test-model")
	})
}

// TestRealignExtraFeatures verifies that engine-granularity extraFeatures are
// correctly converted to canonical-block granularity.
func TestRealignExtraFeatures(t *testing.T) {
	t.Run("1:1 passthrough", func(t *testing.T) {
		features := []*kvblock.BlockExtraFeatures{nil, nil, nil, nil}
		result := realignExtraFeatures(features, 4)
		assert.Equal(t, features, result)
	})

	t.Run("1:many replication (all nil, text-only)", func(t *testing.T) {
		// 2 engine blocks → 8 canonical blocks (ratio 4)
		features := []*kvblock.BlockExtraFeatures{nil, nil}
		result := realignExtraFeatures(features, 8)
		require.Len(t, result, 8)
		for _, f := range result {
			assert.Nil(t, f)
		}
	})

	t.Run("1:many replication (with MM features)", func(t *testing.T) {
		// 2 engine blocks → 4 canonical blocks (ratio 2)
		feat0 := &kvblock.BlockExtraFeatures{MMHashes: []kvblock.MMHash{{Hash: "img0"}}}
		features := []*kvblock.BlockExtraFeatures{feat0, nil}
		result := realignExtraFeatures(features, 4)
		require.Len(t, result, 4)
		// Engine block 0 → canonical blocks 0, 1
		assert.Equal(t, feat0, result[0])
		assert.Equal(t, feat0, result[1])
		// Engine block 1 → canonical blocks 2, 3
		assert.Nil(t, result[2])
		assert.Nil(t, result[3])
	})

	t.Run("many:1 merge (all nil, text-only)", func(t *testing.T) {
		// 8 engine blocks → 2 canonical blocks (ratio 4)
		features := make([]*kvblock.BlockExtraFeatures, 8)
		result := realignExtraFeatures(features, 2)
		require.Len(t, result, 2)
		for _, f := range result {
			assert.Nil(t, f)
		}
	})

	t.Run("many:1 merge (with MM features)", func(t *testing.T) {
		// 4 engine blocks → 2 canonical blocks (ratio 2)
		// Engine blocks 0,1 → canonical 0; engine blocks 2,3 → canonical 1
		features := []*kvblock.BlockExtraFeatures{
			{MMHashes: []kvblock.MMHash{{Hash: "a"}}},
			{MMHashes: []kvblock.MMHash{{Hash: "b"}}},
			nil,
			{MMHashes: []kvblock.MMHash{{Hash: "c"}}},
		}
		result := realignExtraFeatures(features, 2)
		require.Len(t, result, 2)
		// Canonical 0 should merge features from engine blocks 0 and 1
		require.NotNil(t, result[0])
		assert.Len(t, result[0].MMHashes, 2)
		assert.Equal(t, "a", result[0].MMHashes[0].Hash)
		assert.Equal(t, "b", result[0].MMHashes[1].Hash)
		// Canonical 1 should have features from engine block 3 only (block 2 is nil)
		require.NotNil(t, result[1])
		assert.Len(t, result[1].MMHashes, 1)
		assert.Equal(t, "c", result[1].MMHashes[0].Hash)
	})

	t.Run("zero canonical blocks (engine BS < canonical BS)", func(t *testing.T) {
		// 1 engine block → 0 canonical blocks: tokens < canonical block size.
		// realignExtraFeatures returns an empty slice instead of panicking.
		features := []*kvblock.BlockExtraFeatures{
			{MMHashes: []kvblock.MMHash{{Hash: "img0"}}},
		}
		result := realignExtraFeatures(features, 0)
		assert.Empty(t, result)
	})
}

// TestCanonicalWritePath_ExtraKeysOneToMany verifies that events with ExtraKeys
// are correctly processed in the 1:many path (engine BS > canonical BS).
func TestCanonicalWritePath_ExtraKeysOneToMany(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16) // canonical BS = 16

	// 128 tokens, 2 engine keys → engine BS = 64
	// canonical BS = 16 → 8 canonical keys
	// 1:many ratio = 4
	tokens := makeTokens(128)
	engineKeys := makeEngineKeys(2, 300)

	// ExtraKeys: 2 entries (one per engine block), all nil content (text-only)
	extraKeys := make([][]any, 2)

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
				ExtraKeys:   extraKeys,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-extra", "test-model")

	// Compute expected canonical keys (no extra features for text-only)
	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 8)

	// All 8 canonical keys should be present with pod-extra
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "canonical key should have pod-extra")
		assert.Equal(t, "pod-extra", result[ck][0].PodIdentifier)
	}
}

// TestCanonicalWritePath_ExtraKeysManyToOne verifies that events with ExtraKeys
// are correctly processed in the many:1 path (engine BS < canonical BS).
func TestCanonicalWritePath_ExtraKeysManyToOne(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 64) // canonical BS = 64

	// 128 tokens, 8 engine keys → engine BS = 16
	// canonical BS = 64 → 2 canonical keys
	// many:1 ratio = 4
	tokens := makeTokens(128)
	engineKeys := makeEngineKeys(8, 400)

	// ExtraKeys: 8 entries (one per engine block), all nil content (text-only)
	extraKeys := make([][]any, 8)

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
				ExtraKeys:   extraKeys,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-extra-m1", "test-model")

	// Compute expected canonical keys (no extra features for text-only)
	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 2)

	// Both canonical keys should be present with pod-extra-m1
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "canonical key should have pod-extra-m1")
		assert.Equal(t, "pod-extra-m1", result[ck][0].PodIdentifier)
	}
}

// TestBlockStoredEvent_OffloadingEmptyTokens verifies that an offloading event
// (empty Tokens, non-empty BlockHashes, DeviceTier="CPU") correctly updates
// existing index entries with the new device tier rather than being dropped.
func TestBlockStoredEvent_OffloadingEmptyTokens(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 600)

	// Step 1: Store blocks with full tokens (simulates initial GPU event).
	gpuBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, gpuBatch, "pod-a", "test-model")

	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// Verify GPU entry exists.
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1)
		assert.Equal(t, "gpu", result[ck][0].DeviceTier)
	}

	// Step 2: Process offloading event — same engine keys, empty tokens, CPU tier.
	cpuBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      nil,
				ParentHash:  0,
				DeviceTier:  "CPU",
			},
		},
	}
	pool.processEventBatch(ctx, cpuBatch, "pod-a", "test-model")

	// Verify both GPU and CPU entries now exist for each canonical key.
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 2, "should have both gpu and cpu entries")

		tiers := map[string]bool{}
		for _, pe := range result[ck] {
			tiers[pe.DeviceTier] = true
		}
		assert.True(t, tiers["gpu"], "gpu entry should be present")
		assert.True(t, tiers["cpu"], "cpu entry should be present")
	}
}

// TestBlockStoredEvent_OffloadingUnknownEngineKeys verifies that an offloading
// event with engine keys not yet in the index is a graceful no-op.
func TestBlockStoredEvent_OffloadingUnknownEngineKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, _ := newTestPool(t, 16)

	// Offloading event for engine keys that were never stored.
	cpuBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: makeEngineKeys(4, 900),
				Tokens:      nil,
				ParentHash:  0,
				DeviceTier:  "CPU",
			},
		},
	}

	assert.NotPanics(t, func() {
		pool.processEventBatch(ctx, cpuBatch, "pod-x", "test-model")
	})
}

// TestBlockStoredEvent_EvictionOrderGPUThenCPU verifies the full lifecycle:
// GPU store → CPU offload → GPU evict → CPU entry survives → CPU evict → full cleanup.
func TestBlockStoredEvent_EvictionOrderGPUThenCPU(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 700)

	// Step 1: Store blocks on GPU.
	gpuBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, gpuBatch, "pod-a", "test-model")

	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// Step 2: Offload to CPU.
	cpuBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      nil,
				ParentHash:  0,
				DeviceTier:  "CPU",
			},
		},
	}
	pool.processEventBatch(ctx, cpuBatch, "pod-a", "test-model")

	// Verify both tiers present.
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 2)
	}

	// Step 3: Evict from GPU.
	gpuEvict := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: engineKeys,
			},
		},
	}
	pool.processEventBatch(ctx, gpuEvict, "pod-a", "test-model")

	// CPU entries must survive, engine→request mapping must be preserved.
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "cpu entry should survive gpu eviction")
		assert.Equal(t, "cpu", result[ck][0].DeviceTier)
	}
	// Engine→request mapping must still resolve.
	_, err = idx.GetRequestKey(ctx, kvblock.BlockHash(engineKeys[0]))
	require.NoError(t, err, "engine→request mapping should survive gpu eviction")

	// Step 4: Evict from CPU.
	cpuEvict := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: engineKeys,
				DeviceTier:  "CPU",
			},
		},
	}
	pool.processEventBatch(ctx, cpuEvict, "pod-a", "test-model")

	// Everything should be fully cleaned up.
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Empty(t, result[ck], "all entries should be gone after full eviction")
	}
	// Engine→request mapping should be gone.
	_, err = idx.GetRequestKey(ctx, kvblock.BlockHash(engineKeys[0]))
	assert.Error(t, err, "engine→request mapping should be removed after full eviction")
}

func TestHMAGroupMetadataAndEntryOnBlockStored(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 800)
	groupIdx := 0
	slidingWindow := 128

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes:                  engineKeys,
				Tokens:                       tokens,
				ParentHash:                   0,
				GroupIdx:                     &groupIdx,
				KVCacheSpecKind:              KVCacheSpecKindSlidingWindow,
				KVCacheSpecSlidingWindowSize: &slidingWindow,
				BlockSize:                    16,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-hma", "test-model")

	meta, ok := pool.GroupCatalog().Get("pod-hma", kvblock.GroupID(0))
	require.True(t, ok)
	assert.Equal(t, string(KVCacheSpecKindSlidingWindow), meta.Kind)
	assert.Equal(t, 16, meta.BlockSize)
	require.NotNil(t, meta.SlidingWindowSize)
	assert.Equal(t, 128, *meta.SlidingWindowSize)

	canonicalKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.NotEmpty(t, canonicalKeys)

	result, err := idx.Lookup(ctx, canonicalKeys, nil)
	require.NoError(t, err)
	for _, ck := range canonicalKeys {
		entries := result[ck]
		require.Len(t, entries, 1, "each canonical key should have one entry")
		assert.True(t, entries[0].HasGroup)
		assert.Equal(t, kvblock.GroupID(0), entries[0].GroupIdx)
	}
}

// TestHMAGroupLevelEviction_BlockRemoved verifies that a BlockRemoved event with GroupIdx
// performs a group-level eviction, leaving other groups intact.
func TestHMAGroupLevelEviction_BlockRemoved(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 850)

	// Store two groups for the same block (simulates two BlockStored events)
	for _, g := range []int{0, 1} {
		gIdx := g
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes:     engineKeys,
					Tokens:          tokens,
					ParentHash:      0,
					GroupIdx:        &gIdx,
					KVCacheSpecKind: KVCacheSpecKindFullAttention,
					BlockSize:       16,
				},
			},
		}
		pool.processEventBatch(ctx, batch, "pod-hma", "test-model")
	}

	canonicalKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)

	// Verify both groups present
	result, err := idx.Lookup(ctx, canonicalKeys, nil)
	require.NoError(t, err)
	for _, ck := range canonicalKeys {
		entries := result[ck]
		require.Len(t, entries, 2)
		assert.ElementsMatch(t, []kvblock.GroupID{0, 1}, []kvblock.GroupID{entries[0].GroupIdx, entries[1].GroupIdx})
	}

	// Evict group 0 only
	evictGroupIdx := 0
	removeBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: engineKeys,
				GroupIdx:    &evictGroupIdx,
			},
		},
	}
	pool.processEventBatch(ctx, removeBatch, "pod-hma", "test-model")

	// Group 1 should remain; group 0 should be gone
	result, err = idx.Lookup(ctx, canonicalKeys, nil)
	require.NoError(t, err)
	for _, ck := range canonicalKeys {
		entries := result[ck]
		require.Len(t, entries, 1, "pod should still be present after partial eviction")
		assert.True(t, entries[0].HasGroup)
		assert.Equal(t, kvblock.GroupID(1), entries[0].GroupIdx)
	}
}

// TestCanonicalWritePath_PartialBlockDrop verifies that tokens fewer than the canonical block
// size produce zero canonical keys and the event is silently skipped.
func TestCanonicalWritePath_PartialBlockDrop(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, _ := newTestPool(t, 64)

	// 48 tokens < canonical block size (64), so 0 canonical keys
	tokens := makeTokens(48)
	engineKeys := makeEngineKeys(3, 400) // 3 keys -> engine block size 16

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-partial", "test-model")

	// Verify nothing was added to the index
	result, err := idx.Lookup(ctx, []kvblock.BlockHash{kvblock.BlockHash(1)}, nil)
	require.NoError(t, err)
	assert.Empty(t, result[kvblock.BlockHash(1)])
}
