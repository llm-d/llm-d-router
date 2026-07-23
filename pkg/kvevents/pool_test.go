package kvevents //nolint:testpackage // tests use unexported processEventBatch

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
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

// TestAllBlocksCleared_Dispatch verifies the pool wires AllBlocksCleared to
// Index.Clear: the event drops every entry for the emitting pod and leaves
// other pods untouched.
func TestAllBlocksCleared_Dispatch(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	// Same tokens from two pods -> both pods land on the same canonical keys.
	tokens := makeTokens(64)
	storeBatch := func(base uint64) *EventBatch {
		return &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: makeEngineKeys(4, base),
					Tokens:      tokens,
					ParentHash:  0,
				},
			},
		}
	}
	pool.processEventBatch(ctx, storeBatch(500), "pod-cleared", "test-model")
	pool.processEventBatch(ctx, storeBatch(900), "pod-kept", "test-model")

	canonicalKeys, err := tp.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.NotEmpty(t, canonicalKeys)

	clearBatch := &EventBatch{Events: []GenericEvent{&AllBlocksClearedEvent{}}}
	pool.processEventBatch(ctx, clearBatch, "pod-cleared", "test-model")

	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "only the surviving pod should remain on key %s", ck)
		assert.Equal(t, "pod-kept", result[ck][0].PodIdentifier)
	}
}

// TestPool_AllBlocksClearedResetsDedup verifies the filter is reset on
// AllBlocksCleared, so a post-clear store/remove cycle behaves freshly rather
// than carrying a stale reference that would suppress the remove. This is the
// regression guard for p.dedup.clear(); TestAllBlocksCleared_Dispatch only
// proves Index.Clear ran, not that the refcount state was reset.
func TestPool_AllBlocksClearedResetsDedup(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 850)

	storeTwice := func() {
		for range 2 {
			pool.processEventBatch(ctx, &EventBatch{
				Events: []GenericEvent{
					&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
				},
			}, "pod-clr", "test-model")
		}
	}

	// Two stores -> reference count 2 -> would normally need two removes.
	storeTwice()

	// Clear wipes both the index and the dedup counts for the pod.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{&AllBlocksClearedEvent{}},
	}, "pod-clr", "test-model")

	// Re-establish a single reference after the clear.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
		},
	}, "pod-clr", "test-model")

	canonicalKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)

	// A single remove must now fully evict: if the pre-clear count of 2 had
	// survived, this remove would be suppressed and the blocks would linger.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{&BlockRemovedEvent{BlockHashes: engineKeys}},
	}, "pod-clr", "test-model")

	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Empty(t, result[ck], "single remove after clear must evict (dedup count was reset)")
	}
}

// TestPool_DuplicateStoreSurvivesFirstRemove is the end-to-end proof through
// processEventBatch with a real index: two overlapping chunks announce the same
// blocks, so the first BlockRemoved must not evict them and the second must.
func TestPool_DuplicateStoreSurvivesFirstRemove(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 800)

	store := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
			},
		}, "pod-dup", "test-model")
	}
	remove := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{BlockHashes: engineKeys},
			},
		}, "pod-dup", "test-model")
	}

	store() // overlapping chunk A announces these constituent hashes
	store() // overlapping chunk B re-announces the same hashes

	canonicalKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// First remove (chunk A evicted) must NOT drop the blocks.
	remove()
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "block must survive the first of two duplicate removes")
	}

	// Second remove (chunk B evicted) drops them.
	remove()
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Empty(t, result[ck], "block must be evicted after the second duplicate remove")
	}

	// Engine->request mapping should also be gone after full eviction.
	_, err = idx.GetRequestKey(ctx, kvblock.BlockHash(engineKeys[0]))
	assert.Error(t, err, "engine->request mapping should be removed after the second remove")
}

// TestPool_DuplicateCPUOffloadRemovalSurvivesFirstRemove exercises the
// offloading wiring end to end through processEventBatch: a CPU tier is
// established via the empty-token device-tier update path, re-announced by two
// overlapping chunks, and must require two CPU removes before the CPU entry is
// evicted — while the independently reference-counted GPU entry (never removed)
// is untouched throughout. This jointly exercises handleDeviceTierUpdate, the
// deviceTier normalization on both store and remove, and the dedup scope.
func TestPool_DuplicateCPUOffloadRemovalSurvivesFirstRemove(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 900)

	// Step 1: GPU store with tokens establishes the engine->request mapping
	// that the empty-token CPU offload path resolves against.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
		},
	}, "pod-offload", "test-model")

	// Step 2: the same CPU offload (empty tokens, "CPU" tier) announced twice,
	// as two overlapping chunks would re-announce the shared constituent hashes.
	cpuStore := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{BlockHashes: engineKeys, Tokens: nil, ParentHash: 0, DeviceTier: "CPU"},
			},
		}, "pod-offload", "test-model")
	}
	cpuStore()
	cpuStore()

	canonicalKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// Both gpu and cpu entries should now exist for each canonical key.
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 2, "gpu and cpu entries should both exist after offload")
	}

	cpuRemove := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{BlockHashes: engineKeys, DeviceTier: "CPU"},
			},
		}, "pod-offload", "test-model")
	}

	// Step 3: the first CPU remove (one overlapping chunk evicted) must be
	// suppressed — the cpu entry still has an outstanding reference.
	cpuRemove()
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 2, "cpu entry must survive the first of two duplicate CPU removes")
		tiers := map[string]bool{}
		for _, pe := range result[ck] {
			tiers[pe.DeviceTier] = true
		}
		assert.True(t, tiers["cpu"], "cpu entry should still be present after the first CPU remove")
		assert.True(t, tiers["gpu"], "gpu entry should be untouched")
	}

	// Step 4: the second CPU remove releases the last reference and evicts the
	// cpu entry; the gpu entry (never removed) must remain.
	cpuRemove()
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "only the gpu entry should remain after the second CPU remove")
		assert.Equal(t, "gpu", result[ck][0].DeviceTier, "surviving entry must be the untouched gpu copy")
	}
}

// TestPool_DeviceTierUpdateWithNoResolvedKeysDoesNotTrack pins the contract of
// handleDeviceTierUpdate's bool return: a CPU offload for engine keys that were
// never stored resolves nothing, so the store must NOT be reference-counted.
// A subsequent remove then passes straight through (forwarded, not suppressed).
func TestPool_DeviceTierUpdateWithNoResolvedKeysDoesNotTrack(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, _ := newTestPool(t, 16)

	engineKeys := makeEngineKeys(4, 950) // never stored -> nothing resolves

	// CPU offload (empty tokens) for unknown engine keys: handleDeviceTierUpdate
	// returns false, so trackStore must not be called.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{BlockHashes: engineKeys, Tokens: nil, ParentHash: 0, DeviceTier: "CPU"},
		},
	}, "pod-noresolve", "test-model")

	// The dedup filter must hold no references for these hashes, so a remove on
	// the same scope passes through unchanged (nothing was tracked to suppress).
	scope := blockScope{
		podIdentifier:    "pod-noresolve",
		deviceTier:       "cpu",
		groupIdx:         noGroupIdx,
		dataParallelRank: noDataParallelRank,
	}
	kept := pool.dedup.filterRemove(scope, engineKeys)
	assert.Equal(t, engineKeys, kept,
		"a device-tier update that resolved no keys must not be tracked; the remove passes through")
}

// TestPool_DedupMetricsCountBlockHashes verifies the dedup counters record the
// number of constituent block hashes (not BlockRemoved events): two stores
// establish refcount 2 over 4 hashes, so the first remove suppresses all 4 and
// the second forwards all 4. The counters are process-wide globals, so the
// assertions use deltas from a captured baseline.
func TestPool_DedupMetricsCountBlockHashes(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, _ := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 990)

	store := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
			},
		}, "pod-metrics", "test-model")
	}
	remove := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{BlockHashes: engineKeys},
			},
		}, "pod-metrics", "test-model")
	}

	suppressedBefore := counterValue(t, metrics.DedupRemovedHashesSuppressed)
	forwardedBefore := counterValue(t, metrics.DedupRemovedHashesForwarded)

	store()  // overlapping chunk A
	store()  // overlapping chunk B -> refcount 2 for each of the 4 hashes
	remove() // first remove: all 4 hashes suppressed (2 -> 1)
	remove() // second remove: all 4 hashes forwarded (1 -> 0)

	assert.Equal(t, 4.0, counterValue(t, metrics.DedupRemovedHashesSuppressed)-suppressedBefore,
		"first of two duplicate removes must suppress all 4 constituent block hashes")
	assert.Equal(t, 4.0, counterValue(t, metrics.DedupRemovedHashesForwarded)-forwardedBefore,
		"second remove must forward all 4 constituent block hashes")
}

// stubAdapter is a minimal EngineAdapter that shards every message onto the
// same key and decodes to an empty batch, so tasks flow through the pool
// without exercising any engine-specific parsing.
type stubAdapter struct{}

//nolint:gocritic // unnamed results match the EngineAdapter implementations
func (stubAdapter) ParseMessage(_ *RawMessage) (string, string, EventBatch, error) {
	return "pod-1", "model-1", EventBatch{}, nil
}

func (stubAdapter) ShardingKey(_ *RawMessage) string { return "pod-1" }

// TestPool_QueueDepthAccounting verifies that the queue depth gauge tracks
// enqueues and dequeues, and is reset once the pool shuts down.
func TestPool_QueueDepthAccounting(t *testing.T) {
	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: 4, HashSeed: "test",
	})
	require.NoError(t, err)

	cfg := DefaultConfig()
	cfg.Concurrency = 2
	pool := NewPool(cfg, idx, tp, stubAdapter{})

	const tasks = 3
	for i := range uint64(tasks) {
		pool.AddTask(&RawMessage{Topic: "kv@pod-1@model-1", Sequence: i})
	}

	assert.Equal(t, int64(tasks), pool.queueDepth.Load(), "every enqueued task must be counted")
	assert.InDelta(t, float64(tasks), gaugeValue(t, metrics.PoolQueueDepth), 0.001)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pool.Start(ctx)

	assert.Eventually(t, func() bool {
		return pool.queueDepth.Load() == 0
	}, 5*time.Second, 10*time.Millisecond, "workers must decrement the depth as tasks drain")

	pool.Shutdown(ctx)
	assert.Equal(t, int64(0), pool.queueDepth.Load())
	assert.InDelta(t, 0.0, gaugeValue(t, metrics.PoolQueueDepth), 0.001)
	assert.InDelta(t, float64(cfg.Concurrency), gaugeValue(t, metrics.PoolCapacity), 0.001)
}

// gaugeValue reads the current value of a prometheus.Gauge without touching the
// global registry.
func gaugeValue(t *testing.T, g prometheus.Gauge) float64 {
	t.Helper()
	var m dto.Metric
	require.NoError(t, g.Write(&m))
	return m.GetGauge().GetValue()
}

// counterValue reads the current value of a plain prometheus.Counter without
// touching the global registry, using the same dto.Metric.Write pattern as
// pkg/kvcache/metrics.logMetrics.
func counterValue(t *testing.T, c prometheus.Counter) float64 {
	t.Helper()
	var m dto.Metric
	require.NoError(t, c.Write(&m))
	return m.GetCounter().GetValue()
}
