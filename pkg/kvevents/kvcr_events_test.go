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

package kvevents_test

import (
	"context"
	"maps"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

// The tests in this file feed the vLLM adapter the events vLLM publishes for a
// KV Cache Runner (KVCR) secondary tier behind OffloadingConnector with
// self_describing_kv_events, and check what the precise prefix scorer reads.
// KVCR reports its host DRAM tier as medium CPU and its file tier as medium
// STORAGE, both with ownership "kvcr".

const (
	kvcrModel     = "test-model"
	kvcrBlockSize = 16
	kvcrOwnership = "kvcr"
)

type kvcrHarness struct {
	pool    *kvevents.Pool
	indexer *kvcache.Indexer
	tp      kvblock.TokenProcessor
}

func newKVCRHarness(t *testing.T, ctx context.Context) *kvcrHarness {
	t.Helper()

	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: kvcrBlockSize,
		HashSeed:        "test",
	})
	require.NoError(t, err)
	cfg, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	indexer, err := kvcache.NewKVCacheIndexer(ctx, cfg, tp)
	require.NoError(t, err)
	poolConfig := kvevents.DefaultConfig()
	poolConfig.Concurrency = 1
	pool, err := kvevents.NewPool(poolConfig, indexer.KVBlockIndex(), tp, engineadapter.NewVLLMAdapter())
	require.NoError(t, err)
	t.Cleanup(func() { pool.Shutdown(ctx) })
	return &kvcrHarness{pool: pool, indexer: indexer, tp: tp}
}

// publish delivers one vLLM event batch from pod.
func (h *kvcrHarness) publish(t *testing.T, ctx context.Context, pod string, events ...map[string]any) {
	t.Helper()

	payload, err := msgpack.Marshal([]any{1234567890.0, events, nil})
	require.NoError(t, err)
	h.pool.ProcessRawMessageForTest(ctx, &kvevents.RawMessage{
		Topic:   "kv@" + pod + "@" + kvcrModel,
		Payload: payload,
	})
}

// matches returns the prefix match for tokens of every pod holding the first
// block.
func (h *kvcrHarness) matches(t *testing.T, ctx context.Context, tokens []uint32) map[string]kvcache.PodMatch {
	t.Helper()

	keys, err := h.tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, kvcrModel, nil)
	require.NoError(t, err)
	matches, err := h.indexer.MatchBlockKeys(ctx, keys, nil)
	require.NoError(t, err)
	return matches
}

// kvcrBlockStored is BlockStored as vLLM publishes it: a map tagged by "type"
// that carries the fields without defaults, nil when unset, and omits unset
// optional fields. A nil parent marks the first block of a sequence; an empty
// ownership is vLLM's own tier.
func kvcrBlockStored(hashes []uint64, parent any, tokens []uint32, blockSize int, medium, ownership string) map[string]any {
	ev := map[string]any{
		"type":               "BlockStored",
		"block_hashes":       hashes,
		"parent_block_hash":  parent,
		"token_ids":          tokens,
		"block_size":         blockSize,
		"lora_id":            nil,
		"medium":             medium,
		"lora_name":          nil,
		"group_idx":          0,
		"kv_cache_spec_kind": "full_attention",
	}
	if ownership != "" {
		ev["ownership"] = ownership
	}
	return ev
}

// kvcrPlaceholderStored is the BlockStored vLLM emits for an offloaded key it
// holds no request metadata for: one hash, no tokens and block_size 0.
func kvcrPlaceholderStored(hash uint64, medium string) map[string]any {
	return kvcrBlockStored([]uint64{hash}, nil, []uint32{}, 0, medium, kvcrOwnership)
}

func kvcrBlockRemoved(hashes []uint64, medium, ownership string) map[string]any {
	ev := map[string]any{"type": "BlockRemoved", "block_hashes": hashes, "medium": medium, "group_idx": 0}
	if ownership != "" {
		ev["ownership"] = ownership
	}
	return ev
}

func kvcrTokens(n int) []uint32 {
	tokens := make([]uint32, n)
	for i := range tokens {
		tokens[i] = 1 + uint32(i) //#nosec G115 -- test data is small
	}
	return tokens
}

func kvcrHashes(n int, base uint64) []uint64 {
	hashes := make([]uint64, n)
	for i := range hashes {
		hashes[i] = base + uint64(i) //#nosec G115 -- test data is small
	}
	return hashes
}

func TestKVCRInventory_LocalTiersKeepPodScoredAfterGPUEviction(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(4 * kvcrBlockSize)
	hashes := kvcrHashes(4, 100)
	h.publish(t, ctx, "pod-a",
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "GPU", ""),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "CPU", kvcrOwnership),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "STORAGE", kvcrOwnership),
	)
	// pod-b holds the same prompt on GPU only, under its own engine hashes.
	podBHashes := kvcrHashes(4, 500)
	h.publish(t, ctx, "pod-b", kvcrBlockStored(podBHashes, nil, prompt, kvcrBlockSize, "GPU", ""))

	scores, err := h.indexer.ScoreTokens(ctx, prompt, kvcrModel, []string{"pod-a", "pod-b"}, nil)
	require.NoError(t, err)
	assert.ElementsMatch(t, []string{"pod-a", "pod-b"}, slices.Collect(maps.Keys(scores)), "both pods score while the prefix is on their GPUs")

	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "GPU", ""))
	h.publish(t, ctx, "pod-b", kvcrBlockRemoved(podBHashes, "GPU", ""))

	m, ok := h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok, "KVCR tiers must keep the prefix on pod-a after GPU eviction")
	assert.Equal(t, map[string]int{"cpu": 4, "storage": 4}, m.BlocksByTier)
	assert.InDelta(t, 4*0.8, m.WeightedScore, 1e-9, "each block scores its best tier, cpu")

	scores, err = h.indexer.ScoreTokens(ctx, prompt, kvcrModel, []string{"pod-a", "pod-b"}, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{"pod-a"}, slices.Collect(maps.Keys(scores)), "after GPU eviction only the pod holding the prefix in KVCR scores")

	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "CPU", kvcrOwnership))
	m, ok = h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok)
	assert.Equal(t, map[string]int{"storage": 4}, m.BlocksByTier)
	assert.InDelta(t, 4*0.6, m.WeightedScore, 1e-9)

	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "STORAGE", kvcrOwnership))
	_, ok = h.matches(t, ctx, prompt)["pod-a"]
	assert.False(t, ok, "the prefix leaves the index with its last KVCR copy")
}

// vLLM emits a placeholder in whichever KVCR tier the store lands in.
func TestKVCRInventory_PlaceholderStoresResolveThroughEngineKeys(t *testing.T) {
	for _, tc := range []struct {
		medium string
		tier   string
	}{
		{medium: "CPU", tier: "cpu"},
		{medium: "STORAGE", tier: "storage"},
	} {
		t.Run(tc.medium, func(t *testing.T) {
			ctx := logging.NewTestLoggerIntoContext(context.Background())
			h := newKVCRHarness(t, ctx)

			prompt := kvcrTokens(4 * kvcrBlockSize)
			hashes := kvcrHashes(4, 100)
			h.publish(t, ctx, "pod-a", kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "GPU", ""))

			placeholders := make([]map[string]any, 0, len(hashes))
			for _, hash := range hashes {
				placeholders = append(placeholders, kvcrPlaceholderStored(hash, tc.medium))
			}
			h.publish(t, ctx, "pod-a", placeholders...)
			h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "GPU", ""))

			m, ok := h.matches(t, ctx, prompt)["pod-a"]
			require.True(t, ok, "placeholder stores must resolve through the GPU store's engine keys")
			assert.Equal(t, map[string]int{tc.tier: 4}, m.BlocksByTier)

			h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, tc.medium, kvcrOwnership))
			_, ok = h.matches(t, ctx, prompt)["pod-a"]
			assert.False(t, ok)
		})
	}
}

// vLLM's primary CPU tier and KVCR's DRAM both report medium CPU, so one
// owner's removal must not drop the block the other still holds. The pool's
// store reference count is keyed by pod, tier and KV-cache group and ignores
// ownership, so the two stores hold two references to one entry.
func TestKVCRInventory_CPUTierSharedWithPrimaryOffload(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(4 * kvcrBlockSize)
	hashes := kvcrHashes(4, 100)
	h.publish(t, ctx, "pod-a",
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "GPU", ""),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "CPU", ""),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "CPU", kvcrOwnership),
	)
	h.publish(t, ctx, "pod-a",
		kvcrBlockRemoved(hashes, "GPU", ""),
		kvcrBlockRemoved(hashes, "CPU", ""),
	)

	m, ok := h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok, "KVCR's copy must survive the primary CPU tier's removal")
	assert.Equal(t, map[string]int{"cpu": 4}, m.BlocksByTier)

	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "CPU", kvcrOwnership))
	_, ok = h.matches(t, ctx, prompt)["pod-a"]
	assert.False(t, ok)
}

// vLLM can place a primary-tier removal ahead of the KVCR store for the same
// keys in one batch. It keeps a block's tokens until the batch's stores have
// registered, so those stores carry tokens. Removing the last copy drops the
// engine-to-request key mapping, so the KVCR stores must index from their own
// tokens and parents.
func TestKVCRInventory_StoreAfterLastRemovalInOneBatch(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(8 * kvcrBlockSize)
	hashes := kvcrHashes(8, 100)
	h.publish(t, ctx, "pod-a",
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "GPU", ""),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "CPU", ""),
	)

	h.publish(t, ctx, "pod-a",
		kvcrBlockRemoved(hashes, "GPU", ""),
		kvcrBlockRemoved(hashes, "CPU", ""),
		kvcrBlockStored(hashes[:4], nil, prompt[:4*kvcrBlockSize], kvcrBlockSize, "CPU", kvcrOwnership),
		kvcrBlockStored(hashes[4:], hashes[3], prompt[4*kvcrBlockSize:], kvcrBlockSize, "CPU", kvcrOwnership),
	)

	m, ok := h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok)
	assert.Equal(t, map[string]int{"cpu": 8}, m.BlocksByTier)
}
