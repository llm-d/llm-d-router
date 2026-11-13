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
	pool, err := kvevents.NewPool(kvevents.DefaultConfig(), indexer.KVBlockIndex(), tp, engineadapter.NewVLLMAdapter())
	require.NoError(t, err)
	return &kvcrHarness{pool: pool, indexer: indexer, tp: tp}
}

// publish delivers one vLLM event batch from pod.
func (h *kvcrHarness) publish(t *testing.T, ctx context.Context, pod string, events ...[]any) {
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

// kvcrBlockStored is vLLM's array-encoded BlockStored up to ownership. vLLM
// omits the trailing session_id and lookahead_token_ids when unset. A nil
// parent marks the first block of a sequence.
func kvcrBlockStored(hashes []uint64, parent any, tokens []uint32, blockSize int, medium string, ownership any) []any {
	return []any{
		"BlockStored", hashes, parent, tokens, blockSize,
		nil,              // lora_id
		medium,           // medium
		nil,              // lora_name
		nil,              // extra_keys
		0,                // group_idx
		"full_attention", // kv_cache_spec_kind
		nil,              // kv_cache_spec_sliding_window
		nil,              // locality
		ownership,        // ownership
	}
}

// kvcrPlaceholderStored is the BlockStored vLLM emits for an offloaded key it
// holds no request metadata for: one hash, no tokens and block_size 0.
func kvcrPlaceholderStored(hash uint64, medium string) []any {
	return kvcrBlockStored([]uint64{hash}, nil, []uint32{}, 0, medium, kvcrOwnership)
}

func kvcrBlockRemoved(hashes []uint64, medium string, ownership any) []any {
	return []any{"BlockRemoved", hashes, medium, 0, nil, ownership}
}

func kvcrTokens(n int, base uint32) []uint32 {
	tokens := make([]uint32, n)
	for i := range tokens {
		tokens[i] = base + uint32(i) //#nosec G115 -- test data is small
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

	prompt := kvcrTokens(4*kvcrBlockSize, 1)
	hashes := kvcrHashes(4, 100)
	h.publish(t, ctx, "pod-a",
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "GPU", nil),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "CPU", kvcrOwnership),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "STORAGE", kvcrOwnership),
	)
	h.publish(t, ctx, "pod-b",
		kvcrBlockStored(kvcrHashes(4, 500), nil, kvcrTokens(4*kvcrBlockSize, 1000), kvcrBlockSize, "GPU", nil),
	)
	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "GPU", nil))

	m, ok := h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok, "KVCR tiers must keep the prefix on pod-a after GPU eviction")
	assert.Equal(t, map[string]int{"cpu": 4, "storage": 4}, m.BlocksByTier)
	assert.InDelta(t, 4*0.8, m.WeightedScore, 1e-9, "each block scores its best tier, cpu")

	scores, err := h.indexer.ScoreTokens(ctx, prompt, kvcrModel, []string{"pod-a", "pod-b"}, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{"pod-a"}, slices.Collect(maps.Keys(scores)), "only the pod holding the prefix in KVCR scores")

	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "CPU", kvcrOwnership))
	m, ok = h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok)
	assert.Equal(t, map[string]int{"storage": 4}, m.BlocksByTier)
	assert.InDelta(t, 4*0.6, m.WeightedScore, 1e-9)

	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "STORAGE", kvcrOwnership))
	_, ok = h.matches(t, ctx, prompt)["pod-a"]
	assert.False(t, ok, "the prefix leaves the index with its last KVCR copy")
}

func TestKVCRInventory_PlaceholderStoresResolveThroughEngineKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(4*kvcrBlockSize, 1)
	hashes := kvcrHashes(4, 100)
	h.publish(t, ctx, "pod-a", kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "GPU", nil))

	placeholders := make([][]any, 0, len(hashes))
	for _, hash := range hashes {
		placeholders = append(placeholders, kvcrPlaceholderStored(hash, "STORAGE"))
	}
	h.publish(t, ctx, "pod-a", placeholders...)
	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "GPU", nil))

	m, ok := h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok, "placeholder stores must resolve through the GPU store's engine keys")
	assert.Equal(t, map[string]int{"storage": 4}, m.BlocksByTier)

	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "STORAGE", kvcrOwnership))
	_, ok = h.matches(t, ctx, prompt)["pod-a"]
	assert.False(t, ok)
}

// vLLM's primary CPU tier and KVCR's DRAM both report medium CPU, so one
// owner's removal must not drop the block the other still holds.
func TestKVCRInventory_CPUTierSharedWithPrimaryOffload(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(4*kvcrBlockSize, 1)
	hashes := kvcrHashes(4, 100)
	h.publish(t, ctx, "pod-a",
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "GPU", nil),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "CPU", nil),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "CPU", kvcrOwnership),
	)
	h.publish(t, ctx, "pod-a",
		kvcrBlockRemoved(hashes, "GPU", nil),
		kvcrBlockRemoved(hashes, "CPU", nil),
	)

	m, ok := h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok, "KVCR's copy must survive the primary CPU tier's removal")
	assert.Equal(t, map[string]int{"cpu": 4}, m.BlocksByTier)

	h.publish(t, ctx, "pod-a", kvcrBlockRemoved(hashes, "CPU", kvcrOwnership))
	_, ok = h.matches(t, ctx, prompt)["pod-a"]
	assert.False(t, ok)
}

// vLLM can place a primary-tier removal ahead of the KVCR store for the same
// keys in one batch. Removing the last copy drops the engine-to-request key
// mapping, so the KVCR stores must index from their own tokens and parents.
func TestKVCRInventory_StoreAfterLastRemovalInOneBatch(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(8*kvcrBlockSize, 1)
	hashes := kvcrHashes(8, 100)
	h.publish(t, ctx, "pod-a",
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "GPU", nil),
		kvcrBlockStored(hashes, nil, prompt, kvcrBlockSize, "CPU", nil),
	)

	h.publish(t, ctx, "pod-a",
		kvcrBlockRemoved(hashes, "GPU", nil),
		kvcrBlockRemoved(hashes, "CPU", nil),
		kvcrBlockStored(hashes[:4], nil, prompt[:4*kvcrBlockSize], kvcrBlockSize, "CPU", kvcrOwnership),
		kvcrBlockStored(hashes[4:], hashes[3], prompt[4*kvcrBlockSize:], kvcrBlockSize, "CPU", kvcrOwnership),
	)

	m, ok := h.matches(t, ctx, prompt)["pod-a"]
	require.True(t, ok)
	assert.Equal(t, map[string]int{"cpu": 8}, m.BlocksByTier)
}
