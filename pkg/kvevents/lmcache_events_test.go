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
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// The tests in this file feed the vLLM adapter the events vLLM publishes for
// an LMCache multiprocess server behind LMCacheMPConnector, and check what the
// precise prefix scorer reads. LMCache reports its CPU tier as medium CPU with
// one BlockStored per LMCache chunk: one chunk hash, the chunk's tokens, the
// previous chunk's hash as parent, block_size set to the chunk size and no KV
// cache group. The engine reports its GPU blocks of the same prompt under its
// own hashes and block size, so the two tiers of one prefix are separate
// engine-key chains.
//
// The events are built from that wire format and published through the
// harness of kvcr_events_test.go. lmcache_capture_test.go replays messages
// captured from an engine.

const (
	// lmcacheChunkSize is LMCache's default chunk size in tokens.
	lmcacheChunkSize = 256
	// lmcacheEngineBlockSize is the size in tokens of the engine's GPU blocks
	// in testdata/lmcache_mp_kv_events.json.
	lmcacheEngineBlockSize = 64
	// lmcacheBlocksPerChunk is the number of request blocks of one chunk.
	lmcacheBlocksPerChunk = lmcacheChunkSize / kvcrBlockSize
)

// lmcacheChunkStores returns the stores of the chunks of prompt in LMCache's
// CPU tier: chunk i under hashes[i], with chunk i-1 as its parent.
func lmcacheChunkStores(hashes []uint64, prompt []uint32) []map[string]any {
	stores := make([]map[string]any, len(hashes))
	for i, hash := range hashes {
		var parent any
		if i > 0 {
			parent = hashes[i-1]
		}
		stores[i] = map[string]any{
			"type":              "BlockStored",
			"block_hashes":      []uint64{hash},
			"parent_block_hash": parent,
			"token_ids":         prompt[i*lmcacheChunkSize : (i+1)*lmcacheChunkSize],
			"block_size":        lmcacheChunkSize,
			"lora_id":           nil,
			"medium":            "CPU",
			"lora_name":         nil,
		}
	}
	return stores
}

// lmcacheChunksRemoved is the BlockRemoved that withdraws chunks from
// LMCache's CPU tier.
func lmcacheChunksRemoved(hashes ...uint64) map[string]any {
	return map[string]any{"type": "BlockRemoved", "block_hashes": hashes, "medium": "CPU"}
}

// lmcacheChunkHolders returns, for each LMCache chunk of tokens, the sorted
// "pod/tier" entries that hold the chunk's request blocks. It reads each chunk
// on its own, so it reports a chunk that follows one no pod holds, which a
// prefix match does not reach.
func lmcacheChunkHolders(t *testing.T, ctx context.Context, h *kvcrHarness, model string, tokens []uint32) [][]string {
	t.Helper()

	keys, err := h.tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, model, nil)
	require.NoError(t, err)
	holders := make([][]string, len(keys)/lmcacheBlocksPerChunk)
	for i, key := range keys[:len(holders)*lmcacheBlocksPerChunk] {
		held, err := h.indexer.KVBlockIndex().Lookup(ctx, []kvblock.BlockHash{key}, nil)
		require.NoError(t, err)
		entries := make([]string, 0, len(held[key]))
		for _, entry := range held[key] {
			entries = append(entries, entry.PodIdentifier+"/"+entry.DeviceTier)
		}
		slices.Sort(entries)
		if i%lmcacheBlocksPerChunk == 0 {
			holders[i/lmcacheBlocksPerChunk] = entries
			continue
		}
		require.Equal(t, holders[i/lmcacheBlocksPerChunk], entries, "request block %d", i)
	}
	return holders
}

// One chunk hash stands for sixteen request blocks, and a chunk chains on the
// last request block of its parent.
func TestLMCacheEvents_ChunkChain(t *testing.T) {
	ctx := context.Background()
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(3 * lmcacheChunkSize)
	chunks := []uint64{100, 200, 300}
	stores := lmcacheChunkStores(chunks, prompt)
	// Each chunk arrives in its own message and finds its parent in the index.
	for _, store := range stores {
		h.publish(t, ctx, "pod-a", store)
	}

	m := h.matches(t, ctx, prompt)["pod-a"]
	assert.Equal(t, map[string]int{"cpu": 3 * lmcacheBlocksPerChunk}, m.BlocksByTier)
	assert.InDelta(t, 3*lmcacheBlocksPerChunk*0.8, m.WeightedScore, 1e-9)

	// Withdrawing the middle chunk removes its request blocks and no others.
	h.publish(t, ctx, "pod-a", lmcacheChunksRemoved(chunks[1]))
	assert.Equal(t, [][]string{{"pod-a/cpu"}, {}, {"pod-a/cpu"}}, lmcacheChunkHolders(t, ctx, h, kvcrModel, prompt))
	assert.Equal(t, map[string]int{"cpu": lmcacheBlocksPerChunk}, h.matches(t, ctx, prompt)["pod-a"].BlocksByTier)

	h.publish(t, ctx, "pod-a", stores[1])
	assert.Equal(t, map[string]int{"cpu": 3 * lmcacheBlocksPerChunk}, h.matches(t, ctx, prompt)["pod-a"].BlocksByTier)
}

// A chunk whose parent the index does not hold cannot be placed: its request
// blocks chain on the parent's last request block. Whether the chunk is
// indexed once its parent arrives is left open. The test ends with the events
// LMCache publishes when it evicts the chunk and stores it again, which place
// the chunk in either case.
func TestLMCacheEvents_ChunkBeforeItsParent(t *testing.T) {
	ctx := context.Background()
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(2 * lmcacheChunkSize)
	stores := lmcacheChunkStores([]uint64{100, 200}, prompt)
	parent, child := stores[0], stores[1]

	h.publish(t, ctx, "pod-a", child)
	assert.Empty(t, h.matches(t, ctx, prompt[lmcacheChunkSize:]),
		"a chunk with an unknown parent must not be indexed as the start of a prompt")

	h.publish(t, ctx, "pod-a", parent)
	assert.GreaterOrEqual(t, h.matches(t, ctx, prompt)["pod-a"].BlocksByTier["cpu"], lmcacheBlocksPerChunk,
		"the parent chunk is indexed")

	h.publish(t, ctx, "pod-a", lmcacheChunksRemoved(200))
	h.publish(t, ctx, "pod-a", child)
	assert.Equal(t, map[string]int{"cpu": 2 * lmcacheBlocksPerChunk}, h.matches(t, ctx, prompt)["pod-a"].BlocksByTier)
}

// The engine's GPU blocks and LMCache's CPU chunks of one prompt carry
// different hashes and block sizes. Both resolve to the request blocks of the
// prompt's tokens, in either arrival order, and a prefix the engine evicted in
// part from GPU continues in LMCache's CPU tier.
func TestLMCacheEvents_GPUBlocksAndCPUChunksShareRequestBlocks(t *testing.T) {
	prompt := kvcrTokens(2 * lmcacheChunkSize)
	gpuBlocks := kvcrHashes(len(prompt)/lmcacheEngineBlockSize, 1000)
	gpuStore := kvcrBlockStored(gpuBlocks, nil, prompt, lmcacheEngineBlockSize, "GPU", "")
	cpuStores := lmcacheChunkStores([]uint64{100, 200}, prompt)
	const promptBlocks = 2 * lmcacheBlocksPerChunk

	for _, tc := range []struct {
		name   string
		events []map[string]any
	}{
		{name: "GPU blocks first", events: append([]map[string]any{gpuStore}, cpuStores...)},
		{name: "CPU chunks first", events: append(slices.Clone(cpuStores), gpuStore)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			h := newKVCRHarness(t, ctx)
			for _, event := range tc.events {
				h.publish(t, ctx, "pod-a", event)
			}

			m := h.matches(t, ctx, prompt)["pod-a"]
			assert.Equal(t, map[string]int{kvcache.GPUTier: promptBlocks, "cpu": promptBlocks}, m.BlocksByTier)
			assert.InDelta(t, promptBlocks, m.WeightedScore, 1e-9, "each block scores its best tier, gpu")

			for _, hash := range gpuBlocks[len(gpuBlocks)/2:] {
				h.publish(t, ctx, "pod-a", kvcrBlockRemoved([]uint64{hash}, "GPU", ""))
			}
			m = h.matches(t, ctx, prompt)["pod-a"]
			assert.Equal(t, map[string]int{kvcache.GPUTier: promptBlocks / 2, "cpu": promptBlocks}, m.BlocksByTier)
			assert.Equal(t, promptBlocks, m.MatchedBlocks)
			assert.InDelta(t, promptBlocks/2*1.0+promptBlocks/2*0.8, m.WeightedScore, 1e-9)

			for _, hash := range gpuBlocks[:len(gpuBlocks)/2] {
				h.publish(t, ctx, "pod-a", kvcrBlockRemoved([]uint64{hash}, "GPU", ""))
			}
			m = h.matches(t, ctx, prompt)["pod-a"]
			assert.Equal(t, map[string]int{"cpu": promptBlocks}, m.BlocksByTier)
			assert.InDelta(t, promptBlocks*0.8, m.WeightedScore, 1e-9)
		})
	}
}

// A CPU withdrawal names LMCache chunk hashes and medium CPU. It removes those
// chunks from the CPU tier of the pod that sent it. That pod's GPU blocks and
// another pod's copy of the same chunks stay.
func TestLMCacheEvents_RemovalIsScopedToTierAndPod(t *testing.T) {
	ctx := context.Background()
	h := newKVCRHarness(t, ctx)

	prompt := kvcrTokens(2 * lmcacheChunkSize)
	gpuBlocks := kvcrHashes(len(prompt)/lmcacheEngineBlockSize, 1000)
	chunkStores := lmcacheChunkStores([]uint64{100, 200}, prompt)
	holders := func() [][]string { return lmcacheChunkHolders(t, ctx, h, kvcrModel, prompt) }

	// pod-a computed the prompt on GPU. Both pods hold its chunks in LMCache,
	// which hashes a chunk from the prompt's tokens up to the chunk's end, so
	// both announce the same chunk hashes.
	h.publish(t, ctx, "pod-a", kvcrBlockStored(gpuBlocks, nil, prompt, lmcacheEngineBlockSize, "GPU", ""))
	h.publish(t, ctx, "pod-a", chunkStores...)
	h.publish(t, ctx, "pod-b", chunkStores...)
	announced := []string{"pod-a/cpu", "pod-a/gpu", "pod-b/cpu"}
	require.Equal(t, [][]string{announced, announced}, holders())

	// LMCache withdraws each chunk with its own BlockRemoved.
	h.publish(t, ctx, "pod-a", lmcacheChunksRemoved(100), lmcacheChunksRemoved(200))
	withdrawn := []string{"pod-a/gpu", "pod-b/cpu"}
	assert.Equal(t, [][]string{withdrawn, withdrawn}, holders())

	// A repeated withdrawal and one for a chunk the pod never announced
	// change nothing.
	h.publish(t, ctx, "pod-a", lmcacheChunksRemoved(100), lmcacheChunksRemoved(999))
	assert.Equal(t, [][]string{withdrawn, withdrawn}, holders())

	// After an LMCache server restart or event loss the connector withdraws
	// the chunks it announced in one BlockRemoved.
	h.publish(t, ctx, "pod-b", lmcacheChunksRemoved(100, 200))
	assert.Equal(t, [][]string{{"pod-a/gpu"}, {"pod-a/gpu"}}, holders())

	h.publish(t, ctx, "pod-b", chunkStores...)
	assert.Equal(t, [][]string{withdrawn, withdrawn}, holders())
}
