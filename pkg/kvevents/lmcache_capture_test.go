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
	"encoding/binary"
	"encoding/json"
	"os"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

// The tests in this file replay testdata/lmcache_mp_kv_events.json, the
// KV-event messages one vLLM engine with an LMCache multiprocess server
// published for one prompt. testdata/README.md describes the file, the stages
// of the capture and the stack it was taken from.
//
// The tests show that the router parses and indexes the bytes that engine
// published. No engine and no router deployment runs in them.

// lmcacheCapturePod is the endpoint in the topic of the captured messages.
const lmcacheCapturePod = "127.0.0.1:32210"

// lmcacheCapture is the part of the capture file the tests read.
type lmcacheCapture struct {
	Model           string           `json:"model"`
	ChunkSize       int              `json:"lmcache_chunk_size"`
	EngineBlockSize int              `json:"engine_block_size"`
	PromptTokenIDs  []uint32         `json:"prompt_token_ids"`
	Stages          []lmcacheStage   `json:"stages"`
	Messages        []lmcacheMessage `json:"messages"`
}

type lmcacheStage struct {
	Stage string `json:"stage"`
	// LMCacheCachedTokens is set for the stages that sent the prompt.
	LMCacheCachedTokens *int `json:"lmcache_cached_tokens"`
}

type lmcacheMessage struct {
	Stage  string   `json:"stage"`
	Frames [][]byte `json:"frames"`
}

// raw is the message as a subscriber passes it to the pool.
func (m lmcacheMessage) raw() *kvevents.RawMessage {
	return &kvevents.RawMessage{
		Topic:    string(m.Frames[0]),
		Sequence: binary.BigEndian.Uint64(m.Frames[1]),
		Payload:  m.Frames[2],
	}
}

func loadLMCacheCapture(t *testing.T) *lmcacheCapture {
	t.Helper()

	data, err := os.ReadFile("testdata/lmcache_mp_kv_events.json")
	require.NoError(t, err)
	capture := &lmcacheCapture{}
	require.NoError(t, json.Unmarshal(data, capture))
	for i, message := range capture.Messages {
		require.Len(t, message.Frames, 3, "message %d", i)
		require.Len(t, message.Frames[1], 8, "message %d", i)
		require.EqualValues(t, i, message.raw().Sequence, "the capture misses a message")
	}
	return capture
}

// replay passes the messages of the named stages to the pool, stage by stage,
// and returns their number.
func (c *lmcacheCapture) replay(t *testing.T, ctx context.Context, pool *kvevents.Pool, stages ...string) int {
	t.Helper()

	replayed := 0
	for _, stage := range stages {
		before := replayed
		for _, message := range c.Messages {
			if message.Stage == stage {
				pool.ProcessRawMessageForTest(ctx, message.raw())
				replayed++
			}
		}
		require.Greater(t, replayed, before, "the capture has no stage %s", stage)
	}
	return replayed
}

// The captured events have the shape lmcache_events_test.go builds its events
// in. The CPU events, the GPU stores of the prompt and AllBlocksCleared belong
// to the stages testdata/README.md lists them under.
func TestLMCacheCapture_EventsAsPublished(t *testing.T) {
	capture := loadLMCacheCapture(t)
	require.Equal(t, lmcacheChunkSize, capture.ChunkSize)
	require.Equal(t, lmcacheEngineBlockSize, capture.EngineBlockSize)
	prompt := capture.PromptTokenIDs
	chunks := len(prompt) / capture.ChunkSize
	cached := prompt[:chunks*capture.ChunkSize]

	adapter := engineadapter.NewVLLMAdapter()
	var chunkStores, promptStores []*kvevents.BlockStoredEvent
	var promptStages []string
	var chunkRemovals []*kvevents.BlockRemovedEvent
	cleared := 0
	for i, message := range capture.Messages {
		pod, model, batch, err := adapter.ParseMessage(message.raw())
		require.NoError(t, err, "message %d", i)
		assert.Equal(t, lmcacheCapturePod, pod)
		assert.Equal(t, capture.Model, model)
		for _, event := range batch.Events {
			switch ev := event.(type) {
			case *kvevents.BlockStoredEvent:
				if ev.DeviceTier == "CPU" {
					assert.Equal(t, "cpu-stored", message.Stage)
					chunkStores = append(chunkStores, ev)
					continue
				}
				assert.Equal(t, "GPU", ev.DeviceTier)
				assert.Equal(t, capture.EngineBlockSize, ev.BlockSize)
				require.NotNil(t, ev.GroupIdx)
				assert.Zero(t, *ev.GroupIdx)
				assert.Equal(t, kvevents.KVCacheSpecKindFullAttention, ev.KVCacheSpecKind)
				if slices.Equal(ev.Tokens, cached) {
					promptStores = append(promptStores, ev)
					promptStages = append(promptStages, message.Stage)
				}
			case *kvevents.BlockRemovedEvent:
				if ev.DeviceTier == "CPU" {
					assert.Equal(t, "cpu-removed", message.Stage)
					chunkRemovals = append(chunkRemovals, ev)
				}
			case *kvevents.AllBlocksClearedEvent:
				assert.Equal(t, "gpu-reset", message.Stage)
				cleared++
			}
		}
	}

	require.Len(t, chunkStores, chunks, "one CPU BlockStored per chunk")
	for i, store := range chunkStores {
		assert.Equal(t, capture.ChunkSize, store.BlockSize)
		require.Len(t, store.BlockHashes, 1)
		assert.Equal(t, prompt[i*capture.ChunkSize:(i+1)*capture.ChunkSize], store.Tokens,
			"chunk %d carries the tokens the engine renders for the prompt", i)
		assert.Nil(t, store.GroupIdx)
		assert.Empty(t, store.KVCacheSpecKind)
		assert.Nil(t, store.ExtraKeys)
		assert.Nil(t, store.LoraName)
		if i == 0 {
			assert.Zero(t, store.ParentHash)
		} else {
			assert.Equal(t, chunkStores[i-1].BlockHashes[0], store.ParentHash)
		}
	}

	require.Len(t, chunkRemovals, chunks, "one CPU BlockRemoved per chunk")
	for i, removal := range chunkRemovals {
		assert.Equal(t, chunkStores[i].BlockHashes, removal.BlockHashes)
		assert.Nil(t, removal.GroupIdx)
	}

	// The engine stores the prompt on GPU when it computes it and each time
	// it loads it from LMCache, the second time after its reset.
	require.Equal(t, []string{"cold", "cpu-load", "cpu-load-2"}, promptStages)
	loaded := map[string]int{}
	for _, stage := range capture.Stages {
		if stage.LMCacheCachedTokens != nil {
			loaded[stage.Stage] = *stage.LMCacheCachedTokens
		}
	}
	assert.Equal(t, map[string]int{"cold": 0, "cpu-load": len(cached), "cpu-load-2": len(cached)}, loaded)
	for _, store := range promptStores {
		assert.Len(t, store.BlockHashes, len(cached)/capture.EngineBlockSize)
		assert.Equal(t, promptStores[0].BlockHashes, store.BlockHashes)
		for _, chunk := range chunkStores {
			assert.NotContains(t, store.BlockHashes, chunk.BlockHashes[0])
		}
	}
	assert.Equal(t, 1, cleared)
}

// The index after each stage of the capture, with the messages processed in
// the order the engine published them.
func TestLMCacheCapture_IndexAfterEachStage(t *testing.T) {
	ctx := context.Background()
	capture := loadLMCacheCapture(t)
	h := newKVCRHarness(t, ctx)

	keys, err := h.tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, capture.PromptTokenIDs, capture.Model, nil)
	require.NoError(t, err)
	// The chunks and the engine blocks end at the prompt's last full chunk.
	// The request has two more blocks, which no tier holds.
	cached := len(capture.PromptTokenIDs) / capture.ChunkSize * lmcacheBlocksPerChunk
	require.Len(t, keys, cached+2)

	// open marks a value the test does not check.
	const open = -1
	replayed := 0
	for _, want := range []struct {
		stage string
		// gpu and cpu are the blocks of the prompt the pod holds in the tier.
		gpu, cpu int
		score    float64
	}{
		{stage: "warm-up"},
		{stage: "cold", gpu: cached, score: float64(cached)},
		{stage: "cpu-stored", gpu: cached, cpu: cached, score: float64(cached)},
		{stage: "gpu-evicted", cpu: cached, score: float64(cached) * 0.8},
		{stage: "cpu-load", gpu: cached, cpu: cached, score: float64(cached)},
		// The engine reset its own prefix cache and LMCache kept the chunks:
		// the engine loads them in the next stage. AllBlocksCleared carries
		// no medium and the index drops every tier of the pod, and no CPU
		// BlockStored follows the reset. Whether the CPU placement should
		// outlive the reset is open, so the CPU tier is not checked until
		// LMCache withdraws the chunks.
		{stage: "gpu-reset", cpu: open, score: open},
		{stage: "cpu-load-2", gpu: cached, cpu: open, score: float64(cached)},
		{stage: "cpu-removed", gpu: cached, score: float64(cached)},
	} {
		replayed += capture.replay(t, ctx, h.pool, want.stage)
		matches, err := h.indexer.MatchBlockKeys(ctx, keys, nil)
		require.NoError(t, err)
		m := matches[lmcacheCapturePod]
		assert.Equal(t, want.gpu, m.BlocksByTier[kvcache.GPUTier], "gpu blocks after %s", want.stage)
		if want.cpu != open {
			assert.Equal(t, want.cpu, m.BlocksByTier["cpu"], "cpu blocks after %s", want.stage)
		}
		if want.score != open {
			assert.InDelta(t, want.score, m.WeightedScore, 1e-9, "score after %s", want.stage)
		}
	}
	assert.Len(t, capture.Messages, replayed, "a captured message belongs to no stage of the test")
}

// LMCache withdraws the chunks while its CPU tier is the only tier that holds
// the prompt. The engine in the capture loaded the prompt again before the
// withdrawal, so the test replays the withdrawal out of the capture's order.
// The pool does not read sequence numbers.
func TestLMCacheCapture_WithdrawalAfterGPUEviction(t *testing.T) {
	ctx := context.Background()
	capture := loadLMCacheCapture(t)
	h := newKVCRHarness(t, ctx)
	holders := func() [][]string {
		return lmcacheChunkHolders(t, ctx, h, capture.Model, capture.PromptTokenIDs)
	}

	capture.replay(t, ctx, h.pool, "warm-up", "cold", "cpu-stored", "gpu-evicted")
	onCPU := []string{lmcacheCapturePod + "/cpu"}
	require.Equal(t, [][]string{onCPU, onCPU}, holders())

	capture.replay(t, ctx, h.pool, "cpu-removed")
	assert.Equal(t, [][]string{{}, {}}, holders())
}
