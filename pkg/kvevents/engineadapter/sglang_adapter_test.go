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

package engineadapter //nolint:testpackage // Tests access unexported functions

import (
	"bytes"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
)

// TestSGLangShardingKey tests the sharding key extraction.
func TestSGLangShardingKey(t *testing.T) {
	adapter := NewSGLangAdapter()
	assert.Equal(t, "pod-123", adapter.ShardingKey(&kvevents.RawMessage{Topic: "kv@pod-123@llama-2-7b"}))
	assert.Equal(t, "fallback", adapter.ShardingKey(&kvevents.RawMessage{Topic: "fallback"}))
}

// TestSGLangParseMessage_Valid tests full message parsing through the SGLang adapter.
func TestSGLangParseMessage_Valid(t *testing.T) {
	adapter := NewSGLangAdapter()

	// SGLang format: 7 fields (no lora_name, no extra_keys)
	blockStoredEvent := []any{
		"BlockStored",
		[]any{uint64(100), uint64(101)},
		uint64(99),
		[]uint32{1, 2, 3},
		16,
		nil,
		"GPU",
	}

	batch := []any{
		1234567890.0,
		[]any{blockStoredEvent},
		3,
	}
	payload, err := msgpack.Marshal(batch)
	require.NoError(t, err)

	msg := &kvevents.RawMessage{
		Topic:    "kv@pod-1@llama-2-7b",
		Sequence: 42,
		Payload:  payload,
	}

	podID, modelName, eventBatch, err := adapter.ParseMessage(msg)
	require.NoError(t, err)
	assert.Equal(t, "pod-1", podID)
	assert.Equal(t, "llama-2-7b", modelName)
	require.NotNil(t, eventBatch.DataParallelRank)
	assert.Equal(t, 3, *eventBatch.DataParallelRank)
	assert.Len(t, eventBatch.Events, 1)

	blockStored, ok := eventBatch.Events[0].(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{100, 101}, blockStored.BlockHashes)
	assert.Equal(t, uint64(99), blockStored.ParentHash)
}

// TestSGLangParseMessage_InvalidPayload tests error handling for invalid msgpack data.
func TestSGLangParseMessage_InvalidPayload(t *testing.T) {
	adapter := NewSGLangAdapter()

	msg := &kvevents.RawMessage{
		Topic:   "kv@pod-1@model",
		Payload: []byte{0xFF, 0xFF, 0xFF},
	}

	_, _, _, err := adapter.ParseMessage(msg)
	assert.Error(t, err)
}

// TestSGLangBlockStored_FullFields tests decoding with all 9 fields (same as vLLM).
func TestSGLangBlockStored_FullFields(t *testing.T) {
	event := []any{
		"BlockStored",
		[]any{uint64(100), uint64(101)},
		uint64(99),
		[]uint32{1, 2, 3},
		16,
		nil,
		"gpu",
		nil,
		nil,
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeSGLangEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, result)

	blockStored, ok := result.(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{100, 101}, blockStored.BlockHashes)
	assert.Equal(t, uint64(99), blockStored.ParentHash)
	assert.Equal(t, []uint32{1, 2, 3}, blockStored.Tokens)
	assert.Equal(t, 16, blockStored.BlockSize)
	assert.Equal(t, "gpu", blockStored.DeviceTier)
	assert.Nil(t, blockStored.LoraID)
	assert.Nil(t, blockStored.LoraName)
	assert.Nil(t, blockStored.ExtraKeys)
}

// TestSGLangBlockStoredNegativeHashPreservesBits verifies the signed hash
// representation emitted by SGLang survives adapter conversion.
func TestSGLangBlockStoredNegativeHashPreservesBits(t *testing.T) {
	hash := int64(-987654321987654321)
	parentHash := int64(-1234567890123456789)

	// MessagePack: ["BlockStored", [-987654321987654321],
	// -1234567890123456789, [1], 1, nil, "GPU"].
	rawEvent := []byte{
		0x97, // array of 7
		0xab, 'B', 'l', 'o', 'c', 'k', 'S', 't', 'o', 'r', 'e', 'd',
		0x91, 0xd3, 0xf2, 0x4b, 0x25, 0xa0, 0x81, 0x0b, 0xed, 0x4f,
		0xd3, 0xee, 0xdd, 0xef, 0x0b, 0x82, 0x16, 0x7e, 0xeb,
		0x91, 0x01,
		0x01,
		0xc0,
		0xa3, 'G', 'P', 'U',
	}

	event, err := decodeSGLangEvent(rawEvent)
	require.NoError(t, err)

	blockStored, ok := event.(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{uint64(hash)}, blockStored.BlockHashes) // #nosec G115 -- preserve signed hash bit pattern in test expectation
	assert.Equal(t, uint64(parentHash), blockStored.ParentHash)      // #nosec G115 -- preserve signed hash bit pattern in test expectation
}

// TestSGLangBlockStored_7Fields tests decoding with 7 fields (no lora_name, no extra_keys).
func TestSGLangBlockStored_7Fields(t *testing.T) {
	event := []any{
		"BlockStored",
		[]any{uint64(300), uint64(301)},
		uint64(299),
		[]uint32{7, 8, 9},
		64,
		nil,   // lora_id
		"GPU", // medium
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeSGLangEvent(rawBytes)
	require.NoError(t, err, "SGLang 7-field format should decode successfully")
	require.NotNil(t, result)

	blockStored, ok := result.(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{300, 301}, blockStored.BlockHashes)
	assert.Equal(t, uint64(299), blockStored.ParentHash)
	assert.Equal(t, []uint32{7, 8, 9}, blockStored.Tokens)
	assert.Equal(t, "GPU", blockStored.DeviceTier)
	assert.Nil(t, blockStored.LoraID)
	assert.Nil(t, blockStored.LoraName, "SGLang does not send lora_name")
	assert.Nil(t, blockStored.ExtraKeys, "SGLang does not send extra_keys")
}

// TestSGLangBlockStored_MinimalFields tests decoding with only the minimum required fields.
func TestSGLangBlockStored_MinimalFields(t *testing.T) {
	// 7 fields (the current minimum): tag + block_hashes + parent + tokens +
	// block_size + lora_id + medium, with lora_id and medium set to nil.
	event := []any{
		"BlockStored",
		[]any{uint64(400)},
		uint64(399),
		[]uint32{10, 11},
		128,
		nil,
		nil,
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeSGLangEvent(rawBytes)
	require.NoError(t, err, "minimal 7-field BlockStored should decode successfully")
	require.NotNil(t, result)

	blockStored, ok := result.(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{400}, blockStored.BlockHashes)
	assert.Equal(t, uint64(399), blockStored.ParentHash)
	assert.Equal(t, []uint32{10, 11}, blockStored.Tokens)
	assert.Equal(t, 128, blockStored.BlockSize)
	assert.Equal(t, "", blockStored.DeviceTier, "medium should default to empty")
	assert.Nil(t, blockStored.LoraID)
	assert.Nil(t, blockStored.LoraName)
	assert.Nil(t, blockStored.ExtraKeys)
}

// TestSGLangBlockStored_TooFewFields tests that fewer than minimum fields returns an error.
func TestSGLangBlockStored_TooFewFields(t *testing.T) {
	event := []any{
		"BlockStored",
		[]any{uint64(500)},
		uint64(499),
		[]uint32{1},
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	_, err = decodeSGLangEvent(rawBytes)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "need at least")
}

// TestSGLangBlockRemoved_FullFields tests decoding with all 3 fields.
func TestSGLangBlockRemoved_FullFields(t *testing.T) {
	medium := "cpu"
	event := []any{
		"BlockRemoved",
		[]any{uint64(200), uint64(201), uint64(202)},
		&medium,
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeSGLangEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, result)

	blockRemoved, ok := result.(*kvevents.BlockRemovedEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{200, 201, 202}, blockRemoved.BlockHashes)
	assert.Equal(t, "cpu", blockRemoved.DeviceTier)
}

// TestSGLangBlockRemoved_NilMedium tests decoding with medium set to nil.
func TestSGLangBlockRemoved_NilMedium(t *testing.T) {
	event := []any{
		"BlockRemoved",
		[]any{uint64(500), uint64(501)},
		nil,
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeSGLangEvent(rawBytes)
	require.NoError(t, err, "SGLang BlockRemoved with nil medium should decode successfully")
	require.NotNil(t, result)

	blockRemoved, ok := result.(*kvevents.BlockRemovedEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{500, 501}, blockRemoved.BlockHashes)
	assert.Equal(t, "", blockRemoved.DeviceTier, "medium should default to empty")
}

// TestSGLangAllBlocksCleared tests decoding a valid AllBlocksCleared event.
func TestSGLangAllBlocksCleared(t *testing.T) {
	event := []any{"AllBlocksCleared"}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeSGLangEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, result)

	_, ok := result.(*kvevents.AllBlocksClearedEvent)
	require.True(t, ok, "expected AllBlocksClearedEvent")
}

// TestSGLangParseMessage_MapEncodedBlockStored verifies the map encoding emitted
// by SGLang since sgl-project/sglang#37482 dropped msgspec array_like=True:
// events arrive as field-name maps with the tag under "type". cache_salt and
// session_id are attribution fields with no positional slot; they must not
// cause an error and are simply not reflected in the domain event.
func TestSGLangParseMessage_MapEncodedBlockStored(t *testing.T) {
	adapter := NewSGLangAdapter()

	blockStoredEvent := map[string]any{
		"type":              "BlockStored",
		"block_hashes":      []any{uint64(100), uint64(101)},
		"parent_block_hash": uint64(99),
		"token_ids":         []uint32{1, 2, 3},
		"block_size":        16,
		"lora_id":           nil,
		"medium":            "GPU",
		"cache_salt":        "some-salt",
		"session_id":        "some-session",
	}
	payload, err := msgpack.Marshal([]any{1234567890.0, []any{blockStoredEvent}, 3})
	require.NoError(t, err)

	podID, modelName, eventBatch, err := adapter.ParseMessage(&kvevents.RawMessage{
		Topic:   "kv@pod-1@llama-2-7b",
		Payload: payload,
	})
	require.NoError(t, err)
	assert.Equal(t, "pod-1", podID)
	assert.Equal(t, "llama-2-7b", modelName)
	require.Len(t, eventBatch.Events, 1)

	blockStored, ok := eventBatch.Events[0].(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{100, 101}, blockStored.BlockHashes)
	assert.Equal(t, uint64(99), blockStored.ParentHash)
	assert.Equal(t, []uint32{1, 2, 3}, blockStored.Tokens)
	assert.Equal(t, 16, blockStored.BlockSize)
	assert.Equal(t, "GPU", blockStored.DeviceTier)
	assert.Nil(t, blockStored.LoraID)
}

// TestSGLangParseMessage_MapEncodedBlockStored_RealCaptureTwoBlocks replays a
// map-encoded BlockStored event captured from a live engine: two dense
// 64-token blocks with no parent hash and no LoRA.
func TestSGLangParseMessage_MapEncodedBlockStored_RealCaptureTwoBlocks(t *testing.T) {
	adapter := NewSGLangAdapter()

	// Captured payload; the leading 8-byte ZMQ sequence frame is not part of
	// RawMessage.Payload and has already been stripped.
	payload, err := os.ReadFile("testdata/sglang_encoded_kvevent")
	require.NoError(t, err)

	podID, modelName, eventBatch, err := adapter.ParseMessage(&kvevents.RawMessage{
		Topic:   "kv@10.128.6.102:8000@Qwen/Qwen2.5-0.5B-Instruct",
		Payload: payload,
	})

	require.NoError(t, err)
	assert.Equal(t, "10.128.6.102:8000", podID)
	assert.Equal(t, "Qwen/Qwen2.5-0.5B-Instruct", modelName)
	require.Len(t, eventBatch.Events, 1)
	require.NotNil(t, eventBatch.DataParallelRank)
	assert.Equal(t, 0, *eventBatch.DataParallelRank)

	blockStored, ok := eventBatch.Events[0].(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{16196034758909408104, 9213906022183458822}, blockStored.BlockHashes)
	assert.Equal(t, uint64(0), blockStored.ParentHash)
	rawTokens, err := os.ReadFile("testdata/sglang_encoded_kvevent_tokens")
	require.NoError(t, err)
	var wantTokens []uint32
	for s := range strings.SplitSeq(strings.TrimSpace(string(rawTokens)), ",") {
		n, err := strconv.ParseUint(s, 10, 32)
		require.NoError(t, err)
		wantTokens = append(wantTokens, uint32(n))
	}
	assert.Equal(t, wantTokens, blockStored.Tokens)
	assert.Equal(t, 64, blockStored.BlockSize)
	assert.Equal(t, "GPU", blockStored.DeviceTier)
	assert.Nil(t, blockStored.LoraID)
}

// TestSGLangParseMessage_MapEncodedBlockRemovedAndCleared covers the remaining
// map-encoded event kinds, mixed with an array-encoded event in one batch.
func TestSGLangParseMessage_MapEncodedBlockRemovedAndCleared(t *testing.T) {
	adapter := NewSGLangAdapter()

	removed := map[string]any{
		"type":         "BlockRemoved",
		"block_hashes": []any{uint64(100)},
		"medium":       "CPU",
	}
	cleared := map[string]any{"type": "AllBlocksCleared"}
	arrayStored := []any{
		"BlockStored", []any{uint64(7)}, nil, []uint32{9}, 1, nil, "GPU",
	}
	payload, err := msgpack.Marshal([]any{1234567890.0, []any{removed, cleared, arrayStored}, nil})
	require.NoError(t, err)

	_, _, eventBatch, err := adapter.ParseMessage(&kvevents.RawMessage{
		Topic:   "kv@pod-1@m",
		Payload: payload,
	})
	require.NoError(t, err)
	require.Len(t, eventBatch.Events, 3)

	blockRemoved, ok := eventBatch.Events[0].(*kvevents.BlockRemovedEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{100}, blockRemoved.BlockHashes)
	assert.Equal(t, "CPU", blockRemoved.DeviceTier)

	_, ok = eventBatch.Events[1].(*kvevents.AllBlocksClearedEvent)
	require.True(t, ok)

	_, ok = eventBatch.Events[2].(*kvevents.BlockStoredEvent)
	require.True(t, ok)
}

// TestSGLangMapEncodedErrors pins the error behavior for malformed
// map-encoded events: each failure mode reports a distinct, actionable error.
func TestSGLangMapEncodedErrors(t *testing.T) {
	adapter := NewSGLangAdapter()

	for name, tc := range map[string]struct {
		event   any
		wantErr string
	}{
		"unknown tag": {
			event:   map[string]any{"type": "SomethingNew"},
			wantErr: "unknown event tag: SomethingNew",
		},
		"missing tag": {
			event:   map[string]any{"block_hashes": []any{uint64(1)}},
			wantErr: `missing the "type" tag`,
		},
		"non-string tag": {
			event:   map[string]any{"type": 7},
			wantErr: "is not a string",
		},
	} {
		t.Run(name, func(t *testing.T) {
			payload, err := msgpack.Marshal([]any{0.0, []any{tc.event}, nil})
			require.NoError(t, err, name)
			_, _, _, err = adapter.ParseMessage(&kvevents.RawMessage{
				Topic:   "kv@pod-1@m",
				Payload: payload,
			})
			require.ErrorContains(t, err, tc.wantErr, name)
		})
	}
}

// TestSGLangUnknownTag tests error handling for unknown event tags.
func TestSGLangUnknownTag(t *testing.T) {
	event := []any{"UnknownEventType", "some", "data"}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeSGLangEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "unknown event tag")
}

func TestSGLangParseMessage_BoundedUnknownField(t *testing.T) {
	var nested any = nil
	for range maxDecodeDepth + 1 {
		nested = []any{nested}
	}
	payload, err := msgpack.Marshal([]any{0.0, []any{map[string]any{
		"type":       eventTagAllBlocksCleared,
		"cache_salt": nested,
	}}, nil})
	require.NoError(t, err)
	_, _, _, err = NewSGLangAdapter().ParseMessage(&kvevents.RawMessage{Payload: payload})
	require.ErrorContains(t, err, "depth")
}

func BenchmarkParseSGLangMessage_RealCapture(b *testing.B) {
	payload, err := os.ReadFile("testdata/sglang_encoded_kvevent")
	require.NoError(b, err)
	adapter := NewSGLangAdapter()
	message := &kvevents.RawMessage{Topic: "kv@pod@model", Payload: payload}
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	b.ResetTimer()
	for range b.N {
		if _, _, _, err := adapter.ParseMessage(message); err != nil {
			b.Fatal(err)
		}
	}
}

func decodeSGLangEvent(payload []byte) (kvevents.GenericEvent, error) {
	return decodeTypedEventFromDecoder(msgpack.NewDecoder(bytes.NewReader(payload)), true)
}

func TestSGLangParseMessage_EngineSchema(t *testing.T) {
	stored := []any{eventTagBlockStored, []uint64{1}, nil, []uint32{2}, 1, nil, "GPU"}
	for name, event := range map[string]any{
		"array trailing fields": append(stored, 42, "unused", -1),
		"map engine fields": map[string]any{
			"type": eventTagBlockStored, "block_hashes": []uint64{1},
			"token_ids": []uint32{2}, "block_size": 1, "medium": "GPU",
			"lora_name": 42, "extra_keys": "unused", "group_idx": -1,
			"cache_salt": "salt", "session_id": "session",
		},
	} {
		t.Run(name, func(t *testing.T) {
			payload, err := msgpack.Marshal([]any{1.0, []any{event}, nil, "publisher"})
			require.NoError(t, err)
			_, _, batch, err := NewSGLangAdapter().ParseMessage(&kvevents.RawMessage{Payload: payload})
			require.NoError(t, err)
			require.Equal(t, []kvevents.GenericEvent{&kvevents.BlockStoredEvent{
				BlockHashes: []uint64{1}, Tokens: []uint32{2}, BlockSize: 1, DeviceTier: "GPU",
			}}, batch.Events)
		})
	}
}

func TestSGLangParseMessage_InvalidFields(t *testing.T) {
	for name, event := range map[string]any{
		"stored omitted medium":  []any{eventTagBlockStored, []uint64{1}, nil, []uint32{2}, 1, nil},
		"removed omitted medium": []any{eventTagBlockRemoved, []uint64{1}},
		"invalid hash":           []any{eventTagBlockRemoved, []any{true}, nil},
		"invalid tokens":         map[string]any{"type": eventTagBlockStored, "block_hashes": []uint64{1}, "token_ids": "bad", "block_size": 1},
		"missing block size":     map[string]any{"type": eventTagBlockStored, "block_hashes": []uint64{1}, "token_ids": []uint32{2}},
	} {
		t.Run(name, func(t *testing.T) {
			payload, err := msgpack.Marshal([]any{0.0, []any{event}, nil})
			require.NoError(t, err)
			_, _, _, err = NewSGLangAdapter().ParseMessage(&kvevents.RawMessage{Payload: payload})
			require.Error(t, err)
		})
	}
	payload, err := msgpack.Marshal([]any{0.0, []any{}})
	require.NoError(t, err)
	_, _, _, err = NewSGLangAdapter().ParseMessage(&kvevents.RawMessage{Payload: payload})
	require.Error(t, err)
}

func TestSGLangParseMessage_MapTagOrder(t *testing.T) {
	for _, tagFirst := range []bool{true, false} {
		var buffer bytes.Buffer
		encoder := msgpack.NewEncoder(&buffer)
		require.NoError(t, encoder.EncodeArrayLen(3))
		require.NoError(t, encoder.EncodeFloat64(1))
		require.NoError(t, encoder.EncodeArrayLen(1))
		require.NoError(t, encoder.EncodeMapLen(5))
		names := []string{"block_hashes", "token_ids", "block_size", "lora_name", "type"}
		if tagFirst {
			names = []string{"type", "block_hashes", "token_ids", "block_size", "lora_name"}
		}
		values := map[string]any{"type": eventTagBlockStored, "block_hashes": []uint64{1}, "token_ids": []uint32{2}, "block_size": 1, "lora_name": 42}
		for _, name := range names {
			require.NoError(t, encoder.EncodeString(name))
			require.NoError(t, encoder.Encode(values[name]))
		}
		require.NoError(t, encoder.EncodeNil())
		_, _, batch, err := NewSGLangAdapter().ParseMessage(&kvevents.RawMessage{Payload: buffer.Bytes()})
		require.NoError(t, err)
		require.Equal(t, []kvevents.GenericEvent{&kvevents.BlockStoredEvent{
			BlockHashes: []uint64{1}, Tokens: []uint32{2}, BlockSize: 1,
		}}, batch.Events)
	}
}
