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
		nil,
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
	adapter := NewSGLangAdapter()

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

	result, err := decodeEvent(rawBytes, adapter.eventConverters)
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

// TestSGLangBlockStored_7Fields tests decoding with 7 fields (no lora_name, no extra_keys).
func TestSGLangBlockStored_7Fields(t *testing.T) {
	adapter := NewSGLangAdapter()

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

	result, err := decodeEvent(rawBytes, adapter.eventConverters)
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
	adapter := NewSGLangAdapter()

	// Only 5 fields: tag + block_hashes + parent + tokens + block_size
	event := []any{
		"BlockStored",
		[]any{uint64(400)},
		uint64(399),
		[]uint32{10, 11},
		128,
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeEvent(rawBytes, adapter.eventConverters)
	require.NoError(t, err, "minimal 5-field BlockStored should decode successfully")
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
	adapter := NewSGLangAdapter()

	event := []any{
		"BlockStored",
		[]any{uint64(500)},
		uint64(499),
		[]uint32{1},
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	_, err = decodeEvent(rawBytes, adapter.eventConverters)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "too few fields")
}

// TestSGLangBlockRemoved_FullFields tests decoding with all 3 fields.
func TestSGLangBlockRemoved_FullFields(t *testing.T) {
	adapter := NewSGLangAdapter()

	medium := "cpu"
	event := []any{
		"BlockRemoved",
		[]any{uint64(200), uint64(201), uint64(202)},
		&medium,
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeEvent(rawBytes, adapter.eventConverters)
	require.NoError(t, err)
	require.NotNil(t, result)

	blockRemoved, ok := result.(*kvevents.BlockRemovedEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{200, 201, 202}, blockRemoved.BlockHashes)
	assert.Equal(t, "cpu", blockRemoved.DeviceTier)
}

// TestSGLangBlockRemoved_NoMedium tests decoding without the trailing medium field.
func TestSGLangBlockRemoved_NoMedium(t *testing.T) {
	adapter := NewSGLangAdapter()

	event := []any{
		"BlockRemoved",
		[]any{uint64(500), uint64(501)},
	}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeEvent(rawBytes, adapter.eventConverters)
	require.NoError(t, err, "SGLang BlockRemoved without medium should decode successfully")
	require.NotNil(t, result)

	blockRemoved, ok := result.(*kvevents.BlockRemovedEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{500, 501}, blockRemoved.BlockHashes)
	assert.Equal(t, "", blockRemoved.DeviceTier, "medium should default to empty")
}

// TestSGLangAllBlocksCleared tests decoding a valid AllBlocksCleared event.
func TestSGLangAllBlocksCleared(t *testing.T) {
	adapter := NewSGLangAdapter()

	event := []any{"AllBlocksCleared"}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeEvent(rawBytes, adapter.eventConverters)
	require.NoError(t, err)
	require.NotNil(t, result)

	_, ok := result.(*kvevents.AllBlocksClearedEvent)
	require.True(t, ok, "expected AllBlocksClearedEvent")
}

// TestSGLangUnknownTag tests error handling for unknown event tags.
func TestSGLangUnknownTag(t *testing.T) {
	adapter := NewSGLangAdapter()

	event := []any{"UnknownEventType", "some", "data"}

	rawBytes, err := msgpack.Marshal(event)
	require.NoError(t, err)

	result, err := decodeEvent(rawBytes, adapter.eventConverters)
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "unknown event tag")
}
