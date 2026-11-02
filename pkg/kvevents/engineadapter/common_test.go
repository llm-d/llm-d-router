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
	"encoding/binary"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
)

// TestParseTopic_Valid tests topic parsing with valid format.
func TestParseTopic_Valid(t *testing.T) {
	podID, modelName := parseTopic("kv@pod-123@llama-2-7b")
	assert.Equal(t, "pod-123", podID)
	assert.Equal(t, "llama-2-7b", modelName)
}

// TestParseTopic_NoModel tests topic parsing with only two segments.
func TestParseTopic_NoModel(t *testing.T) {
	podID, modelName := parseTopic("pod-123@llama-2-7b")
	assert.Equal(t, "pod-123@llama-2-7b", podID)
	assert.Equal(t, "", modelName)
}

// TestParseTopic_Plain tests topic parsing with no @ separator.
func TestParseTopic_Plain(t *testing.T) {
	podID, modelName := parseTopic("fallback")
	assert.Equal(t, "fallback", podID)
	assert.Equal(t, "", modelName)
}

// TestTopicIdentity tests that a per-pod subscription takes the model from a
// topic that names no pod.
func TestTopicIdentity(t *testing.T) {
	for _, tc := range []struct {
		name, topic, source, wantPod, wantModel string
	}{
		{"model only, per pod", "kv@Qwen/Qwen3-32B", "10.0.0.5:5556", "10.0.0.5:5556", "Qwen/Qwen3-32B"},
		{"pod and model, per pod", "kv@pod-1@Qwen/Qwen3-32B", "10.0.0.5:5556", "pod-1", "Qwen/Qwen3-32B"},
		{"model only, bound socket", "kv@Qwen/Qwen3-32B", "", "kv@Qwen/Qwen3-32B", ""},
		{"empty model, per pod", "kv@", "10.0.0.5:5556", "kv@", ""},
		{"no separator, per pod", "fallback", "10.0.0.5:5556", "fallback", ""},
		{"four parts, per pod", "kv@a@b@c", "10.0.0.5:5556", "kv@a@b@c", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			podID, modelName := topicIdentity(&kvevents.RawMessage{Topic: tc.topic, SourceEndpoint: tc.source})
			assert.Equal(t, tc.wantPod, podID)
			assert.Equal(t, tc.wantModel, modelName)
		})
	}
}

// TestDecodeHash tests hash format conversions.
func TestDecodeHash(t *testing.T) {
	positiveIntegers := []struct {
		name string
		raw  any
		want uint64
	}{
		{name: "uint64", raw: uint64(42), want: 42},
		{name: "uint32", raw: uint32(42), want: 42},
		{name: "uint16", raw: uint16(42), want: 42},
		{name: "uint8", raw: uint8(42), want: 42},
		{name: "int64", raw: int64(42), want: 42},
		{name: "int32", raw: int32(42), want: 42},
		{name: "int16", raw: int16(42), want: 42},
		{name: "int8", raw: int8(42), want: 42},
	}
	for _, tt := range positiveIntegers {
		t.Run(tt.name, func(t *testing.T) {
			result, err := decodeTestHash(tt.raw)
			require.NoError(t, err)
			assert.Equal(t, tt.want, result)
		})
	}

	signedIntegers := []struct {
		name string
		raw  any
		want uint64
	}{
		{name: "int64_negative", raw: int64(-1), want: ^uint64(0)},
		{name: "int32_negative", raw: int32(-1), want: ^uint64(0)},
		{name: "int16_negative", raw: int16(-1), want: ^uint64(0)},
		{name: "int8_negative", raw: int8(-1), want: ^uint64(0)},
	}
	for _, tt := range signedIntegers {
		t.Run(tt.name, func(t *testing.T) {
			result, err := decodeTestHash(tt.raw)
			require.NoError(t, err)
			assert.Equal(t, tt.want, result)
		})
	}

	t.Run("bytes_8", func(t *testing.T) {
		b := make([]byte, 8)
		binary.BigEndian.PutUint64(b, 12345)
		result, err := decodeTestHash(b)
		require.NoError(t, err)
		assert.Equal(t, uint64(12345), result)
	})

	t.Run("bytes_empty", func(t *testing.T) {
		_, err := decodeTestHash([]byte{})
		assert.Error(t, err)
	})

	t.Run("unsupported_type", func(t *testing.T) {
		_, err := decodeTestHash("not a hash")
		assert.Error(t, err)
	})
}

func decodeTestHash(raw any) (uint64, error) {
	payload, err := msgpack.Marshal(raw)
	if err != nil {
		return 0, err
	}
	return decodeHash(msgpack.NewDecoder(bytes.NewReader(payload)))
}
