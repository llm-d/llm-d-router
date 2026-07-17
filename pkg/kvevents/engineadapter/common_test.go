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
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

// TestGetHashAsUint64 tests hash format conversions.
func TestGetHashAsUint64(t *testing.T) {
	t.Run("uint64", func(t *testing.T) {
		result, err := getHashAsUint64(uint64(42))
		require.NoError(t, err)
		assert.Equal(t, uint64(42), result)
	})

	t.Run("int64", func(t *testing.T) {
		result, err := getHashAsUint64(int64(42))
		require.NoError(t, err)
		assert.Equal(t, uint64(42), result)
	})

	t.Run("bytes_8", func(t *testing.T) {
		b := make([]byte, 8)
		binary.BigEndian.PutUint64(b, 12345)
		result, err := getHashAsUint64(b)
		require.NoError(t, err)
		assert.Equal(t, uint64(12345), result)
	})

	t.Run("bytes_empty", func(t *testing.T) {
		_, err := getHashAsUint64([]byte{})
		assert.Error(t, err)
	})

	t.Run("unsupported_type", func(t *testing.T) {
		_, err := getHashAsUint64("not a hash")
		assert.Error(t, err)
	})
}
