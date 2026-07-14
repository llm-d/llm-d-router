/*
Copyright 2025 The llm-d Authors.

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

package kvblock_test

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

func TestNewChunkedTokenDatabase_Validation(t *testing.T) {
	tests := []struct {
		name      string
		config    *kvblock.TokenProcessorConfig
		wantErr   bool
		errSubstr string
	}{
		{
			name:    "BlockSizeTokens zero uses default",
			config:  &kvblock.TokenProcessorConfig{BlockSizeTokens: 0},
			wantErr: false,
		},
		{
			name:      "BlockSizeTokens negative returns error",
			config:    &kvblock.TokenProcessorConfig{BlockSizeTokens: -1},
			wantErr:   true,
			errSubstr: "blockSizeTokens must be greater than 0, got -1",
		},
		{
			name:    "BlockSizeTokens positive succeeds",
			config:  &kvblock.TokenProcessorConfig{BlockSizeTokens: 16},
			wantErr: false,
		},
		{
			name:    "Backward compatibility: BlockSize still works",
			config:  &kvblock.TokenProcessorConfig{BlockSize: 16},
			wantErr: false,
		},
		{
			name:    "BlockSizeTokens takes precedence when both are set",
			config:  &kvblock.TokenProcessorConfig{BlockSize: 8, BlockSizeTokens: 16},
			wantErr: false,
		},
		{
			name:    "nil config uses defaults and succeeds",
			config:  nil,
			wantErr: false,
		},
		{
			name:      "deprecated BlockSize negative reports actual value in error",
			config:    &kvblock.TokenProcessorConfig{BlockSize: -5},
			wantErr:   true,
			errSubstr: "blockSizeTokens must be greater than 0, got -5",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			processor, err := kvblock.NewChunkedTokenDatabase(tc.config)
			if tc.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errSubstr)
				assert.Nil(t, processor)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, processor)
			}
		})
	}
}

func TestTokenProcessorConfig_JSONUnmarshal(t *testing.T) {
	tests := []struct {
		name            string
		json            string
		wantBlockSize   int
		wantBlockTokens int
		wantErr         bool
	}{
		{
			name:            "blockSizeTokens field is decoded",
			json:            `{"blockSizeTokens": 32}`,
			wantBlockTokens: 32,
		},
		{
			name:          "legacy blockSize field is decoded",
			json:          `{"blockSize": 8}`,
			wantBlockSize: 8,
		},
		{
			name:            "both fields present",
			json:            `{"blockSize": 8, "blockSizeTokens": 32}`,
			wantBlockSize:   8,
			wantBlockTokens: 32,
		},
		{
			name:            "hashSeed decoded alongside blockSizeTokens",
			json:            `{"blockSizeTokens": 16, "hashSeed": "abc"}`,
			wantBlockTokens: 16,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var cfg kvblock.TokenProcessorConfig
			err := json.Unmarshal([]byte(tc.json), &cfg)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.wantBlockSize, cfg.BlockSize, "BlockSize mismatch")
			assert.Equal(t, tc.wantBlockTokens, cfg.BlockSizeTokens, "BlockSizeTokens mismatch")
		})
	}
}

func TestTokenProcessorConfig_JSONUnmarshal_EndToEnd(t *testing.T) {
	// Verify that a config decoded from JSON produces a working processor with the correct block size.
	tests := []struct {
		name           string
		json           string
		tokens         int
		wantBlockCount int
	}{
		{
			name:           "blockSizeTokens drives chunking",
			json:           `{"blockSizeTokens": 8}`,
			tokens:         32,
			wantBlockCount: 4, // 32 / 8
		},
		{
			name:           "legacy blockSize drives chunking via compat path",
			json:           `{"blockSize": 16}`,
			tokens:         32,
			wantBlockCount: 2, // 32 / 16
		},
		{
			name:           "partial config with only hashSeed uses default block size of 16",
			json:           `{"hashSeed": "test"}`,
			tokens:         32,
			wantBlockCount: 2, // 32 / 16 (default)
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var cfg kvblock.TokenProcessorConfig
			require.NoError(t, json.Unmarshal([]byte(tc.json), &cfg))

			processor, err := kvblock.NewChunkedTokenDatabase(&cfg)
			require.NoError(t, err)

			tokens := make([]uint32, tc.tokens)
			for i := range tokens {
				tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i is small
			}
			keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
			require.NoError(t, err)
			assert.Len(t, keys, tc.wantBlockCount)
		})
	}
}

func TestTokenProcessorConfig_JSONMarshal(t *testing.T) {
	// Verify field names in the serialized output so a tag rename is caught in both directions.
	tests := []struct {
		name            string
		cfg             kvblock.TokenProcessorConfig
		wantContains    []string
		wantNotContains []string
	}{
		{
			name:            "blockSizeTokens appears in output",
			cfg:             kvblock.TokenProcessorConfig{BlockSizeTokens: 32},
			wantContains:    []string{`"blockSizeTokens":32`},
			wantNotContains: []string{`"blockSize"`}, // omitempty — absent when zero
		},
		{
			name:            "deprecated blockSize appears when non-zero",
			cfg:             kvblock.TokenProcessorConfig{BlockSize: 8, BlockSizeTokens: 16},
			wantContains:    []string{`"blockSize":8`, `"blockSizeTokens":16`},
			wantNotContains: nil,
		},
		{
			name:            "blockSize omitted when zero",
			cfg:             kvblock.TokenProcessorConfig{BlockSizeTokens: 16},
			wantContains:    []string{`"blockSizeTokens":16`},
			wantNotContains: []string{`"blockSize"`},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.cfg)
			require.NoError(t, err)
			out := string(b)
			for _, want := range tc.wantContains {
				assert.Contains(t, out, want)
			}
			for _, notWant := range tc.wantNotContains {
				assert.NotContains(t, out, notWant)
			}
		})
	}
}

func TestTokenProcessorConfig_JSONRoundTrip(t *testing.T) {
	// Marshal then unmarshal; the recovered struct must equal the original.
	original := kvblock.TokenProcessorConfig{
		BlockSize:       8,
		BlockSizeTokens: 32,
		HashSeed:        "round-trip-seed",
	}

	b, err := json.Marshal(original)
	require.NoError(t, err)

	var recovered kvblock.TokenProcessorConfig
	require.NoError(t, json.Unmarshal(b, &recovered))

	assert.Equal(t, original.BlockSize, recovered.BlockSize)
	assert.Equal(t, original.BlockSizeTokens, recovered.BlockSizeTokens)
	assert.Equal(t, original.HashSeed, recovered.HashSeed)
}

func TestBlockSizeTokensPrecedence(t *testing.T) {
	// Test that BlockSizeTokens takes precedence over BlockSize when both are set
	config := &kvblock.TokenProcessorConfig{
		BlockSize:       8,  // deprecated field
		BlockSizeTokens: 16, // new field should take precedence
		HashSeed:        "test",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)
	require.NotNil(t, processor)

	// Create tokens that would create different number of blocks depending on block size
	// With BlockSize=8: 32 tokens = 4 blocks
	// With BlockSizeTokens=16: 32 tokens = 2 blocks
	tokens := make([]uint32, 32)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i is small
	}

	keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)

	// Should create 2 blocks (32/16) not 4 blocks (32/8)
	assert.Len(t, keys, 2, "Should use BlockSizeTokens=16, creating 2 blocks from 32 tokens")
}

func TestNewChunkedTokenDatabase_DoesNotMutateCallerConfig(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16, // deprecated; BlockSizeTokens intentionally left zero
		HashSeed:  "test",
	}

	_, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	assert.Zero(t, config.BlockSizeTokens, "NewChunkedTokenDatabase must not populate BlockSizeTokens on caller's struct")
}

func TestBackwardCompatibility_BlockSize(t *testing.T) {
	// Test that setting only the deprecated BlockSize field still works correctly.
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 8, // deprecated field, BlockSizeTokens not set
		HashSeed:  "test",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)
	require.NotNil(t, processor)

	// 32 tokens / blockSize=8 → 4 blocks
	tokens := make([]uint32, 32)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i is small
	}

	keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)

	assert.Len(t, keys, 4, "BlockSize=8 should produce 4 blocks from 32 tokens")
	assert.Equal(t, 8, processor.BlockSize(), "BlockSize() should report 8")
}

func TestGetInitHash_ConsistentHashesForSameModel(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSizeTokens: 16,
		HashSeed:        "test-seed",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	modelName := "test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} // Full block

	// Get keys multiple times with no parent (should use init hash)
	keys1, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nil)
	require.NoError(t, err)
	keys2, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nil)
	require.NoError(t, err)
	keys3, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nil)
	require.NoError(t, err)

	require.NotEmpty(t, keys1, "Should generate keys")
	require.NotEmpty(t, keys2, "Should generate keys")
	require.NotEmpty(t, keys3, "Should generate keys")

	// All first keys should be identical (derived from same init hash)
	assert.Equal(t, keys1[0], keys2[0], "First key hash should be consistent across calls")
	assert.Equal(t, keys1[0], keys3[0], "First key hash should be consistent across calls")
	assert.NotEqual(t, keys1[0], kvblock.EmptyBlockHash, "Hash should not be zero")
}

func TestGetInitHash_DifferentHashesForDifferentModels(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSizeTokens: 16,
		HashSeed:        "test-seed",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	// Test different model names
	models := []string{
		"gpt-4",
		"llama-2-7b",
		"claude-3",
		"gemini-pro",
		"",  // empty string
		"a", // single character
		"very-long-model-name-with-special-characters-123!@#",
	}

	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} // Full block
	hashes := make(map[string]uint64)

	// Get first key hash for each model (derived from init hash)
	for _, modelName := range models {
		keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nil)
		require.NoError(t, err)
		require.NotEmpty(t, keys, "Should generate keys for model: %s", modelName)

		hashes[modelName] = uint64(keys[0])
		assert.NotZero(t, hashes[modelName], "Hash should not be zero for model: %s", modelName)
	}

	// Verify all hashes are different
	seenHashes := make(map[uint64]string)
	for modelName, hash := range hashes {
		if existingModel, exists := seenHashes[hash]; exists {
			t.Errorf("Hash collision detected: models '%s' and '%s' have the same initial key hash %d",
				modelName, existingModel, hash)
		}
		seenHashes[hash] = modelName
	}
}

func TestGetInitHash_DifferentSeedsProduceDifferentHashes(t *testing.T) {
	modelName := "test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	// Test with different seeds
	seeds := []string{
		"",
		"seed1",
		"seed2",
		"different-seed",
		"123456",
	}

	hashes := make(map[string]uint64)

	for _, seed := range seeds {
		config := &kvblock.TokenProcessorConfig{
			BlockSizeTokens: 16,
			HashSeed:        seed,
		}

		processor, err := kvblock.NewChunkedTokenDatabase(config)
		require.NoError(t, err)
		keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nil)
		require.NoError(t, err)
		require.NotEmpty(t, keys, "Should generate keys for seed: %s", seed)

		hashes[seed] = uint64(keys[0])
		assert.NotZero(t, hashes[seed], "Hash should not be zero for seed: %s", seed)
	}

	// Verify all hashes are different
	seenHashes := make(map[uint64]string)
	for seed, hash := range hashes {
		if existingSeed, exists := seenHashes[hash]; exists {
			t.Errorf("Hash collision detected: seeds '%s' and '%s' produce the same initial hash %d for model %s",
				seed, existingSeed, hash, modelName)
		}
		seenHashes[hash] = seed
	}
}

func TestGetInitHash_ConcurrentAccess(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSizeTokens: 16,
		HashSeed:        "test-seed",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	modelName := "concurrent-test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	numGoroutines := 100

	// Channel to collect results
	results := make(chan uint64, numGoroutines)
	var wg sync.WaitGroup

	// Start multiple goroutines calling TokensToKVBlockKeys (which calls getInitHash)
	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nil)
			if err == nil && len(keys) > 0 {
				results <- uint64(keys[0])
			}
		}()
	}

	wg.Wait()
	close(results)

	// Collect all results
	hashes := make([]uint64, 0, numGoroutines)
	for hash := range results {
		hashes = append(hashes, hash)
	}

	require.Len(t, hashes, numGoroutines, "Should have received hash from all goroutines")

	// Verify all hashes are identical
	expectedHash := hashes[0]
	for i, hash := range hashes {
		assert.Equal(t, expectedHash, hash, "Hash mismatch at index %d", i)
	}

	assert.NotZero(t, expectedHash, "Hash should not be zero")
}

func TestGetInitHash_Deterministic(t *testing.T) {
	// Test that the same configuration always produces the same hash
	modelName := "deterministic-test"
	seed := "deterministic-seed"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	var hashes []uint64

	// Create multiple instances with same config
	for i := 0; i < 5; i++ {
		config := &kvblock.TokenProcessorConfig{
			BlockSizeTokens: 16,
			HashSeed:        seed,
		}

		processor, err := kvblock.NewChunkedTokenDatabase(config)
		require.NoError(t, err)
		keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nil)
		require.NoError(t, err)
		require.NotEmpty(t, keys, "Should generate keys for instance %d", i)

		hashes = append(hashes, uint64(keys[0]))
	}

	// All instances should produce the same hash
	expectedHash := hashes[0]
	for i, hash := range hashes {
		assert.Equal(t, expectedHash, hash, "Hash should be deterministic across instances, mismatch at index %d", i)
	}

	assert.NotZero(t, expectedHash, "Hash should not be zero")
}

func TestHash_ExtraCBORDeterminism(t *testing.T) {
	// Create a canonical CBOR encoder
	// This ensures deterministic encoding (same value -> same bytes, always)
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name  string
		extra interface{}
	}{
		{"nil", nil},
		{"int_zero", 0},
		{"int_positive", 42},
		{"int_negative", -10},
		{"string_empty", ""},
		{"string_medium", "gpu"},
		{"string_long", "very-long-adapter-name-with-special-chars-123"},
		{"map_empty", map[string]interface{}{}},
		{"map_lora_only", map[string]interface{}{"lora_id": 42}},
		{"map_combined", map[string]interface{}{"lora_id": 42, "medium": "gpu"}},
		{"map_nested", map[string]interface{}{"meta": map[string]interface{}{"version": 1}}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Encode the same value 5 times
			var encodings [][]byte
			for i := 0; i < 5; i++ {
				bytes, err := encMode.Marshal(tc.extra)
				require.NoError(t, err)
				encodings = append(encodings, bytes)
			}

			// All encodings must be identical
			for i := 1; i < len(encodings); i++ {
				assert.Equal(t, encodings[0], encodings[i],
					"Encoding %d differs from encoding 0", i)
			}
		})
	}
}

func TestHash_ExtraMapKeyOrdering(t *testing.T) {
	// Create a canonical CBOR encoder
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name string
		maps []map[string]interface{}
		desc string
	}{
		{
			name: "two_keys_different_order",
			maps: []map[string]interface{}{
				{"lora_id": 42, "medium": "gpu"},
				{"medium": "gpu", "lora_id": 42},
			},
			desc: "Same keys inserted in different order",
		},
		{
			name: "three_keys_different_order",
			maps: []map[string]interface{}{
				{"lora_id": 42, "medium": "gpu", "version": 3},
				{"version": 3, "medium": "gpu", "lora_id": 42},
				{"medium": "gpu", "version": 3, "lora_id": 42},
			},
			desc: "Three keys with different permutations",
		},
		{
			name: "nested_maps",
			maps: []map[string]interface{}{
				{"outer": map[string]interface{}{"lora_id": 42, "medium": "gpu"}},
				{"outer": map[string]interface{}{"medium": "gpu", "lora_id": 42}},
			},
			desc: "Nested maps with different key order",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var encodings [][]byte

			// Encode each map
			for _, m := range tc.maps {
				bytes, err := encMode.Marshal(m)
				assert.NoError(t, err)
				encodings = append(encodings, bytes)
			}

			// All encodings must be identical
			expected := encodings[0]
			for i := 1; i < len(encodings); i++ {
				assert.Equal(t, expected, encodings[i],
					"Map %d encoding differs from map 0: %s", i, tc.desc)
			}
		})
	}
}

func TestHash_ExtraDifferentiation(t *testing.T) {
	// Create a canonical CBOR encoder
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name   string
		extra1 interface{}
		extra2 interface{}
		desc   string
	}{
		{
			name:   "nil_vs_zero",
			extra1: nil,
			extra2: 0,
			desc:   "nil should differ from zero",
		},
		{
			name:   "different_ints",
			extra1: 42,
			extra2: 99,
			desc:   "Different LoRA IDs",
		},
		{
			name:   "different_strings",
			extra1: "gpu",
			extra2: "cpu",
			desc:   "Different medium IDs",
		},
		{
			name:   "string_vs_int",
			extra1: "42",
			extra2: 42,
			desc:   "String vs int, type matters",
		},
		{
			name:   "map_different_values",
			extra1: map[string]interface{}{"lora_id": 42},
			extra2: map[string]interface{}{"lora_id": 99},
			desc:   "Maps with different values",
		},
		{
			name:   "map_different_keys",
			extra1: map[string]interface{}{"lora_id": 42},
			extra2: map[string]interface{}{"lora_adapter": 42},
			desc:   "Maps with different values but same values",
		},
		{
			name:   "map_vs_nil",
			extra1: map[string]interface{}{"lora_id": 42},
			extra2: nil,
			desc:   "Maps with LoRA ID vs nil (no LoRA ID)",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			bytes1, err := encMode.Marshal(tc.extra1)
			require.NoError(t, err)

			bytes2, err := encMode.Marshal(tc.extra2)
			require.NoError(t, err)

			// These must be different
			assert.NotEqual(t, bytes1, bytes2,
				"CBOR encodings should differ: %s", tc.desc)
		})
	}
}

func TestHash_ExtraVLLMCompatibility(t *testing.T) {
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name     string
		extra    interface{}
		scenario string
	}{
		{
			name:     "no_lora_no_multimodal",
			extra:    nil,
			scenario: "Standard text-only prompt without LoRA adapter",
		},
		{
			name:     "lora_v0_single_adapter",
			extra:    42,
			scenario: "vLLM v0: single LoRA adapter with hash(lora_int_id)",
		},
		{
			name:     "lora_v1_simple_tuple",
			extra:    map[string]interface{}{"lora_id": 42, "mm_hash": nil, "cache_salt": nil},
			scenario: "vLLM v1: LoRA only (lora_id, mm_hash=None, cache_salt=None)",
		},
		{
			name:     "lora_v1_with_multimodal",
			extra:    map[string]interface{}{"lora_id": 42, "mm_hash": "blake3_abc123", "cache_salt": "xyz"},
			scenario: "vLLM v1: LoRA + multi-modal content with Blake3 hash",
		},
		{
			name:     "medium_identifier",
			extra:    "gpu",
			scenario: "Custom medium identifier for cache segmentation",
		},
		{
			name:     "structured_metadata",
			extra:    map[string]interface{}{"lora_id": 42, "medium": "gpu", "version": 1},
			scenario: "Complex metadata combining multiple differentiation factors",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			bytes, err := encMode.Marshal(tc.extra)
			require.NoError(t, err, "Should successfully encode: %s", tc.scenario)
			assert.NotEmpty(t, bytes, "Encoded bytes should not be empty: %s", tc.scenario)
		})
	}
}

func TestHash_ExtraTypeSupport(t *testing.T) {
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name      string
		extra     interface{}
		shouldErr bool
	}{
		// Supported types that must work
		{"nil", nil, false},
		{"int", 42, false},
		{"int64", int64(9223372036854775807), false},
		{"string", "adapter-name", false},
		{"map_string_int", map[string]interface{}{"id": 42}, false},
		{"map_string_string", map[string]interface{}{"name": "lora"}, false},
		{"map_mixed", map[string]interface{}{"id": 42, "name": "lora"}, false},
		{"bool", true, false},
		{"float", 3.14, false},
		{"slice_int", []interface{}{1, 2, 3}, false},
		{"nested_map", map[string]interface{}{"meta": map[string]interface{}{"v": 1}}, false},

		// Edge cases that should still work
		{"empty_string", "", false},
		{"empty_map", map[string]interface{}{}, false},
		{"zero", 0, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			bytes, err := encMode.Marshal(tc.extra)

			if tc.shouldErr {
				assert.Error(t, err, "Expected encoding to fail")
			} else {
				require.NoError(t, err, "Expected encoding to succeed")
				assert.NotEmpty(t, bytes, "Encoded bytes should not be empty")
			}
		})
	}
}

func TestHeterogeneousBlockSizeSupport(t *testing.T) {
	// Generate enough tokens to fill multiple blocks at various resolutions.
	// 512 tokens = 32 blocks (blockSize=16) = 2 blocks (blockSize=256)
	tokens := make([]uint32, 512)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i is small
	}

	modelName := "test-model"
	parentKey := kvblock.EmptyBlockHash

	// Helper to create a processor with a given block size.
	newProcessor := func(t *testing.T, blockSize int) kvblock.TokenProcessor {
		t.Helper()
		proc, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
			BlockSize: blockSize,
			HashSeed:  "test-seed",
		})
		require.NoError(t, err)
		return proc
	}

	t.Run("different block sizes produce different hashes", func(t *testing.T) {
		proc32 := newProcessor(t, 32)
		proc16 := newProcessor(t, 16)

		keys32, err := proc32.TokensToKVBlockKeys(parentKey, tokens, modelName, nil)
		require.NoError(t, err)
		keys16, err := proc16.TokensToKVBlockKeys(parentKey, tokens, modelName, nil)
		require.NoError(t, err)
		assert.NotEqual(t, keys32[0], keys16[0])
	})

	t.Run("correct key count per resolution", func(t *testing.T) {
		proc256 := newProcessor(t, 256)
		proc16 := newProcessor(t, 16)

		keys256, err := proc256.TokensToKVBlockKeys(parentKey, tokens, modelName, nil)
		require.NoError(t, err)
		keys16, err := proc16.TokensToKVBlockKeys(parentKey, tokens, modelName, nil)
		require.NoError(t, err)
		assert.Equal(t, 2, len(keys256))
		assert.Equal(t, 32, len(keys16))
	})

	t.Run("partial block produces no key", func(t *testing.T) {
		proc256 := newProcessor(t, 256)

		partialTokens := make([]uint32, 300)
		for i := range partialTokens {
			partialTokens[i] = uint32(i + 1) // #nosec G115 -- test data, i is small
		}
		keys, err := proc256.TokensToKVBlockKeys(parentKey, partialTokens, modelName, nil)
		require.NoError(t, err)
		// 300 / 256 = 1 full block, 44 leftover tokens are discarded
		require.Len(t, keys, 1)
	})

	t.Run("hash chains are independent", func(t *testing.T) {
		proc256 := newProcessor(t, 256)
		proc16 := newProcessor(t, 16)

		storageKeys, err := proc256.TokensToKVBlockKeys(parentKey, tokens, modelName, nil)
		require.NoError(t, err)
		gpuKeys, err := proc16.TokensToKVBlockKeys(parentKey, tokens, modelName, nil)
		require.NoError(t, err)

		gpuKeySet := make(map[kvblock.BlockHash]struct{}, len(gpuKeys))
		for _, k := range gpuKeys {
			gpuKeySet[k] = struct{}{}
		}

		for _, sk := range storageKeys {
			_, found := gpuKeySet[sk]
			assert.False(t, found, "storage key %d should not appear in GPU key set", sk)
		}
	})

	t.Run("parentKey propagates correctly", func(t *testing.T) {
		proc256 := newProcessor(t, 256)

		nonEmptyParent := kvblock.BlockHash(999999)
		keysWithParent, err := proc256.TokensToKVBlockKeys(nonEmptyParent, tokens, modelName, nil)
		require.NoError(t, err)
		keysWithoutParent, err := proc256.TokensToKVBlockKeys(parentKey, tokens, modelName, nil)
		require.NoError(t, err)

		require.Len(t, keysWithParent, 2)
		require.Len(t, keysWithoutParent, 2)
		assert.NotEqual(t, keysWithParent[0], keysWithoutParent[0],
			"different parent keys should produce different first hashes")
	})
}
