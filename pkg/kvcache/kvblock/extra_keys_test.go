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

package kvblock_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// ---------------------------------------------------------------------------
// ParseRawExtraKeys
// ---------------------------------------------------------------------------

func TestParseRawExtraKeys_Nil(t *testing.T) {
	result, err := kvblock.ParseRawExtraKeys(nil)
	require.NoError(t, err)
	assert.Nil(t, result)
}

func TestParseRawExtraKeys_NilInnerEntries(t *testing.T) {
	raw := [][]any{nil, nil, nil}
	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.Len(t, result, 3)
	for _, r := range result {
		assert.Nil(t, r)
	}
}

func TestParseRawExtraKeys_BareStringIdentifiers(t *testing.T) {
	// vLLM v0.18.0+ sends bare identifier strings per block.
	raw := [][]any{
		{"hash_A"},           // one MM entry
		{"hash_A"},           // continuation block
		nil,                  // text block
		{"hash_B"},           // different image
		{"hash_B", "hash_C"}, // two overlapping images
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.Len(t, result, 5)

	require.NotNil(t, result[0])
	assert.Equal(t, "hash_A", result[0].MMHashes[0].Hash)

	require.NotNil(t, result[1])
	assert.Equal(t, "hash_A", result[1].MMHashes[0].Hash)

	assert.Nil(t, result[2])

	require.NotNil(t, result[3])
	assert.Equal(t, "hash_B", result[3].MMHashes[0].Hash)

	require.NotNil(t, result[4])
	require.Len(t, result[4].MMHashes, 2)
	assert.Equal(t, "hash_B", result[4].MMHashes[0].Hash)
	assert.Equal(t, "hash_C", result[4].MMHashes[1].Hash)
}

func TestParseRawExtraKeys_LegacyTuples(t *testing.T) {
	// Legacy format: [hash, offset] tuples. Offset is ignored.
	raw := [][]any{
		{[]any{"hash_A", int64(15)}},
		{[]any{"hash_A", int64(-1)}},
		nil,
		{[]any{"hash_B", int64(2)}},
		{[]any{"hash_B", int64(-14)}, []any{"hash_C", int64(5)}},
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.Len(t, result, 5)

	require.NotNil(t, result[0])
	assert.Equal(t, "hash_A", result[0].MMHashes[0].Hash)

	require.NotNil(t, result[1])
	assert.Equal(t, "hash_A", result[1].MMHashes[0].Hash)

	assert.Nil(t, result[2])

	require.NotNil(t, result[3])
	assert.Equal(t, "hash_B", result[3].MMHashes[0].Hash)

	require.NotNil(t, result[4])
	require.Len(t, result[4].MMHashes, 2)
	assert.Equal(t, "hash_B", result[4].MMHashes[0].Hash)
	assert.Equal(t, "hash_C", result[4].MMHashes[1].Hash)
}

func TestParseRawExtraKeys_SkipsUnknownEntryTypes(t *testing.T) {
	// Non-string, non-tuple entries (e.g. numeric LoRA ids) should be skipped.
	raw := [][]any{
		{int64(42), "valid_hash"},
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.NotNil(t, result[0])
	require.Len(t, result[0].MMHashes, 1)
	assert.Equal(t, "valid_hash", result[0].MMHashes[0].Hash)
}

// ---------------------------------------------------------------------------
// ComputeBlockExtraFeatures
// ---------------------------------------------------------------------------

func TestComputeBlockExtraFeatures_NoOverlap(t *testing.T) {
	result := kvblock.ComputeBlockExtraFeatures(nil, nil, 16, 64)
	assert.Nil(t, result)
}

func TestComputeBlockExtraFeatures_SingleImage(t *testing.T) {
	// Image occupies tokens 0..47 (3 full blocks of size 16).
	mmHashes := map[string][]string{"image": {"hash_A"}}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {{Offset: 0, Length: 48}},
	}

	result := kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, 16, 64)
	require.Len(t, result, 4) // 64/16 = 4 blocks

	// Blocks 0-2 should have the image identifier.
	require.NotNil(t, result[0])
	assert.Equal(t, "hash_A", result[0].MMHashes[0].Hash)

	require.NotNil(t, result[1])
	assert.Equal(t, "hash_A", result[1].MMHashes[0].Hash)

	require.NotNil(t, result[2])
	assert.Equal(t, "hash_A", result[2].MMHashes[0].Hash)

	// Block 3 has no image overlap.
	assert.Nil(t, result[3])
}

func TestComputeBlockExtraFeatures_TextOnlyBlocksBetweenImages(t *testing.T) {
	// Two images with a gap: image 1 tokens [0, 32), text tokens [32, 48), image 2 tokens [48, 80).
	mmHashes := map[string][]string{"image": {"hashA", "hashB"}}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {{Offset: 0, Length: 32}, {Offset: 48, Length: 32}},
	}

	result := kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, 16, 80)
	require.Len(t, result, 5) // 80/16 = 5 blocks

	// Blocks 0,1: image A
	require.NotNil(t, result[0])
	assert.Equal(t, "hashA", result[0].MMHashes[0].Hash)
	require.NotNil(t, result[1])
	assert.Equal(t, "hashA", result[1].MMHashes[0].Hash)

	// Block 2: text only [32, 48) — no overlap with either image.
	assert.Nil(t, result[2])

	// Blocks 3,4: image B
	require.NotNil(t, result[3])
	assert.Equal(t, "hashB", result[3].MMHashes[0].Hash)
	require.NotNil(t, result[4])
	assert.Equal(t, "hashB", result[4].MMHashes[0].Hash)
}

// ---------------------------------------------------------------------------
// MM features affect block hashes — distinguishability
// ---------------------------------------------------------------------------

func TestMMFeatures_DifferentImagesProduceDifferentHashes(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSizeTokens: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	tokens := make([]uint32, 16) // one block
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i < 64
	}

	// Same tokens, different image hashes → different block hashes.
	featA := []*kvblock.BlockExtraFeatures{
		{MMHashes: []kvblock.MMHash{{Hash: "image_hash_A"}}},
	}
	featB := []*kvblock.BlockExtraFeatures{
		{MMHashes: []kvblock.MMHash{{Hash: "image_hash_B"}}},
	}

	keysA, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", featA)
	require.NoError(t, err)
	keysB, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", featB)
	require.NoError(t, err)

	assert.NotEqual(t, keysA[0], keysB[0],
		"blocks with different image hashes must produce different block keys")
}

func TestMMFeatures_NilFeaturesSameAsTextOnly(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSizeTokens: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	tokens := make([]uint32, 16)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i < 64
	}

	// nil extraFeatures = text-only
	keysNil, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	// Explicit nil entry per block = also text-only
	keysExplicitNil, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model",
		[]*kvblock.BlockExtraFeatures{nil})
	require.NoError(t, err)

	assert.Equal(t, keysNil[0], keysExplicitNil[0],
		"nil extraFeatures and explicit nil-per-block must produce the same hash")
}

func TestMMFeatures_OnlyAffectOverlappingBlocks(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSizeTokens: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	// 4 blocks of tokens
	tokens := make([]uint32, 64)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i < 64
	}

	// Text-only baseline
	keysTextOnly, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	require.Len(t, keysTextOnly, 4)

	// Image only in block 2 (index 2)
	features := []*kvblock.BlockExtraFeatures{
		nil, // block 0: text
		nil, // block 1: text
		{MMHashes: []kvblock.MMHash{{Hash: "image_X"}}}, // block 2: image
		nil, // block 3: text
	}
	keysWithImage, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", features)
	require.NoError(t, err)
	require.Len(t, keysWithImage, 4)

	// Blocks 0 and 1 should be identical (prefix chain before image block).
	assert.Equal(t, keysTextOnly[0], keysWithImage[0], "block 0 should be unaffected by image in block 2")
	assert.Equal(t, keysTextOnly[1], keysWithImage[1], "block 1 should be unaffected by image in block 2")

	// Block 2 should differ (image taints the hash).
	assert.NotEqual(t, keysTextOnly[2], keysWithImage[2], "block 2 should be affected by image")

	// Block 3 should also differ because the prefix chain changed at block 2.
	assert.NotEqual(t, keysTextOnly[3], keysWithImage[3],
		"block 3 hash should change because it chains from the tainted block 2")
}

func TestMMFeatures_MismatchedLengthReturnsError(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSizeTokens: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	tokens := make([]uint32, 32)                   // 2 chunks
	features := []*kvblock.BlockExtraFeatures{nil} // 1 entry — mismatch

	_, err = processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", features)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not match token chunk count")
}

// ---------------------------------------------------------------------------
// ParseRawExtraKeys ↔ ComputeBlockExtraFeatures consistency
// ---------------------------------------------------------------------------

func TestParseAndComputeProduceSameFeatures(t *testing.T) {
	// Simulate: vLLM sends bare identifiers for 3 blocks where an image
	// overlaps blocks 0-2. ComputeBlockExtraFeatures should produce
	// identical MMHash values.
	raw := [][]any{
		{"img_hash"}, // block 0
		{"img_hash"}, // block 1
		{"img_hash"}, // block 2
		nil,          // block 3: text only
	}

	parsed, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)

	mmHashes := map[string][]string{"image": {"img_hash"}}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {{Offset: 0, Length: 48}}, // spans blocks 0-2 at blockSize=16
	}
	computed := kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, 16, 64)

	require.Len(t, parsed, 4)
	require.Len(t, computed, 4)

	for i := 0; i < 4; i++ {
		if parsed[i] == nil {
			assert.Nil(t, computed[i], "block %d: both should be nil", i)
		} else {
			require.NotNil(t, computed[i], "block %d: both should be non-nil", i)
			assert.Equal(t, parsed[i].MMHashes, computed[i].MMHashes,
				"block %d: parsed and computed MMHashes must match", i)
		}
	}
}
