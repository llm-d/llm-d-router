package prefixstore //nolint:testpackage // convenience

import (
	"testing"

	"github.com/daulet/tokenizers"
	"github.com/stretchr/testify/assert"
)

func TestLRUTokenStore_AddAndRetrieve(t *testing.T) {
	store, err := NewLRUTokenStore(&Config{
		&LRUStoreConfig{CacheSize: defaultMaxCacheSize, BlockSize: 4},
	})
	assert.NoError(t, err)

	modelName := "test-model"
	text := "The capital of France is Paris"
	tokens := []uint32{1, 2, 3, 4, 5, 6}
	offsets := []tokenizers.Offset{
		{0, 3}, {4, 11}, {12, 14}, {15, 21}, {22, 24}, {25, 30},
	}

	// Add tokenization to the store
	err = store.AddTokenization(modelName, text, tokens, offsets)
	assert.NoError(t, err)

	// Retrieve tokens for a matching prefix
	prompt := "The capital of F"
	result := store.FindLongestContainedTokens(prompt, modelName)
	assert.Equal(t, []uint32{1, 2, 3}, result)
}

func TestLRUTokenStore_LRUEviction(t *testing.T) {
	cfg := &Config{
		&LRUStoreConfig{CacheSize: 2, BlockSize: 18}, // Small cache size for testing eviction
	}
	store, err := NewLRUTokenStore(cfg)
	assert.NoError(t, err)

	modelName := "test-model"
	texts := []string{
		"abcdefghjiklmno",
		"123456789011121314",
		"pqrstuvwxyz,./';lp",
	}
	tokens := [][]uint32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	offsets := [][]tokenizers.Offset{
		{{0, 5}, {6, 10}, {11, 15}},
		{{0, 6}, {7, 12}, {13, 18}},
		{{0, 6}, {7, 12}, {13, 18}},
	}

	// Add tokenizations to the store
	for i, text := range texts {
		err = store.AddTokenization(modelName, text, tokens[i], offsets[i])
		assert.NoError(t, err)
	}

	// First text block should be evicted
	prompt := "abcdefghjiklmno"
	result := store.FindLongestContainedTokens(prompt, modelName)
	assert.Empty(t, result, "First text block should be evicted")

	// Third text block should still be in cache
	prompt = "pqrstuvwxyz,./';lp"
	result = store.FindLongestContainedTokens(prompt, modelName)
	assert.Equal(t, []uint32{7, 8, 9}, result)
}
