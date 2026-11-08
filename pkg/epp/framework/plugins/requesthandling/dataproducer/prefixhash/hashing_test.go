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

package prefixhash

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// testMaxPrefixBlocks is a cap large enough not to truncate any prompt in
// these tests.
const testMaxPrefixBlocks = 2048

func TestGetBlockHashes(t *testing.T) {
	tests := []struct {
		name            string
		request         *fwksched.InferenceRequest
		blockSizeTokens int
		expectedBlocks  int
	}{
		{
			name: "TokenizedRequest",
			request: &fwksched.InferenceRequest{
				Body: &fwkrequest.InferenceRequestBody{
					TokenizedRequest: &fwkrequest.TokenizedRequest{
						Prompts: []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}},
					},
				},
			},
			blockSizeTokens: 4,
			expectedBlocks:  3,
		},
		{
			name: "BlockSizeNearMaxInt",
			request: &fwksched.InferenceRequest{
				Body: &fwkrequest.InferenceRequestBody{
					TokenizedRequest: &fwkrequest.TokenizedRequest{
						Prompts: []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}},
					},
				},
			},
			blockSizeTokens: math.MaxInt,
			expectedBlocks:  1,
		},
		{
			name: "MissingTokenizedRequest",
			request: &fwksched.InferenceRequest{
				Body: &fwkrequest.InferenceRequestBody{},
			},
			blockSizeTokens: 4,
			expectedBlocks:  0,
		},
		{
			name: "EmptyTokenIDs",
			request: &fwksched.InferenceRequest{
				Body: &fwkrequest.InferenceRequestBody{
					TokenizedRequest: &fwkrequest.TokenizedRequest{},
				},
			},
			blockSizeTokens: 4,
			expectedBlocks:  0,
		},
		{
			name:            "NilRequest",
			request:         nil,
			blockSizeTokens: 4,
			expectedBlocks:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			perPromptHashes := GetBlockHashes(context.Background(), tt.request, tt.blockSizeTokens, testMaxPrefixBlocks)
			totalBlocks := 0
			for _, ph := range perPromptHashes {
				totalBlocks += len(ph)
			}
			assert.Equal(t, tt.expectedBlocks, totalBlocks)
		})
	}
}

func TestGetBlockHashesCacheSalt(t *testing.T) {
	body := func(salt string) *fwkrequest.InferenceRequestBody {
		return &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts:   []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2, 3, 4}}},
				CacheSalt: salt,
			},
		}
	}

	noSalt := GetBlockHashes(context.Background(), &fwksched.InferenceRequest{
		TargetModel: "m", Body: body(""),
	}, 2, testMaxPrefixBlocks)
	salted := GetBlockHashes(context.Background(), &fwksched.InferenceRequest{
		TargetModel: "m", Body: body("salt"),
	}, 2, testMaxPrefixBlocks)

	assert.Equal(t, len(noSalt), len(salted))
	assert.NotEqual(t, noSalt, salted, "cache salt must change the block hashes")
}

func TestGetBlockHashes_TextToSpeechCacheSalt(t *testing.T) {
	ctx := context.Background()
	p, err := tokenizer.PluginFactory("tokens", plugin.StrictDecoder(nil), plugin.NewEppHandle(ctx, nil))
	require.NoError(t, err)
	producer, ok := p.(*tokenizer.Plugin)
	require.True(t, ok)
	parser := openai.NewOpenAIParser()

	hashesFor := func(salt string) [][]BlockHash {
		payload := map[string]any{
			"model":        "tts",
			"input":        "Welcome to customer support.",
			"instructions": "Speak clearly.",
			"ref_text":     "This is the reference clip.",
		}
		if salt != "" {
			payload["cache_salt"] = salt
		}
		raw, err := json.Marshal(payload)
		require.NoError(t, err)
		parsed, err := parser.ParseRequest(ctx, raw, map[string]string{":path": "/v1/audio/speech"})
		require.NoError(t, err)
		req := &fwksched.InferenceRequest{TargetModel: "tts", Body: parsed.Body}
		require.NoError(t, producer.Produce(ctx, req, nil))
		require.NotNil(t, req.Body.TokenizedRequest)
		hashes := GetBlockHashes(ctx, req, 2, testMaxPrefixBlocks)
		require.NotEmpty(t, hashes)
		return hashes
	}

	tenantA := hashesFor("tenant-a")
	assert.Equal(t, tenantA, hashesFor("tenant-a"))
	assert.NotEqual(t, tenantA, hashesFor("tenant-b"), "different tenants must not match")
	assert.NotEqual(t, tenantA, hashesFor(""), "salted and unsalted requests must not match")
}

// TestGetBlockHashes_ModelSaltBoundary covers (model, salt) pairs whose
// concatenation is identical. The seed must still separate them.
func TestGetBlockHashes_ModelSaltBoundary(t *testing.T) {
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	hashesFor := func(model, salt string) [][]BlockHash {
		return GetBlockHashes(context.Background(), &fwksched.InferenceRequest{
			TargetModel: model,
			Body: &fwkrequest.InferenceRequestBody{
				TokenizedRequest: &fwkrequest.TokenizedRequest{
					Prompts:   []fwkrequest.PromptTokens{{TokenIDs: tokens}},
					CacheSalt: salt,
				},
			},
		}, 2, testMaxPrefixBlocks)
	}

	assert.NotEqual(t, hashesFor("ab", "c"), hashesFor("a", "bc"),
		"model and salt must not collide when their concatenations match")
	assert.NotEqual(t, hashesFor("llama-3-8b", ""), hashesFor("llama-3", "-8b"),
		"empty salt must not collide with a same-length model split")
}

func TestGetBlockHashes_MultiPrompt(t *testing.T) {
	tests := []struct {
		name                    string
		prompts                 []fwkrequest.PromptTokens
		blockSizeTokens         int
		expectedPrompts         int
		expectedBlocksPerPrompt []int
	}{
		{
			name:                    "TwoPrompts",
			prompts:                 []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2, 3, 4}}, {TokenIDs: []uint32{5, 6, 7, 8}}},
			blockSizeTokens:         2,
			expectedPrompts:         2,
			expectedBlocksPerPrompt: []int{2, 2},
		},
		{
			name:                    "ThreePromptsUnevenLengths",
			prompts:                 []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}, {TokenIDs: []uint32{4, 5}}, {TokenIDs: []uint32{6}}},
			blockSizeTokens:         2,
			expectedPrompts:         3,
			expectedBlocksPerPrompt: []int{2, 1, 1},
		},
		{
			name:                    "EmptyPromptSkipped",
			prompts:                 []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2}}, {TokenIDs: []uint32{}}, {TokenIDs: []uint32{3, 4}}},
			blockSizeTokens:         2,
			expectedPrompts:         2,
			expectedBlocksPerPrompt: []int{1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &fwksched.InferenceRequest{
				TargetModel: "model",
				Body: &fwkrequest.InferenceRequestBody{
					TokenizedRequest: &fwkrequest.TokenizedRequest{
						Prompts: tt.prompts,
					},
				},
			}
			result := GetBlockHashes(context.Background(), request, tt.blockSizeTokens, testMaxPrefixBlocks)
			assert.Equal(t, tt.expectedPrompts, len(result))
			for i, expected := range tt.expectedBlocksPerPrompt {
				assert.Equal(t, expected, len(result[i]), "prompt %d block count", i)
			}
		})
	}
}

func TestGetBlockHashes_MultiPromptHashIndependence(t *testing.T) {
	multiPrompt := &fwksched.InferenceRequest{
		TargetModel: "model",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2}}, {TokenIDs: []uint32{3, 4}}},
			},
		},
	}
	singlePrompt := &fwksched.InferenceRequest{
		TargetModel: "model",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: []uint32{1, 2, 3, 4}}},
			},
		},
	}

	multi := GetBlockHashes(context.Background(), multiPrompt, 2, testMaxPrefixBlocks)
	single := GetBlockHashes(context.Background(), singlePrompt, 2, testMaxPrefixBlocks)

	assert.Equal(t, 2, len(multi), "multi-prompt should produce 2 inner slices")
	assert.Equal(t, 1, len(single), "single-prompt should produce 1 inner slice")

	// First block of each starts from the same model seed, so [1,2] hashes match.
	assert.Equal(t, multi[0][0], single[0][0],
		"first block of first prompt should match single prompt's first block")
	// Second prompt's first block [3,4] starts a fresh hash chain from the model
	// seed, while single prompt's second block [3,4] chains from the first block.
	// These must differ - this is the cross-prompt adjacency bug the per-prompt
	// split prevents.
	assert.NotEqual(t, multi[1][0], single[0][1],
		"second prompt must start its own hash chain, not chain from the first prompt")
}

func TestKVCacheBlock_Hash(t *testing.T) {
	tests := []struct {
		name     string
		blockA   HashBlock
		blockB   HashBlock
		shouldEq bool
	}{
		{
			name:     "Identical token IDs",
			blockA:   HashBlock{Tokens: []uint32{1, 2}},
			blockB:   HashBlock{Tokens: []uint32{1, 2}},
			shouldEq: true,
		},
		{
			name:     "Different token IDs",
			blockA:   HashBlock{Tokens: []uint32{1, 2}},
			blockB:   HashBlock{Tokens: []uint32{1, 3}},
			shouldEq: false,
		},
		{
			name:     "Empty fields match",
			blockA:   HashBlock{},
			blockB:   HashBlock{},
			shouldEq: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hashA := tt.blockA.Hash()
			hashB := tt.blockB.Hash()
			if tt.shouldEq {
				assert.Equal(t, hashA, hashB)
			} else {
				assert.NotEqual(t, hashA, hashB)
			}
		})
	}
}

// mmPromptTokens is three 4-token blocks: text in tokens 0-3, one multimodal
// item rendered as a run of mmPlaceholder in tokens 4-9, and text in tokens
// 10-11. Placeholder IDs depend on the item's size, not its content, so only
// the feature hash tells two same-size items apart.
const mmPlaceholder = 9

var mmPromptTokens = []uint32{1, 2, 3, 4, mmPlaceholder, mmPlaceholder, mmPlaceholder, mmPlaceholder, mmPlaceholder, mmPlaceholder, 5, 6}

func mmHashes(t *testing.T, tokens []uint32, features ...fwkrequest.MultiModalFeature) []BlockHash {
	t.Helper()
	hashes := GetBlockHashes(context.Background(), &fwksched.InferenceRequest{
		TargetModel: "m",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens, MultiModalFeatures: features}},
			},
		},
	}, 4, testMaxPrefixBlocks)
	require.Len(t, hashes, 1)
	return hashes[0]
}

func image(hash string, offset, length int) fwkrequest.MultiModalFeature {
	return fwkrequest.MultiModalFeature{Modality: reqcommon.ModalityImage, Hash: hash, Offset: offset, Length: length}
}

func TestGetBlockHashes_MultiModal(t *testing.T) {
	a := mmHashes(t, mmPromptTokens, image("a", 4, 6))
	require.Len(t, a, 3)

	t.Run("same item matches", func(t *testing.T) {
		assert.Equal(t, a, mmHashes(t, mmPromptTokens, image("a", 4, 6)))
	})

	t.Run("different item diverges from the first overlapping block", func(t *testing.T) {
		b := mmHashes(t, mmPromptTokens, image("b", 4, 6))
		assert.Equal(t, a[0], b[0], "block before the item must still match")
		assert.NotEqual(t, a[1], b[1])
		assert.NotEqual(t, a[2], b[2])
	})

	t.Run("blocks before any item hash as text", func(t *testing.T) {
		assert.Equal(t, mmHashes(t, mmPromptTokens)[0], a[0])
	})

	t.Run("item inside the final partial block", func(t *testing.T) {
		tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, mmPlaceholder, mmPlaceholder}
		x := mmHashes(t, tokens, image("x", 8, 2))
		y := mmHashes(t, tokens, image("y", 8, 2))
		require.Len(t, x, 3)
		assert.Equal(t, x[:2], y[:2])
		assert.NotEqual(t, x[2], y[2])
	})

	t.Run("each item in a block shared by two items counts", func(t *testing.T) {
		tokens := []uint32{1, 2, mmPlaceholder, mmPlaceholder, mmPlaceholder, mmPlaceholder, mmPlaceholder, mmPlaceholder, 5, 6, 7, 8}
		base := mmHashes(t, tokens, image("a", 2, 3), image("b", 5, 3))
		otherFirst := mmHashes(t, tokens, image("z", 2, 3), image("b", 5, 3))
		otherSecond := mmHashes(t, tokens, image("a", 2, 3), image("z", 5, 3))
		assert.NotEqual(t, base[0], otherFirst[0])
		assert.Equal(t, base[0], otherSecond[0], "block 0 does not overlap the second item")
		assert.NotEqual(t, base[1], otherSecond[1], "block 1 overlaps both items")
	})
}

func TestWithMMHashes(t *testing.T) {
	const tokenHash = uint64(42)

	t.Run("item ending at the block start does not overlap", func(t *testing.T) {
		assert.Equal(t, tokenHash, withMMHashes(tokenHash, []fwkrequest.MultiModalFeature{image("a", 0, 4)}, 4, 8))
	})

	t.Run("item starting at the block end does not overlap", func(t *testing.T) {
		assert.Equal(t, tokenHash, withMMHashes(tokenHash, []fwkrequest.MultiModalFeature{image("a", 8, 4)}, 4, 8))
	})

	t.Run("item covering one boundary token overlaps", func(t *testing.T) {
		assert.NotEqual(t, tokenHash, withMMHashes(tokenHash, []fwkrequest.MultiModalFeature{image("a", 0, 5)}, 4, 8))
		assert.NotEqual(t, tokenHash, withMMHashes(tokenHash, []fwkrequest.MultiModalFeature{image("a", 7, 4)}, 4, 8))
	})

	t.Run("item hashes are length-prefixed", func(t *testing.T) {
		abC := withMMHashes(tokenHash, []fwkrequest.MultiModalFeature{image("ab", 4, 1), image("c", 5, 1)}, 4, 8)
		aBC := withMMHashes(tokenHash, []fwkrequest.MultiModalFeature{image("a", 4, 1), image("bc", 5, 1)}, 4, 8)
		assert.NotEqual(t, abC, aBC)
	})
}
