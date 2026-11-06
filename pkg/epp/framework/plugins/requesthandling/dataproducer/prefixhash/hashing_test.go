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

// A render-backend prompt's media regions carry the model's content-blind
// placeholder tokens: any media of the same size yields the same token IDs.
// Two prompts that differ only in media content must therefore hash
// differently, which requires the multimodal feature hashes to reach the
// blocks that span them.
func TestGetBlockHashes_MediaContentChangesBlockHashes(t *testing.T) {
	prompt := func(hash string) *fwksched.InferenceRequest {
		return &fwksched.InferenceRequest{
			TargetModel: "m",
			Body: &fwkrequest.InferenceRequestBody{
				TokenizedRequest: &fwkrequest.TokenizedRequest{
					Prompts: []fwkrequest.PromptTokens{{
						// Tokens 4..11 are the media region: content-blind
						// padding, identical for any media of this size.
						TokenIDs:           []uint32{1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 9, 10, 11, 12},
						MultiModalFeatures: []fwkrequest.MultiModalFeature{{Modality: "image", Hash: hash, Offset: 4, Length: 8}},
					}},
				},
			},
		}
	}

	cat := GetBlockHashes(context.Background(), prompt("hash-of-a-cat"), 4, testMaxPrefixBlocks)
	dog := GetBlockHashes(context.Background(), prompt("hash-of-a-dog"), 4, testMaxPrefixBlocks)
	require.NotEmpty(t, cat)
	assert.NotEqual(t, cat, dog, "different media content must change the block hashes")

	same := GetBlockHashes(context.Background(), prompt("hash-of-a-cat"), 4, testMaxPrefixBlocks)
	assert.Equal(t, cat, same, "the same media content must hash stably")
}

// Blocks outside a media span keep the featureless formula, and features that
// hold no tokens (zero length, or a span past the prompt) are ignored.
func TestGetBlockHashes_EmptyAndOutOfRangeFeaturesIgnored(t *testing.T) {
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	base := &fwksched.InferenceRequest{
		TargetModel: "m",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens}},
			},
		},
	}
	withJunk := &fwksched.InferenceRequest{
		TargetModel: "m",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{
					TokenIDs: tokens,
					MultiModalFeatures: []fwkrequest.MultiModalFeature{
						{Modality: "image", Hash: "zero-length", Offset: 2, Length: 0},
						{Modality: "image", Hash: "past-the-end", Offset: 100, Length: 4},
					},
				}},
			},
		},
	}

	baseHashes := GetBlockHashes(context.Background(), base, 4, testMaxPrefixBlocks)
	junkHashes := GetBlockHashes(context.Background(), withJunk, 4, testMaxPrefixBlocks)
	assert.Equal(t, baseHashes, junkHashes, "features holding no tokens must not change any block hash")
}

// A feature spanning a block boundary reaches both blocks' encodings, and two
// features in one block encode in position order.
func TestGetBlockHashes_FeaturesAcrossBlockBoundaries(t *testing.T) {
	request := &fwksched.InferenceRequest{
		TargetModel: "m",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{
					TokenIDs: []uint32{1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 10, 11, 12},
					// Spans blocks 1 and 2 (tokens 4..15 of 20, block size 4).
					MultiModalFeatures: []fwkrequest.MultiModalFeature{
						{Modality: "image", Hash: "spans-two-blocks", Offset: 4, Length: 12},
					},
				}},
			},
		},
	}
	hashes := GetBlockHashes(context.Background(), request, 4, testMaxPrefixBlocks)
	require.Len(t, hashes, 1)
	require.Len(t, hashes[0], 5)

	// Changing the spanning feature's content must change both blocks it
	// covers, and through the chain every block after them.
	changed := *request
	changed.Body = &fwkrequest.InferenceRequestBody{TokenizedRequest: &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{
			TokenIDs:           request.Body.TokenizedRequest.Prompts[0].TokenIDs,
			MultiModalFeatures: []fwkrequest.MultiModalFeature{{Modality: "image", Hash: "other-content", Offset: 4, Length: 12}},
		}},
	}}
	changedHashes := GetBlockHashes(context.Background(), &changed, 4, testMaxPrefixBlocks)
	assert.Equal(t, hashes[0][0], changedHashes[0][0], "the block before the media span is unchanged")
	for i := 1; i < len(hashes[0]); i++ {
		assert.NotEqual(t, hashes[0][i], changedHashes[0][i], "block %d spans or chains after the media", i)
	}
}

// The block encoding must not depend on the order a parser produced the
// feature list in: the same media at the same positions hashes identically
// however the features are ordered on arrival.
func TestGetBlockHashes_FeatureOrderIndependent(t *testing.T) {
	tokens := []uint32{1, 2, 3, 4, 5, 5, 5, 5, 6, 7, 7, 7, 8, 9, 10, 11}
	features := func(first, second fwkrequest.MultiModalFeature) []fwkrequest.MultiModalFeature {
		return []fwkrequest.MultiModalFeature{first, second}
	}
	a := fwkrequest.MultiModalFeature{Modality: "image", Hash: "left-media", Offset: 4, Length: 4}
	b := fwkrequest.MultiModalFeature{Modality: "image", Hash: "right-media", Offset: 8, Length: 4}

	sorted := GetBlockHashes(context.Background(), &fwksched.InferenceRequest{
		TargetModel: "m",
		Body: &fwkrequest.InferenceRequestBody{TokenizedRequest: &fwkrequest.TokenizedRequest{
			Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens, MultiModalFeatures: features(a, b)}},
		}},
	}, 4, testMaxPrefixBlocks)
	reversed := GetBlockHashes(context.Background(), &fwksched.InferenceRequest{
		TargetModel: "m",
		Body: &fwkrequest.InferenceRequestBody{TokenizedRequest: &fwkrequest.TokenizedRequest{
			Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens, MultiModalFeatures: features(b, a)}},
		}},
	}, 4, testMaxPrefixBlocks)

	assert.Equal(t, sorted, reversed, "feature arrival order must not change the block hashes")
}
