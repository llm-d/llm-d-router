/*
Copyright 2026 The Kubernetes Authors.

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
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// testMaxPrefixBlocks is a cap large enough not to truncate any prompt in
// these tests.
const testMaxPrefixBlocks = 2048

func TestGetKVCacheBlocksFromTokens(t *testing.T) {
	tests := []struct {
		name            string
		ids             []uint32
		blockSizeTokens int
		expected        []HashBlock
	}{
		{
			name:            "EvenSplit",
			ids:             []uint32{1, 2, 3, 4, 5, 6, 7, 8},
			blockSizeTokens: 4,
			expected: []HashBlock{
				{Tokens: []uint32{1, 2, 3, 4}},
				{Tokens: []uint32{5, 6, 7, 8}},
			},
		},
		{
			name:            "TrailingPartialBlock",
			ids:             []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			blockSizeTokens: 4,
			expected: []HashBlock{
				{Tokens: []uint32{1, 2, 3, 4}},
				{Tokens: []uint32{5, 6, 7, 8}},
				{Tokens: []uint32{9, 10}},
			},
		},
		{
			name:            "EmptyTokens",
			ids:             nil,
			blockSizeTokens: 4,
			expected:        nil,
		},
		{
			name:            "NonPositiveBlockSize",
			ids:             []uint32{1, 2, 3},
			blockSizeTokens: 0,
			expected:        nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blocks := slices.Collect(getKVCacheBlocksFromTokens(tt.ids, tt.blockSizeTokens))
			assert.Equal(t, tt.expected, blocks)
		})
	}
}

func TestGetBlockHashes(t *testing.T) {
	tests := []struct {
		name            string
		request         *fwksched.InferenceRequest
		blockSizeTokens int
		expectedBlocks  int
	}{
		{
			name: "TokenizedPrompt",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &fwkrh.TokenizedPrompt{
						PerPromptTokens: [][]uint32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
					},
				},
			},
			blockSizeTokens: 4,
			expectedBlocks:  3,
		},
		{
			name: "MissingTokenizedPrompt",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{},
			},
			blockSizeTokens: 4,
			expectedBlocks:  0,
		},
		{
			name: "EmptyTokenIDs",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &fwkrh.TokenizedPrompt{},
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
	body := func(salt string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{{1, 2, 3, 4}},
				CacheSalt:       salt,
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

func TestGetBlockHashes_MultiPrompt(t *testing.T) {
	tests := []struct {
		name                    string
		perPromptTokens         [][]uint32
		blockSizeTokens         int
		expectedPrompts         int
		expectedBlocksPerPrompt []int
	}{
		{
			name:                    "TwoPrompts",
			perPromptTokens:         [][]uint32{{1, 2, 3, 4}, {5, 6, 7, 8}},
			blockSizeTokens:         2,
			expectedPrompts:         2,
			expectedBlocksPerPrompt: []int{2, 2},
		},
		{
			name:                    "ThreePromptsUnevenLengths",
			perPromptTokens:         [][]uint32{{1, 2, 3}, {4, 5}, {6}},
			blockSizeTokens:         2,
			expectedPrompts:         3,
			expectedBlocksPerPrompt: []int{2, 1, 1},
		},
		{
			name:                    "EmptyPromptSkipped",
			perPromptTokens:         [][]uint32{{1, 2}, {}, {3, 4}},
			blockSizeTokens:         2,
			expectedPrompts:         2,
			expectedBlocksPerPrompt: []int{1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &fwksched.InferenceRequest{
				TargetModel: "model",
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &fwkrh.TokenizedPrompt{
						PerPromptTokens: tt.perPromptTokens,
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
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{{1, 2}, {3, 4}},
			},
		},
	}
	singlePrompt := &fwksched.InferenceRequest{
		TargetModel: "model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{{1, 2, 3, 4}},
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
