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

package approximateprefix

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

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
			var blocks []HashBlock
			for block := range getKVCacheBlocksFromTokens(tt.ids, tt.blockSizeTokens) {
				blocks = append(blocks, block)
			}
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
						TokenIDs: []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
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
			hashes := getBlockHashes(context.Background(), tt.request, tt.blockSizeTokens, defaultMaxPrefixBlocks)
			assert.Equal(t, tt.expectedBlocks, len(hashes))
		})
	}
}

func TestGetBlockHashesCacheSalt(t *testing.T) {
	body := func(salt string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{CacheSalt: salt},
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				TokenIDs: []uint32{1, 2, 3, 4},
			},
		}
	}

	noSalt := getBlockHashes(context.Background(), &fwksched.InferenceRequest{
		TargetModel: "m", Body: body(""),
	}, 2, defaultMaxPrefixBlocks)
	salted := getBlockHashes(context.Background(), &fwksched.InferenceRequest{
		TargetModel: "m", Body: body("salt"),
	}, 2, defaultMaxPrefixBlocks)

	assert.Equal(t, len(noSalt), len(salted))
	assert.NotEqual(t, noSalt, salted, "cache salt must change the block hashes")
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
