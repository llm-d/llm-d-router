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

package preciseprefixcache

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

// computeBlockKeys returns one MM content value per returned prompt, aligned
// positionally with the keys; prompts with no keys are skipped from both
// slices.
func TestComputeBlockKeys_PerPromptMMContent(t *testing.T) {
	ctx := utils.NewTestContext(t)

	twoBlocks := make([]uint32, 2*testBlockSize)
	for i := range twoBlocks {
		twoBlocks[i] = uint32(i)
	}
	short := []uint32{1, 2, 3}

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, ts []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			if len(ts) < testBlockSize {
				return nil, nil
			}
			return []kvblock.BlockHash{0xA1, 0xA2}, nil
		},
	}

	img := fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage, Hash: "a", Offset: 0, Length: 16}
	audio := fwkrh.MultiModalFeature{Modality: fwkrh.ModalityAudio, Hash: "b", Offset: 16, Length: 16}

	req := &scheduling.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{
					{TokenIDs: twoBlocks, MultiModalFeatures: []fwkrh.MultiModalFeature{img}},
					{TokenIDs: short},
					{TokenIDs: twoBlocks},
					{TokenIDs: twoBlocks, MultiModalFeatures: []fwkrh.MultiModalFeature{audio}},
				},
			},
		},
	}

	keys, mmContent, err := computeBlockKeys(ctx, idx, req, testBlockSize)
	require.NoError(t, err)
	require.Len(t, keys, 3, "the short prompt yields no keys and is skipped")
	assert.Equal(t, []*mmPromptContent{
		{blockIndices: []int{0}, features: []fwkrh.MultiModalFeature{img}},
		nil,
		{blockIndices: []int{1}, features: []fwkrh.MultiModalFeature{audio}},
	}, mmContent)
}

func TestMultimodalBlockIndices(t *testing.T) {
	tests := []struct {
		name            string
		features        []fwkrh.MultiModalFeature
		blockSizeTokens int
		want            []int
	}{
		{
			name:            "empty features",
			features:        nil,
			blockSizeTokens: 16,
			want:            nil,
		},
		{
			name:            "zero block size",
			features:        []fwkrh.MultiModalFeature{{Offset: 0, Length: 16}},
			blockSizeTokens: 0,
			want:            nil,
		},
		{
			name: "single feature spanning one block",
			features: []fwkrh.MultiModalFeature{
				{Offset: 0, Length: 16},
			},
			blockSizeTokens: 16,
			want:            []int{0},
		},
		{
			name: "single feature spanning multiple blocks",
			features: []fwkrh.MultiModalFeature{
				{Offset: 8, Length: 40},
			},
			blockSizeTokens: 16,
			want:            []int{0, 1, 2},
		},
		{
			name: "multiple features deduplicated and sorted",
			features: []fwkrh.MultiModalFeature{
				{Offset: 32, Length: 16},
				{Offset: 0, Length: 16},
				{Offset: 16, Length: 16},
			},
			blockSizeTokens: 16,
			want:            []int{0, 1, 2},
		},
		{
			name: "zero-length feature skipped",
			features: []fwkrh.MultiModalFeature{
				{Offset: 0, Length: 0},
				{Offset: 16, Length: 16},
			},
			blockSizeTokens: 16,
			want:            []int{1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := multimodalBlockIndices(tt.features, tt.blockSizeTokens)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestCountMMMatchedTokens(t *testing.T) {
	tests := []struct {
		name            string
		features        []fwkrh.MultiModalFeature
		matchLen        int
		blockSizeTokens int
		want            int
	}{
		{name: "no features", features: nil, matchLen: 5, blockSizeTokens: 16, want: 0},
		{name: "no match", features: []fwkrh.MultiModalFeature{{Offset: 0, Length: 16}}, matchLen: 0, blockSizeTokens: 16, want: 0},
		{
			name:            "fully matched feature",
			features:        []fwkrh.MultiModalFeature{{Offset: 0, Length: 256}},
			matchLen:        4,
			blockSizeTokens: 64,
			want:            256,
		},
		{
			name:            "feature ending mid-block counts only its tokens",
			features:        []fwkrh.MultiModalFeature{{Offset: 8, Length: 256}},
			matchLen:        5,
			blockSizeTokens: 64,
			want:            256,
		},
		{
			name:            "feature starting beyond the match counts nothing",
			features:        []fwkrh.MultiModalFeature{{Offset: 320, Length: 256}},
			matchLen:        5,
			blockSizeTokens: 64,
			want:            0,
		},
		{
			// A feature starting past the match edge must not subtract from
			// another feature's count: its overlap clamps at zero.
			name: "feature past the match edge adds nothing",
			features: []fwkrh.MultiModalFeature{
				{Offset: 0, Length: 256},
				{Offset: 400, Length: 256},
			},
			matchLen:        5,
			blockSizeTokens: 64,
			want:            256,
		},
		{
			name:            "partially matched feature counts the overlap",
			features:        []fwkrh.MultiModalFeature{{Offset: 100, Length: 50}},
			matchLen:        2,
			blockSizeTokens: 64,
			want:            28,
		},
		{
			// Two 256-token images at block 64 with only the first cached:
			// the matched MM blocks cover 5 blocks (320 tokens), but the
			// first image holds 256 and the second starts at the match edge.
			name: "two images, only the first matched",
			features: []fwkrh.MultiModalFeature{
				{Offset: 8, Length: 256},
				{Offset: 320, Length: 256},
			},
			matchLen:        5,
			blockSizeTokens: 64,
			want:            256,
		},
		{
			name: "zero-length feature skipped",
			features: []fwkrh.MultiModalFeature{
				{Offset: 0, Length: 0},
				{Offset: 16, Length: 16},
			},
			matchLen:        4,
			blockSizeTokens: 16,
			want:            16,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := countMMMatchedTokens(tt.features, tt.matchLen, tt.blockSizeTokens)
			assert.Equal(t, tt.want, got)
		})
	}
}
func TestCountMMMatchedBlocks(t *testing.T) {
	tests := []struct {
		name          string
		mmBlockIdx    []int
		matchLen      int
		wantMMMatched int
	}{
		{name: "no mm indices", mmBlockIdx: nil, matchLen: 5, wantMMMatched: 0},
		{name: "no match", mmBlockIdx: []int{0, 1, 2}, matchLen: 0, wantMMMatched: 0},
		{name: "all mm indices matched", mmBlockIdx: []int{0, 1, 2}, matchLen: 3, wantMMMatched: 3},
		{name: "partial mm match", mmBlockIdx: []int{0, 5, 10}, matchLen: 6, wantMMMatched: 2},
		{name: "match boundary excludes index equal to matchLen", mmBlockIdx: []int{0, 3, 5}, matchLen: 3, wantMMMatched: 1},
		{name: "no mm in matched range", mmBlockIdx: []int{10, 20}, matchLen: 5, wantMMMatched: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := countMMMatchedBlocks(tt.mmBlockIdx, tt.matchLen)
			assert.Equal(t, tt.wantMMMatched, got)
		})
	}
}
