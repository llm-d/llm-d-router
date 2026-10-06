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

// computeBlockKeys returns one MM block-index slice per returned prompt,
// aligned positionally with the keys; prompts with no keys are skipped from
// both slices.
func TestComputeBlockKeys_PerPromptMMIndices(t *testing.T) {
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

	req := &scheduling.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{
					{TokenIDs: twoBlocks, MultiModalFeatures: []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage, Hash: "a", Offset: 0, Length: 16}}},
					{TokenIDs: short},
					{TokenIDs: twoBlocks},
					{TokenIDs: twoBlocks, MultiModalFeatures: []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityAudio, Hash: "b", Offset: 16, Length: 16}}},
				},
			},
		},
	}

	keys, mmIndices, err := computeBlockKeys(ctx, idx, req, testBlockSize)
	require.NoError(t, err)
	require.Len(t, keys, 3, "the short prompt yields no keys and is skipped")
	assert.Equal(t, [][]int{{0}, nil, {1}}, mmIndices)
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
