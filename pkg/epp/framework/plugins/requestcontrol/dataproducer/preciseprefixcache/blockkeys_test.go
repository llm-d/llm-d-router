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
	"testing"

	"github.com/stretchr/testify/assert"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

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
