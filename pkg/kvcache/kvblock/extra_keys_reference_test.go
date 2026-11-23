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
	"fmt"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// referenceBlockExtraFeatures scans every item for every block, independently
// of the production scan's state across blocks.
func referenceBlockExtraFeatures(hashes map[string][]string, placeholders map[string][]kvblock.PlaceholderRange,
	blockSize, numTokens int,
) []*kvblock.BlockExtraFeatures {
	if len(hashes) == 0 || blockSize <= 0 || numTokens <= 0 {
		return nil
	}
	type item struct {
		hash       string
		start, end int
	}
	var items []item
	for modality, ids := range hashes {
		ranges := placeholders[modality]
		for i := range min(len(ids), len(ranges)) {
			items = append(items, item{ids[i], ranges[i].Offset, ranges[i].Offset + ranges[i].Length})
		}
	}
	if len(items) == 0 {
		return nil
	}
	sort.Slice(items, func(i, j int) bool { return items[i].start < items[j].start })
	result := make([]*kvblock.BlockExtraFeatures, numTokens/blockSize)
	for block := range result {
		start, end := block*blockSize, (block+1)*blockSize
		var overlapping []kvblock.MMHash
		for _, item := range items {
			if item.end > start && item.start < end {
				overlapping = append(overlapping, kvblock.MMHash{Hash: item.hash})
			}
		}
		if len(overlapping) > 0 {
			result[block] = &kvblock.BlockExtraFeatures{MMHashes: overlapping}
		}
	}
	return result
}

func TestComputeBlockExtraFeaturesMatchesReference(t *testing.T) {
	for _, tt := range []struct {
		name      string
		ranges    []kvblock.PlaceholderRange
		blockSize int
		numTokens int
	}{
		{name: "no items", blockSize: 4, numTokens: 24},
		{name: "nonpositive block size", ranges: []kvblock.PlaceholderRange{{Offset: 0, Length: 8}}, numTokens: 24},
		{name: "no tokens", ranges: []kvblock.PlaceholderRange{{Offset: 0, Length: 8}}, blockSize: 4},
		{name: "no full block", ranges: []kvblock.PlaceholderRange{{Offset: 0, Length: 8}}, blockSize: 4, numTokens: 3},
		{
			name: "disjoint with text gaps", blockSize: 4, numTokens: 32,
			ranges: []kvblock.PlaceholderRange{{Offset: 0, Length: 4}, {Offset: 8, Length: 4}, {Offset: 16, Length: 4}},
		},
		{
			name: "spanning blocks", blockSize: 4, numTokens: 24,
			ranges: []kvblock.PlaceholderRange{{Offset: 2, Length: 8}, {Offset: 12, Length: 4}},
		},
		{
			name: "short spans end before an earlier long span", blockSize: 4, numTokens: 40,
			ranges: []kvblock.PlaceholderRange{{Offset: 0, Length: 24}, {Offset: 4, Length: 4}, {Offset: 8, Length: 4}, {Offset: 28, Length: 4}},
		},
		{
			name: "equal starts and different ends", blockSize: 4, numTokens: 32,
			ranges: []kvblock.PlaceholderRange{{Offset: 0, Length: 20}, {Offset: 0, Length: 4}, {Offset: 0, Length: 8}, {Offset: 8, Length: 4}},
		},
		{
			name: "unsorted ranges", blockSize: 4, numTokens: 32,
			ranges: []kvblock.PlaceholderRange{{Offset: 24, Length: 4}, {Offset: 2, Length: 8}, {Offset: 12, Length: 4}},
		},
		{
			name: "nonpositive lengths and negative offsets", blockSize: 4, numTokens: 24,
			ranges: []kvblock.PlaceholderRange{{Offset: 3, Length: 0}, {Offset: 6, Length: -1}, {Offset: -2, Length: 6}, {Offset: 14, Length: 0}},
		},
		{
			name: "partial final block and ranges past prompt", blockSize: 4, numTokens: 23,
			ranges: []kvblock.PlaceholderRange{{Offset: 18, Length: 8}, {Offset: 20, Length: 2}, {Offset: 24, Length: 4}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			hashes := make([]string, len(tt.ranges))
			for i := range hashes {
				hashes[i] = fmt.Sprintf("image-%d", i)
			}
			mmHashes := map[string][]string{"image": hashes}
			placeholders := map[string][]kvblock.PlaceholderRange{"image": tt.ranges}
			want := referenceBlockExtraFeatures(mmHashes, placeholders, tt.blockSize, tt.numTokens)
			got := kvblock.ComputeBlockExtraFeatures(mmHashes, placeholders, tt.blockSize, tt.numTokens)
			require.Equal(t, want, got)
		})
	}
}

func TestComputeBlockExtraFeaturesMixedModalities(t *testing.T) {
	hashes := map[string][]string{
		"image": {"image-1", "image-2", "unpaired"},
		"video": {"video-1", "video-2"},
		"audio": {"missing-placeholders"},
	}
	placeholders := map[string][]kvblock.PlaceholderRange{
		"image": {{Offset: 8, Length: 4}, {Offset: 0, Length: 4}},
		"video": {{Offset: 2, Length: 20}, {Offset: 16, Length: 4}, {Offset: 24, Length: 4}},
	}
	want := referenceBlockExtraFeatures(hashes, placeholders, 4, 32)
	require.Equal(t, want, kvblock.ComputeBlockExtraFeatures(hashes, placeholders, 4, 32))
}

func TestComputeBlockExtraFeaturesRandomized(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	for iter := range 5000 {
		numTokens := rng.IntN(256)
		blockSize := rng.IntN(34) - 1
		ranges := make([]kvblock.PlaceholderRange, rng.IntN(40))
		hashes := make([]string, len(ranges))
		for i := range ranges {
			ranges[i] = kvblock.PlaceholderRange{
				Offset: rng.IntN(numTokens+32) - 8,
				Length: rng.IntN(80) - 4,
			}
			hashes[i] = fmt.Sprintf("image-%d", i)
		}
		mmHashes := map[string][]string{"image": hashes}
		placeholders := map[string][]kvblock.PlaceholderRange{"image": ranges}
		want := referenceBlockExtraFeatures(mmHashes, placeholders, blockSize, numTokens)
		got := kvblock.ComputeBlockExtraFeatures(mmHashes, placeholders, blockSize, numTokens)
		require.Equal(t, want, got, "iteration %d: block size %d, tokens %d, ranges %v", iter, blockSize, numTokens, ranges)
	}
}

func BenchmarkComputeBlockExtraFeatures(b *testing.B) {
	for _, tt := range []struct {
		name        string
		blocks      int
		items       int
		overlapping bool
	}{
		{name: "text", blocks: 2048},
		{name: "small", blocks: 128, items: 4},
		{name: "long", blocks: 2048, items: 256},
		{name: "very_long", blocks: 8192, items: 512},
		{name: "overlapping", blocks: 2048, items: 256, overlapping: true},
	} {
		b.Run(tt.name, func(b *testing.B) {
			const blockSize = 16
			hashes := map[string][]string{}
			placeholders := map[string][]kvblock.PlaceholderRange{}
			for i := range tt.items {
				hashes["image"] = append(hashes["image"], fmt.Sprintf("image-%d", i))
				length := 2 * blockSize
				if tt.overlapping && i == 0 {
					length = tt.blocks * blockSize
				}
				placeholders["image"] = append(placeholders["image"], kvblock.PlaceholderRange{Offset: i * 4 * blockSize, Length: length})
			}
			b.ReportAllocs()
			for b.Loop() {
				kvblock.ComputeBlockExtraFeatures(hashes, placeholders, blockSize, tt.blocks*blockSize)
			}
		})
	}
}
