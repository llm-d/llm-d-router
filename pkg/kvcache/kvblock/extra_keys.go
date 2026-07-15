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

package kvblock

import (
	"sort"
)

// MMHash represents a single multimodal content hash entry.
// This matches vLLM's per-block extra_keys format where each entry is
// the mm_feature.identifier string for an overlapping multimodal item.
type MMHash struct {
	Hash string
}

// BlockExtraFeatures holds per-block extra data that taints the block hash.
// A nil *BlockExtraFeatures means pure text (no taint).
type BlockExtraFeatures struct {
	MMHashes []MMHash
}

// PlaceholderRange describes a contiguous range of placeholder tokens for one
// multimodal item within the full token sequence.
type PlaceholderRange struct {
	Offset int // absolute start token index
	Length int // number of placeholder tokens
}

// ParseRawExtraKeys converts the raw [][]any from BlockStoredEvent.ExtraKeys
// into typed []*BlockExtraFeatures. Each inner []any element is either:
//   - a bare string identifier (vLLM v0.18.0+: mm_feature.identifier), or
//   - a 2-element [string, int] tuple (legacy format, offset is ignored).
//
// nil inner slices produce nil entries. Returns nil if raw is nil.
func ParseRawExtraKeys(raw [][]any) ([]*BlockExtraFeatures, error) {
	if raw == nil {
		return nil, nil
	}

	result := make([]*BlockExtraFeatures, len(raw))
	for blockIdx, blockKeys := range raw {
		if blockKeys == nil {
			continue
		}

		features := &BlockExtraFeatures{}
		for _, entry := range blockKeys {
			switch v := entry.(type) {
			case string:
				// vLLM v0.18.0+: bare identifier string per MM item.
				features.MMHashes = append(features.MMHashes, MMHash{Hash: v})
			case []any:
				// Legacy format: [hash, offset] tuple — extract hash, ignore offset.
				if len(v) >= 1 {
					if hash, ok := v[0].(string); ok {
						features.MMHashes = append(features.MMHashes, MMHash{Hash: hash})
					}
				}
			default:
				// Skip unknown entry types (e.g. LoRA, cache salt).
				continue
			}
		}

		if len(features.MMHashes) > 0 {
			result[blockIdx] = features
		}
	}

	return result, nil
}

// mmItem flattens one multimodal placeholder with its content hash for sorting.
type mmItem struct {
	hash  string
	start int
	end   int
}

// ComputeBlockExtraFeatures converts tokenizer-provided multimodal metadata into
// per-block extra features, matching vLLM's _gen_mm_extra_hash_keys() algorithm.
//
// For each block, it finds overlapping placeholder ranges and emits the
// identifier (hash) of each overlapping multimodal item. This matches vLLM
// v0.18.0+ which appends mm_feature.identifier per overlapping item.
func ComputeBlockExtraFeatures(
	mmHashes map[string][]string,
	mmPlaceholders map[string][]PlaceholderRange,
	blockSize, numTokens int,
) []*BlockExtraFeatures {
	if len(mmHashes) == 0 || blockSize <= 0 || numTokens <= 0 {
		return nil
	}

	// Flatten all placeholder ranges with their hashes, sorted by start position.
	var items []mmItem
	for modality, hashes := range mmHashes {
		ranges, ok := mmPlaceholders[modality]
		if !ok {
			continue
		}
		n := len(hashes)
		if len(ranges) < n {
			n = len(ranges)
		}
		for i := 0; i < n; i++ {
			items = append(items, mmItem{
				hash:  hashes[i],
				start: ranges[i].Offset,
				end:   ranges[i].Offset + ranges[i].Length,
			})
		}
	}

	if len(items) == 0 {
		return nil
	}

	sort.Slice(items, func(i, j int) bool { return items[i].start < items[j].start })

	numBlocks := numTokens / blockSize
	result := make([]*BlockExtraFeatures, numBlocks)

	for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
		blockStart := blockIdx * blockSize
		blockEnd := blockStart + blockSize

		var hashes []MMHash
		for _, item := range items {
			// Placeholder ends before this block — skip.
			if item.end <= blockStart {
				continue
			}
			// Placeholder starts at or after block end — no more overlaps for this block
			// (items are sorted).
			if item.start >= blockEnd {
				break
			}
			// Overlap: emit identifier for this MM item.
			hashes = append(hashes, MMHash{Hash: item.hash})
		}

		if len(hashes) > 0 {
			result[blockIdx] = &BlockExtraFeatures{MMHashes: hashes}
		}
	}

	return result
}
