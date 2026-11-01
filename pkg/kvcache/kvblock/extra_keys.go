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
	// FirstContinuation classifies MMHashes[0] from its event offset.
	// It is not part of the block hash.
	FirstContinuation MMContinuation
}

// MMContinuation reports whether a block's first MM entry continues the item
// that ends the previous block, as read from vLLM's block-relative offset.
type MMContinuation uint8

const (
	MMContinuationUnknown   MMContinuation = iota // no offset (bare identifier)
	MMContinuationNew                             // offset >= 0
	MMContinuationContinues                       // offset < 0
)

// PlaceholderRange describes a contiguous range of placeholder tokens for one
// multimodal item within the full token sequence.
type PlaceholderRange struct {
	Offset int // absolute start token index
	Length int // number of placeholder tokens
}

// ParseRawExtraKeys converts the raw [][]any from BlockStoredEvent.ExtraKeys
// into typed []*BlockExtraFeatures. A first entry equal to loraName is dropped;
// the request key carries the adapter as its model name. Each other element is
// either:
//   - a bare string identifier (vLLM <= v0.18.x: mm_feature.identifier), or
//   - a 2-element [string, int] tuple (vLLM v0.19.0+: identifier and
//     block-relative offset). Only the first entry's offset is kept, as
//     FirstContinuation.
//
// nil inner slices produce nil entries. Returns nil if raw is nil.
func ParseRawExtraKeys(raw [][]any, loraName string) ([]*BlockExtraFeatures, error) {
	if raw == nil {
		return nil, nil
	}

	result := make([]*BlockExtraFeatures, len(raw))
	for blockIdx, blockKeys := range raw {
		if blockKeys == nil {
			continue
		}
		if loraName != "" && len(blockKeys) > 0 && blockKeys[0] == loraName {
			blockKeys = blockKeys[1:]
		}

		features := &BlockExtraFeatures{}
		for _, entry := range blockKeys {
			switch v := entry.(type) {
			case string:
				// vLLM <= v0.18.x: bare identifier string per MM item.
				features.MMHashes = append(features.MMHashes, MMHash{Hash: v})
			case []any:
				// [identifier, offset] tuple — extract the identifier.
				if len(v) >= 1 {
					if hash, ok := v[0].(string); ok {
						if len(features.MMHashes) == 0 && len(v) >= 2 {
							features.FirstContinuation = continuationFromOffset(v[1])
						}
						features.MMHashes = append(features.MMHashes, MMHash{Hash: hash})
					}
				}
			default:
				// Skip unknown entry types.
				continue
			}
		}

		if len(features.MMHashes) > 0 {
			result[blockIdx] = features
		}
	}

	return result, nil
}

func continuationFromOffset(v any) MMContinuation {
	var off int64
	switch n := v.(type) {
	case int:
		off = int64(n)
	case int8:
		off = int64(n)
	case int16:
		off = int64(n)
	case int32:
		off = int64(n)
	case int64:
		off = n
	case uint8, uint16, uint32, uint64:
		return MMContinuationNew
	default:
		return MMContinuationUnknown
	}
	if off < 0 {
		return MMContinuationContinues
	}
	return MMContinuationNew
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
