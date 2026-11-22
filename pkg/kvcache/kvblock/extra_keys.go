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
	"math"
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
	// MMOffsets holds each MMHashes entry's start relative to the block start, or nil without event offsets.
	// It is not part of the block hash.
	MMOffsets []int
	// CacheSalt is the request's cache salt, which vLLM appends to the first
	// block's extra keys after the MM items. The block hash takes it after
	// MMHashes. Events whose MM items are bare identifiers can't tell the salt
	// apart, so it stays in MMHashes there.
	CacheSalt string
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
//     block-relative offset). The offsets are kept in MMOffsets, and the
//     first entry's sign as FirstContinuation.
//
// When any entry of the event is a tuple, a bare string is the cache salt and
// is kept as CacheSalt.
//
// nil inner slices produce nil entries. Returns nil if raw is nil.
func ParseRawExtraKeys(raw [][]any, loraName string) ([]*BlockExtraFeatures, error) {
	if raw == nil {
		return nil, nil
	}

	tupleLayout := hasTupleEntry(raw)
	result := make([]*BlockExtraFeatures, len(raw))
	for blockIdx, blockKeys := range raw {
		if blockKeys == nil {
			continue
		}
		if loraName != "" && len(blockKeys) > 0 && blockKeys[0] == loraName {
			blockKeys = blockKeys[1:]
		}

		features := &BlockExtraFeatures{}
		var offsets []int
		offsetsComplete := true
		for _, entry := range blockKeys {
			switch v := entry.(type) {
			case string:
				if tupleLayout {
					features.CacheSalt = v
					continue
				}
				// vLLM <= v0.18.x: bare identifier string per MM item.
				features.MMHashes = append(features.MMHashes, MMHash{Hash: v})
				offsetsComplete = false
			case []any:
				// [identifier, offset] tuple — extract the identifier.
				if len(v) >= 1 {
					if hash, ok := v[0].(string); ok {
						if len(features.MMHashes) == 0 && len(v) >= 2 {
							features.FirstContinuation = continuationFromOffset(v[1])
						}
						features.MMHashes = append(features.MMHashes, MMHash{Hash: hash})
						offset, ok := tupleOffset(v)
						offsets = append(offsets, offset)
						offsetsComplete = offsetsComplete && ok
					}
				}
			default:
				// Skip unknown entry types.
				continue
			}
		}
		if offsetsComplete && len(offsets) > 0 {
			features.MMOffsets = offsets
		}

		if len(features.MMHashes) > 0 || features.CacheSalt != "" {
			result[blockIdx] = features
		}
	}

	return result, nil
}

func hasTupleEntry(raw [][]any) bool {
	for _, blockKeys := range raw {
		for _, entry := range blockKeys {
			if _, ok := entry.([]any); ok {
				return true
			}
		}
	}
	return false
}

func continuationFromOffset(v any) MMContinuation {
	off, ok := offsetValue(v)
	if !ok {
		return MMContinuationUnknown
	}
	if off < 0 {
		return MMContinuationContinues
	}
	return MMContinuationNew
}

// tupleOffset returns the offset of an [identifier, offset] tuple.
func tupleOffset(tuple []any) (int, bool) {
	if len(tuple) < 2 {
		return 0, false
	}
	return offsetValue(tuple[1])
}

// offsetValue converts a msgpack-decoded integer to an int.
func offsetValue(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int8:
		return int(n), true
	case int16:
		return int(n), true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case uint8:
		return int(n), true
	case uint16:
		return int(n), true
	case uint32:
		return int(n), true
	case uint64:
		return int(min(n, math.MaxInt32)), true
	default:
		return 0, false
	}
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
	remaining := items
	// Active items overwrite only consumed entries in the owned sorted slice.
	active := items[:0]

	for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
		blockStart := blockIdx * blockSize
		blockEnd := blockStart + blockSize

		for len(remaining) > 0 && remaining[0].start < blockEnd {
			active = append(active, remaining[0])
			remaining = remaining[1:]
		}

		var hashes []MMHash
		retained := active[:0]
		for _, item := range active {
			if item.end <= blockStart {
				continue
			}
			retained = append(retained, item)
			hashes = append(hashes, MMHash{Hash: item.hash})
		}
		active = retained

		if len(hashes) > 0 {
			result[blockIdx] = &BlockExtraFeatures{MMHashes: hashes}
		}
	}

	return result
}
