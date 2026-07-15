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
	"sort"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// kvCacheIndexer is the subset of kvcache.Indexer that the producer relies on.
type kvCacheIndexer interface {
	ComputeBlockKeysFromTokens(ctx context.Context, tokens []uint32, modelName string, extraFeatures []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error)
	KVBlockIndex() kvblock.Index
}

// computeBlockKeys hashes the request's TokenizedPrompt into per-prompt
// KV-block keys, folding CacheSalt into each prompt's first block. MM features
// apply only to single-prompt requests; mmBlockIndices (block indices spanned
// by MM content) is populated only in that case for hit attribution.
func computeBlockKeys(ctx context.Context, idx kvCacheIndexer,
	request *scheduling.InferenceRequest, blockSizeTokens int,
) ([][]kvblock.BlockHash, []int, error) {
	if request == nil || request.Body == nil {
		return nil, nil, nil
	}
	tp := request.Body.TokenizedPrompt
	if tp == nil || len(tp.PerPromptTokens) == 0 {
		return nil, nil, nil
	}

	var result [][]kvblock.BlockHash
	var mmBlockIndices []int
	for _, tokens := range tp.PerPromptTokens {
		if len(tokens) == 0 {
			continue
		}
		// MM features apply only to single-prompt requests (chat); multi-prompt
		// completions never carry multimodal content.
		var mmf []fwkrh.MultiModalFeature
		if len(tp.PerPromptTokens) == 1 {
			mmf = tp.MultiModalFeatures
		}
		keys, mmIdx, err := computeBlockKeysForTokens(ctx, idx, tokens, mmf, tp.CacheSalt, request.TargetModel, blockSizeTokens)
		if err != nil {
			return nil, nil, err
		}
		if len(keys) == 0 {
			continue
		}
		result = append(result, keys)
		if len(tp.PerPromptTokens) == 1 {
			mmBlockIndices = mmIdx
		}
	}
	return result, mmBlockIndices, nil
}

func computeBlockKeysForTokens(ctx context.Context, idx kvCacheIndexer,
	tokens []uint32, mmFeatures []fwkrh.MultiModalFeature, cacheSalt, model string, blockSizeTokens int,
) ([]kvblock.BlockHash, []int, error) {
	var extraFeatures []*kvblock.BlockExtraFeatures
	var mmBlockIndices []int
	if len(mmFeatures) > 0 {
		mmHashes, mmPlaceholders := tokenizer.ConvertMMFeaturesFromUpstream(mmFeatures)
		extraFeatures = kvblock.ComputeBlockExtraFeatures(
			mmHashes, mmPlaceholders, blockSizeTokens, len(tokens))
		mmBlockIndices = multimodalBlockIndices(mmFeatures, blockSizeTokens)
	}
	extraFeatures = foldCacheSalt(extraFeatures, cacheSalt, len(tokens)/blockSizeTokens)
	keys, err := idx.ComputeBlockKeysFromTokens(ctx, tokens, model, extraFeatures)
	return keys, mmBlockIndices, err
}

// countMMMatchedBlocks counts entries in (sorted) mmBlockIndices that are
// strictly less than matchLen.
func countMMMatchedBlocks(mmBlockIndices []int, matchLen int) int {
	if matchLen <= 0 || len(mmBlockIndices) == 0 {
		return 0
	}
	for i, idx := range mmBlockIndices {
		if idx >= matchLen {
			return i
		}
	}
	return len(mmBlockIndices)
}

// multimodalBlockIndices returns the sorted unique block indices spanned by
// any of the given features.
func multimodalBlockIndices(features []fwkrh.MultiModalFeature, blockSizeTokens int) []int {
	if blockSizeTokens <= 0 || len(features) == 0 {
		return nil
	}
	seen := map[int]struct{}{}
	for _, f := range features {
		if f.Length <= 0 {
			continue
		}
		start := f.Offset / blockSizeTokens
		end := (f.Offset + f.Length - 1) / blockSizeTokens
		for i := start; i <= end; i++ {
			seen[i] = struct{}{}
		}
	}
	if len(seen) == 0 {
		return nil
	}
	out := make([]int, 0, len(seen))
	for i := range seen {
		out = append(out, i)
	}
	sort.Ints(out)
	return out
}

// foldCacheSalt appends the cache salt to the first block's extra keys, after
// any multimodal hashes. vLLM puts cache_salt in block 0's extra_keys, and
// engine-side KV-event ingestion folds that salt string into the same per-block
// hash list (kvblock.ParseRawExtraKeys); the request side must match for salted
// keys to correlate. No-op without a salt or a full first block.
func foldCacheSalt(extraFeatures []*kvblock.BlockExtraFeatures, salt string, numBlocks int) []*kvblock.BlockExtraFeatures {
	if salt == "" || numBlocks == 0 {
		return extraFeatures
	}
	if extraFeatures == nil {
		extraFeatures = make([]*kvblock.BlockExtraFeatures, numBlocks)
	}
	if extraFeatures[0] == nil {
		extraFeatures[0] = &kvblock.BlockExtraFeatures{}
	}
	extraFeatures[0].MMHashes = append(extraFeatures[0].MMHashes, kvblock.MMHash{Hash: salt})
	return extraFeatures
}
