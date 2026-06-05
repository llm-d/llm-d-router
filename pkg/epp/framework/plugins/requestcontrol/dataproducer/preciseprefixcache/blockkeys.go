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

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// kvCacheIndexer is the subset of kvcache.Indexer that the producer relies on.
type kvCacheIndexer interface {
	ComputeBlockKeysFromTokens(ctx context.Context, tokens []uint32, modelName string, extraFeatures []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error)
	KVBlockIndex() kvblock.Index
}

// computeBlockKeys hashes the request's TokenizedPrompt into KV-block keys.
// When PerPromptTokens has more than one entry (multi-prompt completions),
// each prompt is hashed independently so cross-prompt block adjacency (which
// never exists in the model server cache) is avoided. Single-prompt requests
// produce a length-1 outer slice. A non-empty CacheSalt is folded into each
// prompt's first block. Returns nil when the request carries no tokens or no
// prompt produces full KV blocks.
func computeBlockKeys(ctx context.Context, idx kvCacheIndexer,
	request *scheduling.InferenceRequest, blockSizeTokens int,
) ([][]kvblock.BlockHash, error) {
	if request == nil || request.Body == nil {
		return nil, nil
	}
	tp := request.Body.TokenizedPrompt
	if tp == nil {
		return nil, nil
	}

	// Multi-prompt: score each prompt independently.
	if len(tp.PerPromptTokens) > 1 {
		var result [][]kvblock.BlockHash
		for _, tokens := range tp.PerPromptTokens {
			if len(tokens) == 0 {
				continue
			}
			// MM features are nil: multi-prompt completions don't carry multimodal content.
			keys, err := computeBlockKeysForTokens(ctx, idx, tokens, nil, tp.CacheSalt, request.TargetModel, blockSizeTokens)
			if err != nil {
				return nil, err
			}
			if len(keys) == 0 {
				continue
			}
			result = append(result, keys)
		}
		return result, nil
	}

	// Single prompt.
	if len(tp.TokenIDs) == 0 {
		return nil, nil
	}
	tokens := tp.TokenIDs
	keys, err := computeBlockKeysForTokens(ctx, idx, tokens, tp.MultiModalFeatures, tp.CacheSalt, request.TargetModel, blockSizeTokens)
	if err != nil {
		return nil, err
	}
	if len(keys) == 0 {
		return nil, nil
	}
	return [][]kvblock.BlockHash{keys}, nil
}

func computeBlockKeysForTokens(ctx context.Context, idx kvCacheIndexer,
	tokens []uint32, mmFeatures []fwkrh.MultiModalFeature, cacheSalt, model string, blockSizeTokens int,
) ([]kvblock.BlockHash, error) {
	var extraFeatures []*kvblock.BlockExtraFeatures
	if len(mmFeatures) > 0 {
		mmHashes, mmPlaceholders := tokenizer.ConvertMMFeaturesFromUpstream(mmFeatures)
		extraFeatures = kvblock.ComputeBlockExtraFeatures(
			mmHashes, mmPlaceholders, blockSizeTokens, len(tokens))
	}
	extraFeatures = foldCacheSalt(extraFeatures, cacheSalt, len(tokens)/blockSizeTokens)
	return idx.ComputeBlockKeysFromTokens(ctx, tokens, model, extraFeatures)
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
