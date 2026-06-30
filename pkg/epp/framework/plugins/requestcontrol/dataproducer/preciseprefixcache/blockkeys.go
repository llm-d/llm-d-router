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
	if tp == nil || len(tp.PerPromptTokens) == 0 {
		return nil, nil
	}

	var result [][]kvblock.BlockHash
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
		keys, err := computeBlockKeysForTokens(ctx, idx, tokens, mmf, tp.CacheSalt, request.TargetModel, blockSizeTokens)
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

func computeBlockKeysForTokens(ctx context.Context, idx kvCacheIndexer,
	tokens []uint32, mmFeatures []fwkrh.MultiModalFeature, cacheSalt, model string, blockSizeTokens int,
) ([]kvblock.BlockHash, error) {
	extraFeatures := buildBlockExtraFeatures(mmFeatures, cacheSalt, len(tokens), blockSizeTokens)
	return idx.ComputeBlockKeysFromTokens(ctx, tokens, model, extraFeatures)
}

// buildBlockExtraFeatures constructs the per-block extra features that taint
// the KV-block hash: multimodal hashes (for single-prompt chat requests) with
// any cache salt folded into block 0. Returns nil for text-only requests with
// no salt. Shared by the engine-key and digest block-key paths so both taint
// the hash identically.
func buildBlockExtraFeatures(mmFeatures []fwkrh.MultiModalFeature, cacheSalt string,
	numTokens, blockSizeTokens int,
) []*kvblock.BlockExtraFeatures {
	var extraFeatures []*kvblock.BlockExtraFeatures
	if len(mmFeatures) > 0 {
		mmHashes, mmPlaceholders := tokenizer.ConvertMMFeaturesFromUpstream(mmFeatures)
		extraFeatures = kvblock.ComputeBlockExtraFeatures(
			mmHashes, mmPlaceholders, blockSizeTokens, numTokens)
	}
	return foldCacheSalt(extraFeatures, cacheSalt, numTokens/blockSizeTokens)
}

// computeBlockKeysWithDigests mirrors computeBlockKeys but additionally returns
// the full-width hash digests per block, flattened across prompts in step with
// the returned engine keys. The digests name vLLM fs-connector files (32 bytes
// for sha256_cbor, 8 bytes for fnv64a); the engine keys are the uint64
// truncations used for index lookup. Returns nil when the request carries no
// tokens or no prompt produces full KV blocks.
func computeBlockKeysWithDigests(tp kvblock.TokenProcessor,
	request *scheduling.InferenceRequest, blockSizeTokens int,
) ([]uint64, [][]byte, error) {
	if request == nil || request.Body == nil {
		return nil, nil, nil
	}
	prompt := request.Body.TokenizedPrompt
	if prompt == nil || len(prompt.PerPromptTokens) == 0 {
		return nil, nil, nil
	}

	var engineKeys []uint64
	var digests [][]byte
	for _, tokens := range prompt.PerPromptTokens {
		if len(tokens) == 0 {
			continue
		}
		// MM features apply only to single-prompt requests (chat); multi-prompt
		// completions never carry multimodal content.
		var mmf []fwkrh.MultiModalFeature
		if len(prompt.PerPromptTokens) == 1 {
			mmf = prompt.MultiModalFeatures
		}
		extraFeatures := buildBlockExtraFeatures(mmf, prompt.CacheSalt, len(tokens), blockSizeTokens)

		keys, ds, err := tp.TokensToKVBlockKeysWithDigests(
			kvblock.EmptyBlockHash, tokens, request.TargetModel, extraFeatures)
		if err != nil {
			return nil, nil, err
		}
		for _, k := range keys {
			engineKeys = append(engineKeys, uint64(k))
		}
		digests = append(digests, ds...)
	}
	return engineKeys, digests, nil
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
