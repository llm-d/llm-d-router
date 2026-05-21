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

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// Subset of kvcache.Indexer used by the producer. Tokens-only.
type kvCacheIndexer interface {
	ComputeBlockKeysFromTokens(ctx context.Context, tokens []uint32, modelName string, extraFeatures []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error)
	KVBlockIndex() kvblock.Index
}

// Hashes TokenizedPrompt into KV-block keys. (nil, nil) when tokens absent.
func computeBlockKeys(ctx context.Context, idx kvCacheIndexer,
	request *scheduling.InferenceRequest, blockSizeTokens int,
) ([]kvblock.BlockHash, error) {
	if request == nil || request.Body == nil {
		return nil, nil
	}
	tp := request.Body.TokenizedPrompt
	if tp == nil || len(tp.TokenIDs) == 0 {
		return nil, nil
	}
	var extraFeatures []*kvblock.BlockExtraFeatures
	if len(tp.MultiModalFeatures) > 0 {
		mmHashes, mmPlaceholders := tokenizer.ConvertMMFeaturesFromUpstream(tp.MultiModalFeatures)
		extraFeatures = kvblock.ComputeBlockExtraFeatures(
			mmHashes, mmPlaceholders, blockSizeTokens, len(tp.TokenIDs))
	}
	return idx.ComputeBlockKeysFromTokens(ctx, tp.TokenIDs, request.TargetModel, extraFeatures)
}
