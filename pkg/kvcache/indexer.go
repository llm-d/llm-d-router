/*
Copyright 2025 The llm-d Authors.

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

package kvcache

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization/prefixstore"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/utils/logging"
)

// Config holds the configuration for the Indexer module.
// The configuration cover the different components found in the Indexer
// module.
type Config struct {
	PrefixStoreConfig    *prefixstore.Config           `json:"prefixStoreConfig"`
	TokenProcessorConfig *kvblock.TokenProcessorConfig `json:"tokenProcessorConfig"`
	KVBlockIndexConfig   *kvblock.IndexConfig          `json:"kvBlockIndexConfig"`
	KVBlockScorerConfig  *KVBlockScorerConfig          // not exported
	TokenizersPoolConfig *tokenization.Config          `json:"tokenizersPoolConfig"`
}

// NewDefaultConfig returns a default configuration for the Indexer module.
func NewDefaultConfig() (*Config, error) {
	tokenizerPoolConfig, err := tokenization.DefaultConfig()
	if err != nil {
		return &Config{}, fmt.Errorf("failed to get default tokenizer pool config: %w", err)
	}

	return &Config{
		PrefixStoreConfig:    prefixstore.DefaultConfig(),
		TokenProcessorConfig: kvblock.DefaultTokenProcessorConfig(),
		KVBlockIndexConfig:   kvblock.DefaultIndexConfig(),
		KVBlockScorerConfig:  DefaultKVBlockScorerConfig(),
		TokenizersPoolConfig: tokenizerPoolConfig,
	}, nil
}

// Indexer is a concrete implementation of the KVCacheIndex interface.
type Indexer struct {
	config *Config

	tokensIndexer   prefixstore.Indexer    // gets tokens for a prompt
	tokensProcessor kvblock.TokenProcessor // turns tokens to kv block keys
	kvBlockIndex    kvblock.Index          // looks up pods for block keys
	kvBlockScorer   KVBlockScorer          // scores pods based on block hits

	tokenizersPool *tokenization.Pool
}

// NewKVCacheIndexer creates a KVCacheIndex given a Config.
func NewKVCacheIndexer(ctx context.Context, config *Config) (*Indexer, error) {
	logger := klog.FromContext(ctx)
	if config != nil && config.TokenProcessorConfig != nil {
		logger.Info("NewKVCacheIndexer config", "blockSize", config.TokenProcessorConfig.BlockSize)
	}

	tokensIndexer, err := prefixstore.NewLRUTokenStore(config.PrefixStoreConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create prefixstore.Indexer: %w", err)
	}

	tokensProcessor := kvblock.NewChunkedTokenDatabase(config.TokenProcessorConfig)

	kvBlockIndex, err := kvblock.NewIndex(ctx, config.KVBlockIndexConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create RedisKVBlockIndexer: %w", err)
	}

	scorer, err := NewKVBlockScorer(config.KVBlockScorerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create KVBlockScorer: %w", err)
	}

	tokenizersPool, err := tokenization.NewTokenizationPool(config.TokenizersPoolConfig, tokensIndexer)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizers pool: %w", err)
	}

	return &Indexer{
		config:          config,
		tokensIndexer:   tokensIndexer,
		tokensProcessor: tokensProcessor,
		kvBlockIndex:    kvBlockIndex,
		kvBlockScorer:   scorer,
		tokenizersPool:  tokenizersPool,
	}, nil
}

// Run starts the indexer.
func (k *Indexer) Run(ctx context.Context) {
	k.tokenizersPool.Run(ctx)
}

// KVBlockIndex returns the kvblock.Index used by the Indexer.
func (k *Indexer) KVBlockIndex() kvblock.Index {
	return k.kvBlockIndex
}

// GetPodScores retrieves the pod scores for a given prompt and model name.
// The function receives the mentioned information and a list of relevant pod
// identifiers. A Pod identifier should be its address.
// If the set of pod identifiers is empty, the function assumes all pods are
// relevant.
//
// The function returns a map of pod identifiers to scores.
func (k *Indexer) GetPodScores(ctx context.Context, prompt, modelName string,
	podIdentifiers []string,
) (map[string]int, error) {
	traceLogger := klog.FromContext(ctx).V(logging.TRACE).WithName("kvcache.GetPodScores")

	// 1. tokenize prompt
	tokens := k.tokenizersPool.Tokenize(prompt, modelName)

	// 2. get block keys
	blockKeys := k.tokensProcessor.TokensToKVBlockKeys(tokens, modelName)
	if len(blockKeys) == 0 {
		traceLogger.Info("no block keys found, returning empty scores")
		//nolint:nilnil // no need to return an error
		return nil, nil
	}

	traceLogger.Info("found tokens", "tokens", tokens, "block-keys", blockKeys)

	// 3. query kvblock indexer for pods
	keyToPods, err := k.kvBlockIndex.Lookup(ctx, blockKeys, sets.New(podIdentifiers...))
	if err != nil {
		return nil, fmt.Errorf("failed to query kvblock indexer: %w", err)
	}
	traceLogger.Info("found block keys", "block-keys", blockKeys,
		"pods", podsPerKeyPrintHelper(keyToPods))

	// 4. score pods
	podScores, err := k.kvBlockScorer.Score(blockKeys, keyToPods)
	if err != nil {
		return nil, fmt.Errorf("failed to query kvblock scorer: %w", err)
	}
	traceLogger.Info("found pod scores", "pod-scores", podScores)

	return podScores, nil
}

// podsPerKeyPrintHelper formats a map of keys to pod names for printing.
func podsPerKeyPrintHelper(ks map[kvblock.Key][]string) string {
	flattened := ""
	for k, v := range ks {
		flattened += fmt.Sprintf("%s: %v\n", k.String(), v)
	}

	return flattened
}
