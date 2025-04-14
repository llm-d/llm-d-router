package kvcache

import (
	"context"
	"fmt"

	"github.com/neuralmagic/distributed-kv-cache/pkg/prefixstore"
	"k8s.io/apimachinery/pkg/util/sets"

	"k8s.io/klog/v2"

	"github.com/neuralmagic/distributed-kv-cache/pkg/tokenization"
)

// Config holds the configuration for the Indexer module.
// The configuration cover the different components found in the Indexer
// module.
type Config struct {
	PrefixStoreConfig    *prefixstore.Config
	TokenProcessorConfig *TokenProcessorConfig
	KVBlockIndexerConfig *KVBlockIndexerConfig
	KVBLockScorerConfig  *KVBlockScorerConfig
	TokenizersPoolConfig *tokenization.Config
}

// NewDefaultConfig returns a default configuration for the Indexer module.
func NewDefaultConfig() *Config {
	return &Config{
		PrefixStoreConfig:    prefixstore.DefaultConfig(),
		TokenProcessorConfig: DefaultTokenProcessorConfig(),
		KVBlockIndexerConfig: DefaultKVBlockIndexerConfig(),
		KVBLockScorerConfig:  DefaultKVBlockScorerConfig(),
		TokenizersPoolConfig: tokenization.DefaultConfig(),
	}
}

// Indexer is a concrete implementation of the KVCacheIndex interface.
type Indexer struct {
	tokensIndexer   prefixstore.Indexer // gets tokens for a prompt
	tokensProcessor TokenProcessor      // turns tokens to kv block keys
	kvBlockIndexer  KVBlockIndexer      // looks up pods for block keys
	kvBlockScorer   KVBlockScorer       // scores pods based on block hits

	tokenizersPool *tokenization.Pool
}

// NewKVCacheIndexer creates a KVCacheIndex given a Config.
func NewKVCacheIndexer(config *Config) (*Indexer, error) {
	tokensIndexer, err := prefixstore.NewLRUTokenStore(config.PrefixStoreConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create prefixstore.Indexer: %w", err)
	}

	tokensProcessor := NewChunkedTokenDatabase(config.TokenProcessorConfig)

	kvBlockIndexer, err := NewRedisKVBlockIndexer(config.KVBlockIndexerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create RedisKVBlockIndexer: %w", err)
	}

	scorer, err := NewKVBlockScorer(config.KVBLockScorerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create KVBlockScorer: %w", err)
	}

	tokenizersPool := tokenization.NewTokenizationPool(config.TokenizersPoolConfig, tokensIndexer)

	return &Indexer{
		tokensIndexer:   tokensIndexer,
		tokensProcessor: tokensProcessor,
		kvBlockIndexer:  kvBlockIndexer,
		kvBlockScorer:   scorer,
		tokenizersPool:  tokenizersPool,
	}, nil
}

// Run starts the indexer.
func (k *Indexer) Run(ctx context.Context) {
	k.tokenizersPool.Run(ctx)
}

// GetPodScores retrieves the pod scores for a given prompt and model name.
// The function receives the mentioned information and a list of relevant pod
// identifiers. A Pod identifier should be its address.
// If the set of pod identifiers is empty, the function assumes all pods are
// relevant.
//
// The function returns a slice of PodScores (pod identifier and score),
// where the scores are calculated by the active scoring strategy.
func (k *Indexer) GetPodScores(ctx context.Context, prompt, modelName string,
	podIdentifiers []string,
) ([]PodScore, error) {
	logger := klog.FromContext(ctx)
	// 0. add to tokenizers pool
	k.tokenizersPool.AddTask(prompt, modelName)

	// 1. get available tokens of longest prefix
	tokens := k.tokensIndexer.FindLongestContainedTokens(prompt, modelName)
	if len(tokens) == 0 {
		return nil, nil
	}

	// 2. get block keys
	blockKeys := k.tokensProcessor.TokensToKVBlockKeys(tokens, modelName)
	logger.Info("made block keys", "blockKeys", blockKeys)

	// 3. query kvblock indexer for pods
	strBlockKeys, keyToPods, err := k.kvBlockIndexer.GetPodsForKeys(ctx, blockKeys, sets.New[string](podIdentifiers...))
	if err != nil {
		return nil, fmt.Errorf("failed to query kvblock indexer: %w", err)
	}
	logger.Info("queried kvblock indexer", "strBlockKeys", strBlockKeys, "keyToPods", keyToPods)

	// 4. score pods
	podScores, err := k.kvBlockScorer.Score(strBlockKeys, keyToPods)
	if err != nil {
		return nil, fmt.Errorf("failed to query kvblock scorer: %w", err)
	}

	return podScores, nil
}
