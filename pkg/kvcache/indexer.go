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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	kvctok "github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

// Config holds the configuration for the Indexer module.
// The configuration cover the different components found in the Indexer
// module.
type Config struct {
	KVBlockIndexConfig  *kvblock.IndexConfig    `json:"kvBlockIndexConfig"`
	KVBlockScorerConfig *KVBlockScorerConfig    // not exported
	BackendConfigs      []*KVCacheBackendConfig `json:"kvCacheBackendConfigs"`

	// TokenizersPoolConfig configured the deprecated in-process tokenization
	// pool. The pool itself lives in llm-d-kv-cache; the indexer no longer
	// consumes it, but the field is retained so existing configurations keep
	// parsing. The precise-prefix-cache legacy producer reads it.
	//
	// Deprecated: tokenize externally and call Indexer.ScoreTokens.
	TokenizersPoolConfig *kvctok.Config `json:"tokenizersPoolConfig,omitempty"`
}

// NewDefaultConfig returns a default configuration for the Indexer module.
func NewDefaultConfig() (*Config, error) {
	return &Config{
		KVBlockIndexConfig:  kvblock.DefaultIndexConfig(),
		KVBlockScorerConfig: DefaultKVBlockScorerConfig(),
		BackendConfigs:      DefaultKVCacheBackendConfig(),
	}, nil
}

// Indexer is a concrete implementation of the KVCacheIndex interface.
type Indexer struct {
	config *Config

	tokenProcessor kvblock.TokenProcessor // turns tokens to kv block keys
	kvBlockIndex   kvblock.Index          // looks up pods for block keys
	kvBlockScorer  KVBlockScorer          // scores pods based on block hits
}

// NewKVCacheIndexer creates a KVCacheIndex given a Config. Callers tokenize
// externally and use the tokens-in API (Indexer.ScoreTokens).
func NewKVCacheIndexer(ctx context.Context, config *Config, tokenProcessor kvblock.TokenProcessor) (*Indexer, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	if tokenProcessor == nil {
		return nil, fmt.Errorf("tokenProcessor cannot be nil")
	}

	kvBlockIndex, err := kvblock.NewIndex(ctx, config.KVBlockIndexConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create RedisKVBlockIndexer: %w", err)
	}

	// Wrap index with tracing instrumentation.
	// When tracing is not configured, the tracer is a no-op implementation.
	kvBlockIndex = kvblock.NewTracedIndex(kvBlockIndex)

	// override backend configs with the ones from the config, if the defaults are not used.
	config.KVBlockScorerConfig.BackendConfigs = config.BackendConfigs
	scorer, err := NewKVBlockScorer(config.KVBlockScorerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create KVBlockScorer: %w", err)
	}

	// Wrap scorer with tracing instrumentation.
	// When tracing is not configured, the tracer is a no-op implementation.
	scorer = NewTracedScorer(scorer)

	indexer := &Indexer{
		config:         config,
		tokenProcessor: tokenProcessor,
		kvBlockIndex:   kvBlockIndex,
		kvBlockScorer:  scorer,
	}

	return indexer, nil
}

// Run starts the indexer. Blocks until ctx is cancelled.
func (k *Indexer) Run(ctx context.Context) {
	<-ctx.Done()
}

// KVBlockIndex returns the kvblock.Index used by the Indexer.
func (k *Indexer) KVBlockIndex() kvblock.Index {
	return k.kvBlockIndex
}

// ComputeBlockKeysFromTokens computes the KV-block keys for a pre-tokenized
// prompt. Callers tokenize and truncate externally. extraFeatures provides
// per-block multimodal data that taints the hash; nil means text-only.
func (k *Indexer) ComputeBlockKeysFromTokens(ctx context.Context, tokens []uint32, modelName string,
	extraFeatures []*kvblock.BlockExtraFeatures,
) ([]kvblock.BlockHash, error) {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvcache.ComputeBlockKeysFromTokens")

	blockKeys, err := k.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, extraFeatures)
	if err != nil {
		traceLogger.Error(err, "blockKey conversion failed")
		return nil, fmt.Errorf("blockKey conversion failed: %w", err)
	}
	if len(blockKeys) == 0 {
		traceLogger.Info("no block keys found")
		return nil, nil
	}
	traceLogger.Info("computed block keys", "tokens", tokens, "block-keys", blockKeys)

	return blockKeys, nil
}

// ScoreTokens computes pod scores for the given tokens and model.
// It converts tokens into KV block keys, looks up which pods hold
// matching blocks in the index, and scores each pod based on cache hits.
//
// extraFeatures provides per-block multimodal data that taints the hash;
// nil means text-only. podIdentifiers limits scoring to the given pod addresses.
// If empty, all pods are considered.
func (k *Indexer) ScoreTokens(
	ctx context.Context,
	tokens []uint32,
	modelName string,
	podIdentifiers []string,
	extraFeatures []*kvblock.BlockExtraFeatures,
) (map[string]float64, error) {
	tracer := tracing.Tracer(TracerScope)
	ctx, span := tracer.Start(ctx, "score_tokens",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvcache.ScoreTokens")

	blockKeys, err := k.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, extraFeatures)
	if err != nil {
		return nil, fmt.Errorf("blockKey conversion failed: %w", err)
	}

	span.SetAttributes(
		attribute.String("gen_ai.request.model", modelName),
		attribute.Int("llm_d.kv_cache.pod_count", len(podIdentifiers)),
		attribute.Int("llm_d.kv_cache.token_count", len(tokens)),
		attribute.Int("llm_d.kv_cache.block_keys.count", len(blockKeys)),
	)

	if len(blockKeys) == 0 {
		traceLogger.Info("no block keys found, returning empty scores")
		//nolint:nilnil // no need to return an error
		return nil, nil
	}
	traceLogger.Info("found tokens", "tokens", tokens, "block-keys", blockKeys)

	keyToPods, err := k.kvBlockIndex.Lookup(ctx, blockKeys, sets.New(podIdentifiers...))
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return nil, fmt.Errorf("failed to query kvblock indexer: %w", err)
	}
	traceLogger.Info("found block keys", "block-keys", blockKeys,
		"pods", podsPerKeyPrintHelper(keyToPods))

	// Calculate block-level hit ratio (blocks found / blocks requested).
	blocksFound := 0
	for _, pods := range keyToPods {
		if len(pods) > 0 {
			blocksFound++
		}
	}
	blockHitRatio := 0.0
	if len(blockKeys) > 0 {
		blockHitRatio = float64(blocksFound) / float64(len(blockKeys))
	}
	span.SetAttributes(
		attribute.Float64("llm_d.kv_cache.block_hit_ratio", blockHitRatio),
		attribute.Int("llm_d.kv_cache.blocks_found", blocksFound),
	)

	podScores, err := k.kvBlockScorer.Score(ctx, blockKeys, keyToPods)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return nil, fmt.Errorf("failed to query kvblock scorer: %w", err)
	}

	return podScores, nil
}

// podsPerKeyPrintHelper formats a map of keys to pod entries for printing.
func podsPerKeyPrintHelper(ks map[kvblock.BlockHash][]kvblock.PodEntry) string {
	flattened := ""
	for k, v := range ks {
		entries := make([]string, len(v))
		for i, entry := range v {
			entries[i] = entry.String()
		}
		flattened += fmt.Sprintf("%s: %v\n", k.String(), entries)
	}

	return flattened
}
