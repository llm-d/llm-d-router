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
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/telemetry"
)

const PluginType = "precise-prefix-cache-producer"

// Mirrors the legacy precise-prefix-cache-scorer config shape.
type PluginConfig struct {
	TokenProcessorConfig *kvblock.TokenProcessorConfig `json:"tokenProcessorConfig"`
	IndexerConfig        *kvcache.Config               `json:"indexerConfig"`
	KVEventsConfig       *kvevents.Config              `json:"kvEventsConfig"`
	// Seed predicted cache entries post-routing so the next same-prefix
	// request hits without waiting for engine confirmation.
	SpeculativeIndexing bool `json:"speculativeIndexing"`
	// Go duration string. Defaults to defaultSpeculativeTTL.
	SpeculativeTTL string `json:"speculativeTTL"`
}

var _ requestcontrol.DataProducer = &Producer{}

// Owns the KV-block index and publishes per-endpoint PrefixCacheMatchInfo.
// Speculative indexing in prerequest.go; subscriber lifecycle in extractor.go.
type Producer struct {
	typedName      plugin.TypedName
	kvCacheIndexer kvCacheIndexer

	subscribersManager *kvevents.SubscriberManager
	kvEventsConfig     *kvevents.Config

	kvBlockScorer kvcache.KVBlockScorer

	dk plugin.DataKey

	pluginState *plugin.PluginState

	speculativeCache   *ttlcache.Cache[string, *speculativeEntries]
	speculativeTTL     time.Duration
	speculativeEnabled bool

	blockSizeTokens int

	// Plugin-lifetime, not request-scoped: SubscriberManager binds each
	// subscriber's goroutine to the ctx passed at registration.
	subscriberCtx context.Context
}

func PluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	indexerConfig, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize indexer config: %w", err)
	}

	parameters := PluginConfig{
		IndexerConfig:  indexerConfig,
		KVEventsConfig: kvevents.DefaultConfig(),
	}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", PluginType, err)
		}
	}

	if parameters.IndexerConfig == nil {
		return nil, errors.New("indexerConfig is required")
	}
	// Tokens-only: reject configs that rely on the indexer's internal tokenizer.
	//nolint:staticcheck // SA1019
	if parameters.IndexerConfig.TokenizersPoolConfig != nil {
		return nil, errors.New("tokenizersPoolConfig is not supported; configure a token-producer plugin instead")
	}

	p, err := New(handle.Context(), parameters)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", PluginType, err)
	}

	return p.WithName(name), nil
}

// Indexer and KV-events pool run in background goroutines bound to ctx.
func New(ctx context.Context, config PluginConfig) (*Producer, error) {
	if config.TokenProcessorConfig == nil {
		config.TokenProcessorConfig = kvblock.DefaultTokenProcessorConfig()
	}

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(config.TokenProcessorConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create token processor: %w", err)
	}

	indexer, err := kvcache.NewKVCacheIndexer(ctx, config.IndexerConfig, tokenProcessor)
	if err != nil {
		return nil, fmt.Errorf("failed to create kvcache.Indexer: %w", err)
	}
	go indexer.Run(ctx)

	scorerConfig := kvcache.DefaultKVBlockScorerConfig()
	if config.IndexerConfig != nil && config.IndexerConfig.BackendConfigs != nil {
		scorerConfig.BackendConfigs = config.IndexerConfig.BackendConfigs
	}
	kvBlockScorer, err := kvcache.NewKVBlockScorer(scorerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create KVBlockScorer: %w", err)
	}

	pool := kvevents.NewPool(config.KVEventsConfig, indexer.KVBlockIndex(), tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	subscribersManager := kvevents.NewSubscriberManager(pool)
	if config.KVEventsConfig.ZMQEndpoint != "" {
		if err := subscribersManager.EnsureSubscriber(ctx, "local-subscriber",
			config.KVEventsConfig.ZMQEndpoint, config.KVEventsConfig.TopicFilter, false); err != nil {
			return nil, fmt.Errorf("failed to create local subscriber for global socket mode: %w", err)
		}
	}

	speculativeCache, speculativeTTL, err := buildSpeculativeCache(ctx, config, indexer.KVBlockIndex())
	if err != nil {
		return nil, err
	}

	return &Producer{
		typedName:          plugin.TypedName{Type: PluginType},
		kvCacheIndexer:     indexer,
		kvBlockScorer:      kvBlockScorer,
		subscribersManager: subscribersManager,
		kvEventsConfig:     config.KVEventsConfig,
		dk:                 attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(PluginType),
		pluginState:        plugin.NewPluginState(ctx),
		speculativeCache:   speculativeCache,
		speculativeTTL:     speculativeTTL,
		speculativeEnabled: config.SpeculativeIndexing,
		blockSizeTokens:    config.TokenProcessorConfig.BlockSize,
		subscriberCtx:      ctx,
	}, nil
}

func (p *Producer) TypedName() plugin.TypedName {
	return p.typedName
}

func (p *Producer) WithName(name string) *Producer {
	p.typedName.Name = name
	p.dk = attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name)
	return p
}

func (p *Producer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: attrprefix.PrefixCacheMatchInfo{}}
}

// Declares the TokenizedPrompt dep so the DAG orders token-producer first.
func (p *Producer) Consumes() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: scheduling.TokenizedPrompt{}}
}

func (p *Producer) Produce(ctx context.Context,
	request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint,
) error {
	ctx, span := telemetry.Tracer().Start(ctx, "llm_d.epp.producer.precise_prefix_cache",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(attribute.Int("llm_d.producer.candidate_endpoints", len(endpoints)))
	if request != nil {
		if request.TargetModel != "" {
			span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
		}
		if request.RequestID != "" {
			span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))
		}
	}

	logger := log.FromContext(ctx).WithName(p.typedName.String())

	blockKeys, err := computeBlockKeys(ctx, p.kvCacheIndexer, request, p.blockSizeTokens)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return fmt.Errorf("failed to compute block keys: %w", err)
	}
	if len(blockKeys) == 0 {
		span.SetAttributes(attribute.String("llm_d.producer.result", "skipped_no_tokens"))
		return nil
	}

	endpointSet := extractEndpointSet(endpoints)

	keyToPods, err := p.kvCacheIndexer.KVBlockIndex().Lookup(ctx, blockKeys, endpointSet)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return fmt.Errorf("failed to lookup block keys: %w", err)
	}

	scores, err := p.kvBlockScorer.Score(ctx, blockKeys, keyToPods)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return fmt.Errorf("failed to score block keys: %w", err)
	}

	totalBlocks := len(blockKeys)
	maxMatch := 0
	for _, ep := range endpoints {
		md := ep.GetMetadata()
		if md == nil {
			continue
		}
		addr := fmt.Sprintf("%s:%s", md.Address, md.Port)
		matchLen := int(scores[addr])
		if matchLen > maxMatch {
			maxMatch = matchLen
		}
		ep.Put(p.dk.String(),
			attrprefix.NewPrefixCacheMatchInfo(matchLen, totalBlocks, p.blockSizeTokens))
	}

	if p.speculativeEnabled {
		p.pluginState.Write(request.RequestID, blockKeysStateKey,
			&blockKeysState{blockKeys: blockKeys})
	}

	span.SetAttributes(
		attribute.Int("llm_d.producer.total_blocks", totalBlocks),
		attribute.Int("llm_d.producer.max_match_blocks", maxMatch),
	)

	logger.V(logging.TRACE).Info("Produce completed",
		"blockKeys", totalBlocks, "scores", scores)
	return nil
}
