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
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	rcplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// PluginType is the registered type name of the precise-prefix-cache-producer.
const PluginType = "precise-prefix-cache-producer"

// PluginConfig configures the precise-prefix-cache-producer. Nested fields
// mirror the llm-d-kv-cache configuration shape (see that repo's
// docs/configuration.md for details on TokenProcessorConfig, IndexerConfig,
// and KVEventsConfig).
type PluginConfig struct {
	TokenProcessorConfig *kvblock.TokenProcessorConfig `json:"tokenProcessorConfig"`
	IndexerConfig        *kvcache.Config               `json:"indexerConfig"`
	KVEventsConfig       *kvevents.Config              `json:"kvEventsConfig"`
	// SpeculativeIndexing seeds predicted cache entries for the selected
	// endpoint(s) immediately after a routing decision, so the next
	// same-prefix request hits without waiting for engine confirmation.
	SpeculativeIndexing bool `json:"speculativeIndexing"`
	// SpeculativeTTL bounds how long speculative entries live before
	// eviction. Go duration string; defaults to defaultSpeculativeTTL when
	// empty.
	SpeculativeTTL string `json:"speculativeTTL"`
}

var _ requestcontrol.DataProducer = &Producer{}

// Producer is a DataProducer plugin that maintains a KV-block prefix-cache
// index by subscribing to vLLM KV-events and writes per-endpoint
// PrefixCacheMatchInfo for each request. Operators pair it with the
// generic prefix-cache-scorer (set prefixMatchInfoProducerName to this
// producer's instance name) to route requests by precise cache locality.
//
// Speculative-indexing logic lives in prerequest.go; per-pod ZMQ subscriber
// lifecycle in extractor.go.
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

	// healthMonitor tracks per-endpoint KV-events pipeline health by
	// recording the last time a confirmed (non-speculative) entry was
	// observed in an index lookup and the last time a request was routed
	// to each endpoint. Data-collection only here; the eventual consumer
	// (dynamic speculative TTL) is tracked in
	// https://github.com/llm-d/llm-d-kv-cache/issues/544.
	healthMonitor *KVEventsHealthMonitor

	// Plugin-lifetime, not request-scoped: SubscriberManager binds each
	// subscriber's goroutine to the ctx passed at registration.
	subscriberCtx context.Context
}

// PluginFactory parses the raw plugin configuration and returns a configured
// Producer. Rejects configs with indexerConfig.tokenizersPoolConfig set, since
// this producer is tokens-only and requires an upstream token-producer.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	indexerConfig, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize indexer config: %w", err)
	}

	parameters := PluginConfig{
		IndexerConfig:  indexerConfig,
		KVEventsConfig: kvevents.DefaultConfig(),
	}

	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
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

	p, err := New(handle.Context(), name, parameters)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", PluginType, err)
	}

	return p, nil
}

// New constructs a precise-prefix-cache-producer. The instance name becomes
// the producer name on PrefixCacheMatchInfoDataKey, which downstream
// consumers must match (see prefix-cache-scorer's prefixMatchInfoProducerName).
// The kvcache indexer, KV-events pool, and any local ZMQ subscriber start
// in background goroutines bound to ctx.
func New(ctx context.Context, name string, config PluginConfig) (*Producer, error) {
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

	hasKVEventsCfg := config.KVEventsConfig != nil &&
		(config.KVEventsConfig.ZMQEndpoint != "" || config.KVEventsConfig.DiscoverPods)

	return &Producer{
		typedName:          plugin.TypedName{Type: PluginType, Name: name},
		kvCacheIndexer:     indexer,
		kvBlockScorer:      kvBlockScorer,
		subscribersManager: subscribersManager,
		kvEventsConfig:     config.KVEventsConfig,
		dk:                 attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		pluginState:        plugin.NewPluginState(ctx),
		speculativeCache:   speculativeCache,
		speculativeTTL:     speculativeTTL,
		speculativeEnabled: config.SpeculativeIndexing,
		healthMonitor:      NewKVEventsHealthMonitor(hasKVEventsCfg),
		blockSizeTokens:    tokenProcessor.BlockSize(),
		subscriberCtx:      ctx,
	}, nil
}

// TypedName returns the plugin's registered type and name.
func (p *Producer) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces declares the PrefixCacheMatchInfoDataKey published per endpoint,
// name-bound to this producer instance.
func (p *Producer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: attrprefix.PrefixCacheMatchInfo{}}
}

// Consumes declares the TokenizedPrompt dependency from token-producer so
// the data-layer DAG orders tokenization before this producer runs.
func (p *Producer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: scheduling.TokenizedPrompt{}},
	}
}

// Produce hashes the request's TokenizedPrompt into KV-block keys, looks
// them up in the per-endpoint KV-block index, and writes PrefixCacheMatchInfo
// to each candidate endpoint. No-op when the request carries no tokens.
// With speculativeIndexing enabled, the computed block keys are stashed
// for PreRequest to seed the index after a routing decision is made.
func (p *Producer) Produce(ctx context.Context,
	request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint,
) error {
	ctx, span := tracing.Tracer(rcplugins.TracerScope).Start(ctx, "produce_precise_prefix_cache",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(attribute.Int("llm_d.epp.producer.candidate_endpoints", len(endpoints)))
	if request != nil {
		if request.TargetModel != "" {
			span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
		}
		if request.RequestID != "" {
			span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))
		}
	}

	perPromptKeys, err := computeBlockKeys(ctx, p.kvCacheIndexer, request, p.blockSizeTokens)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return fmt.Errorf("failed to compute block keys: %w", err)
	}
	if len(perPromptKeys) == 0 {
		span.SetAttributes(attribute.String("llm_d.epp.producer.result", "skipped_no_tokens"))
		return nil
	}

	return p.produceFromBlockKeys(ctx, span, request, endpoints, perPromptKeys)
}

func (p *Producer) produceFromBlockKeys(ctx context.Context, span trace.Span,
	request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint,
	perPromptKeys [][]kvblock.BlockHash,
) error {
	logger := log.FromContext(ctx).WithName(p.typedName.String())
	endpointSet := extractEndpointSet(endpoints)

	type promptLookup struct {
		keys      []kvblock.BlockHash
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry
	}

	aggregatedScores := make(map[string]float64)
	totalBlocks := 0
	lookups := make([]promptLookup, 0, len(perPromptKeys))
	for _, blockKeys := range perPromptKeys {
		keyToPods, err := p.kvCacheIndexer.KVBlockIndex().Lookup(ctx, blockKeys, endpointSet)
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
			return fmt.Errorf("failed to lookup block keys: %w", err)
		}
		// Confirmed (non-speculative) entries in the lookup result mean KV events
		// for those endpoints are flowing — surface that to the health monitor so
		// it can distinguish broken pipelines from genuine idle.
		for _, pods := range keyToPods {
			for _, pod := range pods {
				if !pod.Speculative {
					p.healthMonitor.RecordConfirmedEntry(pod.PodIdentifier)
				}
			}
		}
		scores, err := p.kvBlockScorer.Score(ctx, blockKeys, keyToPods)
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
			return fmt.Errorf("failed to score block keys: %w", err)
		}
		for pod, score := range scores {
			aggregatedScores[pod] += score
		}
		totalBlocks += len(blockKeys)
		lookups = append(lookups, promptLookup{keys: blockKeys, keyToPods: keyToPods})
	}

	maxMatch := 0
	for _, ep := range endpoints {
		md := ep.GetMetadata()
		if md == nil {
			continue
		}
		addr := fmt.Sprintf("%s:%s", md.Address, md.Port)
		matchLen := int(aggregatedScores[addr])
		if matchLen > maxMatch {
			maxMatch = matchLen
		}
		cachedBlocks := 0
		for _, lu := range lookups {
			cachedBlocks += matchedBlockCount(lu.keys, lu.keyToPods, addr)
		}
		ep.Put(p.dk.String(),
			attrprefix.NewPrefixCacheMatchInfo(matchLen, totalBlocks, p.blockSizeTokens).WithCachedBlockCount(cachedBlocks))
	}

	if p.speculativeEnabled {
		p.pluginState.Write(request.RequestID, blockKeysStateKey,
			&blockKeysState{perPromptKeys: perPromptKeys})
	}

	span.SetAttributes(
		attribute.Int("llm_d.epp.producer.total_blocks", totalBlocks),
		attribute.Int("llm_d.epp.producer.max_match_blocks", maxMatch),
	)

	logger.V(logging.TRACE).Info("Produce completed",
		"blockKeys", totalBlocks, "scores", aggregatedScores)
	return nil
}
