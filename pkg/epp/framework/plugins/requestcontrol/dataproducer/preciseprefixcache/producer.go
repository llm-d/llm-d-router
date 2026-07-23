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
	"sort"
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

var (
	_ requestcontrol.DataProducer = &Producer{}
	_ plugin.StateDumper          = &Producer{}
)

// subscriberManager is the subset of kvevents.SubscriberManager the producer
// relies on, narrowed so tests can substitute a fake.
type subscriberManager interface {
	EnsureSubscriber(ctx context.Context, podIdentifier, endpoint, topicFilter string, remoteSocket bool) error
	RemoveSubscriber(ctx context.Context, podIdentifier string)
	GetActiveSubscribers() ([]string, []string)
	Shutdown(ctx context.Context)
}

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
	tokenProcessor kvblock.TokenProcessor

	subscribersManager subscriberManager
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

	return &Producer{
		typedName:          plugin.TypedName{Type: PluginType, Name: name},
		kvCacheIndexer:     indexer,
		tokenProcessor:     tokenProcessor,
		kvBlockScorer:      kvBlockScorer,
		subscribersManager: subscribersManager,
		kvEventsConfig:     config.KVEventsConfig,
		dk:                 attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		pluginState:        plugin.NewPluginState(ctx),
		speculativeCache:   speculativeCache,
		speculativeTTL:     speculativeTTL,
		speculativeEnabled: config.SpeculativeIndexing,
		blockSizeTokens:    tokenProcessor.BlockSize(),
		subscriberCtx:      ctx,
	}, nil
}

// GetEngineKeysForRequest computes the KV-block engine keys (uint64 block
// hashes) for the request's tokenized prompt, flattened across all prompts.
// Returns nil when the request carries no tokens or produces no full blocks.
func (p *Producer) GetEngineKeysForRequest(ctx context.Context,
	request *scheduling.InferenceRequest) ([]uint64, error) {
	engineKeys, _, err := p.GetEngineKeysAndDigestsForRequest(ctx, request)
	return engineKeys, err
}

// GetEngineKeysAndDigestsForRequest returns the per-block uint64 engine keys
// (used for index lookup and wire compatibility) alongside the full-width hash
// digests (used by the prefetch plugin to build vLLM fs-connector filenames:
// 32 bytes for sha256_cbor, 8 bytes for fnv64a). Both slices are flattened
// across prompts and aligned index-for-index.
func (p *Producer) GetEngineKeysAndDigestsForRequest(_ context.Context,
	request *scheduling.InferenceRequest) ([]uint64, [][]byte, error) {
	engineKeys, digests, err := computeBlockKeysWithDigests(p.tokenProcessor, request, p.blockSizeTokens)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compute engine keys: %w", err)
	}
	return engineKeys, digests, nil
}

// TypedName returns the plugin's registered type and name.
func (p *Producer) TypedName() plugin.TypedName {
	return p.typedName
}

// Debug-dump caps keep the payload bounded. A list is partial when its matching
// TotalX exceeds MaxX.
const (
	maxDumpSubscribers        = 100
	maxDumpSpeculativeEntries = 100
)

// precisePrefixState is the snapshot returned by DumpState. The KV-block index
// is keyed by prompt-derived block hashes and is not enumerable, so it is not
// reported; the active subscriber pod identities and the live speculative
// request ids are enumerated (sorted and capped) for debugging.
type precisePrefixState struct {
	Subscribers             []string `json:"subscribers"`
	TotalSubscribers        int      `json:"totalSubscribers"`
	MaxSubscribers          int      `json:"maxSubscribers"`
	SpeculativeIndexing     bool     `json:"speculativeIndexing"`
	SpeculativeEntries      []string `json:"speculativeEntries"`
	TotalSpeculativeEntries int      `json:"totalSpeculativeEntries"`
	MaxSpeculativeEntries   int      `json:"maxSpeculativeEntries"`
	BlockSizeTokens         int      `json:"blockSizeTokens"`
}

// DumpState reports the producer's bounded operational state: the active
// KV-event subscriber pod identities, whether speculative indexing is on and
// the live speculative request ids, and the block size in tokens. Both lists
// are sorted and capped; the prompt-derived block index is not exposed.
func (p *Producer) DumpState() (json.RawMessage, error) {
	subscribers := []string{}
	var totalSubscribers int
	if p.subscribersManager != nil {
		ids, _ := p.subscribersManager.GetActiveSubscribers()
		totalSubscribers = len(ids)
		subscribers = sortedCapped(ids, maxDumpSubscribers)
	}
	speculativeEntries := []string{}
	var totalSpeculativeEntries int
	if p.speculativeCache != nil {
		keys := p.speculativeCache.Keys()
		totalSpeculativeEntries = len(keys)
		speculativeEntries = sortedCapped(keys, maxDumpSpeculativeEntries)
	}
	return json.Marshal(precisePrefixState{
		Subscribers:             subscribers,
		TotalSubscribers:        totalSubscribers,
		MaxSubscribers:          maxDumpSubscribers,
		SpeculativeIndexing:     p.speculativeEnabled,
		SpeculativeEntries:      speculativeEntries,
		TotalSpeculativeEntries: totalSpeculativeEntries,
		MaxSpeculativeEntries:   maxDumpSpeculativeEntries,
		BlockSizeTokens:         p.blockSizeTokens,
	})
}

// sortedCapped returns a sorted copy of in (never nil, so empty lists serialize
// as [] not null), truncated to limit entries.
func sortedCapped(in []string, limit int) []string {
	out := append([]string{}, in...)
	sort.Strings(out)
	if len(out) > limit {
		out = out[:limit]
	}
	return out
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
		cachedBlocksByTier := map[string]int{}
		for _, lu := range lookups {
			cachedBlocks += matchedBlockCount(lu.keys, lu.keyToPods, addr)
			for tier, count := range matchedBlockCountByTier(lu.keys, lu.keyToPods, addr) {
				cachedBlocksByTier[tier] += count
			}
		}
		ep.Put(p.dk.String(),
			attrprefix.NewPrefixCacheMatchInfo(matchLen, totalBlocks, p.blockSizeTokens).
				WithCachedBlockCount(cachedBlocks).
				WithCachedBlocksByTier(cachedBlocksByTier))
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
