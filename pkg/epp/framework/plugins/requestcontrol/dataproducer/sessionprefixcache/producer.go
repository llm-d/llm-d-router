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

// Package sessionprefixcache provides the session-prefix-cache-producer: a
// DataProducer that resolves a session producer's engine-block prefixes
// against block residency observed from vLLM KV events and publishes
// per-endpoint PrefixCacheMatchInfo.
package sessionprefixcache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

// PluginType is the registered type name of the session-prefix-cache-producer.
const PluginType = "session-prefix-cache-producer"

// PluginConfig configures the session-prefix-cache-producer.
type PluginConfig struct {
	// SessionCacheRequestProducerName names the plugin instance that publishes
	// SessionCacheRequest for each request.
	SessionCacheRequestProducerName string `json:"sessionCacheRequestProducerName"`
	// IndexConfig configures the in-memory residency index.
	IndexConfig *kvblock.IndexConfig `json:"indexConfig"`
	// KVEventsConfig configures the KV-events pool. Per-endpoint discovery of
	// vLLM engines is required.
	KVEventsConfig *kvevents.Config `json:"kvEventsConfig"`
}

var _ requestcontrol.DataProducer = &Producer{}

// Producer is a DataProducer plugin that keeps engine-block residency per
// endpoint, data-parallel rank, and KV-cache group from vLLM KV events, and
// resolves each request's SessionCacheRequest prefixes against it. It writes
// PrefixCacheMatchInfo per endpoint for the prefix-cache-scorer and stamps
// the request with its session identity in PreRequest. Event handling lives
// in events.go and per-pod subscriber lifecycle in extractor.go.
type Producer struct {
	typedName plugin.TypedName
	dk        plugin.DataKey
	sessionDK plugin.DataKey

	events        *eventConsumer
	subscriptions *kvevents.EndpointSubscriptions
}

// PluginFactory parses the raw plugin configuration and returns a configured
// Producer.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := PluginConfig{
		IndexConfig:    kvblock.DefaultIndexConfig(),
		KVEventsConfig: kvevents.DefaultConfig(),
	}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", PluginType, err)
		}
	}
	p, err := New(handle.Context(), name, parameters)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", PluginType, err)
	}
	return p, nil
}

// New constructs a session-prefix-cache-producer. The instance name becomes
// the producer name on PrefixCacheMatchInfoDataKey, which downstream
// consumers must match. The KV-events pool starts in background goroutines
// bound to ctx.
func New(ctx context.Context, name string, config PluginConfig) (*Producer, error) {
	if config.SessionCacheRequestProducerName == "" {
		return nil, errors.New("sessionCacheRequestProducerName is required")
	}
	if config.IndexConfig == nil {
		return nil, errors.New("indexConfig is required")
	}
	if config.IndexConfig.RedisConfig != nil {
		return nil, errors.New("indexConfig.redisConfig is not supported: residency is local to each producer instance")
	}
	kc := config.KVEventsConfig
	if kc == nil || !kc.DiscoverPods || kc.PodDiscoveryConfig == nil || kc.ZMQEndpoint != "" {
		return nil, errors.New("kvEventsConfig must enable per-endpoint pod discovery without a global zmqEndpoint")
	}
	if kc.EngineType != "" && kc.EngineType != engineadapter.EngineTypeVLLM {
		return nil, fmt.Errorf("kvEventsConfig.engineType %q is not supported: session events require vLLM", kc.EngineType)
	}
	index, err := kvblock.NewIndex(ctx, config.IndexConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create residency index: %w", err)
	}
	adapter, err := engineadapter.NewAdapter(kc.EngineType)
	if err != nil {
		return nil, fmt.Errorf("failed to create KV-events engine adapter: %w", err)
	}
	events := newEventConsumer(kvblock.NewTracedIndex(index))
	pool := kvevents.NewConsumerPool(kc, adapter, events)
	subscriptions, err := kvevents.NewEndpointSubscriptions(ctx, kc, kvevents.NewSubscriberManager(pool))
	if err != nil {
		return nil, err
	}
	pool.Start(ctx)

	return &Producer{
		typedName:     plugin.TypedName{Type: PluginType, Name: name},
		dk:            attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		sessionDK:     attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName(config.SessionCacheRequestProducerName),
		events:        events,
		subscriptions: subscriptions,
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

// Consumes declares the SessionCacheRequest dependency on the configured
// producer so the data-layer DAG orders it before this producer runs.
func (p *Producer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{p.sessionDK: attrsession.SessionCacheRequest{}},
	}
}

// Produce resolves each candidate prefix of the request's SessionCacheRequest
// against every observed cache of the candidate endpoints and writes the best
// match per endpoint as PrefixCacheMatchInfo. Match and coverage are token
// counts through unit-size blocks, so consumers see the same numbers whatever
// block size the engine uses. Endpoints without a match get an empty result.
// No-op for a request without a session identity.
func (p *Producer) Produce(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	if request == nil {
		return nil
	}
	lookup, ok := scheduling.ReadRequestAttribute[attrsession.SessionCacheRequest](request, p.sessionDK)
	if !ok || lookup.SessionID == "" {
		return nil
	}
	if lookup.TotalTokens < 0 {
		return errors.New("session cache request totalTokens must be nonnegative")
	}
	candidates := sets.New[string]()
	for _, ep := range endpoints {
		if md := ep.GetMetadata(); md != nil {
			candidates.Insert(endpointID(md))
		}
	}
	best := make(map[string]*attrprefix.PrefixCacheMatchInfo, len(endpoints))
	for _, prefix := range lookup.Prefixes {
		if len(prefix.BlockHashes) == 0 {
			continue
		}
		if prefix.BlockSizeTokens <= 0 {
			return errors.New("session cache prefix blockSizeTokens must be positive")
		}
		for _, scope := range p.events.scopes(candidates, prefix.BlockSizeTokens) {
			info, err := p.match(ctx, prefix, scope, lookup.TotalTokens)
			if err != nil {
				return err
			}
			if previous := best[scope.Endpoint]; previous == nil || betterMatch(info, previous) {
				best[scope.Endpoint] = info
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	for _, ep := range endpoints {
		md := ep.GetMetadata()
		if md == nil {
			continue
		}
		info := best[endpointID(md)]
		if info == nil {
			info = emptyMatch(lookup.TotalTokens)
		}
		ep.Put(p.dk, info)
	}
	return nil
}

// match counts the leading blocks of prefix resident in scope. With Exact
// set, blocks past the prompt's own length are not queried and the matched
// tokens are also reported as cached GPU tokens.
func (p *Producer) match(ctx context.Context, prefix attrsession.SessionCachePrefix, scope kvblock.EngineScope, totalTokens int) (*attrprefix.PrefixCacheMatchInfo, error) {
	hashes := prefix.BlockHashes
	if prefix.Exact {
		hashes = hashes[:min(len(hashes), totalTokens/prefix.BlockSizeTokens)]
	}
	resident, err := p.events.residentPrefix(ctx, scope, hashes)
	if err != nil {
		return nil, fmt.Errorf("match session cache prefix: %w", err)
	}
	matchedTokens := min(totalTokens, resident*prefix.BlockSizeTokens)
	info := attrprefix.NewPrefixCacheMatchInfo(matchedTokens, totalTokens, 1).
		WithCachedBlockCount(0).
		WithCachedBlocksByTier(map[string]int{}).
		WithTotalTokens(totalTokens)
	if prefix.Exact {
		info.WithCachedBlockCount(matchedTokens).WithCachedBlocksByTier(map[string]int{gpuTier: matchedTokens})
	}
	return info, nil
}

func betterMatch(a, b *attrprefix.PrefixCacheMatchInfo) bool {
	return a.MatchBlocks() > b.MatchBlocks() ||
		(a.MatchBlocks() == b.MatchBlocks() && a.CachedBlockCount() > b.CachedBlockCount())
}

// emptyMatch is the result for an endpoint with no resident candidate prefix.
func emptyMatch(totalTokens int) *attrprefix.PrefixCacheMatchInfo {
	return attrprefix.NewPrefixCacheMatchInfo(0, totalTokens, 1).
		WithCachedBlocksByTier(map[string]int{}).
		WithTotalTokens(totalTokens)
}

// endpointID is the KV-events source identity of an endpoint.
func endpointID(md *fwkdl.EndpointMetadata) string {
	return fmt.Sprintf("%s:%s", md.Address, md.Port)
}
