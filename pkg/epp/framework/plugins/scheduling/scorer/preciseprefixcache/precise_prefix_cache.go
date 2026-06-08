/*
Copyright 2025 The Kubernetes Authors.

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

// Package preciseprefixcache implements the precise-prefix-cache-scorer
// plugin: a Scorer that composes a precise-prefix-cache-producer and a
// prefix-cache-scorer behind a single legacy plugin type.
package preciseprefixcache

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
	prefixscorer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
)

// PrecisePrefixCachePluginType is the registered plugin type name.
const PrecisePrefixCachePluginType = "precise-prefix-cache-scorer"

// PluginConfig is the all-in-one config shape. Fields forward verbatim to
// preciseproducer.PluginConfig.
type PluginConfig struct {
	TokenProcessorConfig *kvblock.TokenProcessorConfig `json:"tokenProcessorConfig"`
	IndexerConfig        *kvcache.Config               `json:"indexerConfig"`
	KVEventsConfig       *kvevents.Config              `json:"kvEventsConfig"`
	SpeculativeIndexing  bool                          `json:"speculativeIndexing"`
	SpeculativeTTL       string                        `json:"speculativeTTL"`
}

// Plugin composes a precise-prefix-cache-producer and a prefix-cache-scorer
// behind the precise-prefix-cache-scorer plugin type, exposing Scorer,
// DataProducer, PreRequest, and EndpointExtractor by delegation.
//
// Deprecated: configure precise-prefix-cache-producer and prefix-cache-scorer
// directly.
type Plugin struct {
	typedName plugin.TypedName
	producer  *legacyProducer
	scorer    *prefixscorer.Plugin
}

var (
	_ scheduling.Scorer           = &Plugin{}
	_ requestcontrol.DataProducer = &Plugin{}
	_ requestcontrol.PreRequest   = &Plugin{}
	_ fwkdl.EndpointExtractor     = &Plugin{}
)

// PluginFactory constructs the plugin. When a precise-prefix-cache-producer
// is already registered in the handle, the factory returns a
// prefix-cache-scorer bound to it and ignores any legacy parameters.
// Otherwise the legacy parameters drive an internal producer + scorer pair.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	ctx := handle.Context()
	logger := log.FromContext(ctx)
	logger.Info("DEPRECATION: precise-prefix-cache-scorer is deprecated; " +
		"configure precise-prefix-cache-producer + prefix-cache-scorer with " +
		"prefixMatchInfoProducerName: precise-prefix-cache-producer")

	existing, err := findExistingPreciseProducer(handle)
	if err != nil {
		return nil, err
	}
	if existing != nil {
		logger.Info("deferring precise-prefix-cache-scorer to existing precise-prefix-cache-producer",
			"producer", existing.TypedName().String())
		return prefixscorer.New(ctx, name, existing.TypedName().Name)
	}

	// Self-host: defaults first, then overlay the operator's YAML — matches
	// the historical factory so partial IndexerConfig (e.g. only
	// tokenizersPoolConfig set) doesn't leave the indexer half-built.
	defaultIndexerCfg, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize indexer config: %w", err)
	}
	legacy := PluginConfig{
		IndexerConfig:  defaultIndexerCfg,
		KVEventsConfig: kvevents.DefaultConfig(),
	}
	if rawParameters != nil {
		if err := rawParameters.Decode(&legacy); err != nil {
			return nil, fmt.Errorf("failed to parse %s config: %w", PrecisePrefixCachePluginType, err)
		}
	}

	producerCfg := preciseproducer.PluginConfig{
		TokenProcessorConfig: legacy.TokenProcessorConfig,
		IndexerConfig:        legacy.IndexerConfig,
		KVEventsConfig:       legacy.KVEventsConfig,
		SpeculativeIndexing:  legacy.SpeculativeIndexing,
		SpeculativeTTL:       legacy.SpeculativeTTL,
	}

	producer, err := newLegacyProducer(ctx, name, producerCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create internal producer for %s: %w", PrecisePrefixCachePluginType, err)
	}

	scorer, err := prefixscorer.New(ctx, name, name)
	if err != nil {
		return nil, fmt.Errorf("failed to create internal scorer for %s: %w", PrecisePrefixCachePluginType, err)
	}

	return &Plugin{
		typedName: plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: name},
		producer:  producer,
		scorer:    scorer,
	}, nil
}

// findExistingPreciseProducer returns the precise-prefix-cache-producer
// registered in the handle, or nil if none is configured. Errors when more
// than one is found, since the legacy plugin has no way to choose between
// multiple producers.
func findExistingPreciseProducer(handle plugin.Handle) (*preciseproducer.Producer, error) {
	var found *preciseproducer.Producer
	for _, p := range handle.GetAllPlugins() {
		prod, ok := p.(*preciseproducer.Producer)
		if !ok {
			continue
		}
		if found != nil {
			return nil, fmt.Errorf("multiple precise-prefix-cache-producer instances configured (%s, %s); "+
				"precise-prefix-cache-scorer cannot disambiguate — configure prefix-cache-scorer directly with an explicit prefixMatchInfoProducerName",
				found.TypedName().String(), prod.TypedName().String())
		}
		found = prod
	}
	return found, nil
}

func (p *Plugin) TypedName() plugin.TypedName { return p.typedName }

func (p *Plugin) Category() scheduling.ScorerCategory { return p.scorer.Category() }

// Score traces under the historical span name "llm_d.epp.scorer.prefix_cache"
// with the original attribute schema so dashboards built against the heavy
// scorer keep working, then delegates to the inner prefix-cache-scorer.
func (p *Plugin) Score(ctx context.Context,
	req *scheduling.InferenceRequest, endpoints []scheduling.Endpoint,
) map[scheduling.Endpoint]float64 {
	ctx, span := tracing.Tracer().Start(ctx, "llm_d.epp.scorer.prefix_cache",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(attribute.Int("llm_d.scorer.candidate_endpoints", len(endpoints)))
	if req != nil {
		if req.TargetModel != "" {
			span.SetAttributes(attribute.String("gen_ai.request.model", req.TargetModel))
		}
		if req.RequestID != "" {
			span.SetAttributes(attribute.String("gen_ai.request.id", req.RequestID))
		}
	}

	scores := p.scorer.Score(ctx, req, endpoints)

	if len(scores) > 0 {
		var maxScore, totalScore float64
		for _, s := range scores {
			if s > maxScore {
				maxScore = s
			}
			totalScore += s
		}
		span.SetAttributes(
			attribute.Float64("llm_d.scorer.score.max", maxScore),
			attribute.Float64("llm_d.scorer.score.avg", totalScore/float64(len(scores))),
			attribute.Int("llm_d.scorer.endpoints_scored", len(scores)),
		)
	}

	return scores
}

func (p *Plugin) Produces() map[plugin.DataKey]any { return p.producer.Produces() }

// Consumes returns the union of the inner producer's and scorer's
// dependencies so the data-layer DAG sees both on this single plugin instance.
func (p *Plugin) Consumes() plugin.DataDependencies {
	merged := plugin.DataDependencies{
		Required: map[plugin.DataKey]any{},
		Optional: map[plugin.DataKey]any{},
	}
	for _, dep := range []plugin.DataDependencies{p.producer.Consumes(), p.scorer.Consumes()} {
		for k, v := range dep.Required {
			merged.Required[k] = v
		}
		for k, v := range dep.Optional {
			merged.Optional[k] = v
		}
	}
	return merged
}

func (p *Plugin) Produce(ctx context.Context,
	req *scheduling.InferenceRequest, endpoints []scheduling.Endpoint,
) error {
	return p.producer.Produce(ctx, req, endpoints)
}

func (p *Plugin) PreRequest(ctx context.Context,
	req *scheduling.InferenceRequest, result *scheduling.SchedulingResult,
) {
	p.producer.PreRequest(ctx, req, result)
}

func (p *Plugin) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	return p.producer.Extract(ctx, event)
}
