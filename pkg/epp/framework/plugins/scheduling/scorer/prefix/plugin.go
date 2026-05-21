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

package prefix

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// Config defines the configuration for the prefix cache scorer plugin.
type Config struct {
	// The name of the data producer that produces PrefixCacheMatchInfo.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName,omitempty"`
}

// Plugin implements the prefix cache aware scoring logic.
type Plugin struct {
	typedName          plugin.TypedName
	prefixMatchDataKey plugin.DataKey
}

// compile-time type assertions
var (
	_ fwksched.Scorer = &Plugin{}
)

const (
	// Type is the unique identifier for the prefix cache scorer plugin.
	PrefixCacheScorerPluginType = "prefix-cache-scorer"
)

// PrefixCachePluginFactory defines the factory function for the Prefix plugin.
func PrefixCachePluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	var cfg Config
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &cfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal prefix cache scorer parameters: %w", err)
		}
	}

	p, err := New(handle.Context(), name, cfg.PrefixMatchInfoProducerName)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// New initializes a new prefix Plugin.
func New(_ context.Context, name string, producerName string) (*Plugin, error) {
	return &Plugin{
		typedName: plugin.TypedName{
			Type: PrefixCacheScorerPluginType,
			Name: name,
		},
		prefixMatchDataKey: attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName),
	}, nil
}

// TypedName returns the type and name of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Category returns the preference the scorer applies (Affinity).
func (p *Plugin) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// Produces returns the data produced by the plugin.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{}
}

// Consumes returns the data consumed by the plugin.
func (p *Plugin) Consumes() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.prefixMatchDataKey: attrprefix.PrefixCacheMatchInfo{}}
}

// Score returns the scoring result for the given list of pods based on prefix cache match info.
func (p *Plugin) Score(ctx context.Context, _ *fwksched.CycleState, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)

	for _, endpoint := range endpoints {
		// Default to score 0 if PrefixCacheMatchInfo is missing or invalid.
		scores[endpoint] = 0.0
		info, ok := endpoint.Get(p.prefixMatchDataKey.String())
		if !ok {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo not found for endpoint, assigning score 0", "endpoint", endpoint, "key", p.prefixMatchDataKey.String())
			continue
		}

		if prefixMatchInfo, ok := info.(*attrprefix.PrefixCacheMatchInfo); ok {
			if prefixMatchInfo.TotalBlocks() != 0 {
				scores[endpoint] = float64(prefixMatchInfo.MatchBlocks()) / float64(prefixMatchInfo.TotalBlocks())
			}
		} else {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo has unexpected type, assigning score 0", "endpoint", endpoint)
		}
	}
	return scores
}
