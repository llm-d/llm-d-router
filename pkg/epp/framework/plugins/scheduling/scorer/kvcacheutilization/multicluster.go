/*
Copyright 2026 The Kubernetes Authors.

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

package kvcacheutilization

import (
	"context"
	"encoding/json"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

// MultiClusterScorerType is the cross-cluster kv-cache-utilization scorer.
const MultiClusterScorerType = "multicluster-kv-cache-utilization-scorer"

var _ fwksched.Scorer = &MultiClusterScorer{}

// MultiClusterScorerFactory builds the cross-cluster kv-cache-utilization scorer.
func MultiClusterScorerFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	inner, err := KvCacheUtilizationScorerFactory(name, params, handle)
	if err != nil {
		return nil, err
	}
	return &MultiClusterScorer{KVCacheUtilizationScorer: inner.(*KVCacheUtilizationScorer)}, nil
}

// MultiClusterScorer scores cluster endpoints from a pool-level KV-cache
// utilization summary, read by name, instead of a single pod's scrape.
type MultiClusterScorer struct {
	*KVCacheUtilizationScorer
}

// TypedName reports the multi-cluster type with this instance's name.
func (s *MultiClusterScorer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: MultiClusterScorerType, Name: s.KVCacheUtilizationScorer.TypedName().Name}
}

var _ fwkplugin.ConsumerPlugin = &MultiClusterScorer{}

// Consumes marks the pool KV-cache utilization attribute Required, so a config missing
// the multicluster metrics extractor fails at load rather than silently no-scoring.
func (s *MultiClusterScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(attrmetrics.MultiClusterKVCacheUtilizationKey, ""): attrmetrics.ScalarMetricValue(0),
		},
	}
}

// Score scores each cluster endpoint as 1 - its pool KV-cache utilization.
// Endpoints without the attribute are left unscored.
func (s *MultiClusterScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		util, ok := attrmetrics.ReadScalarMetricValue(ep, attrmetrics.MultiClusterKVCacheUtilizationKey)
		if !ok {
			continue
		}
		scores[ep] = 1 - float64(util)
	}
	return scores
}
