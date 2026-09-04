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

package kvcacheutilization

import (
	"context"
	"encoding/json"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

// kvCacheUsageDataKey is the attribute this scorer declares and reads. Keeping
// it in one place stops the declaration in Consumes and the read in Score from
// naming different things, which is how they came apart in the first place.
var kvCacheUsageDataKey = fwkplugin.NewDataKey(metrics.KVCacheUsagePercentKey, metrics.MetricsExtractorType)

const (
	KvCacheUtilizationScorerType = "kv-cache-utilization-scorer"
)

// compile-time type assertion
var (
	_ fwksched.Scorer          = &KVCacheUtilizationScorer{}
	_ fwkplugin.ConsumerPlugin = &KVCacheUtilizationScorer{}
)

// KvCacheUtilizationScorerFactory defines the factory function for KVCacheUtilizationScorer.
func KvCacheUtilizationScorerFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewKVCacheUtilizationScorer().WithName(name), nil
}

// NewKVCacheUtilizationScorer initializes a new KVCacheUtilizationScorer and returns its pointer.
func NewKVCacheUtilizationScorer() *KVCacheUtilizationScorer {
	return &KVCacheUtilizationScorer{
		typedName: fwkplugin.TypedName{Type: KvCacheUtilizationScorerType, Name: KvCacheUtilizationScorerType},
	}
}

// KVCacheUtilizationScorer scores list of candidate endpoints based on KV cache utilization.
type KVCacheUtilizationScorer struct {
	typedName fwkplugin.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *KVCacheUtilizationScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *KVCacheUtilizationScorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

// Consumes declares the KV-cache utilization attribute the core metrics
// extractor publishes. Required, so a config with no producer for it fails at
// init rather than scoring on an absent value; the registry validates the
// declared type against the producer's declaration.
func (s *KVCacheUtilizationScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			kvCacheUsageDataKey: attrmetrics.ScalarMetricValue(0),
		},
	}
}

// WithName sets the name of the scorer.
func (s *KVCacheUtilizationScorer) WithName(name string) *KVCacheUtilizationScorer {
	s.typedName.Name = name
	return s
}

// Score scores each endpoint as 1 - its KV-cache utilization, read from the
// attribute Consumes declares.
//
// An endpoint without the attribute is left unscored rather than scored. The
// read went through GetMetrics() before, where an endpoint the extractor has
// not populated is indistinguishable from one reporting an empty cache: both
// give 0, so the unknown endpoint scored 1.0 and outranked every endpoint
// whose utilization was actually known. Reading the attribute makes absence
// observable, and omitting is what the multi-cluster variant in this package
// already does.
func (s *KVCacheUtilizationScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		util, ok := attrmetrics.ReadScalarMetricValue(endpoint, kvCacheUsageDataKey)
		if !ok {
			continue
		}
		scores[endpoint] = 1 - float64(util)
	}
	return scores
}
