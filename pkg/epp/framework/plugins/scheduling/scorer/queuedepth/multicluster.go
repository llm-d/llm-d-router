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

package queuedepth

import (
	"context"
	"encoding/json"
	"math"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

// MultiClusterScorerType is the cross-cluster queue scorer.
const MultiClusterScorerType = "multicluster-queue-scorer"

var _ fwksched.Scorer = &MultiClusterScorer{}

// MultiClusterScorerFactory builds the cross-cluster queue scorer.
func MultiClusterScorerFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	inner, err := QueueScorerFactory(name, params, handle)
	if err != nil {
		return nil, err
	}
	return &MultiClusterScorer{QueueScorer: inner.(*QueueScorer)}, nil
}

// MultiClusterScorer scores cluster endpoints from a pool-level queue-depth
// summary, read by name, instead of a single pod's scrape.
type MultiClusterScorer struct {
	*QueueScorer
}

// TypedName reports the multi-cluster type with this instance's name.
func (s *MultiClusterScorer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: MultiClusterScorerType, Name: s.QueueScorer.TypedName().Name}
}

var _ fwkplugin.ConsumerPlugin = &MultiClusterScorer{}

// Consumes marks the pool queue-size attribute Required, so a config missing the
// multicluster metrics extractor fails at load rather than silently no-scoring.
func (s *MultiClusterScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(attrmetrics.MultiClusterQueueSizeKey, ""): attrmetrics.ScalarMetricValue(0),
		},
	}
}

// Score scores each cluster endpoint by its pool queue depth, min-max normalized
// across the cluster candidates: the shortest queue scores 1, the longest 0, and
// equal queues score a neutral 1. Endpoints without the attribute are unscored.
func (s *MultiClusterScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	queues := make(map[fwksched.Endpoint]float64, len(endpoints))
	minQ, maxQ := math.MaxFloat64, -math.MaxFloat64
	for _, ep := range endpoints {
		q, ok := attrmetrics.ReadScalarMetricValue(ep, attrmetrics.MultiClusterQueueSizeKey)
		if !ok {
			continue
		}
		queues[ep] = float64(q)
		minQ = math.Min(minQ, float64(q))
		maxQ = math.Max(maxQ, float64(q))
	}

	scores := make(map[fwksched.Endpoint]float64, len(queues))
	for ep, q := range queues {
		if maxQ == minQ {
			scores[ep] = 1.0
			continue
		}
		scores[ep] = (maxQ - q) / (maxQ - minQ)
	}
	return scores
}
