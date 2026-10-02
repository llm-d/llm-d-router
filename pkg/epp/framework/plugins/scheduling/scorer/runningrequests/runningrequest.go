/*
Copyright 2025 The Kubernetes Authors.
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

package runningrequests

import (
	"context"
	"encoding/json"
	"math"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

const (
	RunningRequestsSizeScorerType = "running-requests-size-scorer"
)

// compile-time type assertion
var (
	_ fwksched.Scorer          = &RunningRequestsSizeScorer{}
	_ fwkplugin.ConsumerPlugin = &RunningRequestsSizeScorer{}
)

// RunningRequestsSizeScorerFactory defines the factory function for RunningRequestsSizeScorer.
func RunningRequestsSizeScorerFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewRunningRequestsSizeScorer().WithName(name), nil
}

// NewRunningRequestsSizeScorer initializes a new RunningRequestsSizeScorer and returns its pointer.
func NewRunningRequestsSizeScorer() *RunningRequestsSizeScorer {
	return &RunningRequestsSizeScorer{
		typedName: fwkplugin.TypedName{Type: RunningRequestsSizeScorerType, Name: RunningRequestsSizeScorerType},
	}
}

// RunningRequestsSizeScorer scores list of candidate pods based on the pod's running request size.
// the less running request size the pod has, the higher score it will get (since it's more available to serve new request).
type RunningRequestsSizeScorer struct {
	typedName fwkplugin.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *RunningRequestsSizeScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *RunningRequestsSizeScorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

// Consumes declares the scorer reads the running requests size from the
// endpoint's Metrics struct, published by the core-metrics-extractor.
func (s *RunningRequestsSizeScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.RunningRequestsSizeKey, metrics.MetricsExtractorType): int(0),
		},
	}
}

// WithName sets the name of the scorer.
func (s *RunningRequestsSizeScorer) WithName(name string) *RunningRequestsSizeScorer {
	s.typedName.Name = name
	return s
}

// Score scores each endpoint by running-request count, min-max normalized
// across endpoints that have written metrics: the fewest running requests
// score 1, the most score 0, and equal counts score a neutral 1. Endpoints
// with no written metrics are left unscored and do not participate in the range.
func (s *RunningRequestsSizeScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	sizes := make(map[fwksched.Endpoint]int, len(endpoints))
	minSize := math.MaxInt
	maxSize := math.MinInt

	for _, endpoint := range endpoints {
		podMetrics := endpoint.GetMetrics()
		if !podMetrics.Updated() {
			continue
		}
		size := podMetrics.RunningRequestsSize
		sizes[endpoint] = size
		if size < minSize {
			minSize = size
		}
		if size > maxSize {
			maxSize = size
		}
	}

	scores := make(map[fwksched.Endpoint]float64, len(sizes))
	for endpoint, size := range sizes {
		if maxSize == minSize {
			scores[endpoint] = 1.0
			continue
		}
		scores[endpoint] = float64(maxSize-size) / float64(maxSize-minSize)
	}
	return scores
}
