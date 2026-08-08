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

// Package plugins encapsulates an out-of-tree custom scorer plugin implementation.
package plugins

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// Type is the plugin type name used in EndpointPicker configuration.
	Type = "fixed-scorer"
)

var (
	_ scheduling.Scorer     = &FixedScorer{}
	_ plugin.ConsumerPlugin = &FixedScorer{}
)

// Config holds optional parameters for FixedScorer.
type Config struct {
	Score float64 `json:"score"`
}

// FixedScorer assigns a configured fixed score to all candidate endpoints.
type FixedScorer struct {
	typedName plugin.TypedName
	score     float64
}

// Factory instantiates a FixedScorer from configuration parameters.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	cfg := Config{Score: 1.0}
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for %q scorer: %w", Type, err)
		}
	}
	if cfg.Score < 0.0 || cfg.Score > 1.0 {
		return nil, fmt.Errorf("score for %q must be between 0.0 and 1.0, got %f", Type, cfg.Score)
	}
	return New(name, cfg.Score), nil
}

// New creates a FixedScorer.
func New(name string, score float64) *FixedScorer {
	return &FixedScorer{
		typedName: plugin.TypedName{Type: Type, Name: name},
		score:     score,
	}
}

// TypedName returns the plugin type and name.
func (s *FixedScorer) TypedName() plugin.TypedName {
	return s.typedName
}

// Category returns the scorer category.
func (s *FixedScorer) Category() scheduling.ScorerCategory {
	return scheduling.Distribution
}

// Consumes returns data dependencies. FixedScorer requires no endpoint attributes.
func (s *FixedScorer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{}
}

// Score assigns the configured score to all candidate endpoints.
func (s *FixedScorer) Score(ctx context.Context, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	traceLogger := log.FromContext(ctx).V(logging.TRACE)
	scores := make(map[scheduling.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		scores[ep] = s.score
		if traceLogger.Enabled() {
			traceLogger.Info("assigning fixed score", "endpoint", ep, "score", s.score)
		}
	}
	return scores
}
