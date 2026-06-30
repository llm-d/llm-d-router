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

package metric

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

const (
	MetricScorerType = "metric-scorer"

	normalizationLinear = "linear"

	directionLowerIsBetter  = "lower_is_better"
	directionHigherIsBetter = "higher_is_better"
)

// compile-time type assertion
var _ fwksched.Scorer = &MetricScorer{}

type normalizationParameters struct {
	Type      string `json:"type"`
	Direction string `json:"direction"`
}

type parameters struct {
	AttributeKey  string                  `json:"attributeKey"`
	Normalization normalizationParameters `json:"normalization"`
}

// MetricScorerFactory defines the factory function for MetricScorer.
func MetricScorerFactory(name string, decoder *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params parameters
	if decoder != nil {
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to decode metric scorer parameters: %w", err)
		}
	}

	scorer, err := NewMetricScorer(params)
	if err != nil {
		return nil, err
	}
	return scorer.WithName(name), nil
}

// NewMetricScorer validates the given parameters and returns a new MetricScorer.
func NewMetricScorer(params parameters) (*MetricScorer, error) {
	if params.AttributeKey == "" {
		return nil, errors.New("metric scorer requires a non-empty attributeKey")
	}
	if params.Normalization.Type == "" {
		params.Normalization.Type = normalizationLinear
	}
	if params.Normalization.Type != normalizationLinear {
		return nil, fmt.Errorf("metric scorer normalization.type must be %q, got %q",
			normalizationLinear, params.Normalization.Type)
	}
	switch params.Normalization.Direction {
	case directionLowerIsBetter, directionHigherIsBetter:
	default:
		return nil, fmt.Errorf("metric scorer normalization.direction must be %q or %q, got %q",
			directionLowerIsBetter, directionHigherIsBetter, params.Normalization.Direction)
	}

	return &MetricScorer{
		typedName:     fwkplugin.TypedName{Type: MetricScorerType, Name: MetricScorerType},
		attributeKey:  params.AttributeKey,
		lowerIsBetter: params.Normalization.Direction == directionLowerIsBetter,
	}, nil
}

// MetricScorer scores candidate endpoints by a single configured numeric endpoint
// attribute, linearly normalized across the candidates. The attribute is produced
// by the core metrics extractor from a configured model-server metric.
type MetricScorer struct {
	typedName     fwkplugin.TypedName
	attributeKey  string
	lowerIsBetter bool
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *MetricScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *MetricScorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

// Consumes returns the list of data that is consumed by the plugin.
func (s *MetricScorer) Consumes() map[string]any {
	return map[string]any{
		s.attributeKey: attrmetrics.ScalarMetricValue(0),
	}
}

// WithName sets the name of the scorer.
func (s *MetricScorer) WithName(name string) *MetricScorer {
	s.typedName.Name = name
	return s
}

// Score returns the scoring result for the given list of endpoints based on the
// configured endpoint attribute. Endpoints missing the attribute score 0 and do
// not participate in normalization. If all endpoints that have the attribute
// share the same value, they all receive a neutral score of 1.0.
func (s *MetricScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	values := make(map[fwksched.Endpoint]float64, len(endpoints))
	minValue := math.Inf(1)
	maxValue := math.Inf(-1)

	for _, endpoint := range endpoints {
		value, ok := attrmetrics.ReadScalarMetricValue(endpoint, s.attributeKey)
		if !ok {
			continue
		}
		floatValue := float64(value)
		values[endpoint] = floatValue
		if floatValue < minValue {
			minValue = floatValue
		}
		if floatValue > maxValue {
			maxValue = floatValue
		}
	}

	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		value, ok := values[endpoint]
		if !ok {
			scores[endpoint] = 0.0
			continue
		}
		if maxValue == minValue {
			// All endpoints with the attribute have the same value, return a neutral score.
			scores[endpoint] = 1.0
			continue
		}
		normalized := (value - minValue) / (maxValue - minValue)
		if s.lowerIsBetter {
			normalized = 1.0 - normalized
		}
		scores[endpoint] = normalized
	}
	return scores
}
