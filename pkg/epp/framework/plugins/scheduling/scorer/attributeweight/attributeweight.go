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

// Package attributeweight maps string endpoint attributes to static scores.
package attributeweight

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
)

const EndpointAttributeWeightScorerType = "endpoint-attribute-weight-scorer"

var (
	_ fwksched.Scorer          = (*EndpointAttributeWeightScorer)(nil)
	_ fwkplugin.ConsumerPlugin = (*EndpointAttributeWeightScorer)(nil)
)

type parameters struct {
	AttributeKey string             `json:"attributeKey"`
	Producer     *string            `json:"producer"`
	Weights      map[string]float64 `json:"weights"`
}

func Factory(name string, decoder *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params parameters
	if decoder != nil {
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse %q parameters: %w", EndpointAttributeWeightScorerType, err)
		}
	}
	return NewEndpointAttributeWeightScorer(name, params)
}

func NewEndpointAttributeWeightScorer(name string, params parameters) (*EndpointAttributeWeightScorer, error) {
	if name == "" {
		name = EndpointAttributeWeightScorerType
	}
	if params.AttributeKey == "" {
		return nil, fmt.Errorf("%q requires a non-empty 'attributeKey'", EndpointAttributeWeightScorerType)
	}
	if params.Producer == nil {
		return nil, fmt.Errorf("%q requires 'producer'", EndpointAttributeWeightScorerType)
	}
	if len(params.Weights) == 0 {
		return nil, fmt.Errorf("%q requires a non-empty 'weights' map", EndpointAttributeWeightScorerType)
	}

	maxWeight := 0.0
	for value, weight := range params.Weights {
		if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return nil, fmt.Errorf("%q weight for value %q must be finite and positive, got %v",
				EndpointAttributeWeightScorerType, value, weight)
		}
		maxWeight = max(maxWeight, weight)
	}

	scores := make(map[string]float64, len(params.Weights))
	fallbackScore := 1.0
	for value, weight := range params.Weights {
		score := weight / maxWeight
		if score == 0 {
			return nil, fmt.Errorf("%q weight for value %q underflows to zero when normalized against max weight %v",
				EndpointAttributeWeightScorerType, value, maxWeight)
		}
		scores[value] = score
		fallbackScore = min(fallbackScore, score)
	}

	return &EndpointAttributeWeightScorer{
		typedName:     fwkplugin.TypedName{Type: EndpointAttributeWeightScorerType, Name: name},
		dataKey:       fwkplugin.NewDataKey(params.AttributeKey, *params.Producer),
		scores:        scores,
		fallbackScore: fallbackScore,
	}, nil
}

type EndpointAttributeWeightScorer struct {
	typedName     fwkplugin.TypedName
	dataKey       fwkplugin.DataKey
	scores        map[string]float64
	fallbackScore float64
}

func (s *EndpointAttributeWeightScorer) TypedName() fwkplugin.TypedName { return s.typedName }

func (s *EndpointAttributeWeightScorer) Category() fwksched.ScorerCategory { return fwksched.Affinity }

// Static preferences can fall back when their producer is absent.
func (s *EndpointAttributeWeightScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{s.dataKey: attrstring.Value("")},
	}
}

func (s *EndpointAttributeWeightScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		score := s.fallbackScore
		if value, ok := attrstring.ReadValue(endpoint, s.dataKey); ok {
			if configured, ok := s.scores[string(value)]; ok {
				score = configured
			}
		}
		scores[endpoint] = score
	}
	return scores
}
