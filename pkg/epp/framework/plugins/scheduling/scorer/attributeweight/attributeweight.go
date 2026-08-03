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

package attributeweight

import (
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
)

// EndpointAttributeWeightScorerType is the type of the endpoint attribute weight scorer plugin.
const EndpointAttributeWeightScorerType = "endpoint-attribute-weight-scorer"

var _ fwksched.Scorer = &EndpointAttributeWeightScorer{}

// parameters configures the endpoint attribute weight scorer.
type parameters struct {
	// AttributeKey identifies the string endpoint attribute whose value selects a score.
	AttributeKey string `json:"attributeKey"`
	// Weights maps attribute values to positive weights.
	Weights map[string]float64 `json:"weights"`
}

func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", EndpointAttributeWeightScorerType, err)
		}
	}
	return NewEndpointAttributeWeightScorer(name, params)
}

func NewEndpointAttributeWeightScorer(name string, params parameters) (*EndpointAttributeWeightScorer, error) {
	if name == "" {
		name = EndpointAttributeWeightScorerType
	}
	if params.AttributeKey == "" {
		return nil, fmt.Errorf("'%s' requires a non-empty 'attributeKey'", EndpointAttributeWeightScorerType)
	}
	if len(params.Weights) == 0 {
		return nil, fmt.Errorf("'%s' requires a non-empty 'weights' map", EndpointAttributeWeightScorerType)
	}

	maxWeight := 0.0
	for value, weight := range params.Weights {
		if weight <= 0 {
			return nil, fmt.Errorf("'%s' weight for value %q must be positive, got %v", EndpointAttributeWeightScorerType, value, weight)
		}
		if weight > maxWeight {
			maxWeight = weight
		}
	}

	scores := make(map[string]float64, len(params.Weights))
	fallbackScore := 1.0
	for value, weight := range params.Weights {
		normalized := weight / maxWeight
		scores[value] = normalized
		if normalized < fallbackScore {
			fallbackScore = normalized
		}
	}

	return &EndpointAttributeWeightScorer{
		typedName:     fwkplugin.TypedName{Type: EndpointAttributeWeightScorerType, Name: name},
		attributeKey:  params.AttributeKey,
		scores:        scores,
		fallbackScore: fallbackScore,
	}, nil
}

// EndpointAttributeWeightScorer maps string endpoint attribute values to normalized scores.
type EndpointAttributeWeightScorer struct {
	// typedName identifies this plugin instance.
	typedName fwkplugin.TypedName
	// attributeKey is the endpoint attribute to score.
	attributeKey string
	// scores contains normalized scores for configured attribute values.
	scores map[string]float64
	// fallbackScore is used for missing or unknown attribute values.
	fallbackScore float64
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *EndpointAttributeWeightScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the scorer category.
func (s *EndpointAttributeWeightScorer) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// Score returns scores for the configured attribute values.
func (s *EndpointAttributeWeightScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		score := s.fallbackScore
		if value, ok := attrstring.ReadValue(endpoint, s.attributeKey); ok {
			if configuredScore, ok := s.scores[string(value)]; ok {
				score = configuredScore
			}
		}
		scores[endpoint] = score
	}
	return scores
}
