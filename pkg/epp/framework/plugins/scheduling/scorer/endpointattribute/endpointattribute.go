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

package endpointattribute

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
	EndpointAttributeScorerType = "endpoint-attribute-scorer"

	algorithmLinearLowerIsBetter  = "linear_lower_is_better"
	algorithmLinearHigherIsBetter = "linear_higher_is_better"

	sourceEndpoint = "endpoint"
	sourceMetadata = "metadata"
)

// compile-time type assertion
var _ fwksched.Scorer = &EndpointAttributeScorer{}

// fixedRangeParameters normalizes the attribute value against a fixed
// [min, max] range (e.g. kv-cache utilization, which is always in [0, 1]).
type fixedRangeParameters struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

// adaptiveRangeParameters normalizes the attribute value against an adaptive
// [min, max] range computed across the candidate endpoints (e.g. queue depth).
type adaptiveRangeParameters struct{}

// normalizationParameters selects the normalization strategy. At most one
// strategy may be set; adaptiveRange is the default when none is set.
type normalizationParameters struct {
	FixedRange    *fixedRangeParameters    `json:"fixedRange,omitempty"`
	AdaptiveRange *adaptiveRangeParameters `json:"adaptiveRange,omitempty"`
}

type algorithmParameters struct {
	Type          string                  `json:"type"`
	Normalization normalizationParameters `json:"normalization"`
}

type parameters struct {
	AttributeKey      string              `json:"attributeKey"`
	Algorithm         algorithmParameters `json:"algorithm"`
	Source            string              `json:"source"`
	MetadataNamespace string              `json:"metadataNamespace"`
	MetadataField     string              `json:"metadataField"`
}

// EndpointAttributeScorerFactory defines the factory function for EndpointAttributeScorer.
func EndpointAttributeScorerFactory(name string, decoder *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params parameters
	if decoder != nil {
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to decode endpoint attribute scorer parameters: %w", err)
		}
	}

	return NewEndpointAttributeScorer(name, params)
}

// NewEndpointAttributeScorer validates the given parameters and returns a new
// EndpointAttributeScorer with the given name.
func NewEndpointAttributeScorer(name string, params parameters) (*EndpointAttributeScorer, error) {
	if name == "" {
		name = EndpointAttributeScorerType
	}

	source := params.Source
	if source == "" {
		source = sourceEndpoint
	}
	switch source {
	case sourceEndpoint:
		if params.AttributeKey == "" {
			return nil, errors.New("endpoint attribute scorer requires a non-empty attributeKey")
		}
	case sourceMetadata:
		if params.MetadataNamespace == "" || params.MetadataField == "" {
			return nil, errors.New("endpoint attribute scorer with source: metadata requires metadataNamespace and metadataField")
		}
	default:
		return nil, fmt.Errorf("endpoint attribute scorer source must be %q or %q, got %q", sourceEndpoint, sourceMetadata, source)
	}

	switch params.Algorithm.Type {
	case algorithmLinearLowerIsBetter, algorithmLinearHigherIsBetter:
	default:
		return nil, fmt.Errorf("endpoint attribute scorer algorithm.type must be %q or %q, got %q",
			algorithmLinearLowerIsBetter, algorithmLinearHigherIsBetter, params.Algorithm.Type)
	}

	normalization := params.Algorithm.Normalization
	if normalization.FixedRange != nil && normalization.AdaptiveRange != nil {
		return nil, errors.New("endpoint attribute scorer allows at most one normalization strategy, got both fixedRange and adaptiveRange")
	}
	if fixed := normalization.FixedRange; fixed != nil && fixed.Min >= fixed.Max {
		return nil, fmt.Errorf("endpoint attribute scorer normalization.fixedRange requires min < max, got min %v, max %v",
			fixed.Min, fixed.Max)
	}

	return &EndpointAttributeScorer{
		typedName:         fwkplugin.TypedName{Type: EndpointAttributeScorerType, Name: name},
		attributeKey:      params.AttributeKey,
		lowerIsBetter:     params.Algorithm.Type == algorithmLinearLowerIsBetter,
		fixedRange:        normalization.FixedRange,
		source:            source,
		metadataNamespace: params.MetadataNamespace,
		metadataField:     params.MetadataField,
	}, nil
}

// EndpointAttributeScorer scores candidate endpoints by a single numeric value, linearly normalized against
// a fixed [min, max] range or an adaptive range across the candidates. The value is a scraped endpoint
// attribute (source: endpoint) or a per-endpoint map in request metadata (source: metadata).
type EndpointAttributeScorer struct {
	typedName     fwkplugin.TypedName
	attributeKey  string
	lowerIsBetter bool
	// fixedRange selects fixed-range normalization when set; adaptive-range
	// normalization is used otherwise.
	fixedRange *fixedRangeParameters

	source            string
	metadataNamespace string
	metadataField     string
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *EndpointAttributeScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *EndpointAttributeScorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

// Consumes declares the endpoint attribute this scorer reads.
func (s *EndpointAttributeScorer) Consumes() map[string]any {
	if s.source == sourceMetadata {
		// The value comes from request metadata, so there is no endpoint-attribute dependency.
		return map[string]any{}
	}
	return map[string]any{
		s.attributeKey: attrmetrics.ScalarMetricValue(0),
	}
}

// Score returns the scoring result for the given list of endpoints. Endpoints missing the value score 0.
func (s *EndpointAttributeScorer) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	valueOf := s.valueReader(request)
	if s.fixedRange != nil {
		return s.scoreFixedRange(endpoints, valueOf)
	}
	return s.scoreAdaptiveRange(endpoints, valueOf)
}

// valueReader returns the per-endpoint value accessor for the configured source.
func (s *EndpointAttributeScorer) valueReader(request *fwksched.InferenceRequest) func(fwksched.Endpoint) (float64, bool) {
	if s.source == sourceMetadata {
		scores := endpointScoresFromMetadata(request, s.metadataNamespace, s.metadataField)
		return func(ep fwksched.Endpoint) (float64, bool) {
			v, ok := scores[ep.GetMetadata().GetNamespacedName().String()]
			return v, ok
		}
	}
	return func(ep fwksched.Endpoint) (float64, bool) {
		v, ok := attrmetrics.ReadScalarMetricValue(ep, s.attributeKey)
		return float64(v), ok
	}
}

// endpointScoresFromMetadata reads request.Metadata[namespace][field] as an endpoint-identity to score map.
func endpointScoresFromMetadata(request *fwksched.InferenceRequest, namespace, field string) map[string]float64 {
	scores := map[string]float64{}
	if request == nil || request.Metadata == nil {
		return scores
	}
	ns, ok := request.Metadata[namespace].(map[string]any)
	if !ok {
		return scores
	}
	raw, ok := ns[field].(map[string]any)
	if !ok {
		return scores
	}
	for k, v := range raw {
		if f, ok := v.(float64); ok {
			scores[k] = f
		}
	}
	return scores
}

// scoreFixedRange normalizes each endpoint's value against the configured [min, max] range, clamping
// values outside the range.
func (s *EndpointAttributeScorer) scoreFixedRange(endpoints []fwksched.Endpoint, valueOf func(fwksched.Endpoint) (float64, bool)) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		value, ok := valueOf(endpoint)
		if !ok {
			scores[endpoint] = 0.0
			continue
		}
		normalized := (value - s.fixedRange.Min) / (s.fixedRange.Max - s.fixedRange.Min)
		normalized = math.Max(0.0, math.Min(1.0, normalized))
		if s.lowerIsBetter {
			normalized = 1.0 - normalized
		}
		scores[endpoint] = normalized
	}
	return scores
}

// scoreAdaptiveRange normalizes each endpoint's value against the [min, max] range observed across the
// candidates. Endpoints missing the value do not participate in the range. If all endpoints that have the
// value share the same value, they all receive a neutral score of 1.0.
func (s *EndpointAttributeScorer) scoreAdaptiveRange(endpoints []fwksched.Endpoint, valueOf func(fwksched.Endpoint) (float64, bool)) map[fwksched.Endpoint]float64 {
	values := make(map[fwksched.Endpoint]float64, len(endpoints))
	minValue := math.Inf(1)
	maxValue := math.Inf(-1)

	for _, endpoint := range endpoints {
		value, ok := valueOf(endpoint)
		if !ok {
			continue
		}
		values[endpoint] = value
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
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
			// All endpoints with the value share the same value, return a neutral score.
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
