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

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

const (
	// EndpointAttributeFilterType is the type of the EndpointAttributeFilter.
	EndpointAttributeFilterType = "endpoint-attribute-filter"

	onMissingPass = "Pass"
	onMissingFail = "Fail"

	algorithmThreshold = "threshold"

	operatorLessThan           = "LessThan"
	operatorLessThanOrEqual    = "LessThanOrEqual"
	operatorGreaterThan        = "GreaterThan"
	operatorGreaterThanOrEqual = "GreaterThanOrEqual"
	operatorEqual              = "Equal"
	operatorNotEqual           = "NotEqual"
)

// thresholdParameters keeps endpoints whose attribute value compares true
// against the configured value.
type thresholdParameters struct {
	Operator string  `json:"operator"`
	Value    float64 `json:"value"`
}

// algorithmParameters selects the filtering algorithm. threshold is the only
// algorithm currently supported; percentile, topK and range are planned as
// follow-ups.
type algorithmParameters struct {
	Type      string               `json:"type"`
	Threshold *thresholdParameters `json:"threshold,omitempty"`
}

type parameters struct {
	Attribute string `json:"attribute"`
	// OnMissing decides what happens to endpoints that do not have the
	// attribute: "Pass" keeps them (the default), "Fail" drops them.
	OnMissing string `json:"onMissing"`
	// FallbackOnEmpty returns the unfiltered candidates when every endpoint
	// was filtered out, so the request can still be routed somewhere.
	FallbackOnEmpty bool                `json:"fallbackOnEmpty"`
	Algorithm       algorithmParameters `json:"algorithm"`
}

// compile-time type assertion
var _ scheduling.Filter = &EndpointAttributeFilter{}

// EndpointAttributeFilterFactory defines the factory function for EndpointAttributeFilter.
func EndpointAttributeFilterFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var params parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", EndpointAttributeFilterType, err)
		}
	}

	return NewEndpointAttributeFilter(name, params)
}

// NewEndpointAttributeFilter validates the given parameters and returns a new
// EndpointAttributeFilter with the given name.
func NewEndpointAttributeFilter(name string, params parameters) (*EndpointAttributeFilter, error) {
	if name == "" {
		name = EndpointAttributeFilterType
	}
	if params.Attribute == "" {
		return nil, errors.New("endpoint attribute filter requires a non-empty attribute")
	}
	switch params.OnMissing {
	case "":
		params.OnMissing = onMissingPass
	case onMissingPass, onMissingFail:
	default:
		return nil, fmt.Errorf("endpoint attribute filter onMissing must be %q or %q, got %q",
			onMissingPass, onMissingFail, params.OnMissing)
	}
	if params.Algorithm.Type != algorithmThreshold {
		return nil, fmt.Errorf("endpoint attribute filter algorithm.type must be %q, got %q",
			algorithmThreshold, params.Algorithm.Type)
	}
	threshold := params.Algorithm.Threshold
	if threshold == nil {
		return nil, fmt.Errorf("endpoint attribute filter requires algorithm.threshold when algorithm.type is %q",
			algorithmThreshold)
	}
	switch threshold.Operator {
	case operatorLessThan, operatorLessThanOrEqual, operatorGreaterThan,
		operatorGreaterThanOrEqual, operatorEqual, operatorNotEqual:
	default:
		return nil, fmt.Errorf("endpoint attribute filter threshold.operator must be one of %q, %q, %q, %q, %q, %q, got %q",
			operatorLessThan, operatorLessThanOrEqual, operatorGreaterThan,
			operatorGreaterThanOrEqual, operatorEqual, operatorNotEqual, threshold.Operator)
	}

	return &EndpointAttributeFilter{
		typedName:       plugin.TypedName{Type: EndpointAttributeFilterType, Name: name},
		attribute:       params.Attribute,
		passOnMissing:   params.OnMissing == onMissingPass,
		fallbackOnEmpty: params.FallbackOnEmpty,
		threshold:       *threshold,
	}, nil
}

// EndpointAttributeFilter filters candidate endpoints by a single configured
// numeric endpoint attribute (produced by the custom metrics extraction
// layer). Endpoints whose attribute value compares true against the
// configured threshold are kept.
type EndpointAttributeFilter struct {
	typedName plugin.TypedName
	attribute string
	// passOnMissing keeps endpoints that do not have the attribute instead of
	// dropping them.
	passOnMissing bool
	// fallbackOnEmpty returns the unfiltered candidates when every endpoint
	// was filtered out.
	fallbackOnEmpty bool
	threshold       thresholdParameters
}

// TypedName returns the typed name of the plugin.
func (f *EndpointAttributeFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Consumes returns the list of data that is consumed by the plugin.
func (f *EndpointAttributeFilter) Consumes() map[string]any {
	return map[string]any{
		f.attribute: attrmetrics.ScalarMetricValue(0),
	}
}

// Filter keeps the endpoints whose attribute value satisfies the configured
// threshold. Endpoints missing the attribute are kept or dropped according to
// the onMissing policy. When all endpoints are filtered out and
// fallbackOnEmpty is set, the original candidates are returned.
func (f *EndpointAttributeFilter) Filter(_ context.Context, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filtered := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		value, ok := attrmetrics.ReadScalarMetricValue(endpoint, f.attribute)
		if !ok {
			if f.passOnMissing {
				filtered = append(filtered, endpoint)
			}
			continue
		}
		if f.matches(float64(value)) {
			filtered = append(filtered, endpoint)
		}
	}

	if len(filtered) == 0 && f.fallbackOnEmpty {
		return endpoints
	}
	return filtered
}

// matches reports whether the value satisfies the configured threshold.
func (f *EndpointAttributeFilter) matches(value float64) bool {
	switch f.threshold.Operator {
	case operatorLessThan:
		return value < f.threshold.Value
	case operatorLessThanOrEqual:
		return value <= f.threshold.Value
	case operatorGreaterThan:
		return value > f.threshold.Value
	case operatorGreaterThanOrEqual:
		return value >= f.threshold.Value
	case operatorEqual:
		return value == f.threshold.Value
	case operatorNotEqual:
		return value != f.threshold.Value
	default:
		// Unreachable: the operator is validated at construction time.
		return false
	}
}
