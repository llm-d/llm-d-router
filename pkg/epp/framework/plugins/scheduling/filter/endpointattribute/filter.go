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

package endpointattribute

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

const (
	// EndpointAttributeFilterType is the type of the EndpointAttributeFilter.
	EndpointAttributeFilterType = "endpoint-attribute-filter"

	onMissingPass = "Pass"
	onMissingFail = "Fail"

	algorithmThreshold  = "threshold"
	algorithmRange      = "range"
	algorithmTopK       = "topK"
	algorithmPercentile = "percentile"

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

// rangeParameters keeps endpoints whose attribute value is within the
// inclusive [Min, Max] range.
type rangeParameters struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

// topKParameters keeps the K endpoints with the highest (or lowest) attribute
// value. When K exceeds the number of endpoints with the attribute, all of
// them are kept.
type topKParameters struct {
	K              int  `json:"k"`
	HigherIsBetter bool `json:"higherIsBetter"` // true keeps top K highest, false keeps bottom K lowest
}

// percentileParameters keeps endpoints whose attribute value is at or above
// (or at or below) the configured percentile of the values observed across
// the candidate endpoints. Percentile is specified as an integer in [0, 100].
type percentileParameters struct {
	Percentile     float64 `json:"percentile"`
	HigherIsBetter bool    `json:"higherIsBetter"` // true keeps values >= percentile, false keeps values <= percentile
}

// algorithmParameters selects the filtering algorithm.
type algorithmParameters struct {
	Type       string                `json:"type"`
	Threshold  *thresholdParameters  `json:"threshold,omitempty"`
	Range      *rangeParameters      `json:"range,omitempty"`
	TopK       *topKParameters       `json:"topK,omitempty"`
	Percentile *percentileParameters `json:"percentile,omitempty"`
}

type parameters struct {
	Attribute string `json:"attribute"`
	// Producer names the plugin publishing the attribute. Omitted selects the
	// core metrics extractor, the only producer whose attribute names come
	// from configuration; set it to read another producer's attribute, e.g.
	// "dcgm-extractor". Set it to the empty string for a producer-agnostic
	// attribute, such as "llm-d.ai/multicluster-queue-size".
	//
	// A pointer distinguishes an omitted producer from one set to the empty
	// string, which is what lets an attribute name containing "/" be told
	// apart from the combined "Attribute/Producer" spelling.
	Producer *string `json:"producer"`
	// OnMissing decides what happens to endpoints that do not have the
	// attribute: "Pass" keeps them (the default), "Fail" drops them.
	OnMissing string `json:"onMissing"`
	// FallbackOnEmpty returns the unfiltered candidates when every endpoint
	// was filtered out, so the request can still be routed somewhere.
	FallbackOnEmpty bool                `json:"fallbackOnEmpty"`
	Algorithm       algorithmParameters `json:"algorithm"`
}

// compile-time type assertion
var (
	_ scheduling.Filter     = &EndpointAttributeFilter{}
	_ plugin.ConsumerPlugin = &EndpointAttributeFilter{}
)

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
	switch params.Algorithm.Type {
	case algorithmThreshold:
		if err := validateThreshold(params.Algorithm.Threshold); err != nil {
			return nil, err
		}
	case algorithmRange:
		if err := validateRange(params.Algorithm.Range); err != nil {
			return nil, err
		}
	case algorithmTopK:
		if err := validateTopK(params.Algorithm.TopK); err != nil {
			return nil, err
		}
	case algorithmPercentile:
		if err := validatePercentile(params.Algorithm.Percentile); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("endpoint attribute filter algorithm.type must be one of %q, %q, %q, %q, got %q",
			algorithmThreshold, algorithmRange, algorithmTopK, algorithmPercentile, params.Algorithm.Type)
	}

	dataKey, err := attrmetrics.ResolveConfiguredKey("endpoint attribute filter", params.Attribute, params.Producer)
	if err != nil {
		return nil, err
	}

	return &EndpointAttributeFilter{
		typedName:       plugin.TypedName{Type: EndpointAttributeFilterType, Name: name},
		dataKey:         dataKey,
		passOnMissing:   params.OnMissing == onMissingPass,
		fallbackOnEmpty: params.FallbackOnEmpty,
		algorithm:       params.Algorithm,
		threshold:       derefThreshold(params.Algorithm.Threshold),
	}, nil
}

func validateThreshold(t *thresholdParameters) error {
	if t == nil {
		return fmt.Errorf("endpoint attribute filter requires algorithm.threshold when algorithm.type is %q",
			algorithmThreshold)
	}
	switch t.Operator {
	case operatorLessThan, operatorLessThanOrEqual, operatorGreaterThan,
		operatorGreaterThanOrEqual, operatorEqual, operatorNotEqual:
	default:
		return fmt.Errorf("endpoint attribute filter threshold.operator must be one of %q, %q, %q, %q, %q, %q, got %q",
			operatorLessThan, operatorLessThanOrEqual, operatorGreaterThan,
			operatorGreaterThanOrEqual, operatorEqual, operatorNotEqual, t.Operator)
	}
	return nil
}

func validateRange(r *rangeParameters) error {
	if r == nil {
		return fmt.Errorf("endpoint attribute filter requires algorithm.range when algorithm.type is %q",
			algorithmRange)
	}
	if r.Min > r.Max {
		return fmt.Errorf("endpoint attribute filter range requires min <= max, got min %v, max %v",
			r.Min, r.Max)
	}
	return nil
}

func validateTopK(t *topKParameters) error {
	if t == nil {
		return fmt.Errorf("endpoint attribute filter requires algorithm.topK when algorithm.type is %q",
			algorithmTopK)
	}
	if t.K <= 0 {
		return fmt.Errorf("endpoint attribute filter topK.k must be a positive integer, got %d",
			t.K)
	}
	return nil
}

func validatePercentile(p *percentileParameters) error {
	if p == nil {
		return fmt.Errorf("endpoint attribute filter requires algorithm.percentile when algorithm.type is %q",
			algorithmPercentile)
	}
	if p.Percentile < 0 || p.Percentile > 100 {
		return fmt.Errorf("endpoint attribute filter percentile.percentile must be in [0, 100], got %v",
			p.Percentile)
	}
	return nil
}

func derefThreshold(t *thresholdParameters) thresholdParameters {
	if t == nil {
		return thresholdParameters{}
	}
	return *t
}

// EndpointAttributeFilter filters candidate endpoints by a single configured
// numeric endpoint attribute (produced by the custom metrics extraction
// layer). The filtering algorithm is selected by Algorithm.Type: threshold
// compares each endpoint against a fixed value, range keeps endpoints within
// [min, max], topK keeps the K highest or lowest values, and percentile keeps
// endpoints at or above (or at or below) the configured percentile of the
// candidate values.
type EndpointAttributeFilter struct {
	typedName plugin.TypedName
	dataKey   plugin.DataKey
	// passOnMissing keeps endpoints that do not have the attribute instead of
	// dropping them.
	passOnMissing bool
	// fallbackOnEmpty returns the unfiltered candidates when every endpoint
	// is filtered out.
	fallbackOnEmpty bool
	algorithm       algorithmParameters
	threshold       thresholdParameters // resolved from algorithm.Threshold for the threshold algorithm
}

// TypedName returns the typed name of the plugin.
func (f *EndpointAttributeFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Consumes declares the configured attribute as optional: the producer is
// selected in configuration, so a missing one is handled by the onMissing
// policy rather than rejected at init time.
func (f *EndpointAttributeFilter) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Optional: map[plugin.DataKey]any{
			f.dataKey: attrmetrics.ScalarMetricValue(0),
		},
	}
}

// Filter keeps the endpoints whose attribute value satisfies the configured
// algorithm. Endpoints missing the attribute are kept or dropped according
// to the onMissing policy. When all endpoints are filtered out and
// fallbackOnEmpty is set, the original candidates are returned.
func (f *EndpointAttributeFilter) Filter(_ context.Context, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	var filtered []scheduling.Endpoint
	switch f.algorithm.Type {
	case algorithmThreshold:
		filtered = f.filterThreshold(endpoints)
	case algorithmRange:
		filtered = f.filterRange(endpoints)
	case algorithmTopK:
		filtered = f.filterTopK(endpoints)
	case algorithmPercentile:
		filtered = f.filterPercentile(endpoints)
	default:
		// Unreachable: the algorithm type is validated at construction time.
		filtered = endpoints
	}

	if len(filtered) == 0 && f.fallbackOnEmpty {
		return endpoints
	}
	return filtered
}

// filterThreshold keeps endpoints whose attribute value satisfies the
// configured threshold comparison. Endpoints missing the attribute are kept
// or dropped according to the onMissing policy.
func (f *EndpointAttributeFilter) filterThreshold(endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filtered := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		value, ok := attrmetrics.ReadScalarMetricValue(endpoint, f.dataKey)
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
	return filtered
}

// filterRange keeps endpoints whose attribute value is within the inclusive
// [Min, Max] range. Endpoints missing the attribute are kept or dropped
// according to the onMissing policy.
func (f *EndpointAttributeFilter) filterRange(endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	r := f.algorithm.Range
	filtered := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		value, ok := attrmetrics.ReadScalarMetricValue(endpoint, f.dataKey)
		if !ok {
			if f.passOnMissing {
				filtered = append(filtered, endpoint)
			}
			continue
		}
		v := float64(value)
		if v >= r.Min && v <= r.Max {
			filtered = append(filtered, endpoint)
		}
	}
	return filtered
}

// filterTopK keeps the K endpoints with the highest (or lowest) attribute
// value. Endpoints missing the attribute are excluded from ranking but kept
// or dropped according to the onMissing policy after the K selection is
// applied.
func (f *EndpointAttributeFilter) filterTopK(endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	tk := f.algorithm.TopK

	var ranked []epVal
	var missing []scheduling.Endpoint
	for _, endpoint := range endpoints {
		value, ok := attrmetrics.ReadScalarMetricValue(endpoint, f.dataKey)
		if !ok {
			missing = append(missing, endpoint)
			continue
		}
		ranked = append(ranked, epVal{ep: endpoint, value: float64(value)})
	}

	if len(ranked) <= tk.K {
		// All ranked endpoints survive; apply onMissing to the rest.
		return f.appendMissing(append([]scheduling.Endpoint{}, rankedToEndpoints(ranked)...), missing)
	}

	sort.Slice(ranked, func(i, j int) bool {
		if tk.HigherIsBetter {
			return ranked[i].value > ranked[j].value // descending: highest first
		}
		return ranked[i].value < ranked[j].value // ascending: lowest first
	})

	kept := make([]scheduling.Endpoint, 0, tk.K+len(missing))
	for i := 0; i < tk.K; i++ {
		kept = append(kept, ranked[i].ep)
	}
	return f.appendMissing(kept, missing)
}

// filterPercentile keeps endpoints whose attribute value is at or above (or
// at or below) the configured percentile of the values observed across the
// candidate endpoints. Endpoints missing the attribute are kept or dropped
// according to the onMissing policy.
func (f *EndpointAttributeFilter) filterPercentile(endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	p := f.algorithm.Percentile

	var valued []epVal
	var missing []scheduling.Endpoint
	for _, endpoint := range endpoints {
		value, ok := attrmetrics.ReadScalarMetricValue(endpoint, f.dataKey)
		if !ok {
			missing = append(missing, endpoint)
			continue
		}
		valued = append(valued, epVal{ep: endpoint, value: float64(value)})
	}

	if len(valued) == 0 {
		return f.appendMissing(nil, missing)
	}

	sort.Slice(valued, func(i, j int) bool {
		return valued[i].value < valued[j].value // ascending
	})

	// Nearest-rank percentile: the value at the ceil(P/100 * N)th position
	// (1-based) in the ascending sorted array.
	rank := int((p.Percentile/100.0)*float64(len(valued)) + 0.999999)
	if rank < 1 {
		rank = 1
	}
	if rank > len(valued) {
		rank = len(valued)
	}
	cutoff := valued[rank-1].value

	kept := make([]scheduling.Endpoint, 0, len(valued)+len(missing))
	for _, ev := range valued {
		if p.HigherIsBetter {
			if ev.value >= cutoff {
				kept = append(kept, ev.ep)
			}
		} else {
			if ev.value <= cutoff {
				kept = append(kept, ev.ep)
			}
		}
	}
	return f.appendMissing(kept, missing)
}

// appendMissing applies the onMissing policy to endpoints that lack the
// attribute, appending them to kept when passOnMissing is true.
func (f *EndpointAttributeFilter) appendMissing(kept, missing []scheduling.Endpoint) []scheduling.Endpoint {
	if f.passOnMissing {
		kept = append(kept, missing...)
	}
	return kept
}

func rankedToEndpoints(ranked []epVal) []scheduling.Endpoint {
	endpoints := make([]scheduling.Endpoint, len(ranked))
	for i, ev := range ranked {
		endpoints[i] = ev.ep
	}
	return endpoints
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

// epVal is a transport struct used by ranking-based algorithms (topK,
// percentile) to pair endpoints with their attribute values for sorting.
type epVal struct {
	ep    scheduling.Endpoint
	value float64
}
