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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

const testAttribute = "num_requests_running"

func TestEndpointAttributeFilterFactory(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{
			name: "valid threshold with defaulted onMissing",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "threshold",
					"threshold": {"operator": "LessThan", "value": 10}}}`,
		},
		{
			name: "valid threshold with explicit policies",
			parameters: `{"attribute": "num_requests_running",
				"onMissing": "Fail", "fallbackOnEmpty": true,
				"algorithm": {"type": "threshold",
					"threshold": {"operator": "GreaterThanOrEqual", "value": 0.5}}}`,
		},
		{
			name: "missing attribute",
			parameters: `{"algorithm": {"type": "threshold",
				"threshold": {"operator": "LessThan", "value": 10}}}`,
			wantErr: "attribute",
		},
		{
			name: "invalid onMissing",
			parameters: `{"attribute": "num_requests_running", "onMissing": "Maybe",
				"algorithm": {"type": "threshold",
					"threshold": {"operator": "LessThan", "value": 10}}}`,
			wantErr: "onMissing",
		},
		{
			name:       "missing algorithm type",
			parameters: `{"attribute": "num_requests_running"}`,
			wantErr:    "algorithm.type",
		},
		{
			name: "unsupported algorithm type",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "percentile"}}`,
			wantErr: "algorithm.type",
		},
		{
			name: "missing threshold block",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "threshold"}}`,
			wantErr: "algorithm.threshold",
		},
		{
			name: "invalid threshold operator",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "threshold",
					"threshold": {"operator": "Around", "value": 10}}}`,
			wantErr: "threshold.operator",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := json.NewDecoder(strings.NewReader(test.parameters))
			plugin, err := EndpointAttributeFilterFactory("test-filter", decoder, nil)
			if test.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), test.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, EndpointAttributeFilterType, plugin.TypedName().Type)
			assert.Equal(t, "test-filter", plugin.TypedName().Name)
		})
	}
}

func newEndpointWithValue(value float64) scheduling.Endpoint {
	attrs := fwkdl.NewAttributes()
	attrs.Put(testAttribute, attrmetrics.ScalarMetricValue(value))
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, attrs)
}

func newEndpointWithoutValue() scheduling.Endpoint {
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, nil)
}

func TestEndpointAttributeFilterFilter(t *testing.T) {
	tests := []struct {
		name            string
		operator        string
		value           float64
		onMissing       string
		fallbackOnEmpty bool
		endpoints       []scheduling.Endpoint
		wantKept        []int // indexes into endpoints expected to survive
	}{
		{
			name:     "LessThan keeps values below the threshold",
			operator: operatorLessThan,
			value:    10,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(10),
				newEndpointWithValue(15),
			},
			wantKept: []int{0},
		},
		{
			name:     "LessThanOrEqual keeps the boundary value",
			operator: operatorLessThanOrEqual,
			value:    10,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(10),
				newEndpointWithValue(15),
			},
			wantKept: []int{0, 1},
		},
		{
			name:     "GreaterThan keeps values above the threshold",
			operator: operatorGreaterThan,
			value:    10,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(10),
				newEndpointWithValue(15),
			},
			wantKept: []int{2},
		},
		{
			name:     "GreaterThanOrEqual keeps the boundary value",
			operator: operatorGreaterThanOrEqual,
			value:    10,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(10),
				newEndpointWithValue(15),
			},
			wantKept: []int{1, 2},
		},
		{
			name:     "Equal keeps only the matching value",
			operator: operatorEqual,
			value:    10,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(10),
			},
			wantKept: []int{1},
		},
		{
			name:     "NotEqual drops the matching value",
			operator: operatorNotEqual,
			value:    10,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(10),
			},
			wantKept: []int{0},
		},
		{
			name:      "missing attribute passes by default",
			operator:  operatorLessThan,
			value:     10,
			onMissing: onMissingPass,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(15),
				newEndpointWithoutValue(),
			},
			wantKept: []int{1},
		},
		{
			name:      "missing attribute fails when onMissing is Fail",
			operator:  operatorLessThan,
			value:     10,
			onMissing: onMissingFail,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithoutValue(),
			},
			wantKept: []int{0},
		},
		{
			name:     "empty result stays empty without fallback",
			operator: operatorLessThan,
			value:    10,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(15),
				newEndpointWithValue(20),
			},
			wantKept: []int{},
		},
		{
			name:            "empty result returns all candidates with fallbackOnEmpty",
			operator:        operatorLessThan,
			value:           10,
			fallbackOnEmpty: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(15),
				newEndpointWithValue(20),
			},
			wantKept: []int{0, 1},
		},
		{
			name:            "fallbackOnEmpty does not trigger when some endpoints survive",
			operator:        operatorLessThan,
			value:           10,
			fallbackOnEmpty: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(20),
			},
			wantKept: []int{0},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filter, err := NewEndpointAttributeFilter("test-filter", parameters{
				Attribute:       testAttribute,
				OnMissing:       test.onMissing,
				FallbackOnEmpty: test.fallbackOnEmpty,
				Algorithm: algorithmParameters{
					Type: algorithmThreshold,
					Threshold: &thresholdParameters{
						Operator: test.operator,
						Value:    test.value,
					},
				},
			})
			require.NoError(t, err)

			got := filter.Filter(context.Background(), &scheduling.InferenceRequest{}, test.endpoints)

			want := make([]scheduling.Endpoint, 0, len(test.wantKept))
			for _, i := range test.wantKept {
				want = append(want, test.endpoints[i])
			}
			assert.Equal(t, want, got)
		})
	}
}
