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
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

const testAttributeKey = "custom.queue_depth"

func TestEndpointAttributeScorerFactory(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{
			name: "valid lower_is_better with defaulted adaptive range",
			parameters: `{"attributeKey": "custom.queue_depth",
				"algorithm": {"type": "linear_lower_is_better"}}`,
		},
		{
			name: "valid higher_is_better with explicit adaptive range",
			parameters: `{"attributeKey": "custom.tokens_per_second",
				"algorithm": {"type": "linear_higher_is_better",
					"normalization": {"adaptiveRange": {}}}}`,
		},
		{
			name: "valid fixed range",
			parameters: `{"attributeKey": "custom.kv_cache_utilization",
				"algorithm": {"type": "linear_lower_is_better",
					"normalization": {"fixedRange": {"min": 0, "max": 1}}}}`,
		},
		{
			name:       "missing attributeKey",
			parameters: `{"algorithm": {"type": "linear_lower_is_better"}}`,
			wantErr:    "attributeKey",
		},
		{
			name:       "missing algorithm type",
			parameters: `{"attributeKey": "custom.queue_depth"}`,
			wantErr:    "algorithm.type",
		},
		{
			name: "unsupported algorithm type",
			parameters: `{"attributeKey": "custom.queue_depth",
				"algorithm": {"type": "log_lower_is_better"}}`,
			wantErr: "algorithm.type",
		},
		{
			name: "both normalization strategies set",
			parameters: `{"attributeKey": "custom.queue_depth",
				"algorithm": {"type": "linear_lower_is_better",
					"normalization": {"fixedRange": {"min": 0, "max": 1}, "adaptiveRange": {}}}}`,
			wantErr: "at most one normalization strategy",
		},
		{
			name: "fixed range with min not less than max",
			parameters: `{"attributeKey": "custom.queue_depth",
				"algorithm": {"type": "linear_lower_is_better",
					"normalization": {"fixedRange": {"min": 1, "max": 1}}}}`,
			wantErr: "min < max",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := json.NewDecoder(strings.NewReader(test.parameters))
			plugin, err := EndpointAttributeScorerFactory("test-scorer", decoder, nil)
			if test.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), test.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, EndpointAttributeScorerType, plugin.TypedName().Type)
			assert.Equal(t, "test-scorer", plugin.TypedName().Name)
		})
	}
}

func newEndpointWithValue(value float64) fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	attrs.Put(testAttributeKey, attrmetrics.ScalarMetricValue(value))
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, attrs)
}

func newEndpointWithoutValue() fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, nil)
}

func TestEndpointAttributeScorerScoreAdaptiveRange(t *testing.T) {
	tests := []struct {
		name           string
		algorithmType  string
		endpoints      []fwksched.Endpoint
		expectedScores map[int]float64 // endpoint index to expected score
	}{
		{
			name:          "lower_is_better",
			algorithmType: algorithmLinearLowerIsBetter,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithValue(5),
				newEndpointWithValue(0),
			},
			expectedScores: map[int]float64{
				0: 0.0,
				1: 0.5,
				2: 1.0,
			},
		},
		{
			name:          "higher_is_better",
			algorithmType: algorithmLinearHigherIsBetter,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithValue(5),
				newEndpointWithValue(0),
			},
			expectedScores: map[int]float64{
				0: 1.0,
				1: 0.5,
				2: 0.0,
			},
		},
		{
			name:          "equal values get neutral score",
			algorithmType: algorithmLinearLowerIsBetter,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(7),
				newEndpointWithValue(7),
			},
			expectedScores: map[int]float64{
				0: 1.0,
				1: 1.0,
			},
		},
		{
			name:          "endpoint missing the attribute scores zero",
			algorithmType: algorithmLinearLowerIsBetter,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithoutValue(),
				newEndpointWithValue(0),
			},
			expectedScores: map[int]float64{
				0: 0.0,
				1: 0.0,
				2: 1.0,
			},
		},
		{
			name:          "all endpoints missing the attribute score zero",
			algorithmType: algorithmLinearLowerIsBetter,
			endpoints: []fwksched.Endpoint{
				newEndpointWithoutValue(),
				newEndpointWithoutValue(),
			},
			expectedScores: map[int]float64{
				0: 0.0,
				1: 0.0,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scorer, err := NewEndpointAttributeScorer("test-scorer", parameters{
				AttributeKey: testAttributeKey,
				Algorithm: algorithmParameters{
					Type: test.algorithmType,
				},
			})
			require.NoError(t, err)

			scores := scorer.Score(context.Background(), &fwksched.InferenceRequest{}, test.endpoints)

			for i, endpoint := range test.endpoints {
				assert.InDelta(t, test.expectedScores[i], scores[endpoint], 0.0001,
					"endpoint %d should have score %f", i, test.expectedScores[i])
			}
		})
	}
}

func TestEndpointAttributeScorerScoreFixedRange(t *testing.T) {
	tests := []struct {
		name           string
		algorithmType  string
		min            float64
		max            float64
		endpoints      []fwksched.Endpoint
		expectedScores map[int]float64 // endpoint index to expected score
	}{
		{
			name:          "lower_is_better",
			algorithmType: algorithmLinearLowerIsBetter,
			min:           0,
			max:           10,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithValue(5),
				newEndpointWithValue(0),
			},
			expectedScores: map[int]float64{
				0: 0.0,
				1: 0.5,
				2: 1.0,
			},
		},
		{
			name:          "higher_is_better",
			algorithmType: algorithmLinearHigherIsBetter,
			min:           0,
			max:           10,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithValue(5),
				newEndpointWithValue(0),
			},
			expectedScores: map[int]float64{
				0: 1.0,
				1: 0.5,
				2: 0.0,
			},
		},
		{
			name:          "equal values score by range position, not neutrally",
			algorithmType: algorithmLinearLowerIsBetter,
			min:           0,
			max:           10,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(7),
				newEndpointWithValue(7),
			},
			expectedScores: map[int]float64{
				0: 0.3,
				1: 0.3,
			},
		},
		{
			name:          "values outside the range are clamped",
			algorithmType: algorithmLinearLowerIsBetter,
			min:           0,
			max:           10,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(15),
				newEndpointWithValue(-5),
			},
			expectedScores: map[int]float64{
				0: 0.0,
				1: 1.0,
			},
		},
		{
			name:          "endpoint missing the attribute scores zero",
			algorithmType: algorithmLinearHigherIsBetter,
			min:           0,
			max:           10,
			endpoints: []fwksched.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithoutValue(),
			},
			expectedScores: map[int]float64{
				0: 1.0,
				1: 0.0,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scorer, err := NewEndpointAttributeScorer("test-scorer", parameters{
				AttributeKey: testAttributeKey,
				Algorithm: algorithmParameters{
					Type: test.algorithmType,
					Normalization: normalizationParameters{
						FixedRange: &fixedRangeParameters{Min: test.min, Max: test.max},
					},
				},
			})
			require.NoError(t, err)

			scores := scorer.Score(context.Background(), &fwksched.InferenceRequest{}, test.endpoints)

			for i, endpoint := range test.endpoints {
				assert.InDelta(t, test.expectedScores[i], scores[endpoint], 0.0001,
					"endpoint %d should have score %f", i, test.expectedScores[i])
			}
		})
	}
}
