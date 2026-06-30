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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

const testAttributeKey = "custom.queue_depth"

func TestMetricScorerFactory(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{
			name: "valid lower_is_better",
			parameters: `{"attributeKey": "custom.queue_depth",
				"normalization": {"type": "linear", "direction": "lower_is_better"}}`,
		},
		{
			name: "valid higher_is_better with defaulted normalization type",
			parameters: `{"attributeKey": "custom.tokens_per_second",
				"normalization": {"direction": "higher_is_better"}}`,
		},
		{
			name:       "missing attributeKey",
			parameters: `{"normalization": {"type": "linear", "direction": "lower_is_better"}}`,
			wantErr:    "attributeKey",
		},
		{
			name: "unsupported normalization type",
			parameters: `{"attributeKey": "custom.queue_depth",
				"normalization": {"type": "log", "direction": "lower_is_better"}}`,
			wantErr: "normalization.type",
		},
		{
			name: "invalid direction",
			parameters: `{"attributeKey": "custom.queue_depth",
				"normalization": {"type": "linear", "direction": "sideways"}}`,
			wantErr: "normalization.direction",
		},
		{
			name: "missing direction",
			parameters: `{"attributeKey": "custom.queue_depth",
				"normalization": {"type": "linear"}}`,
			wantErr: "normalization.direction",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := json.NewDecoder(strings.NewReader(test.parameters))
			plugin, err := MetricScorerFactory("test-scorer", decoder, nil)
			if test.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), test.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, MetricScorerType, plugin.TypedName().Type)
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

func TestMetricScorerScore(t *testing.T) {
	tests := []struct {
		name           string
		direction      string
		endpoints      []fwksched.Endpoint
		expectedScores map[int]float64 // endpoint index to expected score
	}{
		{
			name:      "lower_is_better",
			direction: directionLowerIsBetter,
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
			name:      "higher_is_better",
			direction: directionHigherIsBetter,
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
			name:      "equal values get neutral score",
			direction: directionLowerIsBetter,
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
			name:      "endpoint missing the attribute scores zero",
			direction: directionLowerIsBetter,
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
			name:      "all endpoints missing the attribute score zero",
			direction: directionLowerIsBetter,
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
			scorer, err := NewMetricScorer(parameters{
				AttributeKey: testAttributeKey,
				Normalization: normalizationParameters{
					Type:      normalizationLinear,
					Direction: test.direction,
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
