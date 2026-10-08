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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrgpu "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/gpu"
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
			name: "attribute and producer set separately",
			parameters: `{"attribute": "GPUUtilization", "producer": "dcgm-extractor",
				"algorithm": {"type": "threshold",
					"threshold": {"operator": "LessThan", "value": 0.8}}}`,
		},
		{
			// Guards the silent regression: the combined spelling built a key
			// matching nothing, so the filter kept every endpoint without error.
			name: "combined Attribute/Producer spelling is rejected",
			parameters: `{"attribute": "GPUUtilization/dcgm-extractor",
				"algorithm": {"type": "threshold",
					"threshold": {"operator": "LessThan", "value": 0.8}}}`,
			wantErr: `split it into attribute: "GPUUtilization" and producer: "dcgm-extractor"`,
		},
		{
			name: "explicit empty producer allows a slash in the attribute name",
			parameters: `{"attribute": "llm-d.ai/multicluster-queue-size", "producer": "",
				"algorithm": {"type": "threshold",
					"threshold": {"operator": "LessThan", "value": 10}}}`,
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
	attrs.Put(attrmetrics.ScalarMetricDataKey(testAttribute), attrmetrics.ScalarMetricValue(value))
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

// TestFilterReadsAttributeOfNamedProducer covers the config documented by the
// DCGM extractor, where producer names a plugin other than the core metrics
// extractor. Resolving to the wrong producer makes the filter miss every
// endpoint and silently degrade to a no-op.
func TestFilterReadsAttributeOfNamedProducer(t *testing.T) {
	params := `{"attribute": "GPUUtilization", "producer": "dcgm-extractor",
		"onMissing": "Fail",
		"algorithm": {"type": "threshold",
			"threshold": {"operator": "LessThan", "value": 0.8}}}`

	plug, err := EndpointAttributeFilterFactory("gpu", json.NewDecoder(strings.NewReader(params)), nil)
	require.NoError(t, err)
	filter := plug.(*EndpointAttributeFilter)

	assert.Equal(t, attrgpu.GPUUtilizationDataKey, filter.dataKey,
		"configured attribute must resolve to the key the DCGM extractor publishes")

	attrs := fwkdl.NewAttributes()
	attrs.Put(attrgpu.GPUUtilizationDataKey, attrmetrics.ScalarMetricValue(0.5))
	busy := fwkdl.NewAttributes()
	busy.Put(attrgpu.GPUUtilizationDataKey, attrmetrics.ScalarMetricValue(0.95))

	endpoints := []scheduling.Endpoint{
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, attrs),
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, busy),
	}

	kept := filter.Filter(context.Background(), nil, endpoints)
	require.Len(t, kept, 1, "only the idle GPU endpoint should survive")
	assert.Equal(t, endpoints[0], kept[0])
}

// ---------------------------------------------------------------------------
// Factory validation tests for the new algorithms (range, topK, percentile)
// ---------------------------------------------------------------------------

func TestEndpointAttributeFilterFactoryNewAlgorithms(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		// range
		{
			name: "valid range",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "range",
					"range": {"min": 0, "max": 100}}}`,
		},
		{
			name: "range with equal min and max",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "range",
					"range": {"min": 50, "max": 50}}}`,
		},
		{
			name: "range missing block",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "range"}}`,
			wantErr: "algorithm.range",
		},
		{
			name: "range min greater than max",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "range",
					"range": {"min": 100, "max": 0}}}`,
			wantErr: "min <= max",
		},
		// topK
		{
			name: "valid topK higherIsBetter",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "topK",
					"topK": {"k": 3, "higherIsBetter": true}}}`,
		},
		{
			name: "valid topK lowerIsBetter",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "topK",
					"topK": {"k": 2, "higherIsBetter": false}}}`,
		},
		{
			name: "topK missing block",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "topK"}}`,
			wantErr: "algorithm.topK",
		},
		{
			name: "topK zero k",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "topK",
					"topK": {"k": 0}}}`,
			wantErr: "positive integer",
		},
		{
			name: "topK negative k",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "topK",
					"topK": {"k": -1}}}`,
			wantErr: "positive integer",
		},
		// percentile
		{
			name: "valid percentile higherIsBetter",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "percentile",
					"percentile": {"percentile": 25, "higherIsBetter": true}}}`,
		},
		{
			name: "valid percentile lowerIsBetter",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "percentile",
					"percentile": {"percentile": 75, "higherIsBetter": false}}}`,
		},
		{
			name: "percentile missing block",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "percentile"}}`,
			wantErr: "algorithm.percentile",
		},
		{
			name: "percentile below zero",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "percentile",
					"percentile": {"percentile": -1}}}`,
			wantErr: "[0, 100]",
		},
		{
			name: "percentile above 100",
			parameters: `{"attribute": "num_requests_running",
				"algorithm": {"type": "percentile",
					"percentile": {"percentile": 101}}}`,
			wantErr: "[0, 100]",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := json.NewDecoder(strings.NewReader(test.parameters))
			_, err := EndpointAttributeFilterFactory("test-filter", decoder, nil)
			if test.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), test.wantErr)
				return
			}
			require.NoError(t, err)
		})
	}
}

// ---------------------------------------------------------------------------
// Filter behavior tests for the new algorithms
// ---------------------------------------------------------------------------

func TestEndpointAttributeFilterRange(t *testing.T) {
	tests := []struct {
		name            string
		min             float64
		max             float64
		onMissing       string
		fallbackOnEmpty bool
		endpoints       []scheduling.Endpoint
		wantKept        []int
	}{
		{
			name: "range keeps values within [10, 50]",
			min:  10,
			max:  50,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(10), // boundary
				newEndpointWithValue(30),
				newEndpointWithValue(50), // boundary
				newEndpointWithValue(80),
			},
			wantKept: []int{1, 2, 3},
		},
		{
			name: "range with single-point [50, 50] keeps only matching value",
			min:  50,
			max:  50,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(30),
				newEndpointWithValue(50),
				newEndpointWithValue(70),
			},
			wantKept: []int{1},
		},
		{
			name:      "range with onMissing Fail drops endpoints without attribute",
			min:       10,
			max:       50,
			onMissing: onMissingFail,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(30),
				newEndpointWithoutValue(),
				newEndpointWithValue(80),
			},
			wantKept: []int{0},
		},
		{
			name:            "range fallbackOnEmpty returns all when nothing matches",
			min:             100,
			max:             200,
			fallbackOnEmpty: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(5),
				newEndpointWithValue(50),
			},
			wantKept: []int{0, 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filter, err := NewEndpointAttributeFilter("test-filter", parameters{
				Attribute:       testAttribute,
				OnMissing:       test.onMissing,
				FallbackOnEmpty: test.fallbackOnEmpty,
				Algorithm: algorithmParameters{
					Type: algorithmRange,
					Range: &rangeParameters{
						Min: test.min,
						Max: test.max,
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

func TestEndpointAttributeFilterTopK(t *testing.T) {
	tests := []struct {
		name            string
		k               int
		higherIsBetter  bool
		onMissing       string
		fallbackOnEmpty bool
		endpoints       []scheduling.Endpoint
		wantKept        []int
	}{
		{
			name:           "topK 3 highest keeps the 3 largest values",
			k:              3,
			higherIsBetter: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10), // 0
				newEndpointWithValue(50), // 1
				newEndpointWithValue(30), // 2
				newEndpointWithValue(80), // 3
				newEndpointWithValue(20), // 4
			},
			wantKept: []int{1, 2, 3}, // 50, 30, 80
		},
		{
			name:           "topK 2 lowest keeps the 2 smallest values",
			k:              2,
			higherIsBetter: false,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10), // 0
				newEndpointWithValue(50), // 1
				newEndpointWithValue(30), // 2
				newEndpointWithValue(80), // 3
			},
			wantKept: []int{0, 2}, // 10, 30
		},
		{
			name:           "topK larger than endpoint count keeps all",
			k:              10,
			higherIsBetter: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithValue(20),
				newEndpointWithValue(30),
			},
			wantKept: []int{0, 1, 2},
		},
		{
			name:           "topK with onMissing Pass includes endpoints without attribute",
			k:              2,
			higherIsBetter: true,
			onMissing:      onMissingPass,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10),  // 0
				newEndpointWithValue(50),  // 1
				newEndpointWithoutValue(), // 2
			},
			wantKept: []int{0, 1, 2}, // 2 ranked + 1 missing (passed)
		},
		{
			name:           "topK with onMissing Fail excludes endpoints without attribute",
			k:              2,
			higherIsBetter: true,
			onMissing:      onMissingFail,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10),  // 0
				newEndpointWithValue(50),  // 1
				newEndpointWithoutValue(), // 2
			},
			wantKept: []int{0, 1},
		},
		{
			name:            "topK fallbackOnEmpty when all endpoints lack attribute",
			k:               2,
			higherIsBetter:  true,
			fallbackOnEmpty: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithoutValue(),
				newEndpointWithoutValue(),
			},
			wantKept: []int{0, 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filter, err := NewEndpointAttributeFilter("test-filter", parameters{
				Attribute:       testAttribute,
				OnMissing:       test.onMissing,
				FallbackOnEmpty: test.fallbackOnEmpty,
				Algorithm: algorithmParameters{
					Type: algorithmTopK,
					TopK: &topKParameters{
						K:              test.k,
						HigherIsBetter: test.higherIsBetter,
					},
				},
			})
			require.NoError(t, err)

			got := filter.Filter(context.Background(), &scheduling.InferenceRequest{}, test.endpoints)

			// topK uses sort which is not stable; compare by set membership, not order.
			assert.ElementsMatch(t, indexesToEndpoints(test.endpoints, test.wantKept), got)
		})
	}
}

func TestEndpointAttributeFilterPercentile(t *testing.T) {
	tests := []struct {
		name            string
		percentile      float64
		higherIsBetter  bool
		onMissing       string
		fallbackOnEmpty bool
		endpoints       []scheduling.Endpoint
		wantKept        []int
	}{
		{
			// 10 endpoints with values 10,20,...,100.
			// 25th percentile nearest-rank = ceil(0.25 * 10) = 3rd value = 30.
			// higherIsBetter: keep values >= 30 → {30,40,...,100} (8 endpoints)
			name:           "percentile 25 higherIsBetter keeps values >= 30",
			percentile:     25,
			higherIsBetter: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10),  // 0
				newEndpointWithValue(20),  // 1
				newEndpointWithValue(30),  // 2
				newEndpointWithValue(40),  // 3
				newEndpointWithValue(50),  // 4
				newEndpointWithValue(60),  // 5
				newEndpointWithValue(70),  // 6
				newEndpointWithValue(80),  // 7
				newEndpointWithValue(90),  // 8
				newEndpointWithValue(100), // 9
			},
			wantKept: []int{2, 3, 4, 5, 6, 7, 8, 9},
		},
		{
			// 75th percentile nearest-rank = ceil(0.75 * 10) = 8th value = 80.
			// higherIsBetter=false: keep values <= 80 → {10,...,80} (8 endpoints)
			name:           "percentile 75 lowerIsBetter keeps values <= 80",
			percentile:     75,
			higherIsBetter: false,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10),  // 0
				newEndpointWithValue(20),  // 1
				newEndpointWithValue(30),  // 2
				newEndpointWithValue(40),  // 3
				newEndpointWithValue(50),  // 4
				newEndpointWithValue(60),  // 5
				newEndpointWithValue(70),  // 6
				newEndpointWithValue(80),  // 7
				newEndpointWithValue(90),  // 8
				newEndpointWithValue(100), // 9
			},
			wantKept: []int{0, 1, 2, 3, 4, 5, 6, 7},
		},
		{
			// 50th percentile of 4 values = ceil(0.5 * 4) = 2nd value = 20.
			// higherIsBetter: keep values >= 20 → {20, 30, 40} (3 endpoints)
			name:           "percentile 50 higherIsBetter with 4 endpoints",
			percentile:     50,
			higherIsBetter: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10), // 0
				newEndpointWithValue(20), // 1
				newEndpointWithValue(30), // 2
				newEndpointWithValue(40), // 3
			},
			wantKept: []int{1, 2, 3},
		},
		{
			name:           "percentile 0 higherIsBetter keeps all valued endpoints",
			percentile:     0,
			higherIsBetter: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithValue(20),
				newEndpointWithValue(30),
			},
			wantKept: []int{0, 1, 2},
		},
		{
			name:           "percentile 100 lowerIsBetter keeps all valued endpoints",
			percentile:     100,
			higherIsBetter: false,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10),
				newEndpointWithValue(20),
				newEndpointWithValue(30),
			},
			wantKept: []int{0, 1, 2},
		},
		{
			name:           "percentile with onMissing Fail drops endpoints without attribute",
			percentile:     50,
			higherIsBetter: true,
			onMissing:      onMissingFail,
			endpoints: []scheduling.Endpoint{
				newEndpointWithValue(10),  // 0
				newEndpointWithValue(20),  // 1
				newEndpointWithValue(30),  // 2
				newEndpointWithValue(40),  // 3
				newEndpointWithoutValue(), // 4
			},
			wantKept: []int{1, 2, 3}, // values >= 20 (2nd of 4)
		},
		{
			name:            "percentile fallbackOnEmpty when all lack attribute",
			percentile:      50,
			higherIsBetter:  true,
			fallbackOnEmpty: true,
			endpoints: []scheduling.Endpoint{
				newEndpointWithoutValue(),
				newEndpointWithoutValue(),
			},
			wantKept: []int{0, 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filter, err := NewEndpointAttributeFilter("test-filter", parameters{
				Attribute:       testAttribute,
				OnMissing:       test.onMissing,
				FallbackOnEmpty: test.fallbackOnEmpty,
				Algorithm: algorithmParameters{
					Type: algorithmPercentile,
					Percentile: &percentileParameters{
						Percentile:     test.percentile,
						HigherIsBetter: test.higherIsBetter,
					},
				},
			})
			require.NoError(t, err)

			got := filter.Filter(context.Background(), &scheduling.InferenceRequest{}, test.endpoints)

			assert.ElementsMatch(t, indexesToEndpoints(test.endpoints, test.wantKept), got)
		})
	}
}

// indexesToEndpoints builds an endpoint slice from the given indexes for test
// assertions.
func indexesToEndpoints(endpoints []scheduling.Endpoint, indexes []int) []scheduling.Endpoint {
	result := make([]scheduling.Endpoint, 0, len(indexes))
	for _, i := range indexes {
		result = append(result, endpoints[i])
	}
	return result
}
