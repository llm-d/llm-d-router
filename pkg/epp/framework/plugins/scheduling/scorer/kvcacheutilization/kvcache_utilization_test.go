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

package kvcacheutilization

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

// endpointWith builds an endpoint carrying the KV-cache utilization attribute,
// the way the core metrics extractor publishes it.
func endpointWith(util float64) fwksched.Endpoint {
	attr := fwkdl.NewAttributes()
	attr.Put(kvCacheUsageDataKey, attrmetrics.ScalarMetricValue(util))
	return fwksched.NewEndpoint(nil, nil, attr)
}

func TestKvCacheUtilizationScorer(t *testing.T) {
	tests := []struct {
		name                   string
		endpoints              []fwksched.Endpoint
		expectedScoresEndpoint map[int]float64 // Map of endpoint index to expected score
	}{
		{
			name:      "Different KV cache utilization",
			endpoints: []fwksched.Endpoint{endpointWith(0.8), endpointWith(0.5), endpointWith(0.0)},
			expectedScoresEndpoint: map[int]float64{
				0: 0.2, // Highest KV cache usage (0.8) gets lowest score (1-0.8=0.2)
				1: 0.5, // Medium KV cache usage (0.5) gets medium score (1-0.5=0.5)
				2: 1.0, // No KV cache usage (0.0) gets highest score (1-0=1.0)
			},
		},
		{
			name:      "Same KV cache utilization",
			endpoints: []fwksched.Endpoint{endpointWith(0.6), endpointWith(0.6)},
			expectedScoresEndpoint: map[int]float64{
				0: 0.4, // Both get same score (1-0.6=0.4)
				1: 0.4,
			},
		},
		{
			name:      "Zero KV cache utilization",
			endpoints: []fwksched.Endpoint{endpointWith(0.0), endpointWith(0.0)},
			expectedScoresEndpoint: map[int]float64{
				0: 1.0, // No KV cache usage gets highest score
				1: 1.0,
			},
		},
		{
			name:      "Full KV cache utilization",
			endpoints: []fwksched.Endpoint{endpointWith(1.0), endpointWith(0.5)},
			expectedScoresEndpoint: map[int]float64{
				0: 0.0, // Full KV cache (1.0) gets lowest score (1-1=0)
				1: 0.5, // Half KV cache (0.5) gets medium score (1-0.5=0.5)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scores := NewKVCacheUtilizationScorer().Score(context.Background(), &fwksched.InferenceRequest{}, test.endpoints)

			for i, endpoint := range test.endpoints {
				expectedScore := test.expectedScoresEndpoint[i]
				assert.InDelta(t, expectedScore, scores[endpoint], 0.0001, "Endpoint %d should have score %f", i, expectedScore)
			}
		})
	}
}

// An endpoint the extractor has not populated is left out of the scores rather
// than scored 1.0. Through GetMetrics() the two were the same reading — an
// absent value and an empty cache both came back 0 — so an endpoint nothing was
// known about outranked every endpoint whose utilization was known.
func TestKvCacheUtilizationScorerSkipsEndpointsWithoutTheAttribute(t *testing.T) {
	known := endpointWith(0.9)
	unknown := fwksched.NewEndpoint(nil, nil, nil)

	scores := NewKVCacheUtilizationScorer().Score(
		context.Background(), &fwksched.InferenceRequest{},
		[]fwksched.Endpoint{known, unknown})

	assert.Len(t, scores, 1, "only the endpoint carrying the attribute is scored")
	assert.InDelta(t, 0.1, scores[known], 0.0001)

	_, scored := scores[unknown]
	assert.False(t, scored, "an endpoint without the attribute must not be scored")
}

// The scorer reads the key it declares. Consumes and Score naming different
// keys is what let the declaration and the read drift apart before.
func TestKvCacheUtilizationScorerReadsTheKeyItDeclares(t *testing.T) {
	s := NewKVCacheUtilizationScorer()

	_, declared := s.Consumes().Required[kvCacheUsageDataKey]
	assert.True(t, declared, "the key Score reads must be the one Consumes declares")

	attr := fwkdl.NewAttributes()
	attr.Put(kvCacheUsageDataKey, attrmetrics.ScalarMetricValue(0.25))
	ep := fwksched.NewEndpoint(nil, nil, attr)

	scores := s.Score(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{ep})
	assert.InDelta(t, 0.75, scores[ep], 0.0001)
}
