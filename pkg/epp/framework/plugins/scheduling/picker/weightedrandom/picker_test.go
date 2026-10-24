/*
Copyright 2025 The Kubernetes Authors.
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

package weightedrandom

import (
	"context"
	"math"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestPickWeightedRandomPicker(t *testing.T) {
	const (
		testIterations = 10000
		tolerance      = 0.05 // Verify within tolerance ±5%
	)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil)
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil)
	endpoint3 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil)
	endpoint4 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod4"}}, nil, nil)
	endpoint5 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod5"}}, nil, nil)

	// A-Res algorithm uses U^(1/w) transformation which introduces statistical variance
	// beyond simple proportional sampling. Generous tolerance is required to prevent
	// flaky tests in CI environments, especially for multi-tier weights.
	tests := []struct {
		name    string
		input   []*fwksched.ScoredEndpoint
		maxPods int // maxNumOfEndpoints for this test
	}{
		{
			name: "High weight dominance test",
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10}, // Lower weight
				{Endpoint: endpoint2, Score: 90}, // Higher weight (should dominate)
			},
			maxPods: 1,
		},
		{
			name: "Equal weights test - A-Res uniform distribution",
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 100}, // Equal weights (higher values for better numerical precision)
				{Endpoint: endpoint2, Score: 100}, // Equal weights should yield uniform distribution
				{Endpoint: endpoint3, Score: 100}, // Equal weights in A-Res
			},
			maxPods: 1,
		},
		{
			name: "Zero weight exclusion test - A-Res edge case",
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 30}, // Normal weight, should be selected
				{Endpoint: endpoint2, Score: 0},  // Zero weight, never selected in A-Res
			},
			maxPods: 1,
		},
		{
			name: "Multi-tier weighted test - A-Res complex distribution",
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 100}, // Highest weight
				{Endpoint: endpoint2, Score: 90},  // High weight
				{Endpoint: endpoint3, Score: 50},  // Medium weight
				{Endpoint: endpoint4, Score: 30},  // Low weight
				{Endpoint: endpoint5, Score: 20},  // Lowest weight
			},
			maxPods: 1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			picker := NewWeightedRandomPicker(test.maxPods)

			// Summarize the total score of all pods
			totalScore := 0.0
			for _, pod := range test.input {
				totalScore += pod.Score
			}

			// Calculate expected probabilities based on scores
			expectedProbabilities := make(map[string]float64)
			for _, endpoint := range test.input {
				podName := endpoint.GetMetadata().ID.Name
				if totalScore > 0 {
					expectedProbabilities[podName] = endpoint.Score / totalScore
				} else {
					expectedProbabilities[podName] = 0.0
				}
			}

			// Initialize selection counters for each pod
			selectionCounts := make(map[string]int)
			for _, endpoint := range test.input {
				endpointName := endpoint.GetMetadata().ID.Name
				selectionCounts[endpointName] = 0
			}

			// Run multiple iterations to gather statistical data
			for range testIterations {
				result := picker.Pick(context.Background(), test.input)

				// Count selections for probability analysis
				selectedEndpointName := result.TargetEndpoints[0].GetMetadata().ID.Name
				selectionCounts[selectedEndpointName]++
			}

			// Verify probability distribution
			for endpointName, expectedProb := range expectedProbabilities {
				actualCount := selectionCounts[endpointName]
				actualProb := float64(actualCount) / float64(testIterations)

				if math.Abs(actualProb-expectedProb) > tolerance {
					t.Errorf("Endpoint %s: expected probability %.3f ±%.1f%%, got %.3f (count: %d/%d)",
						endpointName, expectedProb, tolerance*100, actualProb, actualCount, testIterations)
				} else {
					t.Logf("Endpoint %s: expected %.3f, got %.3f (count: %d/%d) ✓",
						endpointName, expectedProb, actualProb, actualCount, testIterations)
				}
			}
		})
	}
}

func TestPickWeightedRandomPickerTemperature(t *testing.T) {
	const (
		testIterations = 10000
		tolerance      = 0.05 // Verify within tolerance +/-5%
	)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil)
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil)

	// Temperature reshapes the selection distribution: probability becomes proportional to
	// score^(1/T). T<1 sharpens toward the highest score, T>1 flattens toward uniform, T=1 and
	// non-positive (fallback) stay proportional.
	tests := []struct {
		name        string
		temperature float64
		input       []*fwksched.ScoredEndpoint
	}{
		{
			name:        "Low temperature sharpens toward the highest score",
			temperature: 0.3,
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 2},
				{Endpoint: endpoint2, Score: 1},
			},
		},
		{
			name:        "High temperature flattens toward uniform",
			temperature: 5,
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 2},
				{Endpoint: endpoint2, Score: 1},
			},
		},
		{
			name:        "Temperature one stays proportional",
			temperature: 1,
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 3},
				{Endpoint: endpoint2, Score: 1},
			},
		},
		{
			name:        "Non-positive temperature falls back to default",
			temperature: 0,
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 2},
				{Endpoint: endpoint2, Score: 1},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			picker := NewWeightedRandomPicker(1).WithTemperature(test.temperature)

			// Expected probability is proportional to score^(1/T), using the default temperature
			// when the configured value is non-positive (matching WithTemperature's fallback).
			effectiveTemperature := test.temperature
			if effectiveTemperature <= 0 {
				effectiveTemperature = DefaultTemperature
			}

			totalWeight := 0.0
			endpointWeights := make(map[string]float64)
			for _, endpoint := range test.input {
				w := math.Pow(endpoint.Score, 1.0/effectiveTemperature)
				endpointWeights[endpoint.GetMetadata().ID.Name] = w
				totalWeight += w
			}

			expectedProbabilities := make(map[string]float64)
			for name, w := range endpointWeights {
				expectedProbabilities[name] = w / totalWeight
			}

			selectionCounts := make(map[string]int)
			for range testIterations {
				result := picker.Pick(context.Background(), test.input)
				selectionCounts[result.TargetEndpoints[0].GetMetadata().ID.Name]++
			}

			for endpointName, expectedProb := range expectedProbabilities {
				actualProb := float64(selectionCounts[endpointName]) / float64(testIterations)
				if math.Abs(actualProb-expectedProb) > tolerance {
					t.Errorf("Endpoint %s: expected probability %.3f +/-%.1f%%, got %.3f (count: %d/%d)",
						endpointName, expectedProb, tolerance*100, actualProb, selectionCounts[endpointName], testIterations)
				}
			}
		})
	}
}
