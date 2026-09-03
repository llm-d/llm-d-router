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

package maxscore

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestEqualScoreDistribution(t *testing.T) {
	// Verifies that concurrent requests with equal scores are distributed
	// across endpoints via round-robin, not concentrated on one winner.
	// This prevents routing imbalance when prefix cache affinity causes
	// all endpoints to receive identical scores.
	numPods := 8
	endpoints := make([]fwksched.Endpoint, numPods)
	for i := 0; i < numPods; i++ {
		endpoints[i] = fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
			ID: k8stypes.NamespacedName{Name: fmt.Sprintf("pod%d", i)},
		}, nil, nil)
	}

	picker := NewMaxScorePicker(1)
	picks := make(map[string]int)

	// Simulate 80 concurrent requests all seeing the same scores
	for i := 0; i < 80; i++ {
		scored := make([]*fwksched.ScoredEndpoint, numPods)
		for j := 0; j < numPods; j++ {
			scored[j] = &fwksched.ScoredEndpoint{Endpoint: endpoints[j], Score: 50} // all equal
		}
		result := picker.Pick(context.Background(), scored)
		winner := result.TargetEndpoints[0].String()
		picks[winner]++
	}

	// Each pod should get exactly 10 requests (80/8)
	for pod, count := range picks {
		if count != 10 {
			t.Errorf("Pod %s got %d requests, expected 10 (even distribution)", pod, count)
		}
	}

	// All 8 pods should have been picked
	if len(picks) != numPods {
		t.Errorf("Only %d of %d pods were picked — routing imbalance", len(picks), numPods)
	}
}

func TestPickMaxScorePicker(t *testing.T) {
	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil)
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil)
	endpoint3 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil)

	tests := []struct {
		name               string
		picker             fwksched.Picker
		input              []*fwksched.ScoredEndpoint
		output             []fwksched.Endpoint
		tieBreakCandidates int // tie break is random, specify how many candidate with max score
	}{
		{
			name:   "Single max score",
			picker: NewMaxScorePicker(1),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 15},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
		},
		{
			name:   "Multiple max scores, all are equally scored",
			picker: NewMaxScorePicker(2),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 50},
				{Endpoint: endpoint2, Score: 50},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint1, Score: 50},
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 50},
			},
			tieBreakCandidates: 2,
		},
		{
			name:   "Multiple results sorted by highest score, more pods than needed",
			picker: NewMaxScorePicker(2),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 20},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
		},
		{
			name:   "Multiple results sorted by highest score, less pods than needed",
			picker: NewMaxScorePicker(4), // picker is required to return 4 pods at most, but we have only 3.
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 20},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
				&fwksched.ScoredEndpoint{Endpoint: endpoint1, Score: 20},
			},
		},
		{
			name:   "Multiple results sorted by highest score, num of pods exactly needed",
			picker: NewMaxScorePicker(3), // picker is required to return 3 pods at most, we have only 3.
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 30},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint1, Score: 30},
				&fwksched.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
			tieBreakCandidates: 2,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.picker.Pick(context.Background(), test.input)
			got := result.TargetEndpoints

			if test.tieBreakCandidates > 0 {
				testMaxScoredEndpoints := test.output[:test.tieBreakCandidates]
				gotMaxScoredEndpoints := got[:test.tieBreakCandidates]
				diff := cmp.Diff(testMaxScoredEndpoints, gotMaxScoredEndpoints, cmpopts.SortSlices(func(a, b fwksched.Endpoint) bool {
					return a.String() < b.String() // predictable order within the endpoints with equal scores
				}), cmp.Comparer(fwksched.ScoredEndpointComparer))
				if diff != "" {
					t.Errorf("Unexpected output (-want +got): %v", diff)
				}
				test.output = test.output[test.tieBreakCandidates:]
				got = got[test.tieBreakCandidates:]
			}

			if diff := cmp.Diff(test.output, got, cmp.Comparer(fwksched.ScoredEndpointComparer)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
