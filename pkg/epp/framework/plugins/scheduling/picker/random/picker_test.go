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

package random

import (
	"context"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"sync"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestPickRandomPicker(t *testing.T) {
	const (
		testIterations = 10000
		tolerance      = 0.05 // Verify within tolerance ±5%
	)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil)
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil)
	endpoint3 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil)

	tests := []struct {
		name    string
		picker  fwksched.Picker
		input   []*fwksched.ScoredEndpoint
		numPods int // expected number of returned endpoints
	}{
		{
			name:   "Single endpoint returned from multiple candidates",
			picker: NewRandomPicker(1),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 20},
				{Endpoint: endpoint3, Score: 30},
			},
			numPods: 1,
		},
		{
			name:   "Multiple endpoints returned, more candidates than needed",
			picker: NewRandomPicker(2),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 20},
				{Endpoint: endpoint3, Score: 30},
			},
			numPods: 2,
		},
		{
			name:   "Fewer candidates than maxNumOfEndpoints",
			picker: NewRandomPicker(5),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 20},
			},
			numPods: 2,
		},
		{
			name:   "Exact number of candidates as maxNumOfEndpoints",
			picker: NewRandomPicker(3),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 20},
				{Endpoint: endpoint3, Score: 30},
			},
			numPods: 3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Verify correct number of endpoints returned
			result := test.picker.Pick(context.Background(), test.input)
			got := result.TargetEndpoints
			if len(got) != test.numPods {
				t.Errorf("Expected %d endpoints, got %d", test.numPods, len(got))
			}

			// Verify uniform distribution: each endpoint's selection probability
			// should be min(numPods/numCandidates, 1.0), regardless of scores.
			selectionCounts := make(map[string]int)
			for _, ep := range test.input {
				selectionCounts[ep.GetMetadata().ID.Name] = 0
			}

			for range testIterations {
				res := test.picker.Pick(context.Background(), test.input)
				for _, ep := range res.TargetEndpoints {
					selectionCounts[ep.GetMetadata().ID.Name]++
				}
			}

			expectedProb := math.Min(float64(test.numPods)/float64(len(test.input)), 1.0)
			for name, count := range selectionCounts {
				actualProb := float64(count) / float64(testIterations)
				if math.Abs(actualProb-expectedProb) > tolerance {
					t.Errorf("Endpoint %s: expected probability %.3f ±%.1f%%, got %.3f (count: %d/%d)",
						name, expectedProb, tolerance*100, actualProb, count, testIterations)
				}
			}
		})
	}
}

func TestRandomPickerWithRand(t *testing.T) {
	t.Parallel()

	makeInput := func() []*fwksched.ScoredEndpoint {
		input := make([]*fwksched.ScoredEndpoint, 10)
		for i := range input {
			ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod" + strconv.Itoa(i)}}, nil, nil)
			input[i] = &fwksched.ScoredEndpoint{Endpoint: ep, Score: float64(i + 1)}
		}
		return input
	}

	collectPicks := func(ctx context.Context, p *RandomPicker, n int) []string {
		picks := make([]string, 0, n*5)
		for range n {
			res := p.Pick(ctx, makeInput())
			for _, ep := range res.TargetEndpoints {
				picks = append(picks, ep.GetMetadata().ID.Name)
			}
		}
		return picks
	}

	t.Run("reproducible_picks", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		p1 := NewRandomPicker(5).WithRand(rand.NewPCG(42, 99))
		p2 := NewRandomPicker(5).WithRand(rand.NewPCG(42, 99))
		p3 := NewRandomPicker(5).WithRand(rand.NewPCG(100, 200))

		got1 := collectPicks(ctx, p1, 20)
		got2 := collectPicks(ctx, p2, 20)
		got3 := collectPicks(ctx, p3, 20)

		if !slices.Equal(got1, got2) {
			t.Fatalf("identical seeds produced different picks: %v vs %v", got1, got2)
		}
		if slices.Equal(got1, got3) {
			t.Fatalf("distinct seeds produced identical picks: %v", got1)
		}
	})

	t.Run("nil_restores_default", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		seeded := NewRandomPicker(5).WithRand(rand.NewPCG(42, 99))
		reset := NewRandomPicker(5).WithRand(rand.NewPCG(42, 99)).WithRand(nil)

		wantSeeded := collectPicks(ctx, seeded, 20)
		gotReset := collectPicks(ctx, reset, 20)

		if len(gotReset) != 20*5 {
			t.Fatalf("expected %d picked endpoints, got %d", 20*5, len(gotReset))
		}
		if slices.Equal(wantSeeded, gotReset) {
			t.Fatalf("WithRand(nil) still followed the seeded sequence")
		}
	})

	t.Run("concurrent_pick", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		p := NewRandomPicker(3).WithRand(rand.NewPCG(7, 13))

		var wg sync.WaitGroup
		for range 8 {
			wg.Go(func() {
				for range 200 {
					res := p.Pick(ctx, makeInput())
					if len(res.TargetEndpoints) != 3 {
						t.Errorf("expected 3 endpoints, got %d", len(res.TargetEndpoints))
						return
					}
				}
			})
		}
		wg.Wait()
	})
}
