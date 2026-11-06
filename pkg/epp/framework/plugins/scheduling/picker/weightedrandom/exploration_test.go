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

package weightedrandom

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	"strconv"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func explorationCandidates(scores ...float64) []*fwksched.ScoredEndpoint {
	endpoints := make([]*fwksched.ScoredEndpoint, len(scores))
	for i, score := range scores {
		endpoints[i] = &fwksched.ScoredEndpoint{Endpoint: fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: strconv.Itoa(i)}}, nil, nil), Score: score}
	}
	return endpoints
}

func TestExplorationFirst(t *testing.T) {
	plugin, err := WeightedRandomPickerFactory("test", json.NewDecoder(strings.NewReader(`{"explorationBudget":1,"maxNumOfEndpoints":3}`)), nil)
	if err != nil {
		t.Fatal(err)
	}
	result := plugin.(*WeightedRandomPicker).Pick(context.Background(), explorationCandidates(1, 0, -1)).TargetEndpoints
	if len(result) != 2 || result[0].GetMetadata().ID.Name != "1" || result[1].GetMetadata().ID.Name != "0" {
		t.Fatalf("expected zero then positive endpoint, got %v", result)
	}
}

func TestExplorationConfiguration(t *testing.T) {
	for _, tc := range []struct {
		raw     string
		invalid bool
		budget  float64
		count   int
	}{
		{`{}`, false, 0, 1}, {`{"maxNumOfEndpoints":0}`, false, 0, 1},
		{`{"explorationBudget":0}`, false, 0, 1}, {`{"explorationBudget":1}`, false, 1, 1},
		{`{"explorationBudget":0.05,"maxNumOfEndpoints":3}`, false, .05, 3},
		{`{"explorationBudget":-0.1}`, true, 0, 0}, {`{"explorationBudget":1.1}`, true, 0, 0},
		{`{"explorationBudget":1e999}`, true, 0, 0}, {`{"explorationBudget":NaN}`, true, 0, 0},
		{`{"explorationBudget":"0.05"}`, true, 0, 0}, {`{"explorationBudget":Infinity}`, true, 0, 0},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			plugin, err := WeightedRandomPickerFactory("custom", json.NewDecoder(strings.NewReader(tc.raw)), nil)
			if (err != nil) != tc.invalid {
				t.Fatalf("error = %v, want invalid %v", err, tc.invalid)
			}
			if err == nil {
				p := plugin.(*WeightedRandomPicker)
				if p.explorationBudget != tc.budget || p.maxNumOfEndpoints != tc.count || p.TypedName().Name != "custom" {
					t.Fatalf("unexpected configuration: %+v", p)
				}
			}
		})
	}
	for _, budget := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if (weightedRandomParameters{ExplorationBudget: budget}).validate() == nil {
			t.Errorf("accepted %v", budget)
		}
	}
	plugin, err := WeightedRandomPickerFactory("nil", nil, nil)
	if err != nil || plugin.(*WeightedRandomPicker).explorationBudget != 0 {
		t.Fatalf("nil parameters: %v, %v", plugin, err)
	}
}

func TestExplorationLimits(t *testing.T) {
	for _, budget := range []float64{0, 1} {
		for _, k := range []int{1, 2, 5} {
			for _, scores := range [][]float64{nil, {0}, {-1, 0, -2}, {1, 0, -1}, {1, 3, 0, 0, -1}, {1, 3}} {
				t.Run(fmt.Sprint(budget, k, scores), func(t *testing.T) {
					p := NewWeightedRandomPicker(k)
					p.explorationBudget = budget
					input := explorationCandidates(scores...)
					positive, zero := 0, 0
					for _, score := range scores {
						if score > 0 {
							positive++
						}
						if score == 0 {
							zero++
						}
					}
					want := min(k, len(scores))
					if positive > 0 {
						want = min(k, positive)
						if budget == 1 && zero > 0 {
							want = min(k, positive+1)
						}
					}
					for range 20 {
						result := p.Pick(context.Background(), input).TargetEndpoints
						if len(result) != want {
							t.Fatalf("count %d, want %d", len(result), want)
						}
						seen := map[string]bool{}
						for i, ep := range result {
							id := ep.GetMetadata().ID.Name
							if seen[id] {
								t.Fatal("duplicate endpoint", id)
							}
							seen[id] = true
							score := ep.(*fwksched.ScoredEndpoint).Score
							if positive > 0 {
								if score < 0 || (score == 0 && (budget == 0 || i != 0)) {
									t.Fatalf("ineligible score %v at position %d", score, i)
								}
								if budget == 1 && zero > 0 && i == 0 && score != 0 {
									t.Fatal("exploration endpoint must be first")
								}
							}
						}
					}
				})
			}
		}
	}
}

func TestExplorationDistribution(t *testing.T) {
	const trials = 100000
	for _, zeros := range []int{1, 10, 100} {
		for _, k := range []int{1, 2, 8} {
			t.Run(fmt.Sprintf("zeros=%d/k=%d", zeros, k), func(t *testing.T) {
				scores := append([]float64{1, 3, -1}, make([]float64, zeros)...)
				input := explorationCandidates(scores...)
				p := NewWeightedRandomPicker(k)
				p.explorationBudget = .05
				counts := make(map[string]int)
				firstPositive := 0
				for range trials {
					result := p.Pick(context.Background(), input).TargetEndpoints
					counts[result[0].GetMetadata().ID.Name]++
					for i, ep := range result {
						if ep.(*fwksched.ScoredEndpoint).Score == 0 && i != 0 {
							t.Fatal("zero in retry candidates")
						}
					}
					index := 0
					if result[0].(*fwksched.ScoredEndpoint).Score == 0 {
						index++
					}
					if index < len(result) && result[index].GetMetadata().ID.Name == "0" {
						firstPositive++
					}
				}
				// Six binomial standard deviations plus one count for rounding.
				check := func(count, n int, probability float64) {
					t.Helper()
					tolerance := 6*math.Sqrt(float64(n)*probability*(1-probability)) + 1
					if math.Abs(float64(count)-float64(n)*probability) > tolerance {
						t.Errorf("count %d/%d, expected probability %g (+/-%g counts)", count, n, probability, tolerance)
					}
				}
				zeroTotal := 0
				for i := 3; i < len(input); i++ {
					count := counts[strconv.Itoa(i)]
					zeroTotal += count
					check(count, trials, .05/float64(zeros))
				}
				check(zeroTotal, trials, .05)
				check(counts["0"], trials, .95/4)
				check(counts["1"], trials, .95*3/4)
				n := trials
				if k == 1 {
					n -= zeroTotal
				}
				check(firstPositive, n, .25)
				if counts["2"] != 0 {
					t.Fatal("selected negative score")
				}
			})
		}
	}
}

func BenchmarkExploration(b *testing.B) {
	for _, k := range []int{1, 4} {
		for _, budget := range []float64{0, .05, 1} {
			b.Run(fmt.Sprintf("k=%d/budget=%g", k, budget), func(b *testing.B) {
				scores := make([]float64, 100)
				for i := 0; i < 50; i++ {
					scores[i] = float64(i + 1)
				}
				input := explorationCandidates(scores...)
				p := NewWeightedRandomPicker(k)
				p.explorationBudget = budget
				ctx := log.IntoContext(context.Background(), logr.Discard())
				b.ReportAllocs()
				for b.Loop() {
					p.Pick(ctx, input)
				}
			})
		}
	}
}
