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

package bandselection

import (
	"context"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// Saturation detectors report [0.0, 1.0+], so policies must handle values above 1.0.
var conformanceSaturations = []float64{0.0, 0.5, 0.9, 1.0, 1.5}

var conformanceBandCounts = []int{0, 1, 2, 5}

// Ticks per input, enough to exercise stateful policies that rotate across bands.
const conformanceTicks = 16

// TestBandSelectionPolicyConformance runs every registered BandSelectionPolicy against the contract in
// flowcontrol.BandSelectionPolicy. Both methods are called from a single goroutine, matching the dispatch loop.
func TestBandSelectionPolicyConformance(t *testing.T) {
	t.Parallel()

	policies := map[string]fwkplugin.FactoryFunc{
		StrictBandSelectionPolicyType: StrictPolicyFactory,
	}

	for name, factory := range policies {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			t.Run("initialization", func(t *testing.T) {
				t.Parallel()
				policy := newConformancePolicy(t, name, factory)
				assert.Equal(t, name, policy.TypedName().Name, "TypedName().Name should match registered name")
			})

			t.Run("rank_returns_permutation", func(t *testing.T) {
				t.Parallel()
				runRankPermutationTests(t, newConformancePolicy(t, name, factory))
			})

			t.Run("rank_inputs_not_modified", func(t *testing.T) {
				t.Parallel()
				runRankImmutabilityTests(t, newConformancePolicy(t, name, factory))
			})

			t.Run("record_dispatch_before_rank", func(t *testing.T) {
				t.Parallel()
				policy := newConformancePolicy(t, name, factory)
				assert.NotPanics(t, func() {
					policy.RecordDispatch(context.Background(), 0)
					policy.RecordDispatch(context.Background(), -42)
				}, "RecordDispatch must not panic before any Rank call")
			})

			t.Run("record_dispatch_known_and_unknown_priorities", func(t *testing.T) {
				t.Parallel()
				runRecordDispatchTests(t, newConformancePolicy(t, name, factory))
			})
		})
	}
}

func newConformancePolicy(t *testing.T, name string, factory fwkplugin.FactoryFunc) flowcontrol.BandSelectionPolicy {
	t.Helper()
	plugin, err := factory(name, nil, nil)
	require.NoError(t, err, "Factory failed for plugin %s", name)
	require.NotNil(t, plugin, "Factory returned nil for plugin %s", name)
	policy, ok := plugin.(flowcontrol.BandSelectionPolicy)
	require.True(t, ok, "Plugin %s is not a BandSelectionPolicy", name)
	return policy
}

// conformanceParams builds n bands, highest priority first, with ceilings that are open, partially
// gated, or fully gated relative to saturation.
func conformanceParams(n int, saturation float64) map[string]flowcontrol.BandSelectionParameters {
	priorities := make([]int, n)
	open := make([]float64, n)
	partial := make([]float64, n)
	gated := make([]float64, n)
	for i := range n {
		priorities[i] = (n - i) * 10
		open[i] = 1.0
		// The upper half of the bands is open and the lower half is gated.
		if i < (n+1)/2 {
			partial[i] = 1.0
		}
	}
	return map[string]flowcontrol.BandSelectionParameters{
		"open":            {Saturation: saturation, Priorities: priorities, Ceilings: open},
		"partially_gated": {Saturation: saturation, Priorities: priorities, Ceilings: partial},
		"fully_gated":     {Saturation: saturation, Priorities: priorities, Ceilings: gated},
	}
}

// rank mirrors the framework call: the order buffer matches len(Priorities) and is pre-filled with
// the identity permutation.
func rank(policy flowcontrol.BandSelectionPolicy, params flowcontrol.BandSelectionParameters) []int {
	order := make([]int, len(params.Priorities))
	for i := range order {
		order[i] = i
	}
	policy.Rank(context.Background(), params, order)
	return order
}

func runRankPermutationTests(t *testing.T, policy flowcontrol.BandSelectionPolicy) {
	t.Helper()
	for _, n := range conformanceBandCounts {
		for _, saturation := range conformanceSaturations {
			for gating, params := range conformanceParams(n, saturation) {
				for tick := range conformanceTicks {
					order := rank(policy, params)
					sorted := slices.Sorted(slices.Values(order))
					for i, v := range sorted {
						if !assert.Equal(t, i, v, "order %v is not a permutation of 0..%d (n=%d saturation=%v gating=%s tick=%d)",
							order, n-1, n, saturation, gating, tick) {
							break
						}
					}
					// Mirror the dispatch loop: the first ranked band that is not gated dispatches.
					for _, i := range order {
						if i < 0 || i >= n {
							break
						}
						if saturation < params.Ceilings[i] {
							policy.RecordDispatch(context.Background(), params.Priorities[i])
							break
						}
					}
				}
			}
		}
	}
}

func runRankImmutabilityTests(t *testing.T, policy flowcontrol.BandSelectionPolicy) {
	t.Helper()
	for _, n := range conformanceBandCounts {
		for _, saturation := range conformanceSaturations {
			for gating, params := range conformanceParams(n, saturation) {
				priorities := slices.Clone(params.Priorities)
				ceilings := slices.Clone(params.Ceilings)
				rank(policy, params)
				assert.Equal(t, priorities, params.Priorities, "Rank must not modify Priorities (n=%d gating=%s)", n, gating)
				assert.Equal(t, ceilings, params.Ceilings, "Rank must not modify Ceilings (n=%d gating=%s)", n, gating)
			}
		}
	}
}

func runRecordDispatchTests(t *testing.T, policy flowcontrol.BandSelectionPolicy) {
	t.Helper()
	params := conformanceParams(5, 0.5)["open"]
	rank(policy, params)
	assert.NotPanics(t, func() {
		for _, p := range params.Priorities {
			policy.RecordDispatch(context.Background(), p)
		}
	}, "RecordDispatch must not panic for known priorities")
	assert.NotPanics(t, func() {
		policy.RecordDispatch(context.Background(), 999)
		policy.RecordDispatch(context.Background(), -999)
	}, "RecordDispatch must not panic for unknown priorities")
}
