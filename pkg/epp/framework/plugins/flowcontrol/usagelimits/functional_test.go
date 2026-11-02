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

package usagelimits

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits/priorityholdback"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits/softreflectiveceiling"
)

// Saturation detectors report [0.0, 1.0+], so policies must handle values above 1.0.
var conformanceSaturations = []float64{0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 0.99, 1.0, 1.2, 1.5}

// conformancePriorities returns active priority domains, highest first. Each call returns fresh
// slices so a policy that writes to its input cannot affect another subtest.
func conformancePriorities() [][]int {
	return [][]int{
		{},
		{0},
		{10},
		{10, 0},
		{10, 5, 0},
		{100, 10, 1, 0, -5, -100},
	}
}

// Ticks per saturation, enough to cover several periods of dispatch-spreading policies.
const conformanceTicks = 32

type usageLimitPolicyEntry struct {
	name       string
	pluginType string
	factory    fwkplugin.FactoryFunc
	rawParam   string
}

// TestUsageLimitPolicyConformance runs every registered UsageLimitPolicy against the contract in
// flowcontrol.UsageLimitPolicy.
func TestUsageLimitPolicyConformance(t *testing.T) {
	t.Parallel()

	policies := []usageLimitPolicyEntry{
		{
			name:       "static_default",
			pluginType: StaticUsageLimitPolicyType,
			factory:    StaticPolicyFactory,
		},
		{
			name:       "static_threshold",
			pluginType: StaticUsageLimitPolicyType,
			factory:    StaticPolicyFactory,
			rawParam:   `{"threshold": 0.8}`,
		},
		{
			name:       "priority_holdback_rank",
			pluginType: priorityholdback.PolicyType,
			factory:    priorityholdback.PolicyFactory,
			rawParam:   `{"domain": "rank", "minCeiling": 0.2}`,
		},
		{
			name:       "priority_holdback_value",
			pluginType: priorityholdback.PolicyType,
			factory:    priorityholdback.PolicyFactory,
			rawParam:   `{"domain": "value", "minCeiling": 0.2, "maxCeiling": 0.9}`,
		},
		{
			name:       "priority_holdback_explicit",
			pluginType: priorityholdback.PolicyType,
			factory:    priorityholdback.PolicyFactory,
			rawParam:   `{"domain": "explicit", "ceilings": {"10": 1.0, "0": 0.6, "-10": 0.2}}`,
		},
		{
			name:       "soft_reflective_ceiling",
			pluginType: softreflectiveceiling.PolicyType,
			factory:    softreflectiveceiling.Factory,
		},
	}

	for _, entry := range policies {
		t.Run(entry.name, func(t *testing.T) {
			t.Parallel()

			t.Run("initialization", func(t *testing.T) {
				t.Parallel()
				policy := newConformancePolicy(t, entry)
				assert.Equal(t, entry.name, policy.TypedName().Name, "TypedName().Name should match the configured name")
				assert.Equal(t, entry.pluginType, policy.TypedName().Type, "TypedName().Type should match the registered type")
			})

			t.Run("ceilings_bounded_and_monotonic", func(t *testing.T) {
				t.Parallel()
				runCeilingConformanceTests(t, newConformancePolicy(t, entry))
			})

			t.Run("priorities_not_modified", func(t *testing.T) {
				t.Parallel()
				runPrioritiesImmutabilityTests(t, newConformancePolicy(t, entry))
			})

			t.Run("concurrent_compute_limit", func(t *testing.T) {
				t.Parallel()
				runConcurrentComputeLimitTests(t, newConformancePolicy(t, entry))
			})
		})
	}
}

func newConformancePolicy(t *testing.T, entry usageLimitPolicyEntry) flowcontrol.UsageLimitPolicy {
	t.Helper()
	plugin, err := entry.factory(entry.name, fwkplugin.StrictDecoder([]byte(entry.rawParam)), nil)
	require.NoError(t, err, "Factory failed for plugin %s", entry.name)
	require.NotNil(t, plugin, "Factory returned nil for plugin %s", entry.name)
	policy, ok := plugin.(flowcontrol.UsageLimitPolicy)
	require.True(t, ok, "Plugin %s is not a UsageLimitPolicy", entry.name)
	return policy
}

// computeLimit mirrors the framework call: the buffer matches len(priorities) and is pre-filled with 1.0.
func computeLimit(policy flowcontrol.UsageLimitPolicy, saturation float64, priorities []int) []float64 {
	ceilings := make([]float64, len(priorities))
	for i := range ceilings {
		ceilings[i] = 1.0
	}
	policy.ComputeLimit(context.Background(), saturation, priorities, ceilings)
	return ceilings
}

func runCeilingConformanceTests(t *testing.T, policy flowcontrol.UsageLimitPolicy) {
	t.Helper()
	for _, priorities := range conformancePriorities() {
		for _, saturation := range conformanceSaturations {
			for tick := range conformanceTicks {
				ceilings := computeLimit(policy, saturation, priorities)
				assertValidCeilings(t, ceilings, "priorities=%v saturation=%v tick=%d", priorities, saturation, tick)
			}
		}
	}
}

func assertValidCeilings(t *testing.T, ceilings []float64, format string, args ...any) bool {
	t.Helper()
	inputs := fmt.Sprintf(format, args...)
	ok := true
	for i, c := range ceilings {
		if math.IsNaN(c) || math.IsInf(c, 0) || c < 0.0 || c > 1.0 {
			ok = assert.Failf(t, "ceiling out of bounds", "ceilings[%d]=%v must be finite and in [0.0, 1.0] (%s)",
				i, c, inputs)
		}
		if i > 0 && ceilings[i-1] < c {
			ok = assert.Failf(t, "ceilings not monotonic", "ceilings[%d]=%v < ceilings[%d]=%v (%s)",
				i-1, ceilings[i-1], i, c, inputs)
		}
	}
	return ok
}

func runPrioritiesImmutabilityTests(t *testing.T, policy flowcontrol.UsageLimitPolicy) {
	t.Helper()
	for _, priorities := range conformancePriorities() {
		snapshot := slices.Clone(priorities)
		for _, saturation := range conformanceSaturations {
			computeLimit(policy, saturation, priorities)
		}
		assert.Equal(t, snapshot, priorities, "ComputeLimit must not modify the priorities slice")
	}
}

func runConcurrentComputeLimitTests(t *testing.T, policy flowcontrol.UsageLimitPolicy) {
	t.Helper()
	const goroutines = 8
	// The framework shares one priorities snapshot across calls.
	all := conformancePriorities()
	priorities := all[len(all)-1]

	results := make([][][]float64, goroutines)
	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Go(func() {
			for _, saturation := range conformanceSaturations {
				for range conformanceTicks {
					results[g] = append(results[g], computeLimit(policy, saturation, priorities))
				}
			}
		})
	}
	wg.Wait()

	for g, perGoroutine := range results {
		for _, ceilings := range perGoroutine {
			if !assertValidCeilings(t, ceilings, "goroutine=%d", g) {
				return
			}
		}
	}
}
