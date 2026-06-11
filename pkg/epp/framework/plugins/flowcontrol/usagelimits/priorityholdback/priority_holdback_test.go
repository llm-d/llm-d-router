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

package priorityholdback

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// ---------------------------------------------------------------------------
// Factory tests
// ---------------------------------------------------------------------------

func TestPolicyFactory(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		config    []byte
		wantError bool
	}{
		{
			name:      "valid stepwise-spread config",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": 0.5}`),
			wantError: false,
		},
		{
			name:      "valid linear-proportional config",
			config:    []byte(`{"strategy": "linear-proportional", "minCeiling": 0.3, "maxCeiling": 0.9}`),
			wantError: false,
		},
		{
			name:      "valid config with explicit maxCeiling",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": 0.2, "maxCeiling": 0.8}`),
			wantError: false,
		},
		{
			name:      "missing strategy",
			config:    []byte(`{"minCeiling": 0.5}`),
			wantError: true,
		},
		{
			name:      "missing minCeiling",
			config:    []byte(`{"strategy": "stepwise-spread"}`),
			wantError: true,
		},
		{
			name:      "missing both required fields",
			config:    []byte(`{}`),
			wantError: true,
		},
		{
			name:      "nil config",
			config:    nil,
			wantError: true,
		},
		{
			name:      "unsupported strategy",
			config:    []byte(`{"strategy": "unknown", "minCeiling": 0.5}`),
			wantError: true,
		},
		{
			name:      "minCeiling negative",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": -0.1}`),
			wantError: true,
		},
		{
			name:      "minCeiling equals 1.0",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": 1.0}`),
			wantError: true,
		},
		{
			name:      "maxCeiling zero",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": 0.5, "maxCeiling": 0.0}`),
			wantError: true,
		},
		{
			name:      "maxCeiling exceeds 1.0",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": 0.5, "maxCeiling": 1.1}`),
			wantError: true,
		},
		{
			name:      "minCeiling equals maxCeiling",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": 0.5, "maxCeiling": 0.5}`),
			wantError: true,
		},
		{
			name:      "minCeiling greater than maxCeiling",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": 0.8, "maxCeiling": 0.5}`),
			wantError: true,
		},
		{
			name:      "invalid JSON",
			config:    []byte(`{not json}`),
			wantError: true,
		},
		{
			name:      "minCeiling zero is valid",
			config:    []byte(`{"strategy": "stepwise-spread", "minCeiling": 0.0}`),
			wantError: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			p, err := PolicyFactory("test-policy", fwkplugin.StrictDecoder(tc.config), nil)
			if tc.wantError {
				require.Error(t, err)
				require.Nil(t, p)
			} else {
				require.NoError(t, err)
				require.NotNil(t, p)
			}
		})
	}
}

func TestPolicyFactory_DefaultMaxCeiling(t *testing.T) {
	t.Parallel()
	p, err := PolicyFactory("test", fwkplugin.StrictDecoder(
		[]byte(`{"strategy": "stepwise-spread", "minCeiling": 0.5}`)), nil)
	require.NoError(t, err)

	policy := p.(*priorityHoldbackPolicy)
	assert.Equal(t, 1.0, policy.cMax, "maxCeiling should default to 1.0")
}

func TestPolicyFactory_TypedName(t *testing.T) {
	t.Parallel()
	p, err := PolicyFactory("my-policy", fwkplugin.StrictDecoder(
		[]byte(`{"strategy": "stepwise-spread", "minCeiling": 0.5}`)), nil)
	require.NoError(t, err)

	tn := p.(interface{ TypedName() fwkplugin.TypedName }).TypedName()
	assert.Equal(t, PolicyType, tn.Type)
	assert.Equal(t, "my-policy", tn.Name)
}

// ---------------------------------------------------------------------------
// ComputeLimit corner cases
// ---------------------------------------------------------------------------

func TestComputeLimit_EmptyPriorities(t *testing.T) {
	t.Parallel()
	policy := newPriorityHoldbackPolicy(config{
		strategy:   strategyStepwiseSpread,
		minCeiling: 0.5,
		maxCeiling: 1.0,
	})

	ceilings := policy.ComputeLimit(t.Context(), 0.5, []int{})
	assert.Empty(t, ceilings)
	assert.NotNil(t, ceilings, "should return empty slice, not nil")
}

func TestComputeLimit_SinglePriority(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		strategy string
		cMax     float64
	}{
		{"stepwise-spread", strategyStepwiseSpread, 1.0},
		{"linear-proportional", strategyLinearProportional, 1.0},
		{"custom maxCeiling", strategyStepwiseSpread, 0.9},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			policy := newPriorityHoldbackPolicy(config{
				strategy:   tc.strategy,
				minCeiling: 0.5,
				maxCeiling: tc.cMax,
			})
			ceilings := policy.ComputeLimit(t.Context(), 0.5, []int{10})
			require.Len(t, ceilings, 1)
			assert.Equal(t, tc.cMax, ceilings[0], "single priority should bypass holdback with cMax")
		})
	}
}

// ---------------------------------------------------------------------------
// Stepwise-spread tests
// ---------------------------------------------------------------------------

func TestStepwiseSpread_TwoPriorities(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitStepwiseSpread(0.5, 1.0, []int{100, 10})
	require.Len(t, ceilings, 2)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9, "highest priority gets cMax")
	assert.InDelta(t, 0.5, ceilings[1], 1e-9, "lowest priority gets cMin")
}

func TestStepwiseSpread_ThreePriorities(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitStepwiseSpread(0.5, 1.0, []int{100, 50, 10})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9)
	assert.InDelta(t, 0.75, ceilings[1], 1e-9)
	assert.InDelta(t, 0.5, ceilings[2], 1e-9)
}

func TestStepwiseSpread_FourPriorities(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitStepwiseSpread(0.5, 1.0, []int{100, 75, 50, 10})
	require.Len(t, ceilings, 4)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9)
	assert.InDelta(t, 0.8333333, ceilings[1], 1e-6)
	assert.InDelta(t, 0.6666666, ceilings[2], 1e-6)
	assert.InDelta(t, 0.5, ceilings[3], 1e-9)
}

func TestStepwiseSpread_IgnoresNumericalValues(t *testing.T) {
	t.Parallel()

	// Stepwise should produce the same ceilings regardless of priority values,
	// as long as the count and order are the same.
	ceilingsA := computeLimitStepwiseSpread(0.5, 1.0, []int{100, 50, 10})
	ceilingsB := computeLimitStepwiseSpread(0.5, 1.0, []int{1000, 2, 1})

	require.Len(t, ceilingsA, 3)
	require.Len(t, ceilingsB, 3)
	for i := range ceilingsA {
		assert.InDelta(t, ceilingsA[i], ceilingsB[i], 1e-9,
			"stepwise ceilings should be identical regardless of priority values")
	}
}

func TestStepwiseSpread_MonotonicallyDecreasing(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitStepwiseSpread(0.2, 0.9, []int{50, 40, 30, 20, 10})
	for i := 1; i < len(ceilings); i++ {
		assert.Greater(t, ceilings[i-1], ceilings[i],
			"ceiling[%d] (%f) should be greater than ceiling[%d] (%f)", i-1, ceilings[i-1], i, ceilings[i])
	}
}

func TestStepwiseSpread_BoundaryValues(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitStepwiseSpread(0.0, 1.0, []int{10, 5, 1})
	assert.InDelta(t, 1.0, ceilings[0], 1e-9, "highest priority gets cMax")
	assert.InDelta(t, 0.0, ceilings[len(ceilings)-1], 1e-9, "lowest priority gets cMin")
}

func TestStepwiseSpread_NarrowRange(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitStepwiseSpread(0.8, 0.9, []int{10, 5, 1})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 0.9, ceilings[0], 1e-9)
	assert.InDelta(t, 0.85, ceilings[1], 1e-9)
	assert.InDelta(t, 0.8, ceilings[2], 1e-9)
}

// ---------------------------------------------------------------------------
// Linear-proportional tests
// ---------------------------------------------------------------------------

func TestLinearProportional_TwoPriorities(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitLinearProportional(0.5, 1.0, []int{100, 10})
	require.Len(t, ceilings, 2)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9, "highest priority gets cMax")
	assert.InDelta(t, 0.5, ceilings[1], 1e-9, "lowest priority gets cMin")
}

func TestLinearProportional_ThreePriorities_EvenSpacing(t *testing.T) {
	t.Parallel()
	// Priorities {90, 50, 10}: 50 is equidistant from both endpoints.
	ceilings := computeLimitLinearProportional(0.5, 1.0, []int{90, 50, 10})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9)
	assert.InDelta(t, 0.75, ceilings[1], 1e-9)
	assert.InDelta(t, 0.5, ceilings[2], 1e-9)
}

func TestLinearProportional_ThreePriorities_SkewedLow(t *testing.T) {
	t.Parallel()
	// Priorities {100, 50, 10}: 50 is numerically closer to 10 than to 100.
	ceilings := computeLimitLinearProportional(0.5, 1.0, []int{100, 50, 10})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9)
	assert.InDelta(t, 0.7222222, ceilings[1], 1e-6)
	assert.InDelta(t, 0.5, ceilings[2], 1e-9)
}

func TestLinearProportional_SkewedDistribution(t *testing.T) {
	t.Parallel()
	// Priorities {1, 2, 100}: 2 is nearly indistinguishable from 1.
	ceilings := computeLimitLinearProportional(0.5, 1.0, []int{100, 2, 1})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9)
	assert.InDelta(t, 0.505050, ceilings[1], 1e-5, "priority 2 should be close to cMin")
	assert.InDelta(t, 0.5, ceilings[2], 1e-9)
}

func TestLinearProportional_DuplicatePriorities(t *testing.T) {
	t.Parallel()
	// All same value: pRange = 0, should return cMax for all.
	ceilings := computeLimitLinearProportional(0.5, 1.0, []int{10, 10, 10})
	require.Len(t, ceilings, 3)
	for i, c := range ceilings {
		assert.InDelta(t, 1.0, c, 1e-9, "ceiling[%d] should be cMax when all priorities are equal", i)
	}
}

func TestLinearProportional_TwoDuplicatePriorities(t *testing.T) {
	t.Parallel()
	// Two same values at the bottom.
	ceilings := computeLimitLinearProportional(0.5, 1.0, []int{100, 10, 10})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9)
	assert.InDelta(t, 0.5, ceilings[1], 1e-9, "duplicate lowest priorities get cMin")
	assert.InDelta(t, 0.5, ceilings[2], 1e-9)
}

func TestLinearProportional_MonotonicallyDecreasing(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitLinearProportional(0.2, 0.9, []int{50, 40, 30, 20, 10})
	for i := 1; i < len(ceilings); i++ {
		assert.GreaterOrEqual(t, ceilings[i-1], ceilings[i],
			"ceiling[%d] (%f) should be >= ceiling[%d] (%f)", i-1, ceilings[i-1], i, ceilings[i])
	}
}

func TestLinearProportional_NegativePriorities(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitLinearProportional(0.5, 1.0, []int{10, -5, -20})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 1.0, ceilings[0], 1e-9, "highest priority gets cMax")
	assert.InDelta(t, 0.5, ceilings[2], 1e-9, "lowest priority gets cMin")
	assert.Greater(t, ceilings[1], ceilings[2], "middle priority should be between cMin and cMax")
	assert.Less(t, ceilings[1], ceilings[0], "middle priority should be between cMin and cMax")
}

func TestLinearProportional_BoundaryValues(t *testing.T) {
	t.Parallel()
	ceilings := computeLimitLinearProportional(0.0, 1.0, []int{10, 5, 1})
	assert.InDelta(t, 1.0, ceilings[0], 1e-9, "highest priority gets cMax")
	assert.InDelta(t, 0.0, ceilings[len(ceilings)-1], 1e-9, "lowest priority gets cMin")
}

// ---------------------------------------------------------------------------
// Cross-strategy comparison
// ---------------------------------------------------------------------------

func TestStrategies_ConvergeOnEvenSpacing(t *testing.T) {
	t.Parallel()
	// When priorities are evenly distributed relative to their range, both strategies
	// should produce the same ceilings.
	priorities := []int{100, 55, 10}
	stepwise := computeLimitStepwiseSpread(0.5, 1.0, priorities)
	linear := computeLimitLinearProportional(0.5, 1.0, priorities)

	require.Len(t, stepwise, 3)
	require.Len(t, linear, 3)
	for i := range stepwise {
		assert.InDelta(t, stepwise[i], linear[i], 1e-9,
			"strategies should converge for evenly spaced priorities at index %d", i)
	}
}

func TestStrategies_DivergeOnSkewedSpacing(t *testing.T) {
	t.Parallel()
	// When priorities are skewed, linear should give the middle priority a different
	// ceiling than stepwise.
	priorities := []int{100, 2, 1}
	stepwise := computeLimitStepwiseSpread(0.5, 1.0, priorities)
	linear := computeLimitLinearProportional(0.5, 1.0, priorities)

	// Endpoints should be the same.
	assert.InDelta(t, stepwise[0], linear[0], 1e-9, "highest priority should match")
	assert.InDelta(t, stepwise[2], linear[2], 1e-9, "lowest priority should match")

	// Middle priority should differ: stepwise gives 0.75, linear gives ~0.505.
	assert.InDelta(t, 0.75, stepwise[1], 1e-9)
	assert.Less(t, linear[1], 0.51, "linear should give priority 2 a ceiling near cMin")
	assert.Greater(t, stepwise[1]-linear[1], 0.2, "strategies should meaningfully diverge")
}

// ---------------------------------------------------------------------------
// Config validation tests
// ---------------------------------------------------------------------------

func TestBuildConfig_RequiredFields(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		cfg     apiConfig
		wantErr string
	}{
		{
			name:    "both missing",
			cfg:     apiConfig{},
			wantErr: "strategy is required",
		},
		{
			name:    "strategy missing",
			cfg:     apiConfig{MinCeiling: ptrFloat(0.5)},
			wantErr: "strategy is required",
		},
		{
			name:    "minCeiling missing",
			cfg:     apiConfig{Strategy: ptrStr("stepwise-spread")},
			wantErr: "minCeiling is required",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := buildConfig(&tc.cfg)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestBuildConfig_ValidationAggregatesErrors(t *testing.T) {
	t.Parallel()
	// Both required fields missing should report both errors.
	_, err := buildConfig(&apiConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "strategy is required")
	assert.Contains(t, err.Error(), "minCeiling is required")
}

func TestBuildConfig_ValidConfigs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		cfg        apiConfig
		wantMin    float64
		wantMax    float64
		wantStrat  string
	}{
		{
			name:      "stepwise with defaults",
			cfg:       apiConfig{Strategy: ptrStr("stepwise-spread"), MinCeiling: ptrFloat(0.5)},
			wantMin:   0.5,
			wantMax:   1.0,
			wantStrat: "stepwise-spread",
		},
		{
			name:      "linear with explicit maxCeiling",
			cfg:       apiConfig{Strategy: ptrStr("linear-proportional"), MinCeiling: ptrFloat(0.3), MaxCeiling: ptrFloat(0.9)},
			wantMin:   0.3,
			wantMax:   0.9,
			wantStrat: "linear-proportional",
		},
		{
			name:      "minCeiling at zero",
			cfg:       apiConfig{Strategy: ptrStr("stepwise-spread"), MinCeiling: ptrFloat(0.0)},
			wantMin:   0.0,
			wantMax:   1.0,
			wantStrat: "stepwise-spread",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg, err := buildConfig(&tc.cfg)
			require.NoError(t, err)
			assert.Equal(t, tc.wantStrat, cfg.strategy)
			assert.Equal(t, tc.wantMin, cfg.minCeiling)
			assert.Equal(t, tc.wantMax, cfg.maxCeiling)
		})
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func ptrStr(s string) *string     { return &s }
func ptrFloat(f float64) *float64 { return &f }
