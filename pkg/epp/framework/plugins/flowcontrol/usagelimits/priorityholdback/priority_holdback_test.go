/*
Copyright 2026 The Kubernetes Authors.

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
			name:      "valid rank domain config",
			config:    []byte(`{"domain": "rank", "minCeiling": 0.5}`),
			wantError: false,
		},
		{
			name:      "valid value domain config",
			config:    []byte(`{"domain": "value", "minCeiling": 0.3, "maxCeiling": 0.9}`),
			wantError: false,
		},
		{
			name:      "valid config with explicit maxCeiling",
			config:    []byte(`{"domain": "rank", "minCeiling": 0.2, "maxCeiling": 0.8}`),
			wantError: false,
		},
		{
			name:      "defaults applied for shape and domain",
			config:    []byte(`{"minCeiling": 0.5}`),
			wantError: false,
		},
		{
			name:      "explicit shape linear",
			config:    []byte(`{"shape": "linear", "minCeiling": 0.5}`),
			wantError: false,
		},
		{
			name:      "missing minCeiling",
			config:    []byte(`{"domain": "rank"}`),
			wantError: true,
		},
		{
			name:      "empty config",
			config:    []byte(`{}`),
			wantError: true,
		},
		{
			name:      "nil config",
			config:    nil,
			wantError: true,
		},
		{
			name:      "unsupported shape",
			config:    []byte(`{"shape": "sigmoid", "minCeiling": 0.5}`),
			wantError: true,
		},
		{
			name:      "unsupported domain",
			config:    []byte(`{"domain": "unknown", "minCeiling": 0.5}`),
			wantError: true,
		},
		{
			name:      "minCeiling negative",
			config:    []byte(`{"minCeiling": -0.1}`),
			wantError: true,
		},
		{
			name:      "minCeiling equals 1.0",
			config:    []byte(`{"minCeiling": 1.0}`),
			wantError: true,
		},
		{
			name:      "maxCeiling zero",
			config:    []byte(`{"minCeiling": 0.5, "maxCeiling": 0.0}`),
			wantError: true,
		},
		{
			name:      "maxCeiling exceeds 1.0",
			config:    []byte(`{"minCeiling": 0.5, "maxCeiling": 1.1}`),
			wantError: true,
		},
		{
			name:      "minCeiling equals maxCeiling",
			config:    []byte(`{"minCeiling": 0.5, "maxCeiling": 0.5}`),
			wantError: true,
		},
		{
			name:      "minCeiling greater than maxCeiling",
			config:    []byte(`{"minCeiling": 0.8, "maxCeiling": 0.5}`),
			wantError: true,
		},
		{
			name:      "invalid JSON",
			config:    []byte(`{not json}`),
			wantError: true,
		},
		{
			name:      "minCeiling zero is valid",
			config:    []byte(`{"minCeiling": 0.0}`),
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
		[]byte(`{"minCeiling": 0.5}`)), nil)
	require.NoError(t, err)

	policy := p.(*priorityHoldbackPolicy)
	assert.Equal(t, 1.0, policy.cMax, "maxCeiling should default to 1.0")
}

func TestPolicyFactory_TypedName(t *testing.T) {
	t.Parallel()
	p, err := PolicyFactory("my-policy", fwkplugin.StrictDecoder(
		[]byte(`{"minCeiling": 0.5}`)), nil)
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
		shape:      shapeLinear,
		domain:     DomainRank,
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
		name   string
		domain string
		cMax   float64
	}{
		{"rank domain", DomainRank, 1.0},
		{"value domain", DomainValue, 1.0},
		{"custom maxCeiling", DomainRank, 0.9},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			policy := newPriorityHoldbackPolicy(config{
				shape:      shapeLinear,
				domain:     tc.domain,
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
// Stepwise-spread tests (domain: rank)
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
// Linear-proportional tests (domain: value)
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
// Cross-domain comparison
// ---------------------------------------------------------------------------

func TestDomains_ConvergeOnEvenSpacing(t *testing.T) {
	t.Parallel()
	// When priorities are evenly distributed relative to their range, both domains
	// should produce the same ceilings.
	priorities := []int{100, 55, 10}
	rank := computeLimitStepwiseSpread(0.5, 1.0, priorities)
	value := computeLimitLinearProportional(0.5, 1.0, priorities)

	require.Len(t, rank, 3)
	require.Len(t, value, 3)
	for i := range rank {
		assert.InDelta(t, rank[i], value[i], 1e-9,
			"domains should converge for evenly spaced priorities at index %d", i)
	}
}

func TestDomains_DivergeOnSkewedSpacing(t *testing.T) {
	t.Parallel()
	// When priorities are skewed, value domain should give the middle priority a different
	// ceiling than rank domain.
	priorities := []int{100, 2, 1}
	rank := computeLimitStepwiseSpread(0.5, 1.0, priorities)
	value := computeLimitLinearProportional(0.5, 1.0, priorities)

	// Endpoints should be the same.
	assert.InDelta(t, rank[0], value[0], 1e-9, "highest priority should match")
	assert.InDelta(t, rank[2], value[2], 1e-9, "lowest priority should match")

	// Middle priority should differ: rank gives 0.75, value gives ~0.505.
	assert.InDelta(t, 0.75, rank[1], 1e-9)
	assert.Less(t, value[1], 0.51, "value domain should give priority 2 a ceiling near cMin")
	assert.Greater(t, rank[1]-value[1], 0.2, "domains should meaningfully diverge")
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
			name:    "minCeiling missing",
			cfg:     apiConfig{},
			wantErr: "minCeiling is required",
		},
		{
			name:    "minCeiling missing with domain set",
			cfg:     apiConfig{Domain: ptrStr("rank")},
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

func TestBuildConfig_ValidConfigs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		cfg        apiConfig
		wantMin    float64
		wantMax    float64
		wantShape  string
		wantDomain string
	}{
		{
			name:       "defaults for shape and domain",
			cfg:        apiConfig{MinCeiling: ptrFloat(0.5)},
			wantMin:    0.5,
			wantMax:    1.0,
			wantShape:  "linear",
			wantDomain: "rank",
		},
		{
			name:       "explicit rank domain",
			cfg:        apiConfig{Domain: ptrStr("rank"), MinCeiling: ptrFloat(0.5)},
			wantMin:    0.5,
			wantMax:    1.0,
			wantShape:  "linear",
			wantDomain: "rank",
		},
		{
			name:       "value domain with explicit maxCeiling",
			cfg:        apiConfig{Domain: ptrStr("value"), MinCeiling: ptrFloat(0.3), MaxCeiling: ptrFloat(0.9)},
			wantMin:    0.3,
			wantMax:    0.9,
			wantShape:  "linear",
			wantDomain: "value",
		},
		{
			name:       "minCeiling at zero",
			cfg:        apiConfig{MinCeiling: ptrFloat(0.0)},
			wantMin:    0.0,
			wantMax:    1.0,
			wantShape:  "linear",
			wantDomain: "rank",
		},
		{
			name:       "explicit shape and domain",
			cfg:        apiConfig{Shape: ptrStr("linear"), Domain: ptrStr("value"), MinCeiling: ptrFloat(0.4)},
			wantMin:    0.4,
			wantMax:    1.0,
			wantShape:  "linear",
			wantDomain: "value",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg, err := buildConfig(&tc.cfg)
			require.NoError(t, err)
			assert.Equal(t, tc.wantShape, cfg.shape)
			assert.Equal(t, tc.wantDomain, cfg.domain)
			assert.Equal(t, tc.wantMin, cfg.minCeiling)
			assert.Equal(t, tc.wantMax, cfg.maxCeiling)
		})
	}
}

// ---------------------------------------------------------------------------
// Explicit domain — factory / config tests
// ---------------------------------------------------------------------------

func TestPolicyFactory_ExplicitDomain(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		config    []byte
		wantError bool
	}{
		{
			name:      "valid explicit domain config",
			config:    []byte(`{"domain":"explicit","ceilings":{"100":0.95,"50":0.70,"10":0.30}}`),
			wantError: false,
		},
		{
			name:      "valid explicit domain single entry",
			config:    []byte(`{"domain":"explicit","ceilings":{"100":0.80}}`),
			wantError: false,
		},
		{
			name:      "valid explicit domain monotonic equal ceilings",
			config:    []byte(`{"domain":"explicit","ceilings":{"100":0.80,"50":0.80}}`),
			wantError: false,
		},
		{
			name:      "explicit domain missing ceilings",
			config:    []byte(`{"domain":"explicit"}`),
			wantError: true,
		},
		{
			name:      "explicit domain with shape set",
			config:    []byte(`{"domain":"explicit","shape":"linear","ceilings":{"100":0.9,"10":0.3}}`),
			wantError: true,
		},
		{
			name:      "explicit domain with minCeiling set",
			config:    []byte(`{"domain":"explicit","minCeiling":0.3,"ceilings":{"100":0.9,"10":0.3}}`),
			wantError: true,
		},
		{
			name:      "explicit domain with maxCeiling set",
			config:    []byte(`{"domain":"explicit","maxCeiling":0.9,"ceilings":{"100":0.9,"10":0.3}}`),
			wantError: true,
		},
		{
			name:      "explicit domain ceiling above 1.0",
			config:    []byte(`{"domain":"explicit","ceilings":{"100":1.1,"10":0.3}}`),
			wantError: true,
		},
		{
			name:      "explicit domain ceiling below 0.0",
			config:    []byte(`{"domain":"explicit","ceilings":{"100":0.9,"10":-0.1}}`),
			wantError: true,
		},
		{
			name:      "explicit domain non-monotonic ceilings",
			config:    []byte(`{"domain":"explicit","ceilings":{"100":0.30,"50":0.70}}`),
			wantError: true,
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

// ---------------------------------------------------------------------------
// Explicit domain — computeLimitExplicit unit tests
// ---------------------------------------------------------------------------

func TestComputeLimitExplicit_TwoPriorities(t *testing.T) {
	t.Parallel()
	m := map[int]float64{100: 0.95, 10: 0.30}
	ceilings := computeLimitExplicit(m, []int{100, 10})
	require.Len(t, ceilings, 2)
	assert.InDelta(t, 0.95, ceilings[0], 1e-9)
	assert.InDelta(t, 0.30, ceilings[1], 1e-9)
}

func TestComputeLimitExplicit_ThreePriorities(t *testing.T) {
	t.Parallel()
	m := map[int]float64{100: 0.95, 50: 0.70, 10: 0.30}
	ceilings := computeLimitExplicit(m, []int{100, 50, 10})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 0.95, ceilings[0], 1e-9)
	assert.InDelta(t, 0.70, ceilings[1], 1e-9)
	assert.InDelta(t, 0.30, ceilings[2], 1e-9)
}

func TestComputeLimitExplicit_SinglePriority(t *testing.T) {
	t.Parallel()
	m := map[int]float64{100: 0.80}
	ceilings := computeLimitExplicit(m, []int{100})
	require.Len(t, ceilings, 1)
	assert.InDelta(t, 0.80, ceilings[0], 1e-9)
}

// ---------------------------------------------------------------------------
// Explicit domain — ComputeLimit integration tests
// ---------------------------------------------------------------------------

func TestComputeLimit_ExplicitDomain_SinglePriority(t *testing.T) {
	t.Parallel()
	// Explicit domain does not apply the single-priority bypass; it uses the mapped ceiling.
	policy := newPriorityHoldbackPolicy(config{
		domain:   DomainExplicit,
		ceilings: map[int]float64{100: 0.80},
	})

	ceilings := policy.ComputeLimit(t.Context(), 0.5, []int{100})
	require.Len(t, ceilings, 1)
	assert.InDelta(t, 0.80, ceilings[0], 1e-9,
		"explicit domain single priority should return the configured ceiling, not cMax")
}

func TestComputeLimit_ExplicitDomain_MultiplePriorities(t *testing.T) {
	t.Parallel()
	policy := newPriorityHoldbackPolicy(config{
		domain:   DomainExplicit,
		ceilings: map[int]float64{100: 0.95, 50: 0.70, 10: 0.30},
	})

	ceilings := policy.ComputeLimit(t.Context(), 0.5, []int{100, 50, 10})
	require.Len(t, ceilings, 3)
	assert.InDelta(t, 0.95, ceilings[0], 1e-9)
	assert.InDelta(t, 0.70, ceilings[1], 1e-9)
	assert.InDelta(t, 0.30, ceilings[2], 1e-9)
}

func TestComputeLimit_ExplicitDomain_EmptyPriorities(t *testing.T) {
	t.Parallel()
	policy := newPriorityHoldbackPolicy(config{
		domain:   DomainExplicit,
		ceilings: map[int]float64{100: 0.95},
	})

	ceilings := policy.ComputeLimit(t.Context(), 0.5, []int{})
	assert.Empty(t, ceilings)
	assert.NotNil(t, ceilings, "should return empty slice, not nil")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func ptrStr(s string) *string     { return &s }
func ptrFloat(f float64) *float64 { return &f }
