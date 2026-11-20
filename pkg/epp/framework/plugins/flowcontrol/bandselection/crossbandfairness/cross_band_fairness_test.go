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

package crossbandfairness

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
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
			name:   "defaults applied for schedule and distribution",
			config: []byte(`{"activationThreshold": 0.8}`),
		},
		{
			name:   "explicit schedule interleaved",
			config: []byte(`{"activationThreshold": 0.8, "schedule": "interleaved"}`),
		},
		{
			name:   "explicit distribution with ratios",
			config: []byte(`{"activationThreshold": 0.8, "distribution": "explicit", "ratios": {"100": 3, "10": 1}}`),
		},
		{
			name:   "activationThreshold zero is valid",
			config: []byte(`{"activationThreshold": 0.0}`),
		},
		{
			name:      "missing activationThreshold",
			config:    []byte(`{"distribution": "linear"}`),
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
			name:      "activationThreshold negative",
			config:    []byte(`{"activationThreshold": -0.1}`),
			wantError: true,
		},
		{
			name:      "activationThreshold equals 1.0",
			config:    []byte(`{"activationThreshold": 1.0}`),
			wantError: true,
		},
		{
			name:      "unsupported schedule",
			config:    []byte(`{"activationThreshold": 0.8, "schedule": "priority-first"}`),
			wantError: true,
		},
		{
			name:      "unsupported distribution",
			config:    []byte(`{"activationThreshold": 0.8, "distribution": "proportional"}`),
			wantError: true,
		},
		{
			name:      "explicit distribution without ratios",
			config:    []byte(`{"activationThreshold": 0.8, "distribution": "explicit"}`),
			wantError: true,
		},
		{
			name:      "ratios set with linear distribution",
			config:    []byte(`{"activationThreshold": 0.8, "ratios": {"100": 3}}`),
			wantError: true,
		},
		{
			name:      "zero ratio",
			config:    []byte(`{"activationThreshold": 0.8, "distribution": "explicit", "ratios": {"100": 3, "10": 0}}`),
			wantError: true,
		},
		{
			name:      "negative ratio",
			config:    []byte(`{"activationThreshold": 0.8, "distribution": "explicit", "ratios": {"100": -1}}`),
			wantError: true,
		},
		{
			name:      "invalid JSON",
			config:    []byte(`{not json}`),
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
				return
			}
			require.NoError(t, err)
			require.NotNil(t, p)
			require.Implements(t, (*flowcontrol.BandSelectionPolicy)(nil), p)
		})
	}
}

func TestPolicyFactory_TypedName(t *testing.T) {
	t.Parallel()

	p, err := PolicyFactory("my-policy", fwkplugin.StrictDecoder([]byte(`{"activationThreshold": 0.8}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, fwkplugin.TypedName{Type: PolicyType, Name: "my-policy"}, p.TypedName())

	unnamed, err := PolicyFactory("", fwkplugin.StrictDecoder([]byte(`{"activationThreshold": 0.8}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, PolicyType, unnamed.TypedName().Name, "an empty name should fall back to the policy type")
}

// ---------------------------------------------------------------------------
// Dispatch simulation
// ---------------------------------------------------------------------------

// sim reproduces the processor's dispatch cycle: it pre-fills the order buffer with the identity
// permutation, asks the policy to rank, then walks the ranking and dispatches from the first band
// that is ungated and has work.
type sim struct {
	t          *testing.T
	policy     *crossBandFairnessPolicy
	priorities []int
	ceilings   []float64
	saturation float64
	busy       map[int]bool
	counts     map[int]int
	lastOrder  []int
}

func newSim(t *testing.T, cfg string, priorities ...int) *sim {
	t.Helper()
	p, err := PolicyFactory("test", fwkplugin.StrictDecoder([]byte(cfg)), nil)
	require.NoError(t, err)

	s := &sim{
		t:          t,
		policy:     p.(*crossBandFairnessPolicy),
		priorities: priorities,
		ceilings:   make([]float64, len(priorities)),
		busy:       make(map[int]bool, len(priorities)),
		counts:     make(map[int]int, len(priorities)),
	}
	for i := range s.ceilings {
		s.ceilings[i] = 1.0
	}
	for _, priority := range priorities {
		s.busy[priority] = true
	}
	return s
}

// dispatch runs one cycle and returns the priority that dispatched, or -1 when none did.
func (s *sim) dispatch() int {
	s.t.Helper()

	order := make([]int, len(s.priorities))
	for i := range order {
		order[i] = i
	}
	s.policy.Rank(context.Background(), flowcontrol.BandSelectionParameters{
		Saturation: s.saturation,
		Priorities: s.priorities,
		Ceilings:   s.ceilings,
	}, order)
	s.lastOrder = order

	for _, i := range order {
		if s.saturation >= s.ceilings[i] {
			continue
		}
		priority := s.priorities[i]
		if !s.busy[priority] {
			continue
		}
		s.counts[priority]++
		s.policy.RecordDispatch(context.Background(), priority)
		return priority
	}
	return -1
}

func (s *sim) dispatchN(n int) {
	s.t.Helper()
	for range n {
		s.dispatch()
	}
}

// sequence runs n cycles and reports the priority that dispatched on each.
func (s *sim) sequence(n int) []int {
	s.t.Helper()
	seq := make([]int, n)
	for i := range seq {
		seq[i] = s.dispatch()
	}
	return seq
}

// longestRun reports the length of the longest unbroken run of dispatches by one priority.
func longestRun(seq []int, priority int) int {
	longest, run := 0, 0
	for _, p := range seq {
		if p != priority {
			run = 0
			continue
		}
		run++
		longest = max(longest, run)
	}
	return longest
}

// gate marks the band at the given priority as held back by the usage limit policy.
func (s *sim) gate(priority int) {
	s.t.Helper()
	for i, p := range s.priorities {
		if p == priority {
			s.ceilings[i] = 0.0
			return
		}
	}
	s.t.Fatalf("priority %d is not active", priority)
}

func (s *sim) ungate(priority int) {
	s.t.Helper()
	for i, p := range s.priorities {
		if p == priority {
			s.ceilings[i] = 1.0
			return
		}
	}
	s.t.Fatalf("priority %d is not active", priority)
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

// Below the activation threshold the policy must behave exactly like strict band selection, and must
// not touch its scheduling state.
func TestRank_BelowThresholdDispatchesStrictly(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 50, 10)
	s.saturation = 0.79

	s.dispatchN(20)

	assert.Equal(t, []int{0, 1, 2}, s.lastOrder, "order must stay the identity permutation")
	assert.Equal(t, map[int]int{100: 20}, s.counts, "the highest band must take every opportunity")
	assert.Empty(t, s.policy.bands, "scheduling state must not be built while inactive")
}

// A single active band leaves nothing to share, so the policy stays inert.
func TestRank_SingleBandDispatchesStrictly(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100)
	s.saturation = 0.95

	s.dispatchN(5)

	assert.Equal(t, []int{0}, s.lastOrder)
	assert.Empty(t, s.policy.bands)
}

// Saturation crossing back and forth over the threshold must resume the round in progress rather
// than restart it: virtual times are retained while the policy is inactive.
func TestRank_VirtualTimeSurvivesDeactivation(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.5}`, 100, 10)
	s.saturation = 0.9

	require.Equal(t, 100, s.dispatch(), "the first opportunity breaks the virtual time tie by priority")

	s.saturation = 0.1
	require.Equal(t, 100, s.dispatch(), "strict order applies below the threshold")
	require.Equal(t, 100, s.dispatch())

	s.saturation = 0.9
	assert.Equal(t, 10, s.dispatch(), "the lower band keeps the claim it held before deactivation")
}

// ---------------------------------------------------------------------------
// Share convergence
// ---------------------------------------------------------------------------

// Bands with sustained demand must converge on the share their weights describe.
func TestRank_LinearDistributionConvergesOnWeightedShares(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 50, 10)
	s.saturation = 0.9

	// Weights are 3, 2 and 1 by ordinal position, so 6000 opportunities split 3000/2000/1000.
	s.dispatchN(6000)

	assert.InDelta(t, 3000, s.counts[100], 2)
	assert.InDelta(t, 2000, s.counts[50], 2)
	assert.InDelta(t, 1000, s.counts[10], 2)
}

func TestRank_ExplicitDistributionConvergesOnConfiguredRatios(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8, "distribution": "explicit", "ratios": {"100": 3, "10": 1}}`, 100, 10)
	s.saturation = 0.9

	s.dispatchN(4000)

	assert.InDelta(t, 3000, s.counts[100], 2)
	assert.InDelta(t, 1000, s.counts[10], 2)
}

// The first opportunity of a round finds every band at the same virtual time; priority breaks the tie.
func TestRank_TiesFavorHigherPriority(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 50, 10)
	s.saturation = 0.9

	assert.Equal(t, 100, s.dispatch())
	assert.Equal(t, []int{0, 1, 2}, s.lastOrder)
}

// ---------------------------------------------------------------------------
// Gating
// ---------------------------------------------------------------------------

// A gated band cannot dispatch, so it is ranked behind every band that can.
func TestRank_GatedBandsRankLast(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 50, 10)
	s.saturation = 0.9
	s.gate(50)
	s.gate(10)

	// Give the lowest ungated band the strongest claim, then confirm the gated bands still trail it.
	s.dispatchN(3)
	s.ungate(10)
	s.gate(100)
	s.dispatch()

	require.Len(t, s.lastOrder, 3)
	assert.Equal(t, 2, s.lastOrder[0], "the only ungated band must rank first")
}

// A band held back by the usage limit policy banks no claim while it waits: when it is released it
// takes its turn without a burst of catch-up dispatches.
func TestRank_RejoiningBandDoesNotBurst(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 10)
	s.saturation = 0.9
	s.gate(10)

	s.dispatchN(500)
	require.Equal(t, 500, s.counts[100])
	require.Zero(t, s.counts[10])

	s.ungate(10)
	seq := s.sequence(12)
	assert.Contains(t, seq, 10, "a released band must be offered an opportunity promptly")
	assert.LessOrEqual(t, longestRun(seq, 10), 2, "a released band must not monopolize dispatch to catch up")
}

// ---------------------------------------------------------------------------
// Occupancy
// ---------------------------------------------------------------------------

// An idle band is ranked first every cycle because nothing advances its virtual time. Forfeiting the
// opportunities it declines is what keeps it from monopolizing dispatch once its traffic returns.
func TestRecordDispatch_IdleBandForfeitsItsTurn(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 10)
	s.saturation = 0.9
	s.busy[10] = false

	s.dispatchN(500)
	require.Equal(t, 500, s.counts[100], "the busy band must keep dispatching past the idle one")

	s.busy[10] = true
	seq := s.sequence(12)
	assert.Contains(t, seq, 10, "a band returning from idle must be offered an opportunity promptly")
	assert.LessOrEqual(t, longestRun(seq, 10), 2, "a band returning from idle must not monopolize dispatch")
}

// A cycle that dispatches nothing leaves every claim untouched.
func TestRecordDispatch_NoDispatchLeavesStateUntouched(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 10)
	s.saturation = 0.9
	s.busy[100] = false
	s.busy[10] = false

	require.Equal(t, -1, s.dispatch())

	s.busy[100] = true
	s.busy[10] = true
	assert.Equal(t, 100, s.dispatch(), "the round must start where it would have without the empty cycle")
}

// ---------------------------------------------------------------------------
// Dynamic priority sets
// ---------------------------------------------------------------------------

// A band provisioned mid-round enters at the system virtual time: it is served promptly, then
// settles into its share.
func TestRank_NewBandEntersAtSystemVirtualTime(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 10)
	s.saturation = 0.9
	s.dispatchN(500)

	s.priorities = []int{100, 50, 10}
	s.ceilings = []float64{1.0, 1.0, 1.0}
	s.busy[50] = true

	seq := s.sequence(12)
	assert.Contains(t, seq, 50, "a new band must be offered an opportunity promptly")
	assert.LessOrEqual(t, longestRun(seq, 50), 2, "a new band must not claim the service it never deferred")
}

// A band that is garbage-collected must leave no state behind.
func TestRank_DepartedBandIsDropped(t *testing.T) {
	t.Parallel()

	s := newSim(t, `{"activationThreshold": 0.8}`, 100, 50, 10)
	s.saturation = 0.9
	s.dispatchN(10)
	require.Len(t, s.policy.bands, 3)

	s.priorities = []int{100, 10}
	s.ceilings = []float64{1.0, 1.0}
	s.dispatch()

	assert.Len(t, s.policy.bands, 2)
	assert.NotContains(t, s.policy.bands, 50)
}

// ---------------------------------------------------------------------------
// Weight distributions
// ---------------------------------------------------------------------------

func TestComputeWeightsLinear(t *testing.T) {
	t.Parallel()

	priorities := []int{100, 50, 10, -5}
	weights := make([]float64, len(priorities))
	computeWeightsLinear(priorities, weights)

	assert.Equal(t, []float64{4, 3, 2, 1}, weights)
}

func TestComputeWeightsExplicit(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		ratios     map[int]float64
		priorities []int
		want       []float64
	}{
		{
			name:       "configured anchors map directly",
			ratios:     map[int]float64{100: 8, 10: 2},
			priorities: []int{100, 10},
			want:       []float64{8, 2},
		},
		{
			name:       "an interior anchor maps to its configured weight",
			ratios:     map[int]float64{100: 9, 50: 3, 0: 1},
			priorities: []int{100, 50, 0},
			want:       []float64{9, 3, 1},
		},
		{
			name:       "unlisted priority interpolates by value",
			ratios:     map[int]float64{100: 10, 0: 0.5},
			priorities: []int{100, 50, 0},
			want:       []float64{10, 5.25, 0.5},
		},
		{
			name:       "the segment scan picks the anchors bracketing the priority",
			ratios:     map[int]float64{100: 9, 50: 3, 0: 1},
			priorities: []int{75, 25},
			want:       []float64{6, 2},
		},
		{
			name:       "priorities outside the anchor range clamp to the nearest anchor",
			ratios:     map[int]float64{80: 4, 20: 1},
			priorities: []int{100, 80, 20, -10},
			want:       []float64{4, 4, 1, 1},
		},
		{
			name:       "a single anchor applies to every priority",
			ratios:     map[int]float64{50: 7},
			priorities: []int{100, 50, 10},
			want:       []float64{7, 7, 7},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			weights := make([]float64, len(tc.priorities))
			computeWeightsExplicit(buildWeightAnchors(tc.ratios), tc.priorities, weights)
			assert.InDeltaSlice(t, tc.want, weights, 1e-9)
		})
	}
}

// A configured anchor maps to its weight exactly, with no interpolation arithmetic in the path.
// Reconstructing an interior anchor as w + 1.0*(w'-w) rounds to zero when adjacent weights differ by
// many orders of magnitude, and a zero weight would give the band an infinite stride and strand it.
// The ratios here are deliberately pathological: they pin the invariant that realistic inputs cannot
// exercise.
func TestComputeWeightsExplicit_AnchorsAreExact(t *testing.T) {
	t.Parallel()

	priorities := []int{100, 50, 0}
	weights := make([]float64, len(priorities))
	computeWeightsExplicit(buildWeightAnchors(map[int]float64{100: 2, 50: 1, 0: 1e17}), priorities, weights)

	assert.Equal(t, []float64{2, 1, 1e17}, weights)
}

// Shares follow the configured anchors, so a band appearing or disappearing leaves the others alone.
func TestRank_ExplicitSharesAreStableAcrossBandChurn(t *testing.T) {
	t.Parallel()

	cfg := `{"activationThreshold": 0.8, "distribution": "explicit", "ratios": {"100": 3, "10": 1}}`

	full := newSim(t, cfg, 100, 50, 10)
	full.saturation = 0.9
	full.dispatchN(2000)

	partial := newSim(t, cfg, 100, 10)
	partial.saturation = 0.9
	partial.dispatchN(2000)

	// The ratio between the surviving bands is set by their anchors, not by the band count.
	assert.InDelta(t,
		float64(full.counts[100])/float64(full.counts[10]),
		float64(partial.counts[100])/float64(partial.counts[10]),
		0.05)
}
