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

package mrlscorer

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/caio/go-tdigest/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsojourn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/sojourntime"
)

// warmSnapshot builds a snapshot from a list of TTFT samples and decode samples.
// Both digests get the same number of samples for simplicity.
func warmSnapshot(t *testing.T, ttftSamples, decodeSamples []float64) *attrsojourn.SojournEstimatorSnapshot {
	t.Helper()
	ttft, err := tdigest.New(tdigest.Compression(500))
	require.NoError(t, err)
	for _, s := range ttftSamples {
		require.NoError(t, ttft.Add(s))
	}
	tb, err := ttft.AsBytes()
	require.NoError(t, err)

	decode, err := tdigest.New(tdigest.Compression(500))
	require.NoError(t, err)
	for _, s := range decodeSamples {
		require.NoError(t, decode.Add(s))
	}
	db, err := decode.AsBytes()
	require.NoError(t, err)

	return &attrsojourn.SojournEstimatorSnapshot{TtftDigest: tb, DecodeDigest: db}
}

// newEndpoint builds a scheduling endpoint named id and seeds the AttributeMap
// with the sojourn snapshot and the in-flight request list under the scorer's
// default DataKeys. Nil snap or nil inflight leaves that attribute absent.
func newEndpoint(id string, snap *attrsojourn.SojournEstimatorSnapshot, inflight []attrsojourn.InFlightRequest) fwksched.Endpoint {
	attr := fwkdl.NewAttributes()
	if snap != nil {
		attr.Put(attrsojourn.SojournEstimatorSnapshotDataKey, snap)
	}
	if inflight != nil {
		attr.Put(attrsojourn.InFlightRequestsDataKey, &attrsojourn.InFlightRequestsSnapshot{Requests: inflight})
	}
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: id, Namespace: "default"}}
	return fwksched.NewEndpoint(meta, nil, attr)
}

// TestScore_Ties verifies the two tie cases: cold endpoints (no snapshot
// in AttributeMap) and warm endpoints with no in-flight requests both
// land at residual 0, span 0, scores 1.0 across the board — the picker
// falls through to random-shuffle.
func TestScore_Ties(t *testing.T) {
	scorer := NewScorer(DefaultConfig)
	warm := warmSnapshot(t, []float64{1.0, 2.0, 3.0}, []float64{1.0, 2.0, 3.0})

	cases := []struct {
		name     string
		buildEps func() []fwksched.Endpoint
	}{
		{
			name: "cold",
			buildEps: func() []fwksched.Endpoint {
				return []fwksched.Endpoint{
					newEndpoint("a", nil, nil),
					newEndpoint("b", nil, nil),
				}
			},
		},
		{
			name: "warm_empty_inflight",
			buildEps: func() []fwksched.Endpoint {
				return []fwksched.Endpoint{
					newEndpoint("a", warm, []attrsojourn.InFlightRequest{}),
					newEndpoint("b", warm, []attrsojourn.InFlightRequest{}),
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			eps := tc.buildEps()
			scores := scorer.Score(context.Background(), nil, eps)
			for _, ep := range eps {
				assert.Equal(t, 1.0, scores[ep])
			}
		})
	}
}

// TestScore_ArgminFavoured verifies that when one endpoint has higher residual
// than the other, the lower one gets 1.0 and the higher one gets 0.0.
func TestScore_ArgminFavoured(t *testing.T) {
	scorer := NewScorer(DefaultConfig)

	// Both endpoints have the same warm snapshot (same distribution).
	snap := warmSnapshot(t,
		[]float64{0.1, 0.2, 0.5, 1.0, 2.0}, // TTFT samples
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0}, // decode samples
	)
	// heavy has two pre-first-chunk requests; light has none.
	heavyInflight := []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
		{DispatchedAt: time.Now()},
	}
	epLight := newEndpoint("light", snap, []attrsojourn.InFlightRequest{})
	epHeavy := newEndpoint("heavy", snap, heavyInflight)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{epLight, epHeavy})
	assert.Equal(t, 1.0, scores[epLight], "empty in-flight -> residual 0 -> wins argmin")
	assert.Equal(t, 0.0, scores[epHeavy], "two in-flight requests -> higher residual -> loses")
}

// TestScore_TwoTermFormula verifies the residual for a pre-first-chunk
// request matches the hand-computed sum MrlTtft(age)+MeanDecode(). Clock
// is injected via scoreAt so the dispatch age is deterministic.
func TestScore_TwoTermFormula(t *testing.T) {
	scorer := NewScorer(DefaultConfig)

	// Narrow spread so MrlTtft and MeanDecode return stable values;
	// exact values irrelevant, the shape is what's asserted.
	ttftSamples := make([]float64, 500)
	decodeSamples := make([]float64, 500)
	for i := range ttftSamples {
		ttftSamples[i] = 1.0 + float64(i)*0.001
	}
	for i := range decodeSamples {
		decodeSamples[i] = 1.0 + float64(i)*0.001
	}
	snap := warmSnapshot(t, ttftSamples, decodeSamples)

	// Deterministic clock: one in-flight request dispatched 100ms before
	// now. The pre-first-chunk residual must equal MrlTtft(age)+MeanDecode.
	now := time.Unix(1_700_000_000, 0).UTC()
	ageSeconds := 0.1
	dispatchedAt := now.Add(-100 * time.Millisecond)
	inflight := []attrsojourn.InFlightRequest{{DispatchedAt: dispatchedAt}}
	ep := newEndpoint("a", snap, inflight)

	residual := scorer.residualFor(ep, now)
	expected := snap.MrlTtft(ageSeconds) + snap.MeanDecode()
	assert.InDelta(t, expected, residual, 1e-9,
		"pre-first-chunk residual must equal MrlTtft(age)+MeanDecode()")

	// With one endpoint, scoreAt normalization ties to 1.0.
	scores, _, _, _ := scorer.scoreAt(now, []fwksched.Endpoint{ep})
	assert.Equal(t, 1.0, scores[ep], "single-candidate score is always 1.0")

	// Two-endpoint call exposes the argmin span: the empty endpoint has
	// residual 0, the one with the pre-first-chunk request has the
	// expected two-term residual.
	epEmpty := newEndpoint("b", snap, []attrsojourn.InFlightRequest{})
	scores, _, _, _ = scorer.scoreAt(now, []fwksched.Endpoint{ep, epEmpty})
	assert.Equal(t, 0.0, scores[ep], "endpoint with one pre-first-chunk request loses")
	assert.Equal(t, 1.0, scores[epEmpty], "empty endpoint wins")
}

// TestScore_PostFirstChunk verifies the single-term residual for a request
// that has emitted its first chunk: residual = MrlDecode(ageFromFirstChunk).
func TestScore_PostFirstChunk(t *testing.T) {
	scorer := NewScorer(DefaultConfig)
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)

	now := time.Unix(1_700_000_000, 0).UTC()
	ageSeconds := 0.25
	firstChunkAt := now.Add(-250 * time.Millisecond)
	// DispatchedAt is set to an earlier time to make clear it is NOT used
	// once FirstChunkAt is non-zero.
	dispatchedAt := firstChunkAt.Add(-500 * time.Millisecond)
	inflight := []attrsojourn.InFlightRequest{{
		DispatchedAt: dispatchedAt,
		FirstChunkAt: firstChunkAt,
	}}
	ep := newEndpoint("a", snap, inflight)

	residual := scorer.residualFor(ep, now)
	expected := snap.MrlDecode(ageSeconds)
	assert.InDelta(t, expected, residual, 1e-9,
		"post-first-chunk residual must equal MrlDecode(age) alone")
}

// TestScore_MixedInFlight exercises the realistic case of one endpoint
// holding both a pre-first-chunk and a post-first-chunk request. The
// residual is the sum of both branches.
func TestScore_MixedInFlight(t *testing.T) {
	scorer := NewScorer(DefaultConfig)
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)

	now := time.Unix(1_700_000_000, 0).UTC()
	agePreSeconds := 0.1
	ageDecodeSeconds := 0.25
	inflight := []attrsojourn.InFlightRequest{
		// Pre-first-chunk: contributes MrlTtft(agePre) + MeanDecode.
		{DispatchedAt: now.Add(-100 * time.Millisecond)},
		// Post-first-chunk: contributes MrlDecode(ageDecode) alone.
		{
			DispatchedAt: now.Add(-800 * time.Millisecond),
			FirstChunkAt: now.Add(-250 * time.Millisecond),
		},
	}
	ep := newEndpoint("a", snap, inflight)

	residual := scorer.residualFor(ep, now)
	expected := snap.MrlTtft(agePreSeconds) + snap.MeanDecode() + snap.MrlDecode(ageDecodeSeconds)
	assert.InDelta(t, expected, residual, 1e-9,
		"mixed-phase residual must sum the two-term branch and the decode branch")
}

// wrongType is a Cloneable value of a type unrelated to the snapshot type.
// Used by TestReadSnapshot_WrongType to drive the type-assert !ok branch.
type wrongType struct{ v int }

func (w *wrongType) Clone() fwkdl.Cloneable { return &wrongType{v: w.v} }

// TestReadSnapshot_WrongType guards the type-assert !ok branch in
// readSnapshot. If something other than *SojournEstimatorSnapshot is put
// under the DataKey, readSnapshot must return nil (not panic, not coerce).
func TestReadSnapshot_WrongType(t *testing.T) {
	scorer := NewScorer(DefaultConfig)

	attr := fwkdl.NewAttributes()
	// Put a Cloneable value of unrelated type under the snapshot key.
	attr.Put(attrsojourn.SojournEstimatorSnapshotDataKey, &wrongType{v: 42})
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "a", Namespace: "default"}}
	ep := fwksched.NewEndpoint(meta, nil, attr)

	assert.Nil(t, scorer.readSnapshot(ep),
		"readSnapshot must return nil on type-assert mismatch")
}

// TestScore_NilEndpointInSlice guards against a nil entry in the endpoints
// slice: scoreAt must not panic, and the result map must contain only
// non-nil endpoint keys so downstream pickers can deref every key safely.
func TestScore_NilEndpointInSlice(t *testing.T) {
	scorer := NewScorer(DefaultConfig)
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)
	// Warm endpoint with one pre-first-chunk request (positive residual)
	// so the span is non-zero and normalization is exercised.
	warmEp := newEndpoint("warm", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
	})

	now := time.Unix(1_700_000_000, 0).UTC()
	var scores map[fwksched.Endpoint]float64
	assert.NotPanics(t, func() {
		scores, _, _, _ = scorer.scoreAt(now, []fwksched.Endpoint{nil, warmEp})
	}, "scoreAt must tolerate a nil endpoint in the slice")

	_, ok := scores[warmEp]
	assert.True(t, ok, "warm endpoint must appear in the result map")

	_, hasNil := scores[nil]
	assert.False(t, hasNil, "nil endpoint must not be a key in the result map")
}

// noExplorationConfig is DefaultConfig with the exploration coin disabled.
// Used by the mixed-warm-and-cold tests so the deterministic seed-at-minR
// behavior can be asserted without randomness.
var noExplorationConfig = Config{ExplorationRate: 0}

// TestScore_MixedWarmAndCold_NoExploration asserts the primary C-2 fix:
// a cold endpoint ties with — rather than beats — the least-loaded warm
// endpoint. With exploration disabled, cold is seeded to minR; one warm
// and one cold candidate yield minR == maxR, so both score 1.0.
func TestScore_MixedWarmAndCold_NoExploration(t *testing.T) {
	scorer := NewScorer(noExplorationConfig)
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)
	warm := newEndpoint("warm", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
	})
	cold := newEndpoint("cold", nil, nil) // no snapshot => cold

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{warm, cold})
	assert.Equal(t, 1.0, scores[warm], "warm endpoint scores 1.0 (sole warm, defines minR = maxR)")
	assert.Equal(t, 1.0, scores[cold], "cold endpoint ties warm after seeding to minR")
}

// TestScore_MixedWarmAndCold_SpansToTie asserts that cold endpoints
// never widen the normalization span. Two warm endpoints with distinct
// residuals plus one cold candidate: the warm-only min/max defines the
// span, cold is seeded to minR and ties with the lowest-residual warm.
func TestScore_MixedWarmAndCold_SpansToTie(t *testing.T) {
	scorer := NewScorer(noExplorationConfig)
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)
	warmLow := newEndpoint("low", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
	})
	warmHigh := newEndpoint("high", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
		{DispatchedAt: time.Now()},
		{DispatchedAt: time.Now()},
	})
	cold := newEndpoint("cold", nil, nil)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{warmLow, warmHigh, cold})
	assert.Equal(t, 1.0, scores[warmLow], "least-loaded warm scores 1.0")
	assert.Equal(t, 0.0, scores[warmHigh], "heaviest warm scores 0 across the span")
	assert.Equal(t, 1.0, scores[cold], "cold is seeded to minR = warmLow.residual, ties warmLow")
}

// TestScore_ExplorationCoin_DeterministicProbe asserts the coin always
// fires at explorationRate=1.0. Cold endpoints are overridden to 1.0
// regardless of the normalization; warm endpoints are not touched.
func TestScore_ExplorationCoin_DeterministicProbe(t *testing.T) {
	scorer := NewScorer(Config{ExplorationRate: 1.0})
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)
	warm := newEndpoint("warm", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
	})
	cold := newEndpoint("cold", nil, nil)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{warm, cold})
	assert.Equal(t, 1.0, scores[cold], "coin at rate 1.0 forces cold score to 1.0")
	// One warm + one cold-seeded-to-minR gives minR == maxR, so warm
	// ties at 1.0; pinned to catch an accidental coin override on warm.
	assert.Equal(t, 1.0, scores[warm], "warm's normalization score is not overridden by the coin")
}

// TestScore_ExplorationCoin_DeterministicNoProbe asserts explorationRate=0
// disables the coin entirely. Cold's final score comes from the
// seed-at-minR path only; same shape as the no-exploration test.
func TestScore_ExplorationCoin_DeterministicNoProbe(t *testing.T) {
	scorer := NewScorer(Config{ExplorationRate: 0})
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)
	warm := newEndpoint("warm", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
	})
	cold := newEndpoint("cold", nil, nil)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{warm, cold})
	assert.Equal(t, 1.0, scores[warm])
	assert.Equal(t, 1.0, scores[cold], "coin at rate 0 leaves cold at the seed-at-minR score")
}

// TestScore_ColdDropsToZero_WhenCoinFiresAndWarmSpanExists asserts the
// coin's non-probe branch drives cold to 0 when maxR > minR. 100 trials
// at p=0.5 miss either branch with probability ~2 * 0.5^100.
func TestScore_ColdDropsToZero_WhenCoinFiresAndWarmSpanExists(t *testing.T) {
	scorer := NewScorer(Config{ExplorationRate: 0.5})
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)
	// Two warm endpoints with distinct in-flight counts so maxR > minR.
	warmLow := newEndpoint("low", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
	})
	warmHigh := newEndpoint("high", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
		{DispatchedAt: time.Now()},
		{DispatchedAt: time.Now()},
	})
	cold := newEndpoint("cold", nil, nil)

	sawZero, sawOne := false, false
	for i := 0; i < 100; i++ {
		scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{warmLow, warmHigh, cold})
		coldScore := scores[cold]
		switch coldScore {
		case 0.0:
			sawZero = true
		case 1.0:
			sawOne = true
		default:
			t.Fatalf("cold score must be 0 or 1 under coin, got %v on iter %d", coldScore, i)
		}
	}
	assert.True(t, sawZero, "coin non-probe branch must drive cold to 0 at least once over 100 trials")
	assert.True(t, sawOne, "coin probe branch must drive cold to 1.0 at least once over 100 trials")
}

// TestScore_ColdStaysAtSeed_WhenWarmSpanIsZero asserts that when there is
// no warm-side ranking to drop cold out of (maxR == minR, e.g. a single
// warm endpoint), the exploration coin never drives cold to 0. The score
// stays at the seeded tie regardless of coin outcome because the probe
// branch and the seed-tie both yield 1.0.
func TestScore_ColdStaysAtSeed_WhenWarmSpanIsZero(t *testing.T) {
	scorer := NewScorer(Config{ExplorationRate: 0.5})
	snap := warmSnapshot(t,
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
	)
	warm := newEndpoint("warm", snap, []attrsojourn.InFlightRequest{
		{DispatchedAt: time.Now()},
	})
	cold := newEndpoint("cold", nil, nil)

	for i := 0; i < 50; i++ {
		scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{warm, cold})
		assert.Equal(t, 1.0, scores[cold], "cold must stay at seeded tie when warm span is zero (iter %d)", i)
	}
}

// TestScore_ConsumesDataKeys confirms Consumes declares both the snapshot
// and the in-flight-requests DataKey as Required, so the plugin DAG orders
// the producer's construction ahead of the scorer's.
func TestScore_ConsumesDataKeys(t *testing.T) {
	scorer := NewScorer(DefaultConfig)
	deps := scorer.Consumes()
	require.Len(t, deps.Required, 2)
	_, hasSnap := deps.Required[attrsojourn.SojournEstimatorSnapshotDataKey]
	assert.True(t, hasSnap, "SojournEstimatorSnapshotDataKey must be a Required consumption")
	_, hasInflight := deps.Required[attrsojourn.InFlightRequestsDataKey]
	assert.True(t, hasInflight, "InFlightRequestsDataKey must be a Required consumption")
}

// TestScorerFactory covers the scorer's factory: it must be handle-free
// (so the DAG can order it before the producer), decode the configured
// producer-name override, surface malformed JSON as an error, and expose
// the Distribution category.
func TestScorerFactory(t *testing.T) {
	t.Run("handle-free with default config", func(t *testing.T) {
		p, err := ScorerFactory("mrl", nil, nil)
		require.NoError(t, err)
		s, ok := p.(*Scorer)
		require.True(t, ok)
		assert.Equal(t, ScorerType, s.TypedName().Type)
		assert.Equal(t, "mrl", s.TypedName().Name)
		assert.Equal(t, fwksched.Distribution, s.Category())
	})

	t.Run("producer name override selects a different key", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(
			`{"sojournTimeObserverProducerName":"obs-a"}`))
		p, err := ScorerFactory("mrl", params, nil)
		require.NoError(t, err)
		s := p.(*Scorer)

		assert.Contains(t, s.snapshotDataKey.String(), "obs-a")
		assert.Contains(t, s.inFlightRequestsDataKey.String(), "obs-a")
	})

	t.Run("malformed JSON is wrapped with the plugin name", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{`))
		_, err := ScorerFactory("mrl", params, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "mrl",
			"error must be wrapped with the plugin name")
	})

	t.Run("explorationRate override is wired to the Scorer", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{"explorationRate":0.25}`))
		p, err := ScorerFactory("mrl", params, nil)
		require.NoError(t, err)
		assert.InDelta(t, 0.25, p.(*Scorer).explorationRate, 1e-9)
	})

	t.Run("explorationRate above 1 is rejected", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{"explorationRate":1.5}`))
		_, err := ScorerFactory("mrl", params, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "mrl",
			"error must be wrapped with the plugin name")
	})

	t.Run("explorationRate below 0 is rejected", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{"explorationRate":-0.1}`))
		_, err := ScorerFactory("mrl", params, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "mrl",
			"error must be wrapped with the plugin name")
	})
}
