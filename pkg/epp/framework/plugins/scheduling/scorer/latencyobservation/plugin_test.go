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

package latencyobservation

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

type pct = attrlatency.TTFTPercentiles

// trusted returns a snapshot that passes every admissibility check:
// floor 0.20, B at (4, 0.35), C at (9, 0.55).
func trusted() *pct {
	return &attrlatency.TTFTPercentiles{
		FloorTTFT:             0.20,
		LowLoadTTFT:           0.35,
		TypicalLoadTTFT:       0.55,
		InflightAtLowLoad:     4,
		InflightAtTypicalLoad: 9,
		RecentRequestCount:    100,
		Observations:          100,
		CalibrationThreshold:  10,
	}
}

// with applies mutations to a copy of the trusted snapshot.
func with(mutate func(*pct)) *pct {
	m := trusted()
	mutate(m)
	return m
}

func newEndpoint(percentiles *pct, inflight *int64) fwksched.Endpoint {
	attr := fwkdl.NewAttributes()
	if percentiles != nil {
		attr.Put(attrlatency.TTFTPercentilesDataKey, percentiles)
	}
	if inflight != nil {
		attr.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: *inflight})
	}
	return fwksched.NewEndpoint(nil, nil, attr)
}

func inflightPtr(v int64) *int64 { return &v }

func TestScorerFactory(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		p, err := ScorerFactory("ttft", nil, nil)
		require.NoError(t, err)
		s, ok := p.(*Scorer)
		require.True(t, ok)

		assert.Equal(t, ScorerType, s.TypedName().Type)
		assert.Equal(t, "ttft", s.TypedName().Name)
		assert.Equal(t, fwksched.Distribution, s.Category())
		assert.Equal(t, defaultExplorationRate, s.explorationRate)
		assert.Equal(t, defaultMinInflightGap, s.minInflightGap)
		assert.Equal(t, defaultRoundTTFTStep, s.roundTTFTStep)
	})

	t.Run("parses parameters", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(
			`{"explorationRate":0.25,"minInflightGap":3,"roundTTFTStep":0.01}`))
		p, err := ScorerFactory("ttft", params, nil)
		require.NoError(t, err)
		s := p.(*Scorer)

		assert.Equal(t, 0.25, s.explorationRate)
		assert.Equal(t, 3.0, s.minInflightGap)
		assert.Equal(t, 0.01, s.roundTTFTStep)
	})

	t.Run("rejects invalid parameters", func(t *testing.T) {
		for name, raw := range map[string]string{
			"explorationRate above 1": `{"explorationRate":1.5}`,
			"minInflightGap zero":     `{"minInflightGap":0}`,
			"roundTTFTStep negative":  `{"roundTTFTStep":-0.01}`,
		} {
			t.Run(name, func(t *testing.T) {
				_, err := ScorerFactory("ttft", json.NewDecoder(strings.NewReader(raw)), nil)
				require.Error(t, err)
			})
		}
	})

	t.Run("Consumes declares both inputs as required", func(t *testing.T) {
		p, err := ScorerFactory("ttft", nil, nil)
		require.NoError(t, err)
		s := p.(*Scorer)

		required := s.Consumes().Required
		require.Len(t, required, 2)
		assert.Contains(t, required, attrlatency.TTFTPercentilesDataKey)
		assert.Contains(t, required, attrconcurrency.InFlightLoadDataKey)
	})

	t.Run("producer name overrides select a different key", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(
			`{"ttftPercentilesProducerName":"obs-a","inFlightLoadProducerName":"load-b"}`))
		p, err := ScorerFactory("ttft", params, nil)
		require.NoError(t, err)
		s := p.(*Scorer)

		assert.Contains(t, s.percentilesDataKey.String(), "obs-a")
		assert.Contains(t, s.inFlightLoadDataKey.String(), "load-b")
	})
}

func TestPredict(t *testing.T) {
	s := NewScorer(DefaultConfig)

	tests := []struct {
		name            string
		percentiles     *pct
		cur             float64
		wantPred        float64
		wantHasFloor    bool
		wantHasBaseline bool
	}{
		// cold: no usable floor, so nothing is known
		{"zero value", &pct{}, 5, 0, false, false},
		{"below minRequests observations", with(func(m *pct) { m.Observations = 9 }), 5, 0, false, false},

		// a floor but no usable load anchors: predicts at the floor
		{"short window below minRequests", with(func(m *pct) { m.RecentRequestCount = 9 }), 5, 0.20, true, false},
		{"no high in-flight anchor", with(func(m *pct) { m.InflightAtTypicalLoad = 0 }), 5, 0.20, true, false},
		{"high anchor at the floor", with(func(m *pct) { m.TypicalLoadTTFT = 0.20 }), 5, 0.20, true, false},

		// calibrated, low point admissible: two segments
		{"idle predicts the floor", trusted(), 0, 0.20, true, true},
		{"below the low anchor rides A to B", trusted(), 2, 0.275, true, true},
		{"at the low anchor both segments agree", trusted(), 4, 0.35, true, true},
		{"at the high anchor reproduces it", trusted(), 9, 0.55, true, true},
		{"beyond the high anchor extends B to C", trusted(), 20, 0.99, true, true},

		// calibrated, low point inadmissible: the single floor chord A to C
		{"anchors too close in load", with(func(m *pct) { m.InflightAtLowLoad = 8 }), 9, 0.55, true, true},
		{"inverted latency ordering", with(func(m *pct) { m.TypicalLoadTTFT = 0.30 }), 9, 0.30, true, true},
		{"low anchor under the floor", with(func(m *pct) { m.LowLoadTTFT = 0.15 }), 9, 0.55, true, true},
		{"zero low in-flight anchor", with(func(m *pct) { m.InflightAtLowLoad = 0 }), 9, 0.55, true, true},

		// the floor falls back to the short-window P10 before the history fills
		{"short-window P10 as floor", with(func(m *pct) { m.FloorTTFT, m.WindowFloorTTFT = 0, 0.20 }), 9, 0.55, true, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			floor, hasFloor, hasBaseline := calibration(tc.percentiles)
			assert.Equal(t, tc.wantHasFloor, hasFloor, "hasFloor")
			assert.Equal(t, tc.wantHasBaseline, hasBaseline, "hasBaseline")

			// Composed exactly as Score does it.
			var pred float64
			switch {
			case hasBaseline:
				pred = s.predict(tc.percentiles, floor, tc.cur)
			case hasFloor:
				pred = floor
			}
			assert.InDelta(t, tc.wantPred, pred, 1e-9)

			// Never below the endpoint's own floor: a loaded endpoint predicted
			// at its idle latency would win every decision it appeared in.
			if hasFloor {
				assert.GreaterOrEqual(t, pred, floor)
			}
		})
	}
}

// The curve must be monotone non-decreasing in load: more in-flight can never
// predict a faster response, or the most loaded endpoint wins.
func TestPredictIsMonotoneInLoad(t *testing.T) {
	s := NewScorer(DefaultConfig)
	floor, _, hasBaseline := calibration(trusted())
	require.True(t, hasBaseline)

	prev := 0.0
	for cur := 0.0; cur <= 50; cur += 0.5 {
		pred := s.predict(trusted(), floor, cur)
		assert.GreaterOrEqual(t, pred, prev, "prediction dropped at in-flight %v", cur)
		prev = pred
	}
}

func TestScore(t *testing.T) {
	ctx := context.Background()

	t.Run("all cold score equally so the picker spreads traffic", func(t *testing.T) {
		s := NewScorer(DefaultConfig).WithExplorationRate(0)
		endpoints := []fwksched.Endpoint{
			newEndpoint(nil, inflightPtr(0)),
			newEndpoint(&attrlatency.TTFTPercentiles{}, inflightPtr(3)),
		}

		scores := s.Score(ctx, nil, endpoints)
		require.Len(t, scores, 2)
		for _, endpoint := range endpoints {
			assert.Equal(t, 1.0, scores[endpoint])
		}
	})

	t.Run("lowest predicted TTFT scores 1.0", func(t *testing.T) {
		s := NewScorer(DefaultConfig).WithExplorationRate(0)

		// loaded: 20 in flight on the trusted curve -> 0.99s
		loaded := newEndpoint(trusted(), inflightPtr(20))
		// idle but intrinsically slower: floor 0.60, B at (2, 0.75), C at (6, 0.95)
		idle := newEndpoint(&attrlatency.TTFTPercentiles{
			FloorTTFT: 0.60, LowLoadTTFT: 0.75, TypicalLoadTTFT: 0.95,
			InflightAtLowLoad: 2, InflightAtTypicalLoad: 6,
			RecentRequestCount: 100, Observations: 100, CalibrationThreshold: 10,
		}, inflightPtr(1)) // -> 0.60 + 1*(0.75-0.60)/2 = 0.675s

		scores := s.Score(ctx, nil, []fwksched.Endpoint{loaded, idle})

		// The slower-but-idle endpoint wins: 0.675 < 0.99.
		assert.InDelta(t, 0.0, scores[loaded], 1e-9)
		assert.InDelta(t, 1.0, scores[idle], 1e-9)
	})

	t.Run("roundTTFTStep makes near-ties actual ties", func(t *testing.T) {
		fast := newEndpoint(trusted(), inflightPtr(19)) // 0.95s
		slow := newEndpoint(trusted(), inflightPtr(20)) // 0.99s

		unrounded := NewScorer(DefaultConfig).WithExplorationRate(0).WithRoundTTFTStep(0)
		scores := unrounded.Score(ctx, nil, []fwksched.Endpoint{fast, slow})
		assert.Equal(t, 1.0, scores[fast])
		assert.Equal(t, 0.0, scores[slow])

		// Both round to 1.0s, so neither wins on a 40ms difference.
		rounded := NewScorer(DefaultConfig).WithExplorationRate(0).WithRoundTTFTStep(0.5)
		scores = rounded.Score(ctx, nil, []fwksched.Endpoint{fast, slow})
		assert.Equal(t, 1.0, scores[fast])
		assert.Equal(t, 1.0, scores[slow])
	})

	t.Run("exploration always probes an uncalibrated endpoint at rate 1", func(t *testing.T) {
		s := NewScorer(DefaultConfig).WithExplorationRate(1.0)

		calibrated := newEndpoint(trusted(), inflightPtr(20))    // pred 0.99
		fastCalibrated := newEndpoint(trusted(), inflightPtr(0)) // pred 0.20, would win
		uncalibrated := newEndpoint(with(func(m *pct) {
			m.RecentRequestCount = 2 // has a floor, but no trusted load anchor
		}), inflightPtr(50))

		scores := s.Score(ctx, nil, []fwksched.Endpoint{calibrated, fastCalibrated, uncalibrated})
		assert.Equal(t, 1.0, scores[uncalibrated], "probe must be forced to the top score")
	})

	t.Run("exploration suppresses an uncalibrated endpoint when a calibrated one exists", func(t *testing.T) {
		// A rate this small makes a probe on any given call effectively
		// impossible (~1e-9 per iteration), so the suppression branch is what
		// runs. Repeated to catch an accidental inversion of the coin.
		s := NewScorer(DefaultConfig).WithExplorationRate(1e-9)

		for range 200 {
			calibrated := newEndpoint(trusted(), inflightPtr(20))
			uncalibrated := newEndpoint(with(func(m *pct) {
				m.RecentRequestCount = 2
			}), inflightPtr(50))

			scores := s.Score(ctx, nil, []fwksched.Endpoint{calibrated, uncalibrated})
			require.Equal(t, 0.0, scores[uncalibrated])
		}
	})

	t.Run("a missing in-flight reading demotes an endpoint to uncalibrated", func(t *testing.T) {
		s := NewScorer(DefaultConfig).WithExplorationRate(1e-9)

		calibrated := newEndpoint(trusted(), inflightPtr(20))
		// Full anchors, but no InFlightLoad attribute: the curve cannot be
		// evaluated at the endpoint's real load.
		noInflight := newEndpoint(trusted(), nil)

		scores := s.Score(ctx, nil, []fwksched.Endpoint{calibrated, noInflight})
		assert.Equal(t, 0.0, scores[noInflight])
	})

}
