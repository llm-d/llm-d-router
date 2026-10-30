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

package latency

import (
	"context"
	"testing"
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

func makeLatencyScorerEndpoint(name string, kvCache float64, queueSize, runningReqs int) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{
			KVCacheUsagePercent: kvCache,
			WaitingQueueSize:    queueSize,
			RunningRequestsSize: runningReqs,
			UpdateTime:          time.Now(),
		},
		nil,
	)
}

func setLatencyPrediction(ep fwksched.Endpoint, ttftValid, tpotValid bool, ttftHeadroom, tpotHeadroom, ttft, tpot float64) {
	ep.Put(attrlatency.LatencyPredictionInfoDataKey,
		attrlatency.NewLatencyPredictionInfo(ttftValid, tpotValid, ttftHeadroom, tpotHeadroom, ttft, tpot, 0))
}

// noExploreConfig returns default config for deterministic tests.
func noExploreConfig() Config {
	return DefaultConfig
}

func TestScorePositiveHeadroom(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	ep1 := makeLatencyScorerEndpoint("pod1", 0.3, 0, 5)
	ep2 := makeLatencyScorerEndpoint("pod2", 0.5, 0, 8)

	// pod1: lots of headroom, pod2: less headroom. Both positive.
	setLatencyPrediction(ep1, true, true, 100, 20, 50, 10)
	setLatencyPrediction(ep2, true, true, 30, 5, 120, 25)

	endpoints := []fwksched.Endpoint{ep1, ep2}
	scores := scorer.Score(context.Background(), nil, endpoints)

	s1, s2 := scores[ep1], scores[ep2]

	// Both should have non-zero scores (positive tier selected).
	if s1 == 0 || s2 == 0 {
		t.Fatalf("positive endpoints should have non-zero scores: pod1=%f, pod2=%f", s1, s2)
	}

	t.Logf("pod1 score=%f, pod2 score=%f", s1, s2)
}

func TestScoreNegativeOnly(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	ep1 := makeLatencyScorerEndpoint("pod1", 0.5, 0, 5)
	ep2 := makeLatencyScorerEndpoint("pod2", 0.6, 0, 8)

	// Both negative headroom — only negative tier available.
	setLatencyPrediction(ep1, false, false, -10, -5, 110, 35)
	setLatencyPrediction(ep2, false, false, -100, -30, 200, 60)

	endpoints := []fwksched.Endpoint{ep1, ep2}
	scores := scorer.Score(context.Background(), nil, endpoints)

	s1, s2 := scores[ep1], scores[ep2]

	// Both should have non-zero scores.
	if s1 == 0 || s2 == 0 {
		t.Fatalf("negative endpoints should have non-zero scores: pod1=%f, pod2=%f", s1, s2)
	}

	// pod1 has less violation → should score higher.
	if s1 <= s2 {
		t.Errorf("pod1 (less violation) should score higher: pod1=%f, pod2=%f", s1, s2)
	}

	t.Logf("pod1 score=%f, pod2 score=%f", s1, s2)
}

func TestScoreTierSplit(t *testing.T) {
	// The scorer expects upstream filters to have separated tiers.
	// Test with only positive endpoints (typical after tier filter selects positive).
	scorer := NewPlugin(noExploreConfig())

	ep1 := makeLatencyScorerEndpoint("ep1", 0.3, 0, 5)
	ep2 := makeLatencyScorerEndpoint("ep2", 0.5, 0, 8)

	setLatencyPrediction(ep1, true, true, 10, 2, 90, 28)
	setLatencyPrediction(ep2, true, true, 50, 10, 50, 20)

	endpoints := []fwksched.Endpoint{ep1, ep2}
	scores := scorer.Score(context.Background(), nil, endpoints)

	s1, s2 := scores[ep1], scores[ep2]

	// Both should be scored.
	if s1 == 0 {
		t.Errorf("ep1 should have non-zero score: %f", s1)
	}
	if s2 == 0 {
		t.Errorf("ep2 should have non-zero score: %f", s2)
	}

	// ep1 has less headroom (10 vs 50), so with "least" strategy it should score higher.
	if s1 <= s2 {
		t.Errorf("ep1 (less headroom) should score higher with 'least' strategy: ep1=%f, ep2=%f", s1, s2)
	}

	t.Logf("ep1 score=%f, ep2 score=%f", s1, s2)
}

func TestScoreIdlePodPreference(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	epBusy := makeLatencyScorerEndpoint("busy", 0.5, 0, 5)
	epIdle := makeLatencyScorerEndpoint("idle", 0.5, 0, 0) // 0 running requests

	// Same predictions — both negative, same deficit.
	// Use dispatch count for idle detection (matches EPP internal queue).
	epBusy.Put(attrlatency.LatencyPredictionInfoDataKey,
		attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 5))
	epIdle.Put(attrlatency.LatencyPredictionInfoDataKey,
		attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))

	endpoints := []fwksched.Endpoint{epBusy, epIdle}
	scores := scorer.Score(context.Background(), nil, endpoints)

	sBusy, sIdle := scores[epBusy], scores[epIdle]

	// Idle pod should be strictly preferred (selected tier, busy excluded).
	if sIdle == 0 {
		t.Errorf("idle pod should have non-zero score: %f", sIdle)
	}
	if sBusy != 0 {
		t.Errorf("busy pod should have zero score when idle pods available: %f", sBusy)
	}

	t.Logf("busy score=%f, idle score=%f", sBusy, sIdle)
}

func TestScoreHierarchicalBuckets(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	// All negative, all busy — hierarchical buckets apply.
	epBothNeg := makeLatencyScorerEndpoint("both-neg", 0.5, 0, 5)
	epTTFTNeg := makeLatencyScorerEndpoint("ttft-neg", 0.4, 0, 3)
	epTPOTNeg := makeLatencyScorerEndpoint("tpot-neg", 0.6, 0, 4)

	setLatencyPrediction(epBothNeg, false, false, -50, -10, 150, 40) // both negative
	setLatencyPrediction(epTTFTNeg, false, true, -30, 5, 130, 25)    // only TTFT negative
	setLatencyPrediction(epTPOTNeg, true, false, 10, -8, 90, 38)     // only TPOT negative

	endpoints := []fwksched.Endpoint{epBothNeg, epTTFTNeg, epTPOTNeg}
	scores := scorer.Score(context.Background(), nil, endpoints)

	// All should have non-zero scores (they're all in the negative tier).
	for _, ep := range endpoints {
		if scores[ep] == 0 {
			t.Errorf("%s should have non-zero score", ep.GetMetadata().ID.Name)
		}
	}

	t.Logf("both-neg=%f, ttft-neg=%f, tpot-neg=%f",
		scores[epBothNeg], scores[epTTFTNeg], scores[epTPOTNeg])
}

func TestScoreCompositeFallback(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	// No predictions set — should use composite scoring.
	ep1 := makeLatencyScorerEndpoint("pod1", 0.2, 0, 3)
	ep2 := makeLatencyScorerEndpoint("pod2", 0.8, 5, 10)

	endpoints := []fwksched.Endpoint{ep1, ep2}
	scores := scorer.Score(context.Background(), nil, endpoints)

	s1, s2 := scores[ep1], scores[ep2]

	// pod1 has better composite metrics → should score higher.
	if s1 <= s2 {
		t.Errorf("pod1 (lower KV, no queue) should score higher: pod1=%f, pod2=%f", s1, s2)
	}

	t.Logf("pod1 score=%f, pod2 score=%f", s1, s2)
}

func TestScoreCompositeFallbackOmitsUnwrittenMetrics(t *testing.T) {
	for _, tc := range []struct {
		name    string
		metrics *fwkdl.Metrics
	}{
		{name: "never-scraped", metrics: &fwkdl.Metrics{}},
		{name: "unwritten-queue", metrics: &fwkdl.Metrics{WaitingQueueSize: 100}},
		{name: "nil-metrics"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scorer := NewPlugin(noExploreConfig())
			busy := makeLatencyScorerEndpoint("busy", 0.5, 10, 5)
			idle := makeLatencyScorerEndpoint("idle", 0, 0, 0)
			unwritten := fwksched.NewEndpoint(
				&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "unwritten"}},
				tc.metrics,
				nil,
			)

			scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{busy, idle, unwritten})
			if _, ok := scores[unwritten]; ok {
				t.Errorf("endpoint with unwritten metrics should be unscored, got %f", scores[unwritten])
			}
			if len(scores) != 2 {
				t.Errorf("expected two scored endpoints, got %d", len(scores))
			}
			if scores[busy] != 0.17 || scores[idle] != 0.67 {
				t.Errorf("written metrics should keep their scores: busy=%f, idle=%f", scores[busy], scores[idle])
			}

			scores = scorer.Score(context.Background(), nil, []fwksched.Endpoint{unwritten})
			if len(scores) != 0 {
				t.Errorf("expected no scores when all metrics are unwritten, got %v", scores)
			}
		})
	}
}

func TestScorePredictionWithoutMetrics(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod"}},
		nil,
		nil,
	)
	setLatencyPrediction(ep, true, true, 100, 20, 50, 10)

	scores := scorer.Score(context.Background(), nil, []fwksched.Endpoint{ep})
	if scores[ep] <= 0 {
		t.Errorf("endpoint with a prediction should be scored without metrics, got %f", scores[ep])
	}
}

func TestScoreConstantHeadroomPreservesDisabledWeights(t *testing.T) {
	for _, strategy := range []HeadroomSelectionStrategy{StrategyLeast, StrategyMost} {
		t.Run(string(strategy), func(t *testing.T) {
			for _, tc := range []struct {
				name       string
				ttftWeight float64
				tpotWeight float64
				ttft       [2]float64
				tpot       [2]float64
				wantEqual  bool
			}{
				{
					name: "TPOT disabled", ttftWeight: 1, tpotWeight: 0,
					ttft: [2]float64{10, 10}, tpot: [2]float64{5, 20}, wantEqual: true,
				},
				{
					name: "TTFT disabled", ttftWeight: 0, tpotWeight: 1,
					ttft: [2]float64{10, 30}, tpot: [2]float64{5, 5}, wantEqual: true,
				},
				{
					name: "TPOT enabled", ttftWeight: 0.8, tpotWeight: 0.2,
					ttft: [2]float64{10, 10}, tpot: [2]float64{5, 20},
				},
				{
					name: "TTFT enabled", ttftWeight: 0.8, tpotWeight: 0.2,
					ttft: [2]float64{10, 30}, tpot: [2]float64{5, 5},
				},
				{
					name: "TPOT positive below normalized precision", ttftWeight: 1e308, tpotWeight: 1e-308,
					ttft: [2]float64{10, 10}, tpot: [2]float64{5, 20},
				},
				{
					name: "TTFT positive below normalized precision", ttftWeight: 1e-308, tpotWeight: 1e308,
					ttft: [2]float64{10, 30}, tpot: [2]float64{5, 5},
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					cfg := DefaultConfig
					cfg.TTFTWeight, cfg.TPOTWeight = tc.ttftWeight, tc.tpotWeight
					cfg.HeadroomSelectionStrategy = strategy
					scorer := NewPlugin(cfg)
					endpoints := []fwksched.Endpoint{
						makeLatencyScorerEndpoint("pod1", 0.5, 0, 1),
						makeLatencyScorerEndpoint("pod2", 0.5, 0, 1),
					}
					for i, ep := range endpoints {
						setLatencyPrediction(ep, true, true, tc.ttft[i], tc.tpot[i], 0, 0)
					}
					scores := scorer.Score(context.Background(), nil, endpoints)
					equal := scores[endpoints[0]] == scores[endpoints[1]]
					if equal != tc.wantEqual {
						t.Fatalf("scores = (%v, %v), want equal = %v",
							scores[endpoints[0]], scores[endpoints[1]], tc.wantEqual)
					}
					if !tc.wantEqual {
						wantFirstHigher := strategy == StrategyLeast
						if (scores[endpoints[0]] > scores[endpoints[1]]) != wantFirstHigher {
							t.Fatalf("scores = (%v, %v), want first higher = %v",
								scores[endpoints[0]], scores[endpoints[1]], wantFirstHigher)
						}
					}
				})
			}
		})
	}
}

// Note: EpsilonExploreNeg (tier selection) is now handled by the
// slo-headroom-tier-filter, not the scorer. See filter tests.

func TestScoreHeadroomWithinUnitInterval(t *testing.T) {
	for _, strategy := range []HeadroomSelectionStrategy{StrategyLeast, StrategyMost} {
		t.Run(string(strategy), func(t *testing.T) {
			cfg := DefaultConfig
			cfg.TTFTWeight, cfg.TPOTWeight = 1, 0
			cfg.HeadroomSelectionStrategy = strategy
			scorer := NewPlugin(cfg)

			// The best endpoint and one 0.005 behind it on the normalized headroom.
			ttft := []float64{10, 10.5, 110}
			if strategy == StrategyMost {
				ttft = []float64{110, 109.5, 10}
			}
			endpoints := make([]fwksched.Endpoint, len(ttft))
			for i, headroom := range ttft {
				endpoints[i] = makeLatencyScorerEndpoint(string(rune('a'+i)), 0.5, 0, 1)
				setLatencyPrediction(endpoints[i], true, true, headroom, 5, 0, 0)
			}
			scores := scorer.Score(context.Background(), nil, endpoints)

			for _, ep := range endpoints {
				if s := scores[ep]; s < 0 || s > 1 {
					t.Errorf("score of %s = %f, want it in [0, 1]", ep.GetMetadata().ID.Name, s)
				}
			}
			if best, next := scores[endpoints[0]], scores[endpoints[1]]; best <= next {
				t.Errorf("best endpoint should outscore the runner-up: best=%f, runner-up=%f", best, next)
			}
		})
	}
}
