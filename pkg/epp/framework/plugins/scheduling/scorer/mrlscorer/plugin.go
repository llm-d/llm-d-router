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

// Package mrlscorer routes each request to the endpoint with the smallest
// expected remaining in-flight work under a two-term mean-residual-life
// residual formula over TTFT and decode digests. See README.md for the
// derivation.
package mrlscorer

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsojourn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/sojourntime"
)

// ScorerType is the plugin type of the mrl scorer.
const ScorerType = "mrl-scorer-hub"

var (
	_ fwksched.Scorer          = &Scorer{}
	_ fwkplugin.ConsumerPlugin = &Scorer{}
)

// Config holds the scorer's tunables.
type Config struct {
	// ExplorationRate is the probability that a cold endpoint (no
	// published snapshot yet) is probed on a scoring call. Range [0, 1].
	// On a probe the endpoint's final score is overridden to 1.0 so it
	// can win a scheduling decision.
	ExplorationRate float64 `json:"explorationRate,omitempty"`
	// Producer instance to read from. Empty uses the default producer (the
	// type-named instance).
	SojournTimeObserverProducerName string `json:"sojournTimeObserverProducerName,omitempty"`
}

// DefaultConfig is decoded over by the factory.
var DefaultConfig = Config{
	ExplorationRate: 0.1,
}

// Scorer ranks candidate endpoints by expected remaining in-flight work under
// a two-term MRL residual: for each dispatched-not-completed request on the
// candidate, add MrlTtft(age)+MeanDecode if the request has not yet received
// a first chunk, or MrlDecode(age) if it has. Argmin across candidates,
// min-max normalized so the lowest-residual endpoint scores 1.0.
type Scorer struct {
	typedName fwkplugin.TypedName

	snapshotDataKey         fwkplugin.DataKey
	inFlightRequestsDataKey fwkplugin.DataKey

	// explorationRate is the resolved probability that a cold endpoint is
	// probed on a scoring call. Range [0, 1]; 0 disables the coin.
	explorationRate float64
}

// ScorerFactory builds a Scorer from its plugin configuration. It reads only
// its parameters; every runtime input arrives through the endpoint's
// AttributeMap under the DataKeys declared by Consumes.
func ScorerFactory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for plugin %q: %w", name, err)
		}
	}
	if cfg.ExplorationRate < 0 || cfg.ExplorationRate > 1 {
		return nil, fmt.Errorf("plugin %q: explorationRate must be in [0, 1], got %v", name, cfg.ExplorationRate)
	}
	return NewScorer(cfg).WithName(name), nil
}

// NewScorer initializes a Scorer from cfg.
func NewScorer(cfg Config) *Scorer {
	return &Scorer{
		typedName:               fwkplugin.TypedName{Type: ScorerType, Name: ScorerType},
		snapshotDataKey:         attrsojourn.SojournEstimatorSnapshotDataKey.WithNonEmptyProducerName(cfg.SojournTimeObserverProducerName),
		inFlightRequestsDataKey: attrsojourn.InFlightRequestsDataKey.WithNonEmptyProducerName(cfg.SojournTimeObserverProducerName),
		explorationRate:         cfg.ExplorationRate,
	}
}

// TypedName implements fwkplugin.Plugin.
func (s *Scorer) TypedName() fwkplugin.TypedName { return s.typedName }

// WithName sets the instance name.
func (s *Scorer) WithName(name string) *Scorer {
	s.typedName.Name = name
	return s
}

// Category reports that this scorer spreads load rather than seeking
// affinity.
func (s *Scorer) Category() fwksched.ScorerCategory { return fwksched.Distribution }

// Consumes declares both inputs Required so the DAG orders the producer's
// construction ahead of the scorer's and auto-creates it when the config
// omits it.
func (s *Scorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			s.snapshotDataKey:         attrsojourn.SojournEstimatorSnapshot{},
			s.inFlightRequestsDataKey: attrsojourn.InFlightRequestsSnapshot{},
		},
	}
}

// Score ranks endpoints by their two-term MRL residual, normalized argmin,
// with cold-endpoint seeding and an exploration coin so a cold endpoint does
// not deterministically capture traffic.
//
// Per candidate:
//   - Read the SojournEstimatorSnapshot from the endpoint's AttributeMap.
//     A nil snapshot means the endpoint is cold (observer has not yet
//     flushed a warm snapshot).
//   - Read the InFlightRequestsSnapshot from the endpoint's AttributeMap.
//   - Sum the two-term residual across in-flight requests (warm only).
func (s *Scorer) Score(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores, residuals, inflightCounts, colds := s.scoreAt(time.Now(), endpoints)
	if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
		for i, endpoint := range endpoints {
			if endpoint == nil || endpoint.GetMetadata() == nil {
				continue
			}
			debugLogger.Info("mrl-scorer score",
				"endpoint", endpoint.GetMetadata().ID.String(),
				"residual", residuals[i],
				"score", scores[endpoint],
				"inflightCount", inflightCounts[i],
				"cold", colds[i])
		}
	}
	return scores
}

// eval holds pass-1 per-endpoint state. snap is nil for a cold endpoint;
// residual is 0 in that case and pass 2 overwrites it with minR before
// normalization so a cold endpoint cannot beat a warm one on normalization
// alone.
type eval struct {
	snap          *attrsojourn.SojournEstimatorSnapshot
	residual      float64
	inflightCount int
}

// scoreAt is the clock-injected core of Score. Separating "read the clock"
// from "score at a given instant" lets tests drive age-dependent paths with
// a fixed now, matching the clock-threading idiom used elsewhere in the tree.
// Returns the score map plus three per-endpoint slices aligned with the
// endpoints slice — Score consumes them to emit a per-candidate DEBUG trace.
func (s *Scorer) scoreAt(now time.Time, endpoints []fwksched.Endpoint) (map[fwksched.Endpoint]float64, []float64, []int, []bool) {
	evals := make([]eval, len(endpoints))
	minR, maxR := math.MaxFloat64, 0.0
	anyWarm := false

	// Pass 1: classify every candidate warm or cold, record per-endpoint
	// state, and track the min/max residual over WARM endpoints only so
	// cold endpoints cannot widen or narrow the normalization span.
	for i, endpoint := range endpoints {
		var snap *attrsojourn.SojournEstimatorSnapshot
		if endpoint != nil {
			snap = s.readSnapshot(endpoint)
		}
		r := s.residualFor(endpoint, now)
		inflight := 0
		if endpoint != nil {
			inflight = len(s.readInFlight(endpoint))
		}
		evals[i] = eval{snap: snap, residual: r, inflightCount: inflight}
		if snap != nil {
			anyWarm = true
			minR = min(minR, r)
			maxR = max(maxR, r)
		}
	}

	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	residuals := make([]float64, len(endpoints))
	inflightCounts := make([]int, len(endpoints))
	colds := make([]bool, len(endpoints))

	// All candidates cold: nothing to rank. Every endpoint ties at 1.0 and
	// the max-score picker's tie-break spreads traffic.
	if !anyWarm {
		for i, endpoint := range endpoints {
			if endpoint != nil {
				scores[endpoint] = 1.0
			}
			residuals[i] = evals[i].residual
			inflightCounts[i] = evals[i].inflightCount
			colds[i] = evals[i].snap == nil
		}
		return scores, residuals, inflightCounts, colds
	}

	// Pass 2: seed cold endpoints at minR, normalize across all candidates
	// (cold ones now share minR with the least-loaded warm), then apply the
	// per-cold exploration coin on top of the normalized score.
	for i, endpoint := range endpoints {
		e := &evals[i]
		cold := e.snap == nil
		if cold {
			e.residual = minR
		}
		var score float64
		if maxR == minR {
			// All warm residuals equal and all cold seeded to the same
			// value — everyone ties.
			score = 1.0
		} else {
			score = (maxR - e.residual) / (maxR - minR)
		}

		// Independent coin per cold endpoint. Runs AFTER normalization so
		// a probe never shifts the ratio between warm endpoints. The
		// no-probe arm drops cold to 0 only when the warm residuals span
		// a real range: with maxR == minR every warm scored 1.0 via the
		// tie branch, so there is no warm-side ranking the cold would be
		// dropped out of; the seeded score stays.
		if s.explorationRate > 0 && cold {
			if rand.Float64() < s.explorationRate {
				score = 1.0 // probe: cold wins this decision
			} else if maxR > minR {
				score = 0 // no probe: cold loses to the warm ranking
			}
		}

		// Guard the map write so a nil endpoint in the slice does not
		// become a result-map key. The parallel index slices still
		// record data at this slot for the DEBUG log (which itself
		// skips nil endpoints when iterating).
		if endpoint != nil {
			scores[endpoint] = score
		}
		residuals[i] = e.residual
		inflightCounts[i] = e.inflightCount
		colds[i] = cold
	}
	return scores, residuals, inflightCounts, colds
}

// residualFor computes the two-term MRL residual for one endpoint.
func (s *Scorer) residualFor(endpoint fwksched.Endpoint, now time.Time) float64 {
	if endpoint == nil {
		return 0
	}
	snap := s.readSnapshot(endpoint)
	// Cold endpoint (no snapshot yet) reads as residual 0. This lets a
	// newly-added endpoint be considered equally attractive to a fully
	// drained warm endpoint; the observer will warm up within a handful
	// of completions.
	if snap == nil {
		return 0
	}

	inflight := s.readInFlight(endpoint)
	if len(inflight) == 0 {
		return 0
	}

	meanDecode := snap.MeanDecode()
	r := 0.0
	for _, req := range inflight {
		if req.FirstChunkAt.IsZero() {
			// Pre-first-chunk: request is somewhere in the TTFT phase.
			// Expected remaining total = expected remaining TTFT plus a
			// full decode phase to come.
			age := now.Sub(req.DispatchedAt).Seconds()
			r += snap.MrlTtft(age) + meanDecode
		} else {
			// Post-first-chunk: request is in the decode phase. TTFT is
			// done; only the remaining decode contributes.
			age := now.Sub(req.FirstChunkAt).Seconds()
			r += snap.MrlDecode(age)
		}
	}
	return r
}

// readSnapshot returns the endpoint's published snapshot, or nil when the
// endpoint has no entry under the snapshot DataKey or the entry is not a
// *SojournEstimatorSnapshot.
func (s *Scorer) readSnapshot(endpoint fwksched.Endpoint) *attrsojourn.SojournEstimatorSnapshot {
	raw, ok := endpoint.Get(s.snapshotDataKey)
	if !ok || raw == nil {
		return nil
	}
	snap, ok := raw.(*attrsojourn.SojournEstimatorSnapshot)
	if !ok {
		return nil
	}
	return snap
}

// readInFlight returns the endpoint's in-flight request list, or nil when
// the endpoint has no entry under the in-flight DataKey or the entry is not
// an *InFlightRequestsSnapshot.
func (s *Scorer) readInFlight(endpoint fwksched.Endpoint) []attrsojourn.InFlightRequest {
	raw, ok := endpoint.Get(s.inFlightRequestsDataKey)
	if !ok || raw == nil {
		return nil
	}
	snap, ok := raw.(*attrsojourn.InFlightRequestsSnapshot)
	if !ok || snap == nil {
		return nil
	}
	return snap.Requests
}
