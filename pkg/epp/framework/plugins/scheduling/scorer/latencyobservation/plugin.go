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

// Package latencyobservation routes each request to the endpoint with the lowest
// predicted time-to-first-token under its current load. The prediction is a
// piecewise-linear curve through the endpoint's measured points, evaluated at
// its live in-flight count. See README.md for the curve and its rationale.
package latencyobservation

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

// ScorerType is the plugin type of the latency-observation scorer.
const ScorerType = "latency-observation-scorer-hub"

const (
	// Non-zero: at zero an uncalibrated endpoint is seeded at the best observed
	// TTFT, ties for the win, and takes a share of all traffic before anything
	// is known about it.
	defaultExplorationRate = 0.1
	defaultMinInflightGap  = 2.0
	defaultRoundTTFTStep   = 0.0 // disabled
)

var (
	_ fwksched.Scorer          = &Scorer{}
	_ fwkplugin.ConsumerPlugin = &Scorer{}
)

// Config holds the scorer's tunables. The percentiles and the minRequests
// threshold belong to the latency-observer-producer-hub and arrive in its snapshot.
// See README.md for what each field does.
type Config struct {
	// Probability that an under-observed endpoint is probed. Range [0, 1].
	ExplorationRate float64 `json:"explorationRate,omitempty"`
	// Minimum in-flight separation for the low-load anchor to be usable as
	// a curve anchor; below it their slope is noise. Must be > 0.
	MinInflightGap float64 `json:"minInflightGap,omitempty"`
	// Quantization step in seconds, so endpoints closer together than this tie
	// rather than one winning on a meaningless difference. Must be >= 0.
	RoundTTFTStep float64 `json:"roundTTFTStep,omitempty"`

	// Producer instances to read from. Empty uses the default producer.
	TTFTPercentilesProducerName string `json:"ttftPercentilesProducerName,omitempty"`
	InFlightLoadProducerName    string `json:"inFlightLoadProducerName,omitempty"`
}

// DefaultConfig is decoded over by the factory, so an unset field keeps its
// default and an explicitly configured zero is honored.
var DefaultConfig = Config{
	ExplorationRate: defaultExplorationRate,
	MinInflightGap:  defaultMinInflightGap,
	RoundTTFTStep:   defaultRoundTTFTStep,
}

// Scorer ranks candidate endpoints by predicted TTFT. It holds no per-request
// or per-endpoint state: every input is read from endpoint attributes.
type Scorer struct {
	typedName       fwkplugin.TypedName
	explorationRate float64
	minInflightGap  float64
	roundTTFTStep   float64

	percentilesDataKey  fwkplugin.DataKey
	inFlightLoadDataKey fwkplugin.DataKey
}

// ScorerFactory builds a Scorer from its plugin configuration.
func ScorerFactory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for plugin %q: %w", name, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters for plugin %q: %w", name, err)
	}

	return NewScorer(cfg).WithName(name), nil
}

func (c *Config) validate() error {
	if c.ExplorationRate < 0 || c.ExplorationRate > 1 {
		return fmt.Errorf("explorationRate must be in [0, 1], got %v", c.ExplorationRate)
	}
	if c.MinInflightGap <= 0 {
		return fmt.Errorf("minInflightGap must be > 0, got %v", c.MinInflightGap)
	}
	if c.RoundTTFTStep < 0 {
		return fmt.Errorf("roundTTFTStep must be >= 0, got %v", c.RoundTTFTStep)
	}
	return nil
}

// NewScorer initializes a Scorer from cfg. Pass DefaultConfig (optionally with
// fields overridden) rather than a zero Config, which would leave
// minInflightGap at 0.
func NewScorer(cfg Config) *Scorer {
	return &Scorer{
		typedName:           fwkplugin.TypedName{Type: ScorerType, Name: ScorerType},
		explorationRate:     cfg.ExplorationRate,
		minInflightGap:      cfg.MinInflightGap,
		roundTTFTStep:       cfg.RoundTTFTStep,
		percentilesDataKey:  attrlatency.TTFTPercentilesDataKey.WithNonEmptyProducerName(cfg.TTFTPercentilesProducerName),
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.InFlightLoadProducerName),
	}
}

func (s *Scorer) TypedName() fwkplugin.TypedName { return s.typedName }

// Category reports that this scorer spreads load rather than seeking affinity.
func (s *Scorer) Category() fwksched.ScorerCategory { return fwksched.Distribution }

func (s *Scorer) WithName(name string) *Scorer {
	s.typedName.Name = name
	return s
}

func (s *Scorer) WithExplorationRate(r float64) *Scorer {
	s.explorationRate = r
	return s
}

func (s *Scorer) WithMinInflightGap(g float64) *Scorer {
	s.minInflightGap = g
	return s
}

func (s *Scorer) WithRoundTTFTStep(step float64) *Scorer {
	s.roundTTFTStep = step
	return s
}

// Consumes declares both inputs Required so the DAG orders their producers
// ahead of scheduling and auto-creates them when the config omits them.
func (s *Scorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			s.percentilesDataKey:  attrlatency.TTFTPercentiles{},
			s.inFlightLoadDataKey: attrconcurrency.InFlightLoad{},
		},
	}
}

// eval is the per-endpoint working state for one Score call.
type eval struct {
	pred        float64
	hasBaseline bool // load anchors usable, so the full curve applies
	hasFloor    bool // has a service floor, i.e. is not cold
}

// calibration reports what the snapshot alone says is known about an endpoint,
// independent of any prediction.
func calibration(m *attrlatency.TTFTPercentiles) (floor float64, hasFloor, hasBaseline bool) {
	floor = m.Floor()
	if floor == 0 {
		return 0, false, false
	}
	return floor, true, m.RecentRequestCount >= m.CalibrationThreshold && m.InflightAtTypicalLoad > 0 && m.TypicalLoadTTFT > floor
}

// Score ranks endpoints by predicted TTFT, min-max normalized so the lowest
// prediction scores 1.0. Cold endpoints are seeded at the best observed
// prediction; if every endpoint is cold they all score 1.0.
func (s *Scorer) Score(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	evals := make([]eval, len(endpoints))
	minPred, maxPred := math.MaxFloat64, 0.0
	anyObserved, anyTrusted := false, false

	for i, endpoint := range endpoints {
		percentiles, inflight, hasInflight := s.read(endpoint)
		floor, hasFloor, hasBaseline := calibration(percentiles)

		// No live load reading means the curve would be evaluated at an unknown
		// point, so route the endpoint through exploration rather than let it
		// compete on an unanchored value.
		hasBaseline = hasBaseline && hasInflight

		var pred float64
		switch {
		case hasBaseline:
			pred = s.predict(percentiles, floor, inflight)
		case hasFloor:
			pred = floor // uncalibrated: the floor is an optimistic seed
		}
		if s.roundTTFTStep > 0 {
			pred = math.Round(pred/s.roundTTFTStep) * s.roundTTFTStep
		}

		evals[i] = eval{pred: pred, hasBaseline: hasBaseline, hasFloor: hasFloor}
		if hasFloor {
			anyObserved = true
			minPred = min(minPred, pred)
			// Cold endpoints are seeded to minPred below, so they can never be the max.
			maxPred = max(maxPred, pred)
		}
		if hasBaseline {
			anyTrusted = true
		}
	}

	scores := make(map[fwksched.Endpoint]float64, len(endpoints))

	// Nothing has a floor yet, so there is nothing to rank; the picker's random
	// tie-break spreads the traffic.
	if !anyObserved {
		for _, endpoint := range endpoints {
			scores[endpoint] = 1.0
		}
		return scores
	}

	for i, endpoint := range endpoints {
		e := &evals[i]
		if !e.hasFloor {
			e.pred = minPred // optimistic seed for a cold endpoint
		}
		if maxPred == minPred {
			scores[endpoint] = 1.0
		} else {
			scores[endpoint] = (maxPred - e.pred) / (maxPred - minPred)
		}

		// Independent coin per under-observed endpoint. Overrides the final
		// score only, so a probe never distorts the normalization above.
		if s.explorationRate > 0 && !e.hasBaseline {
			if rand.Float64() < s.explorationRate {
				scores[endpoint] = 1.0
			} else if anyTrusted {
				scores[endpoint] = 0
			}
		}
	}

	if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
		for i, endpoint := range endpoints {
			if endpoint == nil || endpoint.GetMetadata() == nil {
				continue
			}
			debugLogger.Info("latency-observation score",
				"endpoint", endpoint.GetMetadata().ID.String(),
				"predictedTTFT", evals[i].pred,
				"score", scores[endpoint],
				"hasBaseline", evals[i].hasBaseline,
				"hasFloor", evals[i].hasFloor)
		}
	}

	return scores
}

// read pulls the endpoint's published snapshot and its live in-flight count.
// A missing or malformed snapshot yields the zero value, which reads as cold.
func (s *Scorer) read(endpoint fwksched.Endpoint) (percentiles *attrlatency.TTFTPercentiles, inflight float64, hasInflight bool) {
	percentiles = &attrlatency.TTFTPercentiles{}
	if endpoint == nil {
		return percentiles, 0, false
	}
	if raw, ok := endpoint.Get(s.percentilesDataKey); ok {
		if typed, ok := raw.(*attrlatency.TTFTPercentiles); ok && typed != nil {
			percentiles = typed
		}
	}

	if raw, ok := endpoint.Get(s.inFlightLoadDataKey); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			return percentiles, float64(load.Requests), true
		}
	}
	return percentiles, 0, false
}

// predict evaluates the endpoint's TTFT-vs-load curve at cur, its live
// in-flight count. The curve runs through A (0, floor), B (InflightAtLowLoad,
// LowLoadTTFT) and C (InflightAtTypicalLoad, TypicalLoadTTFT); every point is measured. See
// README.md for why three points rather than two.
//
// Only meaningful for an endpoint with a baseline; callers use the floor alone
// otherwise.
func (s *Scorer) predict(m *attrlatency.TTFTPercentiles, floor, cur float64) (pred float64) {
	// The low point is admissible only when it is separated from the high point
	// in load, ordered below it in latency, and above the floor. Otherwise its
	// segment slopes on noise, or downwards.
	useLow := m.InflightAtTypicalLoad-m.InflightAtLowLoad >= s.minInflightGap &&
		m.TypicalLoadTTFT > m.LowLoadTTFT &&
		m.LowLoadTTFT > floor &&
		m.InflightAtLowLoad > 0

	switch {
	case useLow && cur < m.InflightAtLowLoad:
		// Segment A->B. Extending the steeper B->C slope backwards instead
		// would predict a draining-but-still-loaded endpoint at its idle
		// latency, so it would win every decision it appeared in.
		pred = floor + cur*(m.LowLoadTTFT-floor)/m.InflightAtLowLoad
	case useLow:
		// Segment B->C, extended beyond C.
		pred = m.LowLoadTTFT + (cur-m.InflightAtLowLoad)*(m.TypicalLoadTTFT-m.LowLoadTTFT)/(m.InflightAtTypicalLoad-m.InflightAtLowLoad)
	default:
		// Low point inadmissible: the single floor chord A->C, extended beyond C.
		pred = floor + cur*(m.TypicalLoadTTFT-floor)/m.InflightAtTypicalLoad
	}
	return max(pred, floor)
}
