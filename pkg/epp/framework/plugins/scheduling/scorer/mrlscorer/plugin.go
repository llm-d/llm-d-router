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
	"errors"
	"fmt"
	"math"
	"time"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsojourn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/sojourntime"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sojourntimeobserver"
)

// ScorerType is the plugin type of the mrl scorer.
const ScorerType = "mrl-scorer-hub"

var (
	_ fwksched.Scorer          = &Scorer{}
	_ fwkplugin.ConsumerPlugin = &Scorer{}
)

// Config holds the scorer's tunables.
type Config struct {
	// Producer instance to read the SojournEstimatorSnapshot from. Empty
	// uses the default producer (the type-named instance).
	SojournTimeObserverProducerName string `json:"sojournTimeObserverProducerName,omitempty"`
}

// DefaultConfig is decoded over by the factory.
var DefaultConfig = Config{}

// Scorer ranks candidate endpoints by expected remaining in-flight work under
// a two-term MRL residual: for each dispatched-not-completed request on the
// candidate, add MrlTtft(age)+MeanDecode if the request has not yet received
// a first chunk, or MrlDecode(age) if it has. Argmin across candidates,
// min-max normalized so the lowest-residual endpoint scores 1.0.
type Scorer struct {
	typedName fwkplugin.TypedName

	snapshotDataKey fwkplugin.DataKey
	observer        *sojourntimeobserver.Observer
}

// ScorerFactory builds a Scorer from its plugin configuration.
//
// The observer handle is resolved at factory time so a misconfigured
// deployment fails at plugin-load rather than silently returning neutral
// scores at request time. The scorer's Consumes declaration also requires
// the snapshot DataKey, so the framework's DAG will already refuse to load
// this scorer without a producer providing that key.
func ScorerFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}

	cfg := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for plugin %q: %w", name, err)
		}
	}

	producerName := cfg.SojournTimeObserverProducerName
	if producerName == "" {
		producerName = sojourntimeobserver.SojournTimeObserverProducerType
	}
	observer, err := fwkplugin.PluginByType[*sojourntimeobserver.Observer](handle, producerName)
	if err != nil {
		return nil, fmt.Errorf("mrl-scorer-hub %q: cannot resolve sojourn-time-observer-hub %q: %w",
			name, producerName, err)
	}

	scorer := NewScorer(observer, cfg).WithName(name)
	return scorer, nil
}

// NewScorer initializes a Scorer bound to the given observer.
func NewScorer(observer *sojourntimeobserver.Observer, cfg Config) *Scorer {
	return &Scorer{
		typedName:       fwkplugin.TypedName{Type: ScorerType, Name: ScorerType},
		snapshotDataKey: attrsojourn.SojournEstimatorSnapshotDataKey.WithNonEmptyProducerName(cfg.SojournTimeObserverProducerName),
		observer:        observer,
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

// Consumes declares the snapshot DataKey as Required so the DAG orders the
// producer's construction ahead of the scorer's and auto-creates it when the
// config omits it.
func (s *Scorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			s.snapshotDataKey: attrsojourn.SojournEstimatorSnapshot{},
		},
	}
}

// Score ranks endpoints by their two-term MRL residual, normalized argmin.
//
// Per candidate:
//   - Read the SojournEstimatorSnapshot from the endpoint's AttributeMap.
//     A nil snapshot means the endpoint is cold (neither digest warm), and
//     that candidate is assigned a residual of 0, so it participates in the
//     normalization on equal footing with a fully drained warm endpoint.
//   - Read live in-flight requests from the observer.
//   - Sum the two-term residual across in-flight requests.
//
// Min-max normalize across candidates so the lowest residual scores 1.0.
// Ties (all residuals equal) yield 1.0 for every endpoint; the max-score
// picker then falls through to its random-shuffle tiebreak.
func (s *Scorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	now := time.Now()
	residuals := make([]float64, len(endpoints))
	minR, maxR := math.MaxFloat64, -math.MaxFloat64

	for i, endpoint := range endpoints {
		r := s.residualFor(endpoint, now)
		residuals[i] = r
		if r < minR {
			minR = r
		}
		if r > maxR {
			maxR = r
		}
	}

	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	span := maxR - minR
	for i, endpoint := range endpoints {
		if span <= 0 {
			// Every endpoint at the same residual, or a single candidate.
			// Neutral score lets the picker random-shuffle.
			scores[endpoint] = 1.0
			continue
		}
		scores[endpoint] = (maxR - residuals[i]) / span
	}
	return scores
}

// residualFor computes the two-term MRL residual for one endpoint.
func (s *Scorer) residualFor(endpoint fwksched.Endpoint, now time.Time) float64 {
	if endpoint == nil || endpoint.GetMetadata() == nil {
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

	inflight := s.observer.InFlightRequestsFor(endpoint.GetMetadata().ID.String())
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
