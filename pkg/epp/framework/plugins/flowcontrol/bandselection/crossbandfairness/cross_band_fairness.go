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

// Package crossbandfairness implements a BandSelectionPolicy that shares dispatch opportunities
// across priority bands in proportion to configured weights once the pool is saturated.
//
// Below activationThreshold, bands are offered a dispatch opportunity from highest to lowest
// priority. At or above it, weighted fair queuing engages and every band receives a bounded share
// of dispatch opportunities, so a lower band makes progress while a higher band still has work.
//
// Weights come from the selected distribution:
//   - "linear": weights decrease linearly by ordinal position, ignoring priority values.
//   - "explicit": operator-supplied anchors, interpolating missing priorities by priority value and
//     clamping outside the anchor range.
package crossbandfairness

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// PolicyType is the registration type for the cross-band fairness band selection policy.
const PolicyType = "cross-band-fairness-policy"

// PolicyFactory creates a crossBandFairnessPolicy from JSON config.
func PolicyFactory(name string, params *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var apiCfg apiConfig
	if params != nil {
		if err := params.Decode(&apiCfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal cross-band fairness policy config: %w", err)
		}
	}
	cfg, err := buildConfig(&apiCfg)
	if err != nil {
		return nil, err
	}
	return newCrossBandFairnessPolicy(*cfg).withName(name), nil
}

// weightAnchor associates a priority level with its configured dispatch weight.
type weightAnchor struct {
	priority int
	weight   float64
}

func buildWeightAnchors(ratios map[int]float64) []weightAnchor {
	anchors := make([]weightAnchor, 0, len(ratios))
	for p, w := range ratios {
		anchors = append(anchors, weightAnchor{priority: p, weight: w})
	}
	sort.Slice(anchors, func(i, j int) bool {
		return anchors[i].priority < anchors[j].priority
	})
	return anchors
}

// bandState is the per-band scheduling state of the interleaved schedule.
type bandState struct {
	// virtualTime is the service a band has already claimed, in units of 1/weight. The band with the
	// smallest virtual time holds the strongest claim on the next dispatch opportunity.
	virtualTime float64
	// weight is the band's share of dispatch opportunities, refreshed from the distribution on every
	// cycle because the active priority set changes as workloads come and go.
	weight float64
	// eligible records whether the band is ungated. Gated bands rank last because they cannot
	// dispatch, and they are excluded from the system virtual time because their virtual time tracks
	// it rather than being earned.
	eligible bool
	// present marks the band as part of the active priority set during the current sync sweep.
	present bool
}

// crossBandFairnessPolicy spreads dispatch opportunities across priority bands under saturation.
// The weight distribution is resolved to a function at construction time to avoid per-dispatch
// branching.
//
// All state is owned by the processor's dispatch goroutine, which is the only caller of Rank and
// RecordDispatch, so it needs no synchronization.
type crossBandFairnessPolicy struct {
	name                string
	activationThreshold float64
	weightFn            func(priorities []int, weights []float64)

	anchors []weightAnchor

	// bands holds the scheduling state of every active priority, keyed by priority level.
	bands map[int]*bandState
	// ranked lists the priorities in the order written by the last Rank call, used to attribute a
	// forfeited opportunity to every band passed over ahead of the one that dispatched.
	ranked []int
	// weights is a reusable buffer sized to the active priority set.
	weights []float64
	// active records whether the last Rank call engaged weighted fair queuing. RecordDispatch does
	// not receive saturation, so it reads the decision taken for the cycle it reports on.
	active bool
}

var _ flowcontrol.BandSelectionPolicy = &crossBandFairnessPolicy{}

func newCrossBandFairnessPolicy(cfg config) *crossBandFairnessPolicy {
	p := &crossBandFairnessPolicy{
		name:                PolicyType,
		activationThreshold: cfg.activationThreshold,
		bands:               make(map[int]*bandState),
	}
	switch cfg.distribution {
	case distributionLinear:
		p.weightFn = computeWeightsLinear
	case distributionExplicit:
		p.anchors = buildWeightAnchors(cfg.ratios)
		p.weightFn = func(priorities []int, weights []float64) {
			computeWeightsExplicit(p.anchors, priorities, weights)
		}
	}
	return p
}

func (p *crossBandFairnessPolicy) withName(name string) *crossBandFairnessPolicy {
	if name != "" {
		p.name = name
	}
	return p
}

func (p *crossBandFairnessPolicy) Name() string {
	return p.name
}

func (p *crossBandFairnessPolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: PolicyType,
		Name: p.name,
	}
}

// Rank orders bands by ascending virtual time, so the band holding the strongest claim on its
// configured share is offered the next dispatch opportunity first. Bands tied on virtual time keep
// priority order, and gated bands are ranked last because they cannot dispatch.
//
// Below the activation threshold, and whenever fewer than two bands are active, the order buffer is
// left untouched: the framework pre-fills it with strict highest-to-lowest order. Virtual times are
// retained across those cycles, so saturation oscillating around the threshold resumes the round it
// left rather than restarting it.
func (p *crossBandFairnessPolicy) Rank(_ context.Context, params flowcontrol.BandSelectionParameters, order []int) {
	p.active = params.Saturation >= p.activationThreshold && len(params.Priorities) > 1
	if !p.active {
		p.ranked = p.ranked[:0]
		return
	}

	weights := p.weightsBuffer(len(params.Priorities))
	p.weightFn(params.Priorities, weights)
	p.syncBands(params, weights)

	sort.SliceStable(order, func(a, b int) bool {
		left := p.bands[params.Priorities[order[a]]]
		right := p.bands[params.Priorities[order[b]]]
		if left.eligible != right.eligible {
			return left.eligible
		}
		if !left.eligible {
			return false
		}
		return left.virtualTime < right.virtualTime
	})

	p.ranked = p.ranked[:0]
	for _, i := range order {
		p.ranked = append(p.ranked, params.Priorities[i])
	}
}

// RecordDispatch charges the dispatching band for the opportunity it consumed, advancing its
// virtual time by one stride (1/weight) so lighter bands fall behind faster than heavier ones.
//
// Bands ranked ahead of the dispatching band were offered the opportunity and did not take it, so
// they were empty or failed to dispatch. Levelling them with the band that was served keeps an idle
// band from banking a claim it would spend monopolizing dispatch once traffic returns, which is the
// only signal the policy has about occupancy.
func (p *crossBandFairnessPolicy) RecordDispatch(_ context.Context, priority int) {
	if !p.active {
		return
	}
	served, ok := p.bands[priority]
	if !ok {
		return
	}
	for _, passed := range p.ranked {
		if passed == priority {
			break
		}
		if b, ok := p.bands[passed]; ok && b.eligible {
			b.virtualTime = served.virtualTime
		}
	}
	served.virtualTime += 1.0 / served.weight
}

// syncBands refreshes per-band scheduling state against the active priority set: departed bands are
// dropped, bands that are new or gated are held at the system virtual time, and all virtual times
// are rebased so the smallest is zero.
//
// Weights are derived from every active band rather than only the ungated ones, so a band held back
// transiently does not reshuffle the shares of the bands still dispatching.
func (p *crossBandFairnessPolicy) syncBands(params flowcontrol.BandSelectionParameters, weights []float64) {
	// The system virtual time is the smallest virtual time among the bands that were in service on
	// the previous cycle and still are, which is the position the next band to be served occupies. A
	// band entering below it would claim service it never deferred. Bands gated on the previous cycle
	// are excluded because their virtual time tracked this value rather than being earned, and
	// including them would hold it at a stale position.
	systemVirtualTime := math.Inf(1)
	for i, priority := range params.Priorities {
		if params.Saturation >= params.Ceilings[i] {
			continue
		}
		if b, ok := p.bands[priority]; ok && b.eligible && b.virtualTime < systemVirtualTime {
			systemVirtualTime = b.virtualTime
		}
	}
	// No band has an earned position: on first activation, or when every band is new or gated, they
	// all start level.
	if math.IsInf(systemVirtualTime, 1) {
		systemVirtualTime = 0
	}

	for _, b := range p.bands {
		b.present = false
	}
	for i, priority := range params.Priorities {
		eligible := params.Saturation < params.Ceilings[i]
		b, ok := p.bands[priority]
		if !ok {
			b = &bandState{}
			p.bands[priority] = b
		}
		// A band that is newly active or currently gated holds no claim. Tracking it at the system
		// virtual time leaves it first in line the moment it can dispatch, without the catch-up burst
		// it would take if it kept the virtual time it held before.
		if !ok || !eligible {
			b.virtualTime = systemVirtualTime
		}
		b.weight = weights[i]
		b.eligible = eligible
		b.present = true
	}
	for priority, b := range p.bands {
		if !b.present {
			delete(p.bands, priority)
		}
	}

	// Rebase to bound virtual time growth over long runs. A uniform shift leaves the order unchanged.
	if systemVirtualTime != 0 {
		for _, b := range p.bands {
			b.virtualTime -= systemVirtualTime
		}
	}
}

func (p *crossBandFairnessPolicy) weightsBuffer(n int) []float64 {
	if cap(p.weights) < n {
		p.weights = make([]float64, n)
	}
	return p.weights[:n]
}

// computeWeightsLinear assigns weights by ordinal position, ignoring priority values.
// w_i = N - i
func computeWeightsLinear(priorities []int, weights []float64) {
	n := len(priorities)
	for i := range priorities {
		weights[i] = float64(n - i)
	}
}

// computeWeightsExplicit maps each priority to a dispatch weight using the configured anchors.
//
// Unlisted priorities are resolved as follows:
//   - If outside the configured anchor range, the weight is clamped to the nearest boundary anchor.
//     When only a single anchor is configured, all priorities receive that anchor's weight.
//   - If between two configured anchors, the weight is interpolated linearly over priority value
//     (not ordinal rank). Value-based interpolation makes the mapping a total function from priority
//     to weight determined entirely by configuration, so shares remain stable regardless of which
//     dynamic priority bands are currently provisioned or garbage-collected.
func computeWeightsExplicit(anchors []weightAnchor, priorities []int, out []float64) {
	if len(anchors) == 0 {
		return
	}
	lowest := anchors[0]
	highest := anchors[len(anchors)-1]

	for i, p := range priorities {
		if p <= lowest.priority {
			out[i] = lowest.weight
			continue
		}
		if p >= highest.priority {
			out[i] = highest.weight
			continue
		}
		for j := 0; j < len(anchors)-1; j++ {
			if p <= anchors[j+1].priority {
				if p == anchors[j+1].priority {
					out[i] = anchors[j+1].weight
					break
				}
				ratio := float64(p-anchors[j].priority) / float64(anchors[j+1].priority-anchors[j].priority)
				out[i] = anchors[j].weight + ratio*(anchors[j+1].weight-anchors[j].weight)
				break
			}
		}
	}
}
