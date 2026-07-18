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

// Package priorityholdback implements a UsageLimitPolicy that computes differentiated admission
// ceilings per priority level. Lower-priority traffic is gated at lower saturation thresholds,
// reserving capacity for higher-priority work as the pool approaches saturation.
//
// Behavior is configured via two independent parameters:
//   - shape: the interpolation curve (currently "linear"; future: sigmoid, exponential, etc.).
//   - domain: how priorities map to positions ("rank" for ordinal, "value" for proportional,
//     "explicit" for a direct operator-supplied map).
package priorityholdback

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// PolicyType is the registration type for the priority holdback usage limit policy.
const PolicyType = "priority-holdback-policy"

// PolicyFactory creates a priorityHoldbackPolicy from JSON config.
func PolicyFactory(name string, params *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var apiCfg apiConfig
	if params != nil {
		if err := params.Decode(&apiCfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal priority holdback policy config: %w", err)
		}
	}
	cfg, err := buildConfig(&apiCfg)
	if err != nil {
		return nil, err
	}
	return newPriorityHoldbackPolicy(*cfg).withName(name), nil
}

// priorityHoldbackPolicy gates lower-priority traffic as saturation rises. The gating strategy
// is resolved to a function at construction time to avoid per-dispatch branching.
type priorityHoldbackPolicy struct {
	name      string
	cMax      float64
	computeFn func(priorities []int) (ceilings []float64)
	// enableSinglePriorityBypass controls whether a single active priority skips holdback
	// and receives cMax directly. This optimization applies to algorithmic domains (rank, value)
	// where the interpolation math degenerates with one input. The explicit domain does not use it
	// because the operator already supplied the ceiling for that priority.
	enableSinglePriorityBypass bool

	domain   string
	ceilings map[int]float64
}

var _ flowcontrol.UsageLimitPolicy = &priorityHoldbackPolicy{}

func newPriorityHoldbackPolicy(cfg config) *priorityHoldbackPolicy {
	p := &priorityHoldbackPolicy{
		name:     PolicyType,
		cMax:     cfg.maxCeiling,
		domain:   cfg.domain,
		ceilings: cfg.ceilings,
	}
	switch cfg.domain {
	case DomainRank:
		p.enableSinglePriorityBypass = true
		p.computeFn = func(priorities []int) []float64 {
			return computeLimitStepwiseSpread(cfg.minCeiling, cfg.maxCeiling, priorities)
		}
	case DomainValue:
		p.enableSinglePriorityBypass = true
		p.computeFn = func(priorities []int) []float64 {
			return computeLimitLinearProportional(cfg.minCeiling, cfg.maxCeiling, priorities)
		}
	case DomainExplicit:
		p.enableSinglePriorityBypass = false
		p.computeFn = func(priorities []int) []float64 {
			return computeLimitExplicit(cfg.ceilings, priorities)
		}
	}
	return p
}

func (p *priorityHoldbackPolicy) Domain() string {
	return p.domain
}

func (p *priorityHoldbackPolicy) Ceilings() map[int]float64 {
	if p.ceilings == nil {
		return nil
	}
	out := make(map[int]float64, len(p.ceilings))
	for k, v := range p.ceilings {
		out[k] = v
	}
	return out
}

func (p *priorityHoldbackPolicy) withName(name string) *priorityHoldbackPolicy {
	if name != "" {
		p.name = name
	}
	return p
}

func (p *priorityHoldbackPolicy) Name() string {
	return p.name
}

func (p *priorityHoldbackPolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: PolicyType,
		Name: p.name,
	}
}

// ComputeLimit returns an admission ceiling for each priority. With a single active priority,
// algorithmic domains bypass holdback (ceiling = cMax) to preserve work-conserving behavior.
// The explicit domain always uses the configured map value regardless of input length.
func (p *priorityHoldbackPolicy) ComputeLimit(_ context.Context, _ float64, priorities []int) (ceilings []float64) {
	if len(priorities) == 0 {
		return []float64{}
	}
	if len(priorities) == 1 && p.enableSinglePriorityBypass {
		return []float64{p.cMax}
	}
	// Ceilings are monotonically decreasing as priorities are ordered from highest to lowest per UsageLimitPolicy contract.
	// New strategies (e.g. sigmoid/static definition) could require explicit monotizing sweep.
	return p.computeFn(priorities)
}

// computeLimitStepwiseSpread divides [cMin, cMax] into equal steps by rank.
// c_i = cMax - i * (cMax - cMin) / (N - 1)
func computeLimitStepwiseSpread(cMin, cMax float64, priorities []int) (ceilings []float64) {
	ceilings = make([]float64, len(priorities))
	spread := cMax - cMin
	n := float64(len(priorities) - 1)
	for i := range priorities {
		ceilings[i] = cMax - float64(i)*spread/n
	}
	return ceilings
}

// computeLimitLinearProportional scales ceilings proportionally to numerical priority values.
// r_i = (p_i - pMin) / (pMax - pMin)
// c_i = cMin + r_i * (cMax - cMin)
func computeLimitLinearProportional(cMin, cMax float64, priorities []int) (ceilings []float64) {
	ceilings = make([]float64, len(priorities))
	pMin := float64(priorities[len(priorities)-1])
	pMax := float64(priorities[0])
	pRange := pMax - pMin
	// All priorities share the same value; no differentiation possible.
	// Also avoid division by zero in the formula for r_i.
	if pRange == 0 {
		for i := range priorities {
			ceilings[i] = cMax
		}
		return ceilings
	}
	spread := cMax - cMin
	for i := range priorities {
		r := (float64(priorities[i]) - pMin) / pRange
		ceilings[i] = cMin + r*spread
	}
	return ceilings
}

// computeLimitExplicit looks up each configured priority ceiling.
// Precondition: configuration validation guarantees that every active
// priority band has a corresponding ceiling entry.
func computeLimitExplicit(ceilings map[int]float64, priorities []int) []float64 {
	result := make([]float64, len(priorities))
	for i, p := range priorities {
		result[i] = ceilings[p]
	}
	return result
}
