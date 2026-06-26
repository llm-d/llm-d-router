// Package priorityholdback implements a UsageLimitPolicy that computes differentiated admission
// ceilings per priority level. Lower-priority traffic is gated at lower saturation thresholds,
// reserving capacity for higher-priority work as the pool approaches saturation.
//
// Behavior is configured via two independent parameters:
//   - shape: the interpolation curve (currently "linear"; future: sigmoid, exponential, etc.).
//   - domain: how priorities map to positions ("rank" for ordinal, "value" for proportional).
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
	cMin      float64
	computeFn func(cMin, cMax float64, priorities []int) (ceilings []float64)
}

var _ flowcontrol.UsageLimitPolicy = &priorityHoldbackPolicy{}

func newPriorityHoldbackPolicy(cfg config) *priorityHoldbackPolicy {
	var fn func(cMin, cMax float64, priorities []int) (ceilings []float64)
	switch cfg.domain {
	case domainRank:
		fn = computeLimitStepwiseSpread
	case domainValue:
		fn = computeLimitLinearProportional
	}
	return &priorityHoldbackPolicy{
		name:      PolicyType,
		cMax:      cfg.maxCeiling,
		cMin:      cfg.minCeiling,
		computeFn: fn,
	}
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
// holdback is bypassed (ceiling = cMax) to preserve work-conserving behavior.
func (p *priorityHoldbackPolicy) ComputeLimit(_ context.Context, _ float64, priorities []int) (ceilings []float64) {
	if len(priorities) == 0 {
		return []float64{}
	}
	if len(priorities) == 1 {
		return []float64{p.cMax}
	}
	// Ceilings are monotonically decreasing as priorities are ordered from highest to lowest per UsageLimitPolicy contract.
	// New strategies (e.g. sigmoid/static definition) could require explicit monotizing sweep.
	return p.computeFn(p.cMin, p.cMax, priorities)
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
