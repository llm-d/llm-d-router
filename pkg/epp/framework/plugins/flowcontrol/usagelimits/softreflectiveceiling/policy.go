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

// Package softreflectiveceiling implements a UsageLimitPolicy that gates
// lower-priority bands proportionally as saturation rises. For each band i
// (priorities ordered highest first) it computes a reflective ceiling
//
//	ceiling[i] = 1 - i*saturation/(N-1)
//
// When saturation reaches a band's ceiling the policy alternates ceiling=1.0
// and ceiling=0.0 across calls so that, on average, each gated band's
// ceiling is open on 1/period of calls, where
// period = round(saturation/(1-saturation)).
//
// A single policy-wide tick counter drives the alternation. All gated bands
// share it, so on any given call either every gated band's ceiling is open
// or every gated band's ceiling is closed. This monotonicity is required by
// the dispatch loop, which stops at the first band whose ceiling is at or
// below current saturation. Per-band counters would drift out of phase as
// bands enter gating at different saturations, so a lower gated band's
// open ticks would repeatedly land on a higher gated band's closed ticks
// and the lower band would starve.
//
// The single tick is small bounded state used only for dispatch spreading,
// which the UsageLimitPolicy contract permits. Signal conditioning (trend
// detection, smoothing) is not permitted here; it belongs in the
// SaturationDetector layer.
package softreflectiveceiling

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync/atomic"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// PolicyType is the registration string. The YAML "type:" field in
// pluginsCustomConfig must equal this value for the loader to find the
// factory.
const PolicyType = "soft-reflective-ceiling-policy"

// Factory creates a soft-reflective ceiling policy instance. The algorithm
// has no tunable parameters, so any provided parameters block must be empty.
// The framework's strict decoder (DisallowUnknownFields) surfaces config
// typos at load time rather than silently ignoring them.
func Factory(name string, rawConfig *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if rawConfig != nil {
		var empty struct{}
		if err := rawConfig.Decode(&empty); err != nil {
			return nil, fmt.Errorf("soft-reflective-ceiling-policy takes no parameters: %w", err)
		}
	}
	logger := logr.Discard()
	if handle != nil {
		logger = log.FromContext(handle.Context())
	}
	return newPolicy(name, logger), nil
}

type policy struct {
	name string

	// tick advances once per ComputeLimit call in which at least one band is
	// in the alternating regime. All gated bands read the same value, so the
	// per-call gating decision is monotone across ranks -- required because
	// the dispatch loop stops at the first band whose ceiling has gated.
	tick atomic.Int64
}

var _ flowcontrol.UsageLimitPolicy = (*policy)(nil)

func newPolicy(name string, logger logr.Logger) *policy {
	p := &policy{name: name}
	logger.WithName(p.TypedName().String()).V(logutil.DEFAULT).Info("Creating new SoftReflectiveCeilingPolicy")
	return p
}

func (p *policy) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: PolicyType, Name: p.name}
}

// ComputeLimit returns per-band ceilings. priorities[0] is the highest
// priority band and receives ceiling=1.0. Lower bands whose reflective
// ceiling has been reached alternate between 1.0 and 0.0 with period
// round(saturation/(1-saturation)), producing a proportional duty cycle.
func (p *policy) ComputeLimit(_ context.Context, saturation float64, priorities []int) []float64 {
	n := len(priorities)
	ceilings := make([]float64, n)
	if n == 0 {
		return ceilings
	}
	if n == 1 {
		ceilings[0] = 1.0
		return ceilings
	}

	// Ceilings decrease monotonically with rank, so any band is at or past
	// its reflective ceiling iff the lowest band is (whose ceiling is
	// 1 - saturation). Advance the shared tick only when at least one band
	// is in the alternating regime; below that saturation the policy is a
	// pure function of its inputs.
	inAlternatingRegime := saturation >= 1.0-saturation && saturation < 1.0

	var open bool
	if inAlternatingRegime {
		// 1e-9 guards against round-off when saturation is very near 1.0.
		period := int64(math.Max(1, math.Round(saturation/(1.0-saturation+1e-9))))
		open = p.tick.Add(1)%period == 0
	}

	for i := range priorities {
		if i == 0 {
			ceilings[i] = 1.0
			continue
		}

		reflectiveCeiling := 1.0 - float64(i)*saturation/float64(n-1)

		switch {
		case saturation < reflectiveCeiling:
			ceilings[i] = 1.0
		case saturation >= 1.0:
			ceilings[i] = 0.0
		default:
			if open {
				ceilings[i] = 1.0
			} else {
				ceilings[i] = 0.0
			}
		}
	}

	return ceilings
}
