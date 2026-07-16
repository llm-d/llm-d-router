// Package softreflectiveceiling implements a UsageLimitPolicy that gates
// lower-priority bands proportionally as saturation rises. For each band i
// (priorities ordered highest first) it computes a reflective ceiling
//
//	ceiling[i] = 1 - i*saturation/(N-1)
//
// When saturation reaches a band's ceiling the policy alternates ceiling=1.0
// and ceiling=0.0 across calls so that, on average, the band dispatches at
// 1/period of the tick rate, where period = round(saturation/(1-saturation)).
// Band 0 (highest priority) is never gated.
//
// The per-band tick counter is bounded state used only to spread dispatch
// evenly across ticks. It is not signal conditioning (trend detection,
// smoothing) -- that responsibility belongs to the SaturationDetector layer
// per the flowcontrol.UsageLimitPolicy contract.
//
// The proportional behavior is encoded entirely in the ComputeLimit return
// values, which requires per-band tick state. This is a deliberate deviation
// from the UsageLimitPolicy interface guidance that policies be stateless:
// the fractional open rate is only observable across ticks. Counters are
// keyed by priority value (not rank) so a band's alternation state is
// preserved when the active priority set changes at runtime -- e.g. when a
// new priority arrives via dynamic InferenceObjective provisioning in the
// flow-control registry. Concurrency safety is provided by sync.Map plus
// pointer-atomic counters.
package softreflectiveceiling

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
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

	// counters maps priority value to a tick counter. Keyed by priority (not
	// rank) so that adding or removing a priority band at runtime does not
	// reassign an existing band's counter to a different priority.
	counters sync.Map
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
// priority band and is never gated. For lower bands, when saturation reaches
// the band's reflective ceiling, the band alternates between ceiling=1.0 and
// ceiling=0.0 across calls with period round(saturation/(1-saturation)),
// approximating proportional dispatch.
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

	for i, priority := range priorities {
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
			// 1e-9 guards against round-off when saturation is very near 1.0.
			period := int64(math.Max(1, math.Round(saturation/(1.0-saturation+1e-9))))
			tick := p.counterFor(priority).Add(1)
			if tick%period == 0 {
				ceilings[i] = 1.0
			} else {
				ceilings[i] = 0.0
			}
		}
	}

	return ceilings
}

// counterFor returns the tick counter for a given priority, creating it on
// first use. sync.Map.LoadOrStore ensures concurrent first-writers converge
// on a single counter instance without an external mutex.
func (p *policy) counterFor(priority int) *atomic.Int64 {
	if v, ok := p.counters.Load(priority); ok {
		return v.(*atomic.Int64)
	}
	fresh := new(atomic.Int64)
	actual, _ := p.counters.LoadOrStore(priority, fresh)
	return actual.(*atomic.Int64)
}
