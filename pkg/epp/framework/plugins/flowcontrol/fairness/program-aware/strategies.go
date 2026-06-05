package programaware

import (
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/prometheus/client_golang/prometheus"
)

// ScoringStrategy determines how program queues are prioritized for dispatch.
// All methods must be safe for concurrent use; Pick() and OnCompleted() may
// execute on different goroutines.
type ScoringStrategy interface {
	Name() string

	// Pick receives the priority band and all queues in that band keyed by
	// program ID (including empty ones for bookkeeping) and returns the
	// selected queue plus per-queue scores for observability.
	// Returns (nil, nil) if no queue is eligible.
	Pick(bandPriority int, queues map[string]QueueInfo) (selected flowcontrol.FlowQueueAccessor, scores map[string]float64)

	// OnPreRequest is called before each request dispatch to reset per-cycle state.
	OnPreRequest(metrics *ProgramMetrics, request *fwksched.InferenceRequest)

	// OnCompleted is called when a response finishes with actual token usage.
	OnCompleted(metrics *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response)

	// EvictProgram drops any per-program state held by this strategy.
	// Called by the plugin's eviction sweep after the program is removed from
	// the central programMetrics map. Strategies with no per-program state
	// (RR) supply a no-op.
	EvictProgram(id string)

	// Collectors returns the Prometheus collectors this strategy owns so the
	// plugin can register them at startup. Strategies with no metrics return
	// nil. Called once during plugin init, after the strategy is constructed.
	Collectors() []prometheus.Collector
}

// QueueInfo bundles read-only data for each queue passed to Pick.
type QueueInfo struct {
	Queue   flowcontrol.FlowQueueAccessor
	Metrics *ProgramMetrics
	Len     int
}

// newStrategy constructs a ScoringStrategy from the plugin config. The
// caller is responsible for passing a Config already merged onto
// DefaultConfig and validated, so every numeric field is known-good here.
func newStrategy(cfg Config) (ScoringStrategy, error) {
	switch cfg.Strategy {
	case "drr":
		return &DRRStrategy{
			weightDeficit:          cfg.WeightDeficit,
			weightHeadWait:         cfg.WeightDRRHeadWait,
			quantumTokens:          cfg.QuantumTokens,
			deficitHalfLifeSeconds: cfg.DeficitHalfLifeSeconds,
			decayFactor:            cfg.DeficitDecayFactor,
		}, nil
	case "", "las":
		return &LASStrategy{
			weightService:   cfg.WeightService,
			weightHeadWait:  cfg.WeightServiceHeadWait,
			decayFactor:     cfg.ServiceDecayFactor,
			halfLifeSeconds: cfg.ServiceHalfLifeSeconds,
		}, nil
	case "rr":
		return &RRStrategy{}, nil
	default:
		return nil, fmt.Errorf("unknown scoring strategy %q: valid values are \"drr\", \"las\", \"rr\"", cfg.Strategy)
	}
}

// rangeNormalize performs min-max normalization: (v - min) / (max - min) → [0, 1].
// Returns 0.5 when min == max (no discriminative signal for this dimension).
func rangeNormalize(v, min, max float64) float64 {
	if max == min {
		return 0.5
	}
	return (v - min) / (max - min)
}

// programIDFromRequest mirrors the fallback rule applied by the request hooks
// so strategies can resolve their per-program state key from the request alone.
func programIDFromRequest(req *fwksched.InferenceRequest) string {
	if req == nil || req.FairnessID == "" {
		return metadata.DefaultFairnessID
	}
	return req.FairnessID
}
