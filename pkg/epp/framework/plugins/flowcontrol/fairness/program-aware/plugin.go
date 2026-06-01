// Package programaware implements a flow-control fairness policy that schedules
// programs using their accumulated metrics using scoring strategies (LAS, DRR, or RR).
package programaware

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

// ProgramAwarePluginType is the registered type name for this plugin.
const ProgramAwarePluginType = "program-aware-fairness"

// Config holds the JSON-decoded configuration for the plugin.
type Config struct {
	// Strategy selects the fairness scoring algorithm used by Pick().
	// Valid values: "las" (default), "drr", "rr".
	//
	//   "las"    — attained service fairness: tracks time-decayed weighted tokens
	//              consumed per program. Programs with lower attained service are
	//              promoted. Directly targets fair resource allocation.
	//
	//   "drr"    — Deficit Round Robin adapted for tokens [Shreedhar & Varghese 1995].
	//              Each round every active queue earns a token quantum; actual token
	//              usage is deducted at response completion. Provides provably
	//              proportional fairness independent of request rate or size.
	//
	//   "rr"     — Simple round-robin: cycles through program queues in sorted order,
	//              skipping empty queues. Matches the upstream round-robin fairness
	//              policy. No token or service tracking.
	Strategy string `json:"strategy"`

	// --- DRR weights (only used when strategy == "drr") ---

	// WeightDeficit is the weight for the deficit counter signal.
	// Default: 0.7.
	WeightDeficit *float64 `json:"weightDeficit,omitempty"`

	// WeightDRRHeadWait is the weight for head-of-queue age in DRR.
	// Default: 0.3.
	WeightDRRHeadWait *float64 `json:"weightDrrHeadWait,omitempty"`

	// QuantumTokens is the token budget added to each non-empty queue per Pick() cycle.
	// Default: 1000.
	QuantumTokens *int64 `json:"quantumTokens,omitempty"`

	// DeficitHalfLifeSeconds enables time-based decay for the DRR deficit counter.
	// Defines the half-life in seconds: deficit decays to 50% after this duration.
	// Prevents unbounded deficit accumulation for programs that stop sending requests.
	// Default: 60 (deficit halves every 60s). Set to 0 to disable decay.
	DeficitHalfLifeSeconds *float64 `json:"deficitHalfLifeSeconds,omitempty"`

	// --- Service weights (only used when strategy == "las") ---

	// WeightService is the weight for the inverted attained service signal.
	// Programs with lower attained service score higher. Default: 0.8.
	WeightService *float64 `json:"weightService,omitempty"`

	// WeightServiceHeadWait is the weight for head-of-queue age in service strategy.
	// Acts as a tiebreaker for cold start. Default: 0.2.
	WeightServiceHeadWait *float64 `json:"weightServiceHeadWait,omitempty"`

	// ServiceDecayFactor controls how quickly old service is forgotten.
	// Applied to each program's attained service every Pick() cycle.
	// Higher values (closer to 1.0) = longer memory. Default: 0.995.
	// Ignored when ServiceHalfLifeSeconds is set.
	ServiceDecayFactor *float64 `json:"serviceDecayFactor,omitempty"`

	// ServiceHalfLifeSeconds enables time-based decay for the service strategy.
	// Defines the half-life in seconds: service decays to 50% after this duration.
	// When set (> 0), overrides ServiceDecayFactor with wall-clock based decay.
	// Example: 30 = service halves every 30s regardless of Pick() frequency.
	ServiceHalfLifeSeconds *float64 `json:"serviceHalfLifeSeconds,omitempty"`

	// --- RR options (only used when strategy == "rr") ---

	// DeferRRCursor controls whether the round-robin cursor advances in
	// Pick() (false, default) or is deferred to OnPreRequest() so the
	// cursor only moves after a real dispatch. Default: false.
	DeferRRCursor *bool `json:"deferRRCursor,omitempty"`
}

// Compile-time interface assertions.
var (
	_ flowcontrol.FairnessPolicy  = &ProgramAwarePlugin{}
	_ fwkrc.DataProducer          = &ProgramAwarePlugin{}
	_ fwkrc.PreRequest            = &ProgramAwarePlugin{}
	_ fwkrc.ResponseBodyProcessor = &ProgramAwarePlugin{}
)

// ProgramAwarePluginFactory creates a new ProgramAwarePlugin from JSON config.
// Example config: {"strategy": "drr"}
//
//nolint:revive
func ProgramAwarePluginFactory(name string, parameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	cfg := Config{Strategy: "las"}
	if parameters != nil {
		if err := parameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", ProgramAwarePluginType, name, err)
		}
	}
	strategy, err := newStrategy(cfg)
	if err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", ProgramAwarePluginType, name, err)
	}
	p := &ProgramAwarePlugin{
		name:     name,
		strategy: strategy,
	}
	// Register Prometheus collectors via the framework's recorder.
	// Both handle and handle.Metrics() may be nil in test paths.
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			for _, c := range GetCollectors() {
				reg.MustRegister(c)
			}
		}
	}
	return p, nil
}

// ProgramAwarePlugin implements a FairnessPolicy that selects which program's
// queue to service next, and request lifecycle hooks that track per-program metrics.
//
// Fairness behaviour is determined by the configured ScoringStrategy (default: LAS).
// Program identity comes from the x-gateway-inference-fairness-id request header.
//
//nolint:revive
type ProgramAwarePlugin struct {
	name     string
	strategy ScoringStrategy

	// programMetrics stores aggregated metrics per program.
	// Key: program ID (string), Value: *ProgramMetrics.
	programMetrics sync.Map

	// requestTimestamps tracks when Pick() dispatched each request,
	// used to compute flow-control queue wait time in PreRequest.
	// Key: request ID (string), Value: time.Time.
	requestTimestamps sync.Map
}

// TypedName returns the plugin type and instance name.
func (p *ProgramAwarePlugin) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: ProgramAwarePluginType,
		Name: p.name,
	}
}

// getStrategy returns the configured strategy, falling back to LAS for zero-value
// plugin instances constructed directly in tests.
func (p *ProgramAwarePlugin) getStrategy() ScoringStrategy {
	if p.strategy == nil {
		return &LASStrategy{
			weightService:  defaultServiceWeightService,
			weightHeadWait: defaultServiceWeightHeadWait,
			decayFactor:    defaultServiceDecayFactor,
		}
	}
	return p.strategy
}

// --- FairnessPolicy interface ---

// NewState creates per-PriorityBand state. This plugin uses its own sync.Map
// for all state, so no per-band state is needed.
func (p *ProgramAwarePlugin) NewState(_ context.Context) any {
	return nil
}

// Pick selects which program queue to service next by delegating to the
// configured ScoringStrategy. The strategy receives all queues and returns
// the selected queue plus per-queue scores for observability.
func (p *ProgramAwarePlugin) Pick(_ context.Context, band flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error) {
	start := time.Now()
	defer func() {
		pickLatencyUs.Observe(float64(time.Since(start).Microseconds()))
	}()

	if band == nil {
		return nil, nil //nolint:nilnil
	}

	strategy := p.getStrategy()

	// Build QueueInfo map for the strategy.
	infos := make(map[string]QueueInfo)
	band.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) (keepIterating bool) {
		if queue == nil {
			return true
		}
		id := queue.FlowKey().ID
		infos[id] = QueueInfo{
			Queue:   queue,
			Metrics: p.getOrCreateMetrics(id),
			Len:     queue.Len(),
		}
		return true
	})

	// Strategy owns scoring, normalization, and internal bookkeeping.
	bestQueue, scores := strategy.Pick(band.Priority(), infos)

	// Emit per-queue scores for observability.
	for id, score := range scores {
		queueScore.WithLabelValues(id).Set(score)
	}

	// Record the selected item's enqueue time so PreRequest can compute
	// the actual flow-control queue wait time (enqueue → dispatch).
	if bestQueue != nil {
		if head := bestQueue.PeekHead(); head != nil {
			p.requestTimestamps.Store(head.OriginalRequest().ID(), head.EnqueueTime())
		}
	}

	fairnessIndex.Set(p.computeFairnessIndex())

	return bestQueue, nil
}

// getOrCreateMetrics returns the ProgramMetrics for the given program ID, creating if needed.
func (p *ProgramAwarePlugin) getOrCreateMetrics(programID string) *ProgramMetrics {
	if metricsRaw, ok := p.programMetrics.Load(programID); ok {
		return metricsRaw.(*ProgramMetrics)
	}
	m := &ProgramMetrics{}
	actual, _ := p.programMetrics.LoadOrStore(programID, m)
	return actual.(*ProgramMetrics)
}

// computeFairnessIndex returns Jain's Fairness Index over the service rate
// (weighted tokens/sec) for each program. Equal service rates = perfect fairness.
// Returns 1.0 when fewer than 2 programs have rate data.
func (p *ProgramAwarePlugin) computeFairnessIndex() float64 {
	var sum, sumSq float64
	var n float64
	p.programMetrics.Range(func(_, value any) bool {
		m := value.(*ProgramMetrics)
		x := m.ServiceRate()
		if x == 0 {
			return true
		}
		sum += x
		sumSq += x * x
		n++
		return true
	})
	if n <= 1 || sumSq == 0 {
		return 1.0
	}
	return (sum * sum) / (n * sumSq)
}
