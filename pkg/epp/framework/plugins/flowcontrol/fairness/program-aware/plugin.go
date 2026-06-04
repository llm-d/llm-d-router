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

// enqueueTimeAttributeKey is the per-request attribute key under which Pick
// stashes the flow-control enqueue timestamp for PreRequest to read back.
const enqueueTimeAttributeKey = "program-aware/enqueue-time"

// Config holds the JSON-decoded configuration for the plugin. JSON parameters
// are merged onto a copy of DefaultConfig, so any field omitted from the
// user's JSON keeps its default value.
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
	WeightDeficit float64 `json:"weightDeficit,omitempty"`

	// WeightDRRHeadWait is the weight for head-of-queue age in DRR.
	WeightDRRHeadWait float64 `json:"weightDrrHeadWait,omitempty"`

	// QuantumTokens is the token budget added to each non-empty queue per Pick() cycle.
	QuantumTokens int64 `json:"quantumTokens,omitempty"`

	// DeficitHalfLifeSeconds is the half-life of the DRR deficit counter.
	// Deficit decays to 50% after this duration. 0 disables time-based decay
	// (DeficitDecayFactor takes over). When > 0, this takes precedence over
	// DeficitDecayFactor.
	DeficitHalfLifeSeconds float64 `json:"deficitHalfLifeSeconds,omitempty"`

	// DeficitDecayFactor is the per-Pick factor decay for the DRR deficit
	// counter when DeficitHalfLifeSeconds is 0. Each Pick() multiplies the
	// deficit of inactive queues (Len==0 and no in-flight requests) by this
	// factor. Must be in [0, 1); 0 disables factor decay.
	DeficitDecayFactor float64 `json:"deficitDecayFactor,omitempty"`

	// --- Service weights (only used when strategy == "las") ---

	// WeightService is the weight for the inverted attained service signal.
	// Programs with lower attained service score higher.
	WeightService float64 `json:"weightService,omitempty"`

	// WeightServiceHeadWait is the weight for head-of-queue age in service strategy.
	// Acts as a tiebreaker for cold start.
	WeightServiceHeadWait float64 `json:"weightServiceHeadWait,omitempty"`

	// ServiceDecayFactor controls how quickly old service is forgotten.
	// Applied to each program's attained service every Pick() cycle.
	// Higher values (closer to 1.0) = longer memory. Must be in (0, 1].
	// Ignored when ServiceHalfLifeSeconds is set.
	ServiceDecayFactor float64 `json:"serviceDecayFactor,omitempty"`

	// ServiceHalfLifeSeconds is the half-life of the LAS attained-service
	// counter when set (> 0); overrides ServiceDecayFactor with wall-clock
	// based decay. Service decays to 50% after this duration.
	ServiceHalfLifeSeconds float64 `json:"serviceHalfLifeSeconds,omitempty"`
}

// DefaultConfig is the canonical Config used when JSON parameters are absent
// or partial. The factory makes a copy of this value before decoding, so
// every fairness-plugin default lives in one place.
var DefaultConfig = Config{
	Strategy:               "las",
	WeightDeficit:          0.8,
	WeightDRRHeadWait:      0.2,
	QuantumTokens:          1000,
	DeficitHalfLifeSeconds: 60,
	DeficitDecayFactor:     0,
	WeightService:          0.8,
	WeightServiceHeadWait:  0.2,
	ServiceDecayFactor:     0.995,
	ServiceHalfLifeSeconds: 0,
}

// validate checks that numeric fields fall in the ranges the scoring
// strategies assume. Defaults from DefaultConfig already satisfy every rule;
// validation only catches user overrides that fall outside the safe range.
func (c Config) validate() error {
	if c.WeightDeficit < 0 {
		return fmt.Errorf("weightDeficit must be >= 0, got %v", c.WeightDeficit)
	}
	if c.WeightDRRHeadWait < 0 {
		return fmt.Errorf("weightDrrHeadWait must be >= 0, got %v", c.WeightDRRHeadWait)
	}
	if c.WeightService < 0 {
		return fmt.Errorf("weightService must be >= 0, got %v", c.WeightService)
	}
	if c.WeightServiceHeadWait < 0 {
		return fmt.Errorf("weightServiceHeadWait must be >= 0, got %v", c.WeightServiceHeadWait)
	}
	if c.QuantumTokens <= 0 {
		return fmt.Errorf("quantumTokens must be > 0, got %d", c.QuantumTokens)
	}
	if c.DeficitHalfLifeSeconds < 0 {
		return fmt.Errorf("deficitHalfLifeSeconds must be >= 0, got %v", c.DeficitHalfLifeSeconds)
	}
	if c.ServiceHalfLifeSeconds < 0 {
		return fmt.Errorf("serviceHalfLifeSeconds must be >= 0, got %v", c.ServiceHalfLifeSeconds)
	}
	if c.DeficitDecayFactor < 0 || c.DeficitDecayFactor >= 1 {
		return fmt.Errorf("deficitDecayFactor must be in [0, 1), got %v", c.DeficitDecayFactor)
	}
	if c.ServiceDecayFactor <= 0 || c.ServiceDecayFactor > 1 {
		return fmt.Errorf("serviceDecayFactor must be in (0, 1], got %v", c.ServiceDecayFactor)
	}
	return nil
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
// The qualified name matches sibling fairness factories
// (roundrobin.RoundRobinFairnessPolicyFactory, globalstrict.GlobalStrictFairnessPolicyFactory).
//
//nolint:revive // factory name matches sibling fairness plugins; see comment above.
func ProgramAwarePluginFactory(name string, parameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	cfg := DefaultConfig
	if parameters != nil {
		if err := parameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", ProgramAwarePluginType, name, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", ProgramAwarePluginType, name, err)
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
}

// TypedName returns the plugin type and instance name.
func (p *ProgramAwarePlugin) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: ProgramAwarePluginType,
		Name: p.name,
	}
}

// getStrategy returns the configured strategy, falling back to a strategy
// built from DefaultConfig for zero-value plugin instances constructed
// directly in tests. DefaultConfig is known-valid so newStrategy cannot fail.
func (p *ProgramAwarePlugin) getStrategy() ScoringStrategy {
	if p.strategy == nil {
		s, _ := newStrategy(DefaultConfig)
		return s
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

	// Emit per-queue scores for non-empty queues.
	for id, score := range scores {
		queueScore.WithLabelValues(id).Set(score)
	}
	// Emit deficit for all queues, including empty ones, so decay is observable.
	for id, qi := range infos {
		if qi.Metrics != nil {
			deficitTokens.WithLabelValues(id).Set(float64(qi.Metrics.Deficit()))
		}
	}

	// Stash the selected item's enqueue time on the InferenceRequest's own
	// attribute store so PreRequest can compute the flow-control queue wait
	// time (enqueue → dispatch). The attribute lifetime is the request
	// lifetime, so an abandoned request cannot leak into a side map.
	// Pick precedes PreRequest on a single request goroutine; no concurrent writes.
	if bestQueue != nil {
		if head := bestQueue.PeekHead(); head != nil {
			if req := head.OriginalRequest().InferenceRequest(); req != nil {
				req.PutAttribute(enqueueTimeAttributeKey, head.EnqueueTime())
			}
		}
	}

	fairnessIndex.Set(p.computeFairnessIndex())

	return bestQueue, nil
}

// getOrCreateMetrics returns the ProgramMetrics for the given program ID, creating if needed.
// Type assertions use the comma-ok form so a stray non-*ProgramMetrics entry
// (only reachable via a future bug) degrades to a fresh metrics object instead
// of panicking the scheduler.
func (p *ProgramAwarePlugin) getOrCreateMetrics(programID string) *ProgramMetrics {
	if metricsRaw, ok := p.programMetrics.Load(programID); ok {
		if m, ok := metricsRaw.(*ProgramMetrics); ok {
			return m
		}
	}
	m := &ProgramMetrics{}
	actual, _ := p.programMetrics.LoadOrStore(programID, m)
	if existing, ok := actual.(*ProgramMetrics); ok {
		return existing
	}
	return m
}

// computeFairnessIndex returns Jain's Fairness Index over the service rate
// (weighted tokens/sec) for each program. Equal service rates = perfect fairness.
// Returns 1.0 when fewer than 2 programs have rate data.
func (p *ProgramAwarePlugin) computeFairnessIndex() float64 {
	var sum, sumSq float64
	var n float64
	p.programMetrics.Range(func(_, value any) bool {
		m, ok := value.(*ProgramMetrics)
		if !ok {
			return true
		}
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
