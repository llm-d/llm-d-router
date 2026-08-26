// Package programscore implements a ScoringOrderingPolicy that orders requests within a single
// queue by the originating program/agent's decayed turns taken and tokens consumed, dispatching
// the least-served program first.
//
// For detailed documentation, see README.md.
package programscore

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// ProgramScoreOrderingPolicyType is the registration type for the program-score ordering policy.
const ProgramScoreOrderingPolicyType = "program-score-ordering-policy"

// Config is the JSON configuration for the program-score ordering policy.
type Config struct {
	// WeightTurns weights the decayed turn count in the combined cost.
	WeightTurns float64 `json:"weightTurns,omitempty"`
	// WeightTokens weights the decayed token count in the combined cost.
	WeightTokens float64 `json:"weightTokens,omitempty"`
	// HalfLifeSeconds is the exponential decay half-life applied to both turns and tokens.
	HalfLifeSeconds float64 `json:"halfLifeSeconds,omitempty"`
	// RescoreIntervalSeconds is how often the FlowController re-heapifies queues using this
	// policy; see flowcontrol.ScoringOrderingPolicy.RescoreInterval.
	RescoreIntervalSeconds float64 `json:"rescoreIntervalSeconds,omitempty"`
	// EvictionTTLSeconds is how long an idle program's state is retained before eviction. 0
	// disables eviction.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds,omitempty"`
	// EvictionSweepSeconds is the interval between idle-eviction sweeps.
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds,omitempty"`
}

// DefaultConfig returns the default configuration. WeightTokens is small relative to WeightTurns
// since token counts typically run one to two orders of magnitude higher than turn counts;
// operators should tune both to their own workload.
func DefaultConfig() Config {
	return Config{
		WeightTurns:            1,
		WeightTokens:           0.01,
		HalfLifeSeconds:        60,
		RescoreIntervalSeconds: 1,
		EvictionTTLSeconds:     3600,
		EvictionSweepSeconds:   300,
	}
}

func (c Config) validate() error {
	if c.WeightTurns < 0 {
		return fmt.Errorf("weightTurns must be >= 0, got %v", c.WeightTurns)
	}
	if c.WeightTokens < 0 {
		return fmt.Errorf("weightTokens must be >= 0, got %v", c.WeightTokens)
	}
	if c.HalfLifeSeconds < 0 {
		return fmt.Errorf("halfLifeSeconds must be >= 0, got %v", c.HalfLifeSeconds)
	}
	if c.RescoreIntervalSeconds < 0 {
		return fmt.Errorf("rescoreIntervalSeconds must be >= 0, got %v", c.RescoreIntervalSeconds)
	}
	if c.EvictionTTLSeconds < 0 {
		return fmt.Errorf("evictionTtlSeconds must be >= 0, got %v", c.EvictionTTLSeconds)
	}
	if c.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", c.EvictionSweepSeconds)
	}
	return nil
}

var (
	_ flowcontrol.ScoringOrderingPolicy = &ProgramScorePolicy{}
	_ fwkrc.PreRequest                  = &ProgramScorePolicy{}
	_ fwkrc.ResponseBodyProcessor       = &ProgramScorePolicy{}
)

// ProgramScoreOrderingPolicyFactory creates a new ProgramScorePolicy instance. handle may be nil
// (e.g. conformance tests), in which case metrics registration and the eviction sweep are skipped.
func ProgramScoreOrderingPolicyFactory(name string, parameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	cfg := DefaultConfig()
	if parameters != nil {
		if err := parameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", ProgramScoreOrderingPolicyType, name, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", ProgramScoreOrderingPolicyType, name, err)
	}
	p := newProgramScorePolicy(cfg).withName(name)
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			for _, c := range GetCollectors() {
				reg.MustRegister(c)
			}
		}
		if cfg.EvictionTTLSeconds > 0 {
			interval := time.Duration(cfg.EvictionSweepSeconds * float64(time.Second))
			ttl := time.Duration(cfg.EvictionTTLSeconds * float64(time.Second))
			go p.runEviction(handle.Context(), interval, ttl)
		}
	}
	return p, nil
}

// ProgramScorePolicy orders requests within a single queue by decayed turns taken and tokens
// consumed by the originating program (FairnessID), lower cost dispatching first. See the
// documentation for ProgramScoreOrderingPolicyType for detailed behavioral guarantees.
type ProgramScorePolicy struct {
	name string
	cfg  Config

	states sync.Map // key: program ID (string), value: *programState
}

func newProgramScorePolicy(cfg Config) *ProgramScorePolicy {
	return &ProgramScorePolicy{name: ProgramScoreOrderingPolicyType, cfg: cfg}
}

func (p *ProgramScorePolicy) withName(name string) *ProgramScorePolicy {
	if name != "" {
		p.name = name
	}
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *ProgramScorePolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: ProgramScoreOrderingPolicyType, Name: p.name}
}

// RescoreInterval implements flowcontrol.ScoringOrderingPolicy.
func (p *ProgramScorePolicy) RescoreInterval() time.Duration {
	return time.Duration(p.cfg.RescoreIntervalSeconds * float64(time.Second))
}

// Less returns true if item 'a' should be dispatched before item 'b': the program with the lower
// weighted, decayed cost (turns taken and tokens consumed) goes first, so a tenant running many
// agents does not out-dispatch one running few merely by occupying more of the shared queue.
// FCFS (enqueue time) breaks ties.
func (p *ProgramScorePolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
	if a == nil && b == nil {
		return false
	}
	if a == nil {
		return false
	}
	if b == nil {
		return true
	}

	now := time.Now()
	costA := p.cost(a, now)
	costB := p.cost(b, now)
	if costA != costB {
		return costA < costB
	}
	return a.EnqueueTime().Before(b.EnqueueTime())
}

func (p *ProgramScorePolicy) cost(item flowcontrol.QueueItemAccessor, now time.Time) float64 {
	id := programID(item.OriginalRequest().InferenceRequest())
	turns, tokens := p.getOrCreateState(id).Cost(now, p.cfg.HalfLifeSeconds)
	return p.cfg.WeightTurns*turns + p.cfg.WeightTokens*tokens
}

// programID returns req's FairnessID, or metadata.DefaultFairnessID if req is nil or has none.
func programID(req *fwksched.InferenceRequest) string {
	if req == nil || req.FairnessID == "" {
		return metadata.DefaultFairnessID
	}
	return req.FairnessID
}

func (p *ProgramScorePolicy) getOrCreateState(id string) *programState {
	if a, ok := p.states.Load(id); ok {
		if st, ok := a.(*programState); ok {
			return st
		}
	}
	fresh := &programState{lastActive: time.Now()}
	actual, _ := p.states.LoadOrStore(id, fresh)
	if st, ok := actual.(*programState); ok {
		return st
	}
	p.states.Store(id, fresh)
	return fresh
}

// PreRequest records one turn for the request's program.
func (p *ProgramScorePolicy) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	if request == nil {
		return nil
	}
	id := programID(request)
	turns := p.getOrCreateState(id).AddTurn(time.Now(), p.cfg.HalfLifeSeconds)
	decayedTurns.WithLabelValues(id).Set(turns)
	return nil
}

// ResponseBody records the request's token cost against its program once the response completes.
// Intermediate stream chunks are no-ops.
func (p *ProgramScorePolicy) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	id := programID(request)
	cost := float64(response.Usage.PromptTokens + response.Usage.CompletionTokens)
	tokens := p.getOrCreateState(id).AddTokens(cost, time.Now(), p.cfg.HalfLifeSeconds)
	decayedTokens.WithLabelValues(id).Set(tokens)
}

func (p *ProgramScorePolicy) runEviction(ctx context.Context, interval, ttl time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.evictIdle(ttl)
		}
	}
}

// evictIdle is best-effort: a request landing strictly after the gate can recreate a
// freshly-deleted entry via getOrCreateState.
func (p *ProgramScorePolicy) evictIdle(ttl time.Duration) {
	now := time.Now()
	p.states.Range(func(key, value any) bool {
		st, ok := value.(*programState)
		if !ok {
			p.states.Delete(key)
			return true
		}
		if now.Sub(st.LastActive()) <= ttl {
			return true
		}
		p.states.Delete(key)
		if id, ok := key.(string); ok {
			DeleteSharedSeries(id)
		}
		return true
	})
}
