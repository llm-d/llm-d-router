// Package programaware implements a flow-control fairness policy that
// schedules per-program queues using a swappable scoring strategy.
package programaware

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

const ProgramAwarePluginType = "program-aware-fairness"

// enqueueTimeAttributeKey is the per-request attribute under which Pick
// stashes the flow-control enqueue timestamp for PreRequest to read back.
const enqueueTimeAttributeKey = "program-aware/enqueue-time"

type Config struct {
	Strategy             string  `json:"strategy,omitempty"`
	EvictionTTLSeconds   float64 `json:"evictionTtlSeconds,omitempty"`
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds,omitempty"`

	LASWeightService   float64 `json:"lasWeightService,omitempty"`
	LASWeightHeadWait  float64 `json:"lasWeightHeadWait,omitempty"`
	LASDecayFactor     float64 `json:"lasDecayFactor,omitempty"`
	LASHalfLifeSeconds float64 `json:"lasHalfLifeSeconds,omitempty"`
}

func DefaultConfig() Config {
	return Config{
		Strategy:             "las",
		EvictionTTLSeconds:   3600,
		EvictionSweepSeconds: 300,
		LASWeightService:     0.8,
		LASWeightHeadWait:    0.2,
		LASDecayFactor:       0.99997,
		LASHalfLifeSeconds:   0,
	}
}

func (c Config) validate() error {
	if c.EvictionTTLSeconds < 0 {
		return fmt.Errorf("evictionTtlSeconds must be >= 0, got %v", c.EvictionTTLSeconds)
	}
	if c.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", c.EvictionSweepSeconds)
	}
	if c.LASWeightService < 0 {
		return fmt.Errorf("lasWeightService must be >= 0, got %v", c.LASWeightService)
	}
	if c.LASWeightHeadWait < 0 {
		return fmt.Errorf("lasWeightHeadWait must be >= 0, got %v", c.LASWeightHeadWait)
	}
	if c.LASDecayFactor <= 0 || c.LASDecayFactor > 1 {
		return fmt.Errorf("lasDecayFactor must be in (0, 1], got %v", c.LASDecayFactor)
	}
	if c.LASHalfLifeSeconds < 0 {
		return fmt.Errorf("lasHalfLifeSeconds must be >= 0, got %v", c.LASHalfLifeSeconds)
	}
	return nil
}

var (
	_ flowcontrol.FairnessPolicy  = &ProgramAwarePlugin{}
	_ fwkrc.PreRequest            = &ProgramAwarePlugin{}
	_ fwkrc.ResponseBodyProcessor = &ProgramAwarePlugin{}
)

//nolint:revive // factory name matches sibling fairness plugins.
func ProgramAwarePluginFactory(name string, parameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	cfg := DefaultConfig()
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
	p := &ProgramAwarePlugin{name: name, strategy: strategy}
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			for _, c := range GetCollectors() {
				reg.MustRegister(c)
			}
			for _, c := range strategy.Collectors() {
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

//nolint:revive
type ProgramAwarePlugin struct {
	name     string
	strategy Strategy

	programMetrics sync.Map // key: program ID (string), value: *ProgramMetrics
}

func (p *ProgramAwarePlugin) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: ProgramAwarePluginType, Name: p.name}
}

// getStrategy falls back to a default LAS strategy for zero-value plugin
// instances constructed in tests.
func (p *ProgramAwarePlugin) getStrategy() Strategy {
	if p.strategy == nil {
		s, _ := newStrategy(DefaultConfig())
		return s
	}
	return p.strategy
}

func (p *ProgramAwarePlugin) getOrCreateMetrics(programID string) *ProgramMetrics {
	// Seed lastCompletionTime so a program seen but never completing still
	// becomes evictable after ttl.
	fresh := &ProgramMetrics{lastCompletionTime: time.Now()}
	actual, _ := p.programMetrics.LoadOrStore(programID, fresh)
	if m, ok := actual.(*ProgramMetrics); ok {
		return m
	}
	p.programMetrics.Store(programID, fresh)
	return fresh
}

func programIDFor(req *fwksched.InferenceRequest) string {
	if req == nil || req.FairnessID == "" {
		return metadata.DefaultFairnessID
	}
	return req.FairnessID
}

func (p *ProgramAwarePlugin) NewState(_ context.Context) any { return nil }

func (p *ProgramAwarePlugin) Pick(_ context.Context, band flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}

	infos := make(map[string]QueueInfo)
	band.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) bool {
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

	best := p.getStrategy().Pick(band.Priority(), infos)

	// Stash the selected item's enqueue time on the request so PreRequest
	// can compute the queue wait time. Attribute lifetime tracks the
	// request, so abandoned requests cannot leak.
	if best != nil {
		if head := best.PeekHead(); head != nil {
			if req := head.OriginalRequest().InferenceRequest(); req != nil {
				req.PutAttribute(enqueueTimeAttributeKey, head.EnqueueTime())
			}
		}
	}

	fairnessIndex.Set(p.computeFairnessIndex())
	return best, nil
}

func (p *ProgramAwarePlugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) {
	if request == nil {
		return
	}
	id := programIDFor(request)
	metrics := p.getOrCreateMetrics(id)

	enqueueTime, _ := fwksched.ReadRequestAttribute[time.Time](request, enqueueTimeAttributeKey)
	metrics.RecordDispatched(enqueueTime)
	avgWaitTimeMs.WithLabelValues(id).Set(metrics.AverageWaitTime())

	p.getStrategy().OnPreRequest(metrics, request)
}

// ResponseBody acts on the final stream chunk only; intermediate chunks are
// no-ops.
func (p *ProgramAwarePlugin) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	id := programIDFor(request)
	metrics := p.getOrCreateMetrics(id)

	p.getStrategy().OnCompleted(metrics, request, response)
	metrics.RecordCompletion(time.Now())
}

func (p *ProgramAwarePlugin) runEviction(ctx context.Context, interval, ttl time.Duration) {
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

// evictIdle is best-effort: a request landing strictly after the gate can
// recreate a freshly-deleted entry via getOrCreateMetrics.
func (p *ProgramAwarePlugin) evictIdle(ttl time.Duration) {
	now := time.Now()
	p.programMetrics.Range(func(key, value any) bool {
		m, ok := value.(*ProgramMetrics)
		if !ok {
			p.evictKey(key)
			return true
		}
		if m.InFlight() != 0 {
			return true
		}
		if now.Sub(m.LastCompletionTime()) <= ttl {
			return true
		}
		p.evictKey(key)
		return true
	})
}

func (p *ProgramAwarePlugin) evictKey(key any) {
	p.programMetrics.Delete(key)
	if id, ok := key.(string); ok {
		p.getStrategy().EvictProgram(id)
		DeleteSharedSeries(id)
	}
}

// computeFairnessIndex returns Jain's Fairness Index over the average wait
// time per program. Programs with no wait observations are skipped.
func (p *ProgramAwarePlugin) computeFairnessIndex() float64 {
	var sum, sumSq, n float64
	p.programMetrics.Range(func(_, value any) bool {
		m, ok := value.(*ProgramMetrics)
		if !ok {
			return true
		}
		if m.WaitCount() == 0 {
			return true
		}
		x := m.AverageWaitTime()
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
