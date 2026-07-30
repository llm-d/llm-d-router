// Package programaware implements a flow-control fairness policy that
// schedules per-program queues using a swappable scoring strategy.
package programaware

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
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
	_ plugin.StateDumper          = &ProgramAwarePlugin{}
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

// maxDebugDumpPrograms bounds the per-program list in the DumpState payload so
// a large program population cannot produce an unbounded debug response.
const maxDebugDumpPrograms = 1000

// serviceLookup is the optional capability a Strategy may implement to expose a
// single program's attained service to DumpState. It is a per-program lookup
// rather than a full-map snapshot so a large (see #1625) program map does not
// force an O(N) copy on every debug request.
type serviceLookup interface {
	ServiceForProgram(id string) float64
}

// sanitizeProgramID hashes the user-controlled program ID before it enters the
// debug dump. #1839 deliberately omits raw program IDs (user-controlled,
// high-cardinality); hashing keeps per-program granularity without echoing raw
// tenant strings into the unauthenticated metrics/admin surface. This is surface
// hygiene and a stable per-program pseudonym, not a confidentiality control: a
// low-entropy ID (for example a short tenant name) is recoverable by hashing
// candidates, and the same ID is already a program_id metric label. Emitting raw
// IDs instead would reverse #1839's deliberate omission and #1798's "no sensitive
// data" acceptance criterion, so it is a maintainer decision, not a local tweak.
func sanitizeProgramID(id string) string {
	sum := sha256.Sum256([]byte(id))
	return hex.EncodeToString(sum[:])
}

// fairnessDumpState is the snapshot returned by DumpState. It carries the #1839
// bounded aggregates plus a bounded, sorted per-program list. Program IDs are
// sanitized (see sanitizeProgramID) because they come from a user-controlled
// request header.
type fairnessDumpState struct {
	TotalPrograms int            `json:"totalPrograms"`
	TotalInFlight int64          `json:"totalInFlight"`
	FairnessIndex float64        `json:"fairnessIndex"`
	Strategy      string         `json:"strategy"`
	Programs      []programState `json:"programs"`
	MaxPrograms   int            `json:"maxPrograms"`
	Truncated     bool           `json:"truncated"`
}

// programState is one program's per-program accounting in the debug dump. Its
// ProgramID is the sanitized (hashed) form of the user-controlled ID.
type programState struct {
	ProgramID          string  `json:"programID"`
	DispatchedCount    int64   `json:"dispatchedCount"`
	InFlight           int64   `json:"inFlight"`
	AverageWaitTimeMs  float64 `json:"averageWaitTimeMs"`
	WaitCount          int64   `json:"waitCount"`
	AttainedService    float64 `json:"attainedService"`
	LastCompletionTime string  `json:"lastCompletionTime,omitempty"`
}

// dumpRow is a cheap per-program record collected during the single map pass.
// It carries the raw program ID and the values needed for sorting; the
// expensive per-program work (hashing the ID, looking up attained service,
// formatting the timestamp) is deferred until after the list is capped, so a
// large program map (see #1625, where the ID is an attacker-controlled header)
// cannot turn each debug request into O(N) hashing and service lookups.
type dumpRow struct {
	id             string
	dispatched     int64
	inFlight       int64
	avgWaitMs      float64
	waitCount      int64
	lastCompletion time.Time
	completed      bool
}

// DumpState reports fairness health: the #1839 aggregates (program count, total
// in-flight, Jain's fairness index) plus a bounded, sorted per-program list.
//
// A single pass over the program map accumulates the aggregates and collects
// cheap per-program rows; the rows are then sorted and capped, and only the
// retained rows pay for ID hashing and attained-service lookup. Each program's
// ProgramMetrics fields and its strategy attained service are read under their
// own per-entry locks, so a row is not guaranteed to reflect a single instant
// and a program may appear in the metrics map but be absent from the service
// map (reported as attainedService 0). This best-effort visibility is preferred
// over a global lock contending the scheduling hot path.
func (p *ProgramAwarePlugin) DumpState() (json.RawMessage, error) {
	strategy := p.getStrategy()
	lookup, _ := strategy.(serviceLookup)

	var totalInFlight int64
	var sum, sumSq, n float64
	rows := make([]dumpRow, 0)
	p.programMetrics.Range(func(key, value any) bool {
		id, ok := key.(string)
		if !ok {
			return true
		}
		m, ok := value.(*ProgramMetrics)
		if !ok {
			return true
		}
		inFlight := m.InFlight()
		avgWait := m.AverageWaitTime()
		waitCount := m.WaitCount()
		totalInFlight += inFlight
		if waitCount > 0 {
			sum += avgWait
			sumSq += avgWait * avgWait
			n++
		}
		r := dumpRow{
			id:         id,
			dispatched: m.DispatchedCount(),
			inFlight:   inFlight,
			avgWaitMs:  avgWait,
			waitCount:  waitCount,
		}
		// getOrCreateMetrics seeds lastCompletionTime with the creation time, so
		// a non-zero value alone does not mean a request completed. Record it only
		// after at least one completion so never-completed programs report no
		// completion time rather than the seed timestamp.
		if r.dispatched-r.inFlight > 0 {
			if t := m.LastCompletionTime(); !t.IsZero() {
				r.lastCompletion = t
				r.completed = true
			}
		}
		rows = append(rows, r)
		return true
	})

	totalPrograms := len(rows)

	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].inFlight != rows[j].inFlight {
			return rows[i].inFlight > rows[j].inFlight
		}
		if rows[i].dispatched != rows[j].dispatched {
			return rows[i].dispatched > rows[j].dispatched
		}
		return rows[i].id < rows[j].id
	})

	truncated := false
	if len(rows) > maxDebugDumpPrograms {
		rows = rows[:maxDebugDumpPrograms]
		truncated = true
	}

	// Hash IDs and resolve attained service only for the retained rows.
	programs := make([]programState, 0, len(rows))
	for _, r := range rows {
		ps := programState{
			ProgramID:         sanitizeProgramID(r.id),
			DispatchedCount:   r.dispatched,
			InFlight:          r.inFlight,
			AverageWaitTimeMs: r.avgWaitMs,
			WaitCount:         r.waitCount,
		}
		if lookup != nil {
			ps.AttainedService = lookup.ServiceForProgram(r.id)
		}
		if r.completed {
			ps.LastCompletionTime = r.lastCompletion.UTC().Format(time.RFC3339Nano)
		}
		programs = append(programs, ps)
	}

	return json.Marshal(fairnessDumpState{
		TotalPrograms: totalPrograms,
		TotalInFlight: totalInFlight,
		FairnessIndex: jainFairnessIndex(sum, sumSq, n),
		Strategy:      strategy.Name(),
		Programs:      programs,
		MaxPrograms:   maxDebugDumpPrograms,
		Truncated:     truncated,
	})
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
	if a, ok := p.programMetrics.Load(programID); ok {
		if m, ok := a.(*ProgramMetrics); ok {
			return m
		}
	}
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
		if head := best.Peek(); head != nil {
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

// jainFairnessIndex returns Jain's fairness index for the given sum, sum of
// squares, and count of per-program wait observations. It is 1.0 (perfectly
// fair) when fewer than two programs have observations.
func jainFairnessIndex(sum, sumSq, n float64) float64 {
	if n <= 1 || sumSq == 0 {
		return 1.0
	}
	return (sum * sum) / (n * sumSq)
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
	return jainFairnessIndex(sum, sumSq, n)
}
