package programaware

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
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
}

// QueueInfo bundles read-only data for each queue passed to Pick.
type QueueInfo struct {
	Queue   flowcontrol.FlowQueueAccessor
	Metrics *ProgramMetrics
	Len     int
}

// newStrategy constructs a ScoringStrategy from the plugin config.
func newStrategy(cfg Config) (ScoringStrategy, error) {
	switch cfg.Strategy {
	case "drr":
		return &DRRStrategy{
			weightDeficit:          floatOr(cfg.WeightDeficit, defaultDRRWeightDeficit),
			weightHeadWait:         floatOr(cfg.WeightDRRHeadWait, defaultDRRWeightHeadWait),
			quantumTokens:          int64Or(cfg.QuantumTokens, defaultDRRQuantumTokens),
			deficitHalfLifeSeconds: floatOr(cfg.DeficitHalfLifeSeconds, defaultDRRDeficitHalfLifeSeconds),
		}, nil
	case "", "las":
		return &LASStrategy{
			weightService:   floatOr(cfg.WeightService, defaultServiceWeightService),
			weightHeadWait:  floatOr(cfg.WeightServiceHeadWait, defaultServiceWeightHeadWait),
			decayFactor:     floatOr(cfg.ServiceDecayFactor, defaultServiceDecayFactor),
			halfLifeSeconds: floatOr(cfg.ServiceHalfLifeSeconds, 0),
		}, nil
	case "rr":
		return &RRStrategy{
			deferCursor: boolOr(cfg.DeferRRCursor, false),
		}, nil
	default:
		return nil, fmt.Errorf("unknown scoring strategy %q: valid values are \"drr\", \"las\", \"rr\"", cfg.Strategy)
	}
}

// floatOr returns *p if non-nil, otherwise the default.
func floatOr(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}

// int64Or returns *p if non-nil, otherwise the default.
func int64Or(p *int64, def int64) int64 {
	if p != nil {
		return *p
	}
	return def
}

// boolOr returns *p if non-nil, otherwise the default.
func boolOr(p *bool, def bool) bool {
	if p != nil {
		return *p
	}
	return def
}

// rangeNormalize performs min-max normalization: (v - min) / (max - min) → [0, 1].
// Returns 0.5 when min == max (no discriminative signal for this dimension).
func rangeNormalize(v, min, max float64) float64 {
	if max == min {
		return 0.5
	}
	return (v - min) / (max - min)
}

// =============================================================================
// DRR Strategy
// =============================================================================

// Default DRR strategy values.
const (
	defaultDRRQuantumTokens          int64   = 1000
	defaultDRRWeightDeficit          float64 = 0.8
	defaultDRRWeightHeadWait         float64 = 0.2
	defaultDRRDeficitHalfLifeSeconds float64 = 60
)

// DRRStrategy implements Deficit Round Robin adapted for token-based LLM fwksched.
//
// Classic DRR (https://dl.acm.org/doi/pdf/10.1145/217391.217453) assigns each active flow a fixed
// byte quantum per round, serves the highest-deficit flow first, and deducts actual bytes
// served from the deficit counter. This guarantees proportional bandwidth allocation
// — in contrast to EWMA which counts requests, not compute.
//
// Mapping for program-aware scheduler:
//   - "bytes"   = prompt + completion tokens (actual cost known at response completion)
//   - "quantum" = quantumTokens added per Pick() cycle per non-empty queue
//   - Actual token cost is deducted in OnCompleted() (ResponseComplete hook)
//   - Idle (empty) queues do not receive quantum; instead their deficit is
//     time-decayed (half-life configurable) so stale credit shrinks toward zero
//
// headWaitMs is used as a secondary signal to prevent starvation of
// new or returning programs that start with deficit=0.
//
// Weights and quantum are configurable via the plugin config; defaults are 0.8/0.2/1000.
type DRRStrategy struct {
	weightDeficit          float64
	weightHeadWait         float64
	quantumTokens          int64
	deficitHalfLifeSeconds float64  // 0 = no decay
	addQuantum             sync.Map // key: int (band priority) → bool
}

// Name returns "drr".
func (s *DRRStrategy) Name() string { return "drr" }

// Pick selects the queue with the highest deficit-weighted score.
//
// Quantum allocation is per-band and cycle-aware: the first Pick() for a
// given priority band allocates quantum to non-empty queues in that band;
// subsequent Pick() calls for the same band in the same cycle skip
// allocation. OnPreRequest() resets the flag for the dispatched band.
// Empty queues receive no quantum; instead their deficit is time-decayed.
func (s *DRRStrategy) Pick(bandPriority int, queues map[string]QueueInfo) (flowcontrol.FlowQueueAccessor, map[string]float64) {
	type entry struct {
		deficit    float64
		headWaitMs float64
	}

	// Check whether quantum should be allocated for this band.
	// Missing key means first call for this band — allocate.
	shouldAdd := true
	if v, ok := s.addQuantum.Load(bandPriority); ok {
		shouldAdd = v.(bool)
	}

	// Pass 1: bookkeeping + collect raw values for non-empty queues.
	entries := make(map[string]entry)
	minDeficit, maxDeficit := 0.0, 0.0
	minWait, maxWait := 0.0, 0.0
	first := true
	now := time.Now()

	for id, qi := range queues {
		if qi.Metrics == nil {
			continue
		}

		if qi.Len == 0 {
			// Empty queues: decay existing deficit so stale credit from
			// long-idle programs does not accumulate unboundedly.
			if s.deficitHalfLifeSeconds > 0 {
				qi.Metrics.DecayDeficitTimed(s.deficitHalfLifeSeconds, now)
			}
			continue
		}

		// Non-empty queues: allocate quantum once per cycle per band.
		if shouldAdd {
			qi.Metrics.AddDeficit(s.quantumTokens)
		}

		deficit := float64(qi.Metrics.Deficit())
		var headWaitMs float64
		if head := qi.Queue.PeekHead(); head != nil {
			headWaitMs = float64(time.Since(head.EnqueueTime()).Milliseconds())
		}

		entries[id] = entry{deficit: deficit, headWaitMs: headWaitMs}

		if first {
			minDeficit, maxDeficit = deficit, deficit
			minWait, maxWait = headWaitMs, headWaitMs
			first = false
		} else {
			if deficit < minDeficit {
				minDeficit = deficit
			}
			if deficit > maxDeficit {
				maxDeficit = deficit
			}
			if headWaitMs < minWait {
				minWait = headWaitMs
			}
			if headWaitMs > maxWait {
				maxWait = headWaitMs
			}
		}
	}

	if len(entries) == 0 {
		return nil, nil
	}

	// Pass 2: normalize, score, select.
	scores := make(map[string]float64, len(entries))
	var best flowcontrol.FlowQueueAccessor
	bestScore := math.Inf(-1)

	for id, e := range entries {
		normDeficit := rangeNormalize(e.deficit, minDeficit, maxDeficit)
		normWait := rangeNormalize(e.headWaitMs, minWait, maxWait)
		score := s.weightDeficit*normDeficit + s.weightHeadWait*normWait
		scores[id] = score
		if score > bestScore {
			bestScore = score
			best = queues[id].Queue
		}
	}

	s.addQuantum.Store(bandPriority, false)
	return best, scores
}

// OnPreRequest resets the quantum flag for the dispatched band.
func (s *DRRStrategy) OnPreRequest(_ *ProgramMetrics, request *fwksched.InferenceRequest) {
	s.addQuantum.Store(request.Objectives.Priority, true)
}

// OnCompleted deducts actual token usage from the deficit counter.
func (s *DRRStrategy) OnCompleted(metrics *ProgramMetrics, _ *fwksched.InferenceRequest, response *fwkrc.Response) {
	if metrics == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	metrics.DeductTokens(weightInputToken*promptTokens + weightOutputToken*completionTokens)
}

// =============================================================================
// LAS (Least Attained Service) Strategy
// =============================================================================

// Default LAS strategy values.
const (
	defaultServiceWeightService  float64 = 0.8
	defaultServiceWeightHeadWait float64 = 0.2
	defaultServiceDecayFactor    float64 = 0.995
)

// LASStrategy scores queues by equalizing attained service (weighted tokens
// consumed) across programs. Programs with lower attained service receive higher
// scores, directly targeting fair resource allocation.
//
//   - attainedService (inverted): time-decayed accumulator of weighted tokens
//     consumed — lower service → higher score (underserved programs promoted).
//   - headWait: age of the oldest request — tiebreaker for cold start when
//     all programs have zero attained service.
//
// On each Pick() cycle, every queue's attained service is decayed by decayFactor,
// causing old service to be gradually forgotten. On each completion, the actual
// weighted token cost is added to the program's attained service.
//
// Weights and decay factor are configurable via the plugin config.
type LASStrategy struct {
	weightService   float64
	weightHeadWait  float64
	decayFactor     float64
	halfLifeSeconds float64 // if > 0, use time-based decay instead of per-cycle decayFactor
}

// Name returns "service".
func (s *LASStrategy) Name() string { return "las" }

// Pick selects the queue with the lowest attained service (highest need).
//
// First decays every queue's attained service, then uses two-pass adaptive
// normalization across non-empty queues. The service dimension is inverted
// so that lower attained service maps to a higher score.
func (s *LASStrategy) Pick(_ int, queues map[string]QueueInfo) (flowcontrol.FlowQueueAccessor, map[string]float64) {
	type entry struct {
		service    float64
		headWaitMs float64
	}

	// Pass 1: decay service for all queues, collect raw values for non-empty.
	entries := make(map[string]entry)
	minService, maxService := 0.0, 0.0
	minWait, maxWait := 0.0, 0.0
	first := true
	now := time.Now()

	for id, qi := range queues {
		if qi.Metrics == nil {
			continue
		}
		// Decay runs for every queue, including empty ones.
		if s.halfLifeSeconds > 0 {
			qi.Metrics.DecayServiceTimed(s.halfLifeSeconds, now)
		} else {
			qi.Metrics.DecayService(s.decayFactor)
		}

		if qi.Len == 0 {
			continue
		}

		service := qi.Metrics.AttainedService()
		var headWaitMs float64
		if head := qi.Queue.PeekHead(); head != nil {
			headWaitMs = float64(time.Since(head.EnqueueTime()).Milliseconds())
		}

		entries[id] = entry{service: service, headWaitMs: headWaitMs}

		if first {
			minService, maxService = service, service
			minWait, maxWait = headWaitMs, headWaitMs
			first = false
		} else {
			if service < minService {
				minService = service
			}
			if service > maxService {
				maxService = service
			}
			if headWaitMs < minWait {
				minWait = headWaitMs
			}
			if headWaitMs > maxWait {
				maxWait = headWaitMs
			}
		}
	}

	if len(entries) == 0 {
		return nil, nil
	}

	// Pass 2: normalize (invert service), score, select.
	scores := make(map[string]float64, len(entries))
	var best flowcontrol.FlowQueueAccessor
	bestScore := math.Inf(-1)

	for id, e := range entries {
		// Invert service: lower attained service → higher normalized score.
		normService := 1 - rangeNormalize(e.service, minService, maxService)
		normWait := rangeNormalize(e.headWaitMs, minWait, maxWait)
		score := s.weightService*normService + s.weightHeadWait*normWait
		scores[id] = score
		if score > bestScore {
			bestScore = score
			best = queues[id].Queue
		}
	}

	return best, scores
}

// OnPreRequest is a no-op for LAS.
func (s *LASStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {}

// OnCompleted accumulates the weighted token cost into the program's attained service.
func (s *LASStrategy) OnCompleted(metrics *ProgramMetrics, _ *fwksched.InferenceRequest, response *fwkrc.Response) {
	if metrics == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	cost := float64(weightInputToken*promptTokens + weightOutputToken*completionTokens)
	metrics.AddService(cost)
}

// =============================================================================
// RR (Round-Robin) Strategy
// =============================================================================

// RRStrategy implements a simple round-robin scheduling strategy that matches
// the upstream gateway-api-inference-extension round-robin fairness policy.
//
// It maintains a cursor (lastSelected) that tracks which program was last
// dispatched. On each Pick() cycle, programs are sorted deterministically
// and the one immediately after the cursor gets the highest score.
// Empty queues are naturally skipped because only non-empty queues are scored.
type RRStrategy struct {
	lastSelected sync.Map // key: int (band priority) → string (program ID)
	moveCursor   sync.Map // key: int (band priority) → bool
	deferCursor  bool     // when true, cursor advances once per cycle (reset by OnPreRequest)
}

// Name returns "rr".
func (s *RRStrategy) Name() string { return "rr" }

// Pick selects the next non-empty queue in deterministic round-robin order.
// Walks forward from the per-band cursor and returns the first non-empty queue found.
func (s *RRStrategy) Pick(bandPriority int, queues map[string]QueueInfo) (flowcontrol.FlowQueueAccessor, map[string]float64) {
	// Sort all program IDs for deterministic ordering.
	allKeys := make([]string, 0, len(queues))
	for id := range queues {
		allKeys = append(allKeys, id)
	}
	slices.Sort(allKeys)

	n := len(allKeys)
	if n == 0 {
		return nil, nil
	}

	// Load per-band cursor.
	cursor := ""
	if v, ok := s.lastSelected.Load(bandPriority); ok {
		cursor = v.(string)
	}

	// When deferCursor is active and cursor already moved this cycle,
	// return the same queue without advancing.
	if s.deferCursor && cursor != "" {
		canMove := true
		if v, ok := s.moveCursor.Load(bandPriority); ok {
			canMove = v.(bool)
		}
		if !canMove {
			if qi, ok := queues[cursor]; ok && qi.Len > 0 {
				return qi.Queue, nil
			}
		}
	}

	// Find the start index (next after cursor).
	start := 0
	if cursor != "" {
		if idx := slices.Index(allKeys, cursor); idx != -1 {
			start = (idx + 1) % n
		}
	}

	// Walk forward from start, pick the first non-empty queue.
	for i := range n {
		id := allKeys[(start+i)%n]
		if queues[id].Len > 0 {
			s.lastSelected.Store(bandPriority, id)
			if s.deferCursor {
				s.moveCursor.Store(bandPriority, false)
			}
			return queues[id].Queue, nil
		}
	}

	s.lastSelected.Delete(bandPriority)
	return nil, nil
}

// OnPreRequest resets the moveCursor flag for the dispatched band.
func (s *RRStrategy) OnPreRequest(_ *ProgramMetrics, request *fwksched.InferenceRequest) {
	if !s.deferCursor {
		return
	}
	s.moveCursor.Store(request.Objectives.Priority, true)
}

// OnCompleted is a no-op for round-robin (no token tracking needed).
func (s *RRStrategy) OnCompleted(_ *ProgramMetrics, _ *fwksched.InferenceRequest, _ *fwkrc.Response) {
}
