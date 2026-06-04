package programaware

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
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

// =============================================================================
// DRR Strategy
// =============================================================================

// drrState holds DRRStrategy's per-program state. Callers must hold mu for the
// duration of any read-modify-write on deficit/lastDecay. The struct is
// declared with no methods so the math at the call site stays visible and
// the lock contract is explicit; a single mutex covers both fields together
// because the time-based decay path mutates them as a pair.
type drrState struct {
	mu        sync.Mutex
	deficit   int64
	lastDecay time.Time
}

// DRRStrategy implements Deficit Round Robin adapted for token-based LLM fwksched.
//
// Classic DRR (https://dl.acm.org/doi/pdf/10.1145/217391.217453) assigns each active flow a fixed
// byte quantum per round, serves the highest-deficit flow first, and deducts actual bytes
// served from the deficit counter. This guarantees proportional bandwidth allocation
// — in contrast to EWMA which counts requests, not compute.
//
// Mapping for program-aware scheduler:
//   - "bytes"   = prompt + completion tokens (actual cost known at response completion)
//   - "quantum" = quantumTokens added per Pick() cycle to each non-empty queue
//   - Actual token cost is deducted in OnCompleted() (ResponseComplete hook)
//   - Inactive queues (Len==0 and no in-flight requests) do not receive quantum;
//     their deficit is decayed so stale credit shrinks toward zero. Decay is
//     time-based when deficitHalfLifeSeconds > 0, otherwise a per-cycle factor
//     (decayFactor) is applied if it is in (0, 1).
//
// headWaitMs is used as a secondary signal to prevent starvation of
// new or returning programs that start with deficit=0.
//
// Weights and quantum are configurable via the plugin config; defaults live in DefaultConfig.
type DRRStrategy struct {
	weightDeficit          float64
	weightHeadWait         float64
	quantumTokens          int64
	deficitHalfLifeSeconds float64 // > 0 enables time-based decay
	decayFactor            float64 // in (0, 1) enables per-cycle factor decay; ignored if half-life is set

	// state holds per-program deficit state, populated in Step 4 once Pick
	// and OnCompleted are cut over. Allocated but unused until then while
	// ProgramMetrics still holds the deficit.
	state sync.Map // key: program ID (string), value: *drrState
}

func (s *DRRStrategy) getState(id string) *drrState {
	if v, ok := s.state.Load(id); ok {
		return v.(*drrState)
	}
	actual, _ := s.state.LoadOrStore(id, &drrState{})
	return actual.(*drrState)
}

// Name returns "drr".
func (s *DRRStrategy) Name() string { return "drr" }

// Pick selects the queue with the highest deficit-weighted score.
//
// Each Pick() call allocates the configured quantum to every non-empty queue
// (under the dispatch loop's one-Pick-per-dispatch contract this preserves
// classic DRR's proportional-fairness guarantee). Inactive queues — Len==0
// and no in-flight requests — receive no quantum and have their deficit
// decayed instead, so stale credit from long-idle programs does not
// accumulate. The "no in-flight" gate prevents decay from racing with the
// upcoming OnCompleted() deduction for a request that is mid-flight.
func (s *DRRStrategy) Pick(_ int, queues map[string]QueueInfo) (flowcontrol.FlowQueueAccessor, map[string]float64) {
	type entry struct {
		deficit    float64
		headWaitMs float64
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
		st := s.getState(id)

		if qi.Len == 0 {
			// Inactive (no queued and no in-flight) queues: decay deficit so
			// stale credit shrinks. Skip decay while a request is in flight
			// to preserve the upcoming OnCompleted() deduction.
			if qi.Metrics.InFlight() == 0 {
				st.mu.Lock()
				if s.deficitHalfLifeSeconds > 0 {
					if st.lastDecay.IsZero() {
						st.lastDecay = now
					} else if elapsed := now.Sub(st.lastDecay).Seconds(); elapsed > 0 {
						st.deficit = int64(float64(st.deficit) * math.Pow(0.5, elapsed/s.deficitHalfLifeSeconds))
						st.lastDecay = now
					}
				} else if s.decayFactor > 0 && s.decayFactor < 1.0 {
					st.deficit = int64(float64(st.deficit) * s.decayFactor)
				}
				deficit := st.deficit
				st.mu.Unlock()
				deficitTokensGauge.WithLabelValues(id).Set(float64(deficit))
			}
			continue
		}

		// Non-empty queues: allocate quantum each Pick.
		st.mu.Lock()
		st.deficit += s.quantumTokens
		deficit := float64(st.deficit)
		st.mu.Unlock()
		deficitTokensGauge.WithLabelValues(id).Set(deficit)
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

	return best, scores
}

// OnPreRequest is a no-op for DRR.
func (s *DRRStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {}

// OnCompleted deducts actual token usage from the deficit counter.
func (s *DRRStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	cost := weightInputToken*promptTokens + weightOutputToken*completionTokens
	st := s.getState(programIDFromRequest(request))
	st.mu.Lock()
	st.deficit -= cost
	st.mu.Unlock()
}

// EvictProgram drops the per-program deficit state.
func (s *DRRStrategy) EvictProgram(id string) {
	s.state.Delete(id)
}

// Collectors returns the Prometheus collectors owned by DRR.
func (s *DRRStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{deficitTokensGauge}
}

// deficitTokensGauge tracks the DRR deficit per program; only DRRStrategy writes to it.
var deficitTokensGauge = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: programAwareSubsystem,
		Name:      "deficit_tokens",
		Help:      metricsutil.HelpMsgWithStability("DRR deficit counter per program (positive = owed service, negative = overserved); decays exponentially when the queue is empty", compbasemetrics.ALPHA),
	},
	[]string{"program_id"},
)

// =============================================================================
// LAS (Least Attained Service) Strategy
// =============================================================================

// lasState holds LASStrategy's per-program state. Callers must hold mu for
// the duration of any read-modify-write on attainedService/lastDecay; the
// time-based decay path mutates them as a pair.
type lasState struct {
	mu              sync.Mutex
	attainedService float64
	lastDecay       time.Time
}

// LASStrategy scores queues by equalizing attained service (weighted tokens
// consumed) across programs. Programs with lower attained service receive higher
// scores, directly targeting fair resource allocation.
//
//   - attainedService (inverted): accumulator of weighted tokens consumed,
//     decayed when the program is inactive — lower service → higher score
//     (underserved programs promoted).
//   - headWait: age of the oldest request — tiebreaker for cold start when
//     all programs have zero attained service.
//
// Decay is applied only to inactive programs (Len==0 and no in-flight
// requests). Active programs accumulate service without decay so persistent
// heavy users stay deprioritized; idle programs lose stale service so they
// can compete on return. On each completion the weighted token cost is added
// to the program's attained service.
//
// Weights and decay factor are configurable via the plugin config.
type LASStrategy struct {
	weightService   float64
	weightHeadWait  float64
	decayFactor     float64
	halfLifeSeconds float64 // if > 0, use time-based decay instead of per-cycle decayFactor

	state sync.Map // key: program ID (string), value: *lasState
}

func (s *LASStrategy) getState(id string) *lasState {
	if v, ok := s.state.Load(id); ok {
		return v.(*lasState)
	}
	actual, _ := s.state.LoadOrStore(id, &lasState{})
	return actual.(*lasState)
}

// Name returns "service".
func (s *LASStrategy) Name() string { return "las" }

// Pick selects the queue with the lowest attained service (highest need).
//
// Decays attained service only for inactive queues (Len==0 and no in-flight
// requests), then uses two-pass adaptive normalization across non-empty
// queues. The service dimension is inverted so that lower attained service
// maps to a higher score.
func (s *LASStrategy) Pick(_ int, queues map[string]QueueInfo) (flowcontrol.FlowQueueAccessor, map[string]float64) {
	type entry struct {
		service    float64
		headWaitMs float64
	}

	// Pass 1: decay inactive queues, collect raw values for non-empty.
	entries := make(map[string]entry)
	minService, maxService := 0.0, 0.0
	minWait, maxWait := 0.0, 0.0
	first := true
	now := time.Now()

	for id, qi := range queues {
		if qi.Metrics == nil {
			continue
		}

		st := s.getState(id)

		if qi.Len == 0 {
			// Inactive (no queued and no in-flight) queues: decay attained
			// service so stale usage shrinks. Skip decay while a request is
			// in flight to preserve the upcoming OnCompleted() AddService.
			if qi.Metrics.InFlight() == 0 {
				st.mu.Lock()
				if s.halfLifeSeconds > 0 {
					if st.lastDecay.IsZero() {
						st.lastDecay = now
					} else if elapsed := now.Sub(st.lastDecay).Seconds(); elapsed > 0 {
						st.attainedService *= math.Pow(0.5, elapsed/s.halfLifeSeconds)
						st.lastDecay = now
					}
				} else {
					st.attainedService *= s.decayFactor
				}
				st.mu.Unlock()
			}
			continue
		}

		st.mu.Lock()
		service := st.attainedService
		st.mu.Unlock()
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
func (s *LASStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	cost := float64(weightInputToken*promptTokens + weightOutputToken*completionTokens)
	id := programIDFromRequest(request)
	st := s.getState(id)
	st.mu.Lock()
	st.attainedService += cost
	service := st.attainedService
	st.mu.Unlock()
	attainedServiceTokensGauge.WithLabelValues(id).Set(service)
}

// EvictProgram drops the per-program attained-service state.
func (s *LASStrategy) EvictProgram(id string) {
	s.state.Delete(id)
}

// Collectors returns the Prometheus collectors owned by LAS.
func (s *LASStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{attainedServiceTokensGauge}
}

// attainedServiceTokensGauge tracks decayed attained service per program; only LASStrategy writes to it.
var attainedServiceTokensGauge = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: programAwareSubsystem,
		Name:      "attained_service_tokens",
		Help:      metricsutil.HelpMsgWithStability("Time-decayed attained service (weighted tokens consumed) per program", compbasemetrics.ALPHA),
	},
	[]string{"program_id"},
)

// =============================================================================
// RR (Round-Robin) Strategy
// =============================================================================

// RRStrategy implements a simple round-robin scheduling strategy that matches
// the upstream gateway-api-inference-extension round-robin fairness policy.
//
// It maintains a cursor (lastSelected) per priority band that tracks which
// program was last dispatched. On each Pick() cycle, programs are sorted
// deterministically and the one immediately after the cursor is selected.
// Empty queues are naturally skipped.
type RRStrategy struct {
	lastSelected sync.Map // key: int (band priority) → string (program ID)
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
			return queues[id].Queue, nil
		}
	}

	s.lastSelected.Delete(bandPriority)
	return nil, nil
}

// OnPreRequest is a no-op for RR.
func (s *RRStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {}

// OnCompleted is a no-op for round-robin (no token tracking needed).
func (s *RRStrategy) OnCompleted(_ *ProgramMetrics, _ *fwksched.InferenceRequest, _ *fwkrc.Response) {
}

// EvictProgram is a no-op. The per-band lastSelected cursor is keyed by band
// priority, not program ID, and self-corrects on the next Pick() if the
// cursor's program disappears.
func (s *RRStrategy) EvictProgram(_ string) {}

// Collectors returns no collectors — RR has no strategy-specific metrics.
func (s *RRStrategy) Collectors() []prometheus.Collector { return nil }
