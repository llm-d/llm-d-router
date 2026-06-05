package programaware

import (
	"math"
	"sync"
	"time"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
)

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
//     their deficit is decayed so stale credit shrinks over time. Decay is
//     time-based when deficitHalfLifeSeconds > 0 (predictable wall-clock
//     half-life, recommended for production), otherwise a per-cycle factor
//     (decayFactor) is applied if it is in (0, 1) — note that factor decay is
//     coupled to Pick() cadence, so its effective half-life depends on the
//     cluster's pick rate.
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

	state sync.Map // key: program ID (string), value: *drrState
}

func (s *DRRStrategy) getState(id string) *drrState {
	if v, ok := s.state.Load(id); ok {
		if st, ok := v.(*drrState); ok {
			return st
		}
	}
	st := &drrState{}
	actual, _ := s.state.LoadOrStore(id, st)
	if existing, ok := actual.(*drrState); ok {
		return existing
	}
	return st
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

// EvictProgram drops the per-program deficit state and its Prom label series.
func (s *DRRStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	deficitTokensGauge.DeleteLabelValues(id)
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
