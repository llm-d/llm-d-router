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
// heavy users stay deprioritized; idle programs lose stale service over time
// so they can compete on return. Decay is time-based when halfLifeSeconds > 0
// (predictable wall-clock half-life, recommended for production), otherwise a
// per-Pick factor (decayFactor) is applied — note that factor decay is
// coupled to Pick() cadence, so its effective half-life depends on the
// cluster's pick rate. On each completion the weighted token cost is added
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

// EvictProgram drops the per-program attained-service state and its Prom
// label series.
func (s *LASStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	attainedServiceTokensGauge.DeleteLabelValues(id)
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
