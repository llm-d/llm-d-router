package programaware

import (
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Output tokens are weighted ~2x input tokens to reflect their relative cost.
const (
	weightInputToken  = 1
	weightOutputToken = 2
)

type lasState struct {
	mu              sync.Mutex
	attainedService float64
	// decayAnchor marks the instant up to which time-based decay has been folded into
	// attainedService. Decay is applied lazily by Service and AddService, so an idle program's
	// service ages out in wall-clock time without the program ever being visited by Pick, and it
	// accrues continuously, including while requests are in flight.
	decayAnchor time.Time
}

// decayLocked folds in the decay accrued since decayAnchor and advances the anchor to now.
// halfLifeSeconds > 0 selects half-life decay; otherwise factor is a per-second decay rate.
// The caller must hold mu.
func (s *lasState) decayLocked(now time.Time, halfLifeSeconds, factor float64) {
	if s.decayAnchor.IsZero() {
		s.decayAnchor = now
		return
	}
	elapsed := now.Sub(s.decayAnchor).Seconds()
	if elapsed <= 0 {
		return
	}
	if halfLifeSeconds > 0 {
		s.attainedService *= math.Pow(0.5, elapsed/halfLifeSeconds)
	} else {
		s.attainedService *= math.Pow(factor, elapsed)
	}
	s.decayAnchor = now
}

// Service returns the attained service with decay up to now applied.
func (s *lasState) Service(now time.Time, halfLifeSeconds, factor float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked(now, halfLifeSeconds, factor)
	return s.attainedService
}

// AddService folds in pending decay, then accumulates cost, and returns the new total.
func (s *lasState) AddService(cost float64, now time.Time, halfLifeSeconds, factor float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked(now, halfLifeSeconds, factor)
	s.attainedService += cost
	return s.attainedService
}

var attainedServiceTokens = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "program_aware_attained_service_tokens",
		Help:      metricsutil.HelpMsgWithStability("Time-decayed attained service (weighted tokens consumed) per program.", compbasemetrics.ALPHA),
	},
	[]string{"program_id"},
)

var _ Strategy = &LASStrategy{}

type LASStrategy struct {
	weightService   float64
	weightHeadWait  float64
	decayFactor     float64
	halfLifeSeconds float64

	state sync.Map // key: program ID (string), value: *lasState
}

func (s *LASStrategy) getOrCreateState(id string) *lasState {
	if a, ok := s.state.Load(id); ok {
		if st, ok := a.(*lasState); ok {
			return st
		}
	}
	fresh := &lasState{}
	actual, _ := s.state.LoadOrStore(id, fresh)
	if st, ok := actual.(*lasState); ok {
		return st
	}
	s.state.Store(id, fresh)
	return fresh
}

func (s *LASStrategy) Name() string { return "las" }

func (s *LASStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	type entry struct {
		service    float64
		headWaitMs float64
	}

	entries := make(map[string]entry)
	minService, maxService := 0.0, 0.0
	minWait, maxWait := 0.0, 0.0
	first := true
	now := time.Now()

	for id, qi := range queues {
		// Empty entries can appear transiently (a queue drained between iteration and scoring); they
		// carry nothing to score, and skipping them before getOrCreateState avoids resurrecting state
		// for a program the eviction sweep is concurrently removing.
		if qi.Metrics == nil || qi.Len == 0 {
			continue
		}

		service := s.getOrCreateState(id).Service(now, s.halfLifeSeconds, s.decayFactor)
		var headWaitMs float64
		if head := qi.Queue.Peek(); head != nil {
			headWaitMs = float64(time.Since(head.EnqueueTime()).Milliseconds())
		}

		entries[id] = entry{service: service, headWaitMs: headWaitMs}

		if first {
			minService, maxService = service, service
			minWait, maxWait = headWaitMs, headWaitMs
			first = false
		} else {
			minService = min(minService, service)
			maxService = max(maxService, service)
			minWait = min(minWait, headWaitMs)
			maxWait = max(maxWait, headWaitMs)
		}
	}

	if len(entries) == 0 {
		return nil
	}

	var best flowcontrol.FlowQueueAccessor
	bestScore := math.Inf(-1)

	for id, e := range entries {
		// Invert service: lower attained service → higher score.
		normService := 1 - rangeNormalize(e.service, minService, maxService)
		normWait := rangeNormalize(e.headWaitMs, minWait, maxWait)
		score := s.weightService*normService + s.weightHeadWait*normWait
		if score > bestScore {
			bestScore = score
			best = queues[id].Queue
		}
	}

	return best
}

func (s *LASStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {}

func (s *LASStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	cost := float64(weightInputToken*promptTokens + weightOutputToken*completionTokens)
	id := programIDFor(request)
	service := s.getOrCreateState(id).AddService(cost, time.Now(), s.halfLifeSeconds, s.decayFactor)
	attainedServiceTokens.WithLabelValues(id).Set(service)
}

func (s *LASStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	attainedServiceTokens.DeleteLabelValues(id)
}

func (s *LASStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{attainedServiceTokens}
}
