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
	lastDecay       time.Time
}

func (s *lasState) Service() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.attainedService
}

func (s *lasState) AddService(cost float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.attainedService += cost
	return s.attainedService
}

// Decay applies time-based decay when halfLifeSeconds > 0; otherwise it
// applies factor once per call.
func (s *lasState) Decay(now time.Time, halfLifeSeconds, factor float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if halfLifeSeconds > 0 {
		if s.lastDecay.IsZero() {
			s.lastDecay = now
			return
		}
		elapsed := now.Sub(s.lastDecay).Seconds()
		if elapsed <= 0 {
			return
		}
		s.attainedService *= math.Pow(0.5, elapsed/halfLifeSeconds)
		s.lastDecay = now
		return
	}
	s.attainedService *= factor
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

func (s *LASStrategy) getState(id string) *lasState {
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
		if qi.Metrics == nil {
			continue
		}

		st := s.getState(id)

		if qi.Len == 0 {
			// Skip decay while a request is in flight to preserve the
			// upcoming OnCompleted accumulation.
			if qi.Metrics.InFlight() == 0 {
				st.Decay(now, s.halfLifeSeconds, s.decayFactor)
			}
			continue
		}

		service := st.Service()
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
	id := programIDFromRequest(request)
	service := s.getState(id).AddService(cost)
	attainedServiceTokens.WithLabelValues(id).Set(service)
}

func (s *LASStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	attainedServiceTokens.DeleteLabelValues(id)
}

func (s *LASStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{attainedServiceTokens}
}
