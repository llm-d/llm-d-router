/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package programaware

import (
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Output tokens are weighted ~2x input tokens to reflect their relative cost.
const (
	weightInputToken  = 1
	weightOutputToken = 2
)

// servedBaselineTTL is how long LAS keeps a session's served-token baseline
// after the session was last read at a queue head. It matches the default idle
// eviction TTL of session-state-producer.
const servedBaselineTTL = time.Hour

// servedBaseline is a session's served totals as of its last SyncSession.
type servedBaseline struct {
	input    int64
	output   int64
	syncedAt time.Time
}

type lasState struct {
	mu              sync.Mutex
	attainedService float64
	// decayAnchor marks the instant up to which time-based decay has been folded into
	// attainedService. Decay is applied lazily by Service and AddService, so an idle program's
	// service ages out in wall-clock time without the program ever being visited by Pick, and it
	// accrues continuously, including while requests are in flight.
	decayAnchor time.Time
	// served holds each session's served totals as of its last SyncSession,
	// keyed by SessionState.SessionID. A flow can hold several sessions, for
	// example sub-agents of one program or sessions sharing a tenant fairness
	// ID, so each session keeps its own baseline.
	served map[string]servedBaseline
	// prunedAt is when served was last scanned for stale baselines.
	prunedAt time.Time
}

// decayLocked folds in the decay accrued since decayAnchor and advances the anchor to now.
// halfLifeSeconds of 0 disables decay. The caller must hold mu.
func (s *lasState) decayLocked(now time.Time, halfLifeSeconds float64) {
	if s.decayAnchor.IsZero() {
		s.decayAnchor = now
		return
	}
	elapsed := now.Sub(s.decayAnchor).Seconds()
	if elapsed <= 0 {
		return
	}
	if halfLifeSeconds > 0 {
		s.attainedService *= math.Exp2(-elapsed / halfLifeSeconds)
	}
	s.decayAnchor = now
}

// Service returns the attained service with decay up to now applied. Folding the decay in writes
// the decayed value back and advances the anchor, so Service mutates lasState.
func (s *lasState) Service(now time.Time, halfLifeSeconds float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked(now, halfLifeSeconds)
	return s.attainedService
}

// AddService folds in pending decay, then accumulates cost, and returns the new total.
func (s *lasState) AddService(cost float64, now time.Time, halfLifeSeconds float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked(now, halfLifeSeconds)
	s.attainedService += cost
	return s.attainedService
}

// SyncSession folds in the service a session recorded since the previous sync
// and returns the new total. The new cost is aged from LastResponseAt, the end
// of the most recent response, which approximates when it was served.
func (s *lasState) SyncSession(state sessionstate.SessionState, now time.Time, halfLifeSeconds float64) (float64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked(now, halfLifeSeconds)
	prev := s.served[state.SessionID]
	deltaInput := state.ServedInputTokens - prev.input
	deltaOutput := state.ServedOutputTokens - prev.output
	if deltaInput < 0 || deltaOutput < 0 {
		// The session record was evicted and recreated.
		deltaInput, deltaOutput = state.ServedInputTokens, state.ServedOutputTokens
	}
	if s.served == nil {
		s.served = make(map[string]servedBaseline)
	}
	s.served[state.SessionID] = servedBaseline{input: state.ServedInputTokens, output: state.ServedOutputTokens, syncedAt: now}
	s.pruneServedLocked(now)
	cost := float64(weightInputToken*deltaInput + weightOutputToken*deltaOutput)
	if cost <= 0 {
		return s.attainedService, false
	}
	if age := now.Sub(state.LastResponseAt).Seconds(); halfLifeSeconds > 0 && age > 0 {
		cost *= math.Exp2(-age / halfLifeSeconds)
	}
	s.attainedService += cost
	return s.attainedService, true
}

// pruneServedLocked drops baselines of sessions not read for servedBaselineTTL,
// scanning at most once per minute. A session that returns after its baseline
// was dropped is charged its full served totals, aged from LastResponseAt.
// The caller must hold mu.
func (s *lasState) pruneServedLocked(now time.Time) {
	if now.Sub(s.prunedAt) < time.Minute {
		return
	}
	s.prunedAt = now
	for id, baseline := range s.served {
		if now.Sub(baseline.syncedAt) > servedBaselineTTL {
			delete(s.served, id)
		}
	}
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

		head := qi.Queue.Peek()
		st := s.getOrCreateState(id)
		var service float64
		if session, ok := headSessionState(head); ok {
			var changed bool
			if service, changed = st.SyncSession(session, now, s.halfLifeSeconds); changed {
				attainedServiceTokens.WithLabelValues(id).Set(service)
			}
		} else {
			service = st.Service(now, s.halfLifeSeconds)
		}
		var headWaitMs float64
		if head != nil {
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

// OnCompleted charges the response to its program when no session state is
// published. With session state, the registry records the served tokens and
// Pick folds them in.
func (s *LASStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrh.Response) {
	if request == nil || response == nil {
		return
	}
	if _, ok := sessionstate.ReadSessionState(request); ok {
		return
	}
	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)
	if completionTokens <= 0 && response.StreamedEvents > 0 {
		completionTokens = int64(response.StreamedEvents)
	}
	cost := float64(weightInputToken*promptTokens + weightOutputToken*completionTokens)
	id := programIDFor(request)
	service := s.getOrCreateState(id).AddService(cost, time.Now(), s.halfLifeSeconds)
	setProgramGauge(attainedServiceTokens, id, service)
}

// headSessionState returns the session state of a queue head, if published.
func headSessionState(head flowcontrol.QueueItemAccessor) (sessionstate.SessionState, bool) {
	if head == nil {
		return sessionstate.SessionState{}, false
	}
	orig := head.OriginalRequest()
	if orig == nil {
		return sessionstate.SessionState{}, false
	}
	req := orig.InferenceRequest()
	if req == nil {
		return sessionstate.SessionState{}, false
	}
	return sessionstate.ReadSessionState(req)
}

func (s *LASStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	attainedServiceTokens.DeleteLabelValues(id)
}

func (s *LASStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{attainedServiceTokens}
}
