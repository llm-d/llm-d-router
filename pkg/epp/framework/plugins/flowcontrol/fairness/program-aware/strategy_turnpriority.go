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
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
)

const turnPriorityStrategyName = "turn-priority"

var _ Strategy = &turnPriorityStrategy{}

// turnPriorityStrategy scores each flow by its head request's priority
//
//	score = turnNumber + timeWeight*headWait
//
// and picks the highest, favoring deeper sessions whose prefix is most likely
// still resident in the KV cache. A shallower flow ages up as its head waits, but
// only up to timeWeight*requestTTL turn-equivalents, since the flow controller
// sheds the request once the TTL elapses. That product bounds the depth a newcomer
// can overcome: 3 turns at the default weight, so under sustained contention a new
// session is shed rather than dispatched ahead of deeper ones. Covering a deeper
// workload means raising the weight, which trades prefix retention for admission
// latency.
//
// When session-state-producer publishes SessionState on the head request, depth
// comes from the head's own session, so a flow that carries several sessions is
// scored by the session whose request is next. Otherwise depth is the flow's
// dispatched count, which requires one fairness ID per session. That holds for
// traffic carrying a session header. Requests with no fairness ID share a single
// flow whose dispatched count aggregates unrelated clients, so that count is not
// a session depth and the ordering it produces is not meaningful. Such traffic
// belongs in a priority band on a different fairness policy.
//
// The turn counter resets once the session or program has been inactive for
// inactivitySeconds. The strategy keeps no accumulated per-program state.
type turnPriorityStrategy struct {
	timeWeight        float64
	inactivitySeconds float64
}

func (s *turnPriorityStrategy) Name() string { return turnPriorityStrategyName }

func (s *turnPriorityStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	var best flowcontrol.FlowQueueAccessor
	bestScore := math.Inf(-1)
	bestWait := math.Inf(-1)
	now := time.Now()

	for _, qi := range queues {
		if qi.Len == 0 {
			continue
		}
		head := qi.Queue.Peek()
		if head == nil {
			continue
		}

		headEnqueue := head.EnqueueTime()
		headWait := now.Sub(headEnqueue).Seconds()
		if headWait < 0 {
			headWait = 0
		}

		score := float64(s.headTurnNumber(head, qi.Metrics)) + s.timeWeight*headWait
		// Tie-break on the longer head wait so equal depth dispatches in arrival
		// order rather than by map iteration order.
		if score > bestScore || (score == bestScore && headWait > bestWait) {
			bestScore = score
			bestWait = headWait
			best = qi.Queue
		}
	}

	return best
}

func (s *turnPriorityStrategy) OnPreRequest(_ *ProgramMetrics, _ *fwksched.InferenceRequest) {}

func (s *turnPriorityStrategy) OnCompleted(_ *ProgramMetrics, _ *fwksched.InferenceRequest, _ *fwkrh.Response) {
}

func (s *turnPriorityStrategy) EvictProgram(_ string) {}

func (s *turnPriorityStrategy) Collectors() []prometheus.Collector { return nil }

// headTurnNumber returns the turn number of a flow's head request, counted over
// the head's session when its SessionState is published and over the program's
// metrics otherwise.
func (s *turnPriorityStrategy) headTurnNumber(head flowcontrol.QueueItemAccessor, metrics *ProgramMetrics) int64 {
	if session, ok := headSessionState(head); ok {
		return s.turnNumber(session.TurnsTaken, session.InFlightRequests, session.LastResponseAt, head.EnqueueTime())
	}
	return s.turnNumberFor(metrics, head.EnqueueTime())
}

func (s *turnPriorityStrategy) turnNumberFor(metrics *ProgramMetrics, headEnqueue time.Time) int64 {
	if metrics == nil {
		return 1
	}
	return s.turnNumber(metrics.DispatchedCount(), metrics.InFlight(), metrics.LastCompletionTime(), headEnqueue)
}

// turnNumber returns the turn number of a head request: the count of requests
// already dispatched plus the waiting request itself. headEnqueue is the head
// request's arrival instant, so the idle gap is measured from the previous
// completion to that arrival and excludes the head's own queue wait.
//
// A session idle for longer than inactivitySeconds has its next request scored as
// turn one, on the grounds that a session dormant that long has stopped competing
// for its own prefix. Dispatching that request re-prefills the conversation, and
// the session competes at its full dispatched count again.
//
// The count is read here but incremented in PreRequest, off the dispatch path, so
// consecutive picks for one session can score against a count that lags by the
// requests already dispatched from it.
func (s *turnPriorityStrategy) turnNumber(dispatched, inFlight int64, lastCompletion, headEnqueue time.Time) int64 {
	if s.inactivitySeconds > 0 && inFlight == 0 && !lastCompletion.IsZero() &&
		headEnqueue.Sub(lastCompletion).Seconds() > s.inactivitySeconds {
		return 1
	}
	return dispatched + 1
}

// headSessionState returns the session state of a queue head, if published.
func headSessionState(head flowcontrol.QueueItemAccessor) (sessionstate.SessionState, bool) {
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
