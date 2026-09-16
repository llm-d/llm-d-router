package programaware

import (
	"math"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
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
// The strategy requires one fairness ID per session, which holds for traffic
// carrying a session header. Requests with no fairness ID share a single flow
// whose dispatched count aggregates unrelated clients, so that count is not a
// session depth and the ordering it produces is not meaningful. Such traffic
// belongs in a priority band on a different fairness policy.
//
// A program's turn counter resets once it has been inactive for
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

		score := float64(s.turnNumberFor(qi.Metrics, headEnqueue)) + s.timeWeight*headWait
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

func (s *turnPriorityStrategy) OnCompleted(_ *ProgramMetrics, _ *fwksched.InferenceRequest, _ *fwkrc.Response) {
}

func (s *turnPriorityStrategy) EvictProgram(_ string) {}

func (s *turnPriorityStrategy) Collectors() []prometheus.Collector { return nil }

// turnNumberFor returns the turn number of a program's head request: the count of
// requests already dispatched for the program plus the waiting request itself.
// headEnqueue is the head request's arrival instant, so the idle gap is measured
// from the previous completion to that arrival and excludes the head's own queue
// wait.
//
// A program idle for longer than inactivitySeconds has its next request scored as
// turn one, on the grounds that a session dormant that long has stopped competing
// for its own prefix. Dispatching that request re-prefills the conversation, and
// the program competes at its full dispatched count again.
//
// The count is read here but incremented in PreRequest, off the dispatch path, so
// consecutive picks for one program can score against a count that lags by the
// requests already dispatched from it.
func (s *turnPriorityStrategy) turnNumberFor(metrics *ProgramMetrics, headEnqueue time.Time) int64 {
	if metrics == nil {
		return 1
	}
	if s.inactivitySeconds > 0 && metrics.InFlight() == 0 {
		last := metrics.LastCompletionTime()
		if !last.IsZero() && headEnqueue.Sub(last).Seconds() > s.inactivitySeconds {
			return 1
		}
	}
	return metrics.DispatchedCount() + 1
}
