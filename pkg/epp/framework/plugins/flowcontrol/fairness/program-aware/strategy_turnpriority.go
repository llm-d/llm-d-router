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
//	prio = turnNumber + timeWeight * elapsedSeconds
//
// and picks the highest. The elapsed term ages waiting flows up so a flow with
// a lower turn number is not starved indefinitely. The strategy keeps no
// accumulated per-program state.
type turnPriorityStrategy struct {
	timeWeight float64
}

func (s *turnPriorityStrategy) Name() string { return turnPriorityStrategyName }

func (s *turnPriorityStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	// Collect the flows with pending work. Scoring only orders flows against
	// each other, so it is meaningful only under contention: with no waiting
	// flow there is nothing to dispatch, and with exactly one the choice is
	// forced. Score only when two or more flows are waiting.
	waiting := make([]QueueInfo, 0, len(queues))
	for _, qi := range queues {
		if qi.Len == 0 || qi.Queue.Peek() == nil {
			continue
		}
		waiting = append(waiting, qi)
	}

	switch len(waiting) {
	case 0:
		return nil
	case 1:
		return waiting[0].Queue
	}

	var best flowcontrol.FlowQueueAccessor
	bestScore := math.Inf(-1)
	now := time.Now()

	for _, qi := range waiting {
		head := qi.Queue.Peek()
		elapsed := now.Sub(head.EnqueueTime()).Seconds()
		if elapsed < 0 {
			elapsed = 0
		}
		score := float64(turnNumberFor(qi.Metrics)) + s.timeWeight*elapsed
		if score > bestScore {
			bestScore = score
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

// turnNumberFor returns the turn number of a program's head request: the count
// of requests already dispatched for that program plus the waiting request
// itself (dispatchedCount+1).
//
// The counter is program-scoped: each program (fairness ID) has its own count,
// and every dispatch advances it by one. The count resets to zero when idle
// program state is evicted: a program that goes idle past the eviction TTL has
// likely lost its KV cache, so on return it restarts from turn one and re-earns
// priority from a cold state.
func turnNumberFor(metrics *ProgramMetrics) int64 {
	if metrics == nil {
		return 1
	}
	return metrics.DispatchedCount() + 1
}
