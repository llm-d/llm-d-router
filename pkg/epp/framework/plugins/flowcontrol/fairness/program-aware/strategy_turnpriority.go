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
// still resident in the KV cache. headWait is unbounded, so a flow with a lower
// turn number is not starved: it overtakes any deeper rival once it has waited
// long enough.
//
// A program's turn counter resets once it has been inactive for
// inactivitySeconds, so a session that returns with a cold cache competes from
// turn one. The strategy keeps no accumulated per-program state.
type turnPriorityStrategy struct {
	timeWeight        float64
	inactivitySeconds float64
}

func (s *turnPriorityStrategy) Name() string { return turnPriorityStrategyName }

func (s *turnPriorityStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	// Collect the flows with pending work: with no waiting flow there is nothing
	// to dispatch, and with exactly one the choice is forced.
	type candidate struct {
		queue    flowcontrol.FlowQueueAccessor
		metrics  *ProgramMetrics
		headWait float64
	}

	waiting := make([]candidate, 0, len(queues))
	for _, qi := range queues {
		if qi.Len == 0 {
			continue
		}
		head := qi.Queue.Peek()
		if head == nil {
			continue
		}
		headWait := time.Since(head.EnqueueTime()).Seconds()
		if headWait < 0 {
			headWait = 0
		}
		waiting = append(waiting, candidate{queue: qi.Queue, metrics: qi.Metrics, headWait: headWait})
	}

	switch len(waiting) {
	case 0:
		return nil
	case 1:
		return waiting[0].queue
	}

	var best flowcontrol.FlowQueueAccessor
	bestScore := math.Inf(-1)

	for _, c := range waiting {
		score := float64(s.turnNumberFor(c.metrics)) + s.timeWeight*c.headWait
		if score > bestScore {
			bestScore = score
			best = c.queue
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
//
// A program idle for longer than inactivitySeconds counts as turn one and
// re-earns depth from a cold state: a session dormant that long has stopped
// competing for its own prefix. The threshold is independent of the metrics
// eviction TTL, which governs only when per-program bookkeeping is reclaimed.
func (s *turnPriorityStrategy) turnNumberFor(metrics *ProgramMetrics) int64 {
	if metrics == nil {
		return 1
	}
	if s.inactivitySeconds > 0 && metrics.InFlight() == 0 {
		last := metrics.LastCompletionTime()
		if !last.IsZero() && time.Since(last).Seconds() > s.inactivitySeconds {
			return 1
		}
	}
	return metrics.DispatchedCount() + 1
}
