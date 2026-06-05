package programaware

import (
	"slices"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/prometheus/client_golang/prometheus"
)

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
		if c, ok := v.(string); ok {
			cursor = c
		}
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
