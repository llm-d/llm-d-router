package programaware

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTurnPriority_Name(t *testing.T) {
	assert.Equal(t, "turn-priority", (&turnPriorityStrategy{}).Name())
}

func TestTurnPriority_Pick_PrefersLongerWait(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0.5}
	now := time.Now()

	// Both programs have zero dispatches (equal turn), so the longer-waiting
	// head wins on the elapsed term.
	idA, qA := makeInfo("alpha", now.Add(-5*time.Second))
	idB, qB := makeInfo("beta", now.Add(-1*time.Second))
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID)
}

func TestTurnPriority_Pick_PrefersHigherTurn(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0.5}
	now := time.Now()

	// Equal wait, but beta has more dispatched requests, so its turn number is
	// higher and it wins.
	idA, qA := makeInfo("alpha", now.Add(-1*time.Second))
	idB, qB := makeInfo("beta", now.Add(-1*time.Second))
	for i := 0; i < 10; i++ {
		qB.Metrics.RecordDispatched(time.Time{})
	}
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID)
}

func TestTurnPriority_TurnNumberFromDispatchedCount(t *testing.T) {
	// Waiting request is the next turn: dispatchedCount + 1.
	assert.Equal(t, int64(1), turnNumberFor(&ProgramMetrics{}))
	assert.Equal(t, int64(1), turnNumberFor(nil))

	m := &ProgramMetrics{}
	m.RecordDispatched(time.Time{})
	m.RecordDispatched(time.Time{})
	assert.Equal(t, int64(3), turnNumberFor(m))
}

func TestTurnPriority_Pick_SkipsEmptyQueues(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0.5}
	queues := map[string]QueueInfo{
		"empty": {Queue: makeQueue("empty", 0, time.Time{}), Metrics: &ProgramMetrics{}, Len: 0},
	}
	assert.Nil(t, s.Pick(0, queues))
}

func TestTurnPriority_Pick_SingleWaitingFlowDispatchesDirectly(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0.5}

	// A just-enqueued head scores ~0; with only one waiting flow it must still
	// be picked, proving the single-flow path bypasses scoring.
	id, qi := makeInfo("only", time.Now())
	queues := map[string]QueueInfo{
		id:      qi,
		"empty": {Queue: makeQueue("empty", 0, time.Time{}), Metrics: &ProgramMetrics{}, Len: 0},
	}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "only", got.FlowKey().ID)
}

func TestTurnPriority_Pick_ZeroWeightIgnoresElapsed(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0}
	now := time.Now()

	// With timeWeight 0 and equal turn numbers, all scores tie; a flow is still
	// selected rather than nil.
	idA, qA := makeInfo("alpha", now.Add(-5*time.Second))
	idB, qB := makeInfo("beta", now.Add(-1*time.Second))
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
}
