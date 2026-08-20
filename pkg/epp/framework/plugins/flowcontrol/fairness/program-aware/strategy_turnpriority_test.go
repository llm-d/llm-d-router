package programaware

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// seedTurns advances a program to the given turn depth, leaving nothing in flight
// and the last completion at completedAt.
func seedTurns(turns int, completedAt time.Time) *ProgramMetrics {
	m := &ProgramMetrics{}
	for range turns {
		m.RecordDispatched(time.Time{})
		m.RecordCompletion(completedAt)
	}
	return m
}

func turnInfo(id string, metrics *ProgramMetrics, headEnqueue time.Time) (string, QueueInfo) {
	return id, QueueInfo{
		Queue:   makeQueue(id, 1, headEnqueue),
		Metrics: metrics,
		Len:     1,
	}
}

func TestTurnPriority_Name(t *testing.T) {
	assert.Equal(t, "turn-priority", (&turnPriorityStrategy{}).Name())
}

func TestTurnPriority_PrefersDeeperSession(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0}
	now := time.Now()

	idA, qA := turnInfo("shallow", seedTurns(1, now), now)
	idB, qB := turnInfo("deep", seedTurns(20, now), now)

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "deep", got.FlowKey().ID)
}

func TestTurnPriority_PrefersLongerWaitWhenTimeWeighted(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 1.0}
	now := time.Now()

	idA, qA := turnInfo("deep", seedTurns(20, now), now)
	idB, qB := turnInfo("waiting", seedTurns(1, now), now.Add(-30*time.Second))

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "waiting", got.FlowKey().ID)
}

// A long enough wait must overcome depth at the default weighting, otherwise a
// shallow session starves behind deep ones.
func TestTurnPriority_WaitOvercomesDepthAtDefaultWeight(t *testing.T) {
	cfg := DefaultConfig()
	s := &turnPriorityStrategy{timeWeight: cfg.TurnPriorityTimeWeight}
	now := time.Now()

	// At timeWeight 0.05 a turn-50 lead needs ~1000s of wait to overcome.
	idA, qA := turnInfo("deep", seedTurns(50, now), now)
	idB, qB := turnInfo("starving", seedTurns(1, now), now.Add(-30*time.Minute))

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "starving", got.FlowKey().ID)
}

func TestTurnPriority_SingleWaitingFlowBypassesScoring(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0.5}
	now := time.Now()

	id, qi := turnInfo("only", seedTurns(1, now), now)
	empty := QueueInfo{Queue: makeQueue("idle", 0, time.Time{}), Metrics: &ProgramMetrics{}, Len: 0}

	got := s.Pick(0, map[string]QueueInfo{id: qi, "idle": empty})
	require.NotNil(t, got)
	assert.Equal(t, "only", got.FlowKey().ID)
}

func TestTurnPriority_NoWaitingFlows(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0.5}
	queues := map[string]QueueInfo{
		"idle": {Queue: makeQueue("idle", 0, time.Time{}), Metrics: &ProgramMetrics{}, Len: 0},
	}
	assert.Nil(t, s.Pick(0, queues))
	assert.Nil(t, s.Pick(0, map[string]QueueInfo{}))
}

// A positive Len with a nil head can appear when a queue drains between
// iteration and scoring.
func TestTurnPriority_SkipsNilHead(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0.5}
	now := time.Now()

	drained := makeQueue("drained", 1, now)
	drained.PeekV = nil

	idA, qA := turnInfo("live", seedTurns(1, now), now)
	queues := map[string]QueueInfo{
		idA:       qA,
		"drained": {Queue: drained, Metrics: &ProgramMetrics{}, Len: 1},
	}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "live", got.FlowKey().ID)
}

func TestTurnPriority_NilMetricsCountsAsFirstTurn(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0}
	now := time.Now()

	idA, qA := turnInfo("deep", seedTurns(5, now), now)
	queues := map[string]QueueInfo{
		idA:      qA,
		"absent": {Queue: makeQueue("absent", 1, now), Metrics: nil, Len: 1},
	}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "deep", got.FlowKey().ID)
}

func TestTurnPriority_InactivityResetsTurnCount(t *testing.T) {
	s := &turnPriorityStrategy{inactivitySeconds: 60}

	idle := seedTurns(30, time.Now().Add(-10*time.Minute))
	assert.Equal(t, int64(1), s.turnNumberFor(idle))

	active := seedTurns(30, time.Now())
	assert.Equal(t, int64(31), s.turnNumberFor(active))
}

// An in-flight request means the program is active regardless of how old its last
// completion is.
func TestTurnPriority_InFlightSuppressesReset(t *testing.T) {
	s := &turnPriorityStrategy{inactivitySeconds: 60}

	m := seedTurns(5, time.Now().Add(-10*time.Minute))
	m.RecordDispatched(time.Time{})

	assert.Equal(t, int64(7), s.turnNumberFor(m))
}

func TestTurnPriority_ZeroInactivityDisablesReset(t *testing.T) {
	s := &turnPriorityStrategy{inactivitySeconds: 0}
	m := seedTurns(9, time.Now().Add(-24*time.Hour))
	assert.Equal(t, int64(10), s.turnNumberFor(m))
}

func TestTurnPriority_EqualCandidatesResolve(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0.5}
	now := time.Now()

	idA, qA := turnInfo("alpha", seedTurns(3, now), now)
	idB, qB := turnInfo("beta", seedTurns(3, now), now)

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Contains(t, []string{"alpha", "beta"}, got.FlowKey().ID)
}
