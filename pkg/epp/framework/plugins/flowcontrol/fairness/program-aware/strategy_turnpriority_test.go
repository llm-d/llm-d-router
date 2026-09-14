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

// Wait overcomes depth only inside timeWeight*requestTTL, since the flow
// controller sheds the head once the TTL elapses. At the default weight that
// bound is 0.05*60 = 3 turn-equivalents.
func TestTurnPriority_WaitOvercomesShallowDepthWithinRequestTTL(t *testing.T) {
	cfg := DefaultConfig()
	s := &turnPriorityStrategy{timeWeight: cfg.TurnPriorityTimeWeight}
	now := time.Now()

	idA, qA := turnInfo("deep", seedTurns(2, now), now)
	idB, qB := turnInfo("starving", seedTurns(1, now), now.Add(-50*time.Second))

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "starving", got.FlowKey().ID)
}

// Past that bound the deeper session keeps the slot for the whole TTL, so the
// newcomer is shed rather than dispatched.
func TestTurnPriority_DeepSessionHoldsSlotBeyondTTLBound(t *testing.T) {
	cfg := DefaultConfig()
	s := &turnPriorityStrategy{timeWeight: cfg.TurnPriorityTimeWeight}
	now := time.Now()

	idA, qA := turnInfo("deep", seedTurns(20, now), now)
	idB, qB := turnInfo("newcomer", seedTurns(1, now), now.Add(-60*time.Second))

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "deep", got.FlowKey().ID)
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
	now := time.Now()

	idle := seedTurns(30, now.Add(-10*time.Minute))
	assert.Equal(t, int64(1), s.turnNumberFor(idle, now))

	active := seedTurns(30, now)
	assert.Equal(t, int64(31), s.turnNumberFor(active, now))
}

// The idle gap runs from the previous completion to the head's arrival, so a head
// that has since waited past the threshold keeps its depth.
func TestTurnPriority_QueueWaitDoesNotTriggerReset(t *testing.T) {
	s := &turnPriorityStrategy{inactivitySeconds: 120}
	now := time.Now()

	m := seedTurns(40, now.Add(-130*time.Second))
	headEnqueue := now.Add(-129 * time.Second)

	assert.Equal(t, int64(41), s.turnNumberFor(m, headEnqueue))
}

// An in-flight request means the program is active regardless of how old its last
// completion is.
func TestTurnPriority_InFlightSuppressesReset(t *testing.T) {
	s := &turnPriorityStrategy{inactivitySeconds: 60}

	m := seedTurns(5, time.Now().Add(-10*time.Minute))
	m.RecordDispatched(time.Time{})

	assert.Equal(t, int64(7), s.turnNumberFor(m, time.Now()))
}

func TestTurnPriority_ZeroInactivityDisablesReset(t *testing.T) {
	s := &turnPriorityStrategy{inactivitySeconds: 0}
	m := seedTurns(9, time.Now().Add(-24*time.Hour))
	assert.Equal(t, int64(10), s.turnNumberFor(m, time.Now()))
}

// With depth alone, equal-depth flows fall back to arrival order rather than map
// iteration order.
func TestTurnPriority_EqualDepthBreaksTieOnHeadWait(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0}
	now := time.Now()

	for range 20 {
		idA, qA := turnInfo("earlier", seedTurns(3, now), now.Add(-10*time.Second))
		idB, qB := turnInfo("later", seedTurns(3, now), now.Add(-2*time.Second))

		got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
		require.NotNil(t, got)
		assert.Equal(t, "earlier", got.FlowKey().ID)
	}
}
