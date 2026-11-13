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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
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

// sessionTurnInfo builds a flow whose head request carries session state the way
// session-state-producer publishes it, resolved on every read.
func sessionTurnInfo(id string, metrics *ProgramMetrics, headEnqueue time.Time, state func() sessionstate.SessionState) (string, QueueInfo) {
	req := &fwksched.InferenceRequest{FairnessID: id}
	req.PutAttribute(sessionstate.SessionStateDataKey, &fwksched.DynamicRequestAttribute{Get: func() any { return state() }})
	q := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: id},
		PeekV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     headEnqueue,
			OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{InferenceRequestV: req},
		},
	}
	return id, QueueInfo{Queue: q, Metrics: metrics, Len: 1}
}

func fixedSession(state sessionstate.SessionState) func() sessionstate.SessionState {
	return func() sessionstate.SessionState { return state }
}

func TestTurnPriority_ReadsDepthFromSessionState(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0}
	now := time.Now()

	idA, qA := sessionTurnInfo("a", seedTurns(20, now), now,
		fixedSession(sessionstate.SessionState{TurnsTaken: 1, LastResponseAt: now}))
	idB, qB := sessionTurnInfo("b", seedTurns(1, now), now,
		fixedSession(sessionstate.SessionState{TurnsTaken: 10, LastResponseAt: now}))

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "b", got.FlowKey().ID, "session state supplies the depth when the head carries it")
}

func TestTurnPriority_FallsBackToProgramMetricsWithoutSessionState(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0}
	now := time.Now()

	idA, qA := sessionTurnInfo("session", &ProgramMetrics{}, now,
		fixedSession(sessionstate.SessionState{TurnsTaken: 5, LastResponseAt: now}))
	idB, qB := turnInfo("program", seedTurns(10, now), now)

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "program", got.FlowKey().ID)
}

// One flow can carry several sessions, for example sessions sharing a tenant
// fairness ID with composeFairnessIDs disabled. The head's own session is the
// prefix the next dispatch reuses, so its depth is scored rather than the
// flow's dispatched total.
func TestTurnPriority_SharedFlowScoresHeadSession(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0}
	now := time.Now()

	idA, qA := sessionTurnInfo("tenant", seedTurns(40, now), now,
		fixedSession(sessionstate.SessionState{TurnsTaken: 2, LastResponseAt: now}))
	idB, qB := sessionTurnInfo("solo", seedTurns(5, now), now,
		fixedSession(sessionstate.SessionState{TurnsTaken: 5, LastResponseAt: now}))

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "solo", got.FlowKey().ID)
}

func TestTurnPriority_SessionStateInactivityResetsTurnCount(t *testing.T) {
	s := &turnPriorityStrategy{inactivitySeconds: 60}
	now := time.Now()

	// The idle flow's head waited longer, so without the reset it would win.
	idle := sessionstate.SessionState{TurnsTaken: 30, LastResponseAt: now.Add(-10 * time.Minute)}
	idA, qA := sessionTurnInfo("idle", &ProgramMetrics{}, now.Add(-time.Second),
		func() sessionstate.SessionState { return idle })
	idB, qB := sessionTurnInfo("active", &ProgramMetrics{}, now,
		fixedSession(sessionstate.SessionState{TurnsTaken: 5, LastResponseAt: now}))
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "active", got.FlowKey().ID, "a session idle beyond the threshold scores as turn one")

	idle.InFlightRequests = 1
	got = s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "idle", got.FlowKey().ID, "an in-flight request keeps the session's depth")
}

// The producer publishes a getter, so a queued head observes turns its session
// records after the head was enqueued.
func TestTurnPriority_ReadsLiveSessionStateWhileQueued(t *testing.T) {
	s := &turnPriorityStrategy{timeWeight: 0}
	now := time.Now()

	live := sessionstate.SessionState{TurnsTaken: 2, LastResponseAt: now}
	idA, qA := sessionTurnInfo("growing", &ProgramMetrics{}, now,
		func() sessionstate.SessionState { return live })
	idB, qB := sessionTurnInfo("steady", &ProgramMetrics{}, now,
		fixedSession(sessionstate.SessionState{TurnsTaken: 4, LastResponseAt: now}))
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "steady", got.FlowKey().ID)

	live.TurnsTaken = 6
	got = s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "growing", got.FlowKey().ID)
}
