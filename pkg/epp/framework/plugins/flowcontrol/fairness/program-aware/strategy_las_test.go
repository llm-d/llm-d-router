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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
)

func makeQueue(id string, length int, headEnqueue time.Time) *fwkfcmocks.MockFlowQueueAccessor {
	q := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     length,
		FlowKeyV: flowcontrol.FlowKey{ID: id},
	}
	if length > 0 {
		q.PeekV = &fwkfcmocks.MockQueueItemAccessor{EnqueueTimeV: headEnqueue}
	}
	return q
}

func makeInfo(id string, headEnqueue time.Time) (string, QueueInfo) {
	return id, QueueInfo{
		Queue:   makeQueue(id, 1, headEnqueue),
		Metrics: &ProgramMetrics{},
		Len:     1,
	}
}

func TestLAS_Name(t *testing.T) {
	assert.Equal(t, "las", (&LASStrategy{}).Name())
}

func TestLAS_Pick_PrefersLowerService(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0}
	now := time.Now()

	// Seed alpha with high attained service, beta with low.
	s.getOrCreateState("alpha").AddService(1000, now, 0)
	s.getOrCreateState("beta").AddService(10, now, 0)

	idA, qA := makeInfo("alpha", now)
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID)
}

func TestLAS_Pick_ColdStartUsesHeadWait(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 1.0}
	now := time.Now()

	// Both have zero service; alpha's head waited longer, so it wins on tiebreak.
	idA, qA := makeInfo("alpha", now.Add(-500*time.Millisecond))
	idB, qB := makeInfo("beta", now.Add(-50*time.Millisecond))
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID)
}

// TestLAS_Pick_UsesDecayedService is the regression test for idle-service freeze: the band
// accessor skips empty queues, so an idle program is never visited by Pick, and its decay must
// happen lazily when its service is next read. A program with high raw service that idled long
// enough must outrank a fresh low-service competitor.
func TestLAS_Pick_UsesDecayedService(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0, halfLifeSeconds: 1.0}
	now := time.Now()

	// alpha accrued heavy service, then idled for 10 half-lives (never visited while idle);
	// beta accrued light service just now.
	alpha := s.getOrCreateState("alpha")
	alpha.attainedService = 1000
	alpha.decayAnchor = now.Add(-10 * time.Second)
	beta := s.getOrCreateState("beta")
	beta.attainedService = 10
	beta.decayAnchor = now

	idA, qA := makeInfo("alpha", now)
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID,
		"alpha's service must have decayed to ~1 during its idle period, below beta's 10")
}

// TestLAS_Pick_DoesNotCreateStateForEmptyEntries guards against resurrecting strategy state for a
// program the eviction sweep is concurrently removing: entries with Len == 0 must not reach
// getOrCreateState.
func TestLAS_Pick_DoesNotCreateStateForEmptyEntries(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0}
	queues := map[string]QueueInfo{
		"ghost": {Queue: makeQueue("ghost", 0, time.Time{}), Metrics: &ProgramMetrics{}, Len: 0},
	}
	s.Pick(0, queues)

	_, exists := s.state.Load("ghost")
	assert.False(t, exists, "an empty entry must not allocate strategy state")
}

func TestLAS_OnCompleted_AccumulatesWeightedCost(t *testing.T) {
	s := &LASStrategy{}
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	resp := &fwkrh.Response{EndOfStream: true}
	resp.Usage.PromptTokens = 100
	resp.Usage.CompletionTokens = 50

	s.OnCompleted(nil, req, resp)

	// cost = 1*100 + 2*50 = 200
	assert.Equal(t, 200.0, s.getOrCreateState("alpha").Service(time.Now(), 0))
}

func TestLAS_OnCompleted_StreamedEventsFallback(t *testing.T) {
	tests := []struct {
		name             string
		promptTokens     int
		completionTokens int
		streamedEvents   int
		wantService      float64
	}{
		{
			name:             "falls back to StreamedEvents when CompletionTokens is zero",
			promptTokens:     0,
			completionTokens: 0,
			streamedEvents:   35,
			wantService:      70.0, // 1*0 + 2*35
		},
		{
			name:             "combines PromptTokens with StreamedEvents fallback",
			promptTokens:     100,
			completionTokens: 0,
			streamedEvents:   25,
			wantService:      150.0, // 1*100 + 2*25
		},
		{
			name:             "reported CompletionTokens takes precedence over StreamedEvents",
			promptTokens:     100,
			completionTokens: 40,
			streamedEvents:   45,
			wantService:      180.0, // 1*100 + 2*40
		},
		{
			name:             "zero CompletionTokens and zero StreamedEvents charges only prompt",
			promptTokens:     100,
			completionTokens: 0,
			streamedEvents:   0,
			wantService:      100.0, // 1*100 + 2*0
		},
		{
			name:             "negative StreamedEvents is ignored",
			promptTokens:     100,
			completionTokens: 0,
			streamedEvents:   -5,
			wantService:      100.0, // 1*100 + 2*0
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := &LASStrategy{}
			req := &fwksched.InferenceRequest{FairnessID: "alpha"}
			resp := &fwkrh.Response{
				EndOfStream:    true,
				StreamedEvents: tc.streamedEvents,
			}
			resp.Usage.PromptTokens = tc.promptTokens
			resp.Usage.CompletionTokens = tc.completionTokens

			s.OnCompleted(nil, req, resp)

			assert.Equal(t, tc.wantService, s.getOrCreateState("alpha").Service(time.Now(), 0))
		})
	}
}

func TestLAS_Pick_DeprioritizesProgramAfterTruncatedStreamWithEvents(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0}
	now := time.Now()

	// alpha disconnects mid-stream after 500 streamed events (no Usage block);
	// beta completes a small request with reported usage (10 prompt + 5 completion = 20 cost).
	s.OnCompleted(nil, &fwksched.InferenceRequest{FairnessID: "alpha"}, &fwkrh.Response{
		EndOfStream:      true,
		StreamedEvents:   500,
		TerminationCause: fwkrh.TerminationCauseClientDisconnect,
	})
	betaResp := &fwkrh.Response{EndOfStream: true}
	betaResp.Usage.PromptTokens = 10
	betaResp.Usage.CompletionTokens = 5
	s.OnCompleted(nil, &fwksched.InferenceRequest{FairnessID: "beta"}, betaResp)

	idA, qA := makeInfo("alpha", now)
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID,
		"alpha accrued 2*500=1000 service from StreamedEvents and must rank behind beta (20)")
}

func TestLAS_OnCompleted_NilSafe(t *testing.T) {
	s := &LASStrategy{}
	s.OnCompleted(nil, nil, &fwkrh.Response{EndOfStream: true})
	s.OnCompleted(nil, &fwksched.InferenceRequest{}, nil)
}

func TestLAS_TimedDecay_HalvesAtHalfLife(t *testing.T) {
	st := &lasState{attainedService: 100}
	now := time.Now()
	st.decayAnchor = now.Add(-1 * time.Second) // one half-life ago

	assert.InDelta(t, 50.0, st.Service(now, 1.0), 0.001)
}

func TestLAS_ZeroHalfLife_DisablesDecay(t *testing.T) {
	st := &lasState{attainedService: 100}
	now := time.Now()
	st.decayAnchor = now.Add(-1 * time.Hour)

	assert.Equal(t, 100.0, st.Service(now, 0), "a half-life of 0 must hold service flat")
}

// TestLAS_AddService_AppliesPendingDecayFirst pins that a completion does not forfeit the decay
// accrued during the preceding idle window: the pending decay is folded in before the new cost is
// accumulated.
func TestLAS_AddService_AppliesPendingDecayFirst(t *testing.T) {
	st := &lasState{attainedService: 100}
	now := time.Now()
	st.decayAnchor = now.Add(-1 * time.Second) // one half-life ago

	got := st.AddService(10, now, 1.0)

	assert.InDelta(t, 60.0, got, 0.001, "100 must halve to 50 before the +10 lands")
}

func TestLAS_EvictProgram_DropsState(t *testing.T) {
	s := &LASStrategy{}
	now := time.Now()
	s.getOrCreateState("alpha").AddService(100, now, 0)

	s.EvictProgram("alpha")

	// A subsequent getOrCreateState returns a fresh zero entry.
	assert.Equal(t, 0.0, s.getOrCreateState("alpha").Service(now, 0))
}

func TestRangeNormalize(t *testing.T) {
	assert.Equal(t, 0.5, rangeNormalize(5, 10, 10), "min == max returns 0.5")
	assert.Equal(t, 0.0, rangeNormalize(0, 0, 10))
	assert.Equal(t, 1.0, rangeNormalize(10, 0, 10))
	assert.InDelta(t, 0.25, rangeNormalize(2.5, 0, 10), 0.001)
	assert.True(t, math.IsNaN(rangeNormalize(0, 1, 1)) || rangeNormalize(0, 1, 1) == 0.5)
}

func makeSessionInfo(id string, headEnqueue time.Time, state func() sessionstate.SessionState) (string, QueueInfo) {
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
	return id, QueueInfo{Queue: q, Metrics: &ProgramMetrics{}, Len: 1}
}

func TestLAS_Pick_ReadsServedTokensFromSessionState(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0}
	now := time.Now()

	idA, qA := makeSessionInfo("alpha", now, func() sessionstate.SessionState {
		return sessionstate.SessionState{ServedInputTokens: 900, ServedOutputTokens: 50, LastResponseAt: now}
	})
	idB, qB := makeSessionInfo("beta", now, func() sessionstate.SessionState {
		return sessionstate.SessionState{ServedInputTokens: 10, ServedOutputTokens: 5, LastResponseAt: now}
	})

	got := s.Pick(0, map[string]QueueInfo{idA: qA, idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID)
	// cost = 1*900 + 2*50
	assert.Equal(t, 1000.0, s.getOrCreateState("alpha").Service(now, 0))
}

func TestLAS_Pick_SyncsOnlyNewServedTokens(t *testing.T) {
	s := &LASStrategy{weightService: 1.0}
	now := time.Now()
	served := sessionstate.SessionState{ServedInputTokens: 100, LastResponseAt: now}
	id, q := makeSessionInfo("alpha", now, func() sessionstate.SessionState { return served })
	queues := map[string]QueueInfo{id: q}

	s.Pick(0, queues)
	s.Pick(0, queues)
	assert.Equal(t, 100.0, s.getOrCreateState("alpha").Service(now, 0), "a repeated sync must not charge the same tokens twice")

	served.ServedInputTokens = 150
	served.ServedOutputTokens = 10
	s.Pick(0, queues)
	assert.Equal(t, 170.0, s.getOrCreateState("alpha").Service(now, 0))
}

func TestLAS_Pick_AgesNewServiceFromLastResponse(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, halfLifeSeconds: 1}
	now := time.Now()
	id, q := makeSessionInfo("alpha", now, func() sessionstate.SessionState {
		return sessionstate.SessionState{ServedInputTokens: 200, LastResponseAt: now.Add(-time.Second)}
	})

	s.Pick(0, map[string]QueueInfo{id: q})
	assert.InDelta(t, 100.0, s.getOrCreateState("alpha").Service(time.Now(), 0), 1.0,
		"tokens served one half-life ago count for half")
}

func TestLAS_Pick_RecreatedSessionRestartsServedTotals(t *testing.T) {
	s := &LASStrategy{weightService: 1.0}
	now := time.Now()
	served := sessionstate.SessionState{ServedInputTokens: 500, LastResponseAt: now}
	id, q := makeSessionInfo("alpha", now, func() sessionstate.SessionState { return served })
	queues := map[string]QueueInfo{id: q}
	s.Pick(0, queues)

	served.ServedInputTokens = 20
	s.Pick(0, queues)
	assert.Equal(t, 520.0, s.getOrCreateState("alpha").Service(now, 0))
}

func TestLAS_Pick_KeepsBaselinePerSessionInSharedFlow(t *testing.T) {
	s := &LASStrategy{weightService: 1.0}
	now := time.Now()
	// Two sessions share one flow, for example sub-agents of one program, and
	// take turns at the queue head.
	sessionA := sessionstate.SessionState{SessionID: "a", ServedInputTokens: 500, LastResponseAt: now}
	sessionB := sessionstate.SessionState{SessionID: "b", ServedInputTokens: 20, LastResponseAt: now}
	head := sessionA
	id, q := makeSessionInfo("tenant", now, func() sessionstate.SessionState { return head })
	queues := map[string]QueueInfo{id: q}

	s.Pick(0, queues)
	head = sessionB
	s.Pick(0, queues)
	head = sessionA
	s.Pick(0, queues)
	assert.Equal(t, 520.0, s.getOrCreateState("tenant").Service(now, 0),
		"alternating heads must not charge a session's served totals again")

	sessionB.ServedInputTokens = 50
	head = sessionB
	s.Pick(0, queues)
	assert.Equal(t, 550.0, s.getOrCreateState("tenant").Service(now, 0))
}

func TestLAS_PrunesStaleSessionBaselines(t *testing.T) {
	st := &lasState{}
	now := time.Now()
	st.SyncSession(sessionstate.SessionState{SessionID: "old", ServedInputTokens: 1}, now, 0)
	later := now.Add(servedBaselineTTL + time.Minute)
	st.SyncSession(sessionstate.SessionState{SessionID: "new", ServedInputTokens: 1}, later, 0)

	st.mu.Lock()
	defer st.mu.Unlock()
	assert.NotContains(t, st.served, "old")
	assert.Contains(t, st.served, "new")
}

func TestLAS_OnCompleted_SkipsRequestsWithSessionState(t *testing.T) {
	s := &LASStrategy{}
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	req.PutAttribute(sessionstate.SessionStateDataKey, &fwksched.DynamicRequestAttribute{Get: func() any { return sessionstate.SessionState{} }})
	resp := &fwkrh.Response{EndOfStream: true}
	resp.Usage.PromptTokens = 100

	s.OnCompleted(nil, req, resp)
	assert.Zero(t, s.getOrCreateState("alpha").Service(time.Now(), 0), "the registry records the tokens of a session-tracked request")
}
