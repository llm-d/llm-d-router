package programaware

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
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
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0, decayFactor: 1.0}
	now := time.Now()

	// Seed alpha with high attained service, beta with low.
	s.getOrCreateState("alpha").AddService(1000, now, 0, 1.0)
	s.getOrCreateState("beta").AddService(10, now, 0, 1.0)

	idA, qA := makeInfo("alpha", now)
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID)
}

func TestLAS_Pick_ColdStartUsesHeadWait(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 1.0, decayFactor: 1.0}
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
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0, halfLifeSeconds: 1.0, decayFactor: 1.0}
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
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0, decayFactor: 1.0}
	queues := map[string]QueueInfo{
		"ghost": {Queue: makeQueue("ghost", 0, time.Time{}), Metrics: &ProgramMetrics{}, Len: 0},
	}
	s.Pick(0, queues)

	_, exists := s.state.Load("ghost")
	assert.False(t, exists, "an empty entry must not allocate strategy state")
}

func TestLAS_OnCompleted_AccumulatesWeightedCost(t *testing.T) {
	s := &LASStrategy{decayFactor: 1.0}
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	resp := &fwkrc.Response{EndOfStream: true}
	resp.Usage.PromptTokens = 100
	resp.Usage.CompletionTokens = 50

	s.OnCompleted(nil, req, resp)

	// cost = 1*100 + 2*50 = 200
	assert.Equal(t, 200.0, s.getOrCreateState("alpha").Service(time.Now(), 0, 1.0))
}

func TestLAS_OnCompleted_NilSafe(t *testing.T) {
	s := &LASStrategy{}
	s.OnCompleted(nil, nil, &fwkrc.Response{EndOfStream: true})
	s.OnCompleted(nil, &fwksched.InferenceRequest{}, nil)
}

func TestLAS_TimedDecay_HalvesAtHalfLife(t *testing.T) {
	st := &lasState{attainedService: 100}
	now := time.Now()
	st.decayAnchor = now.Add(-1 * time.Second) // one half-life ago

	assert.InDelta(t, 50.0, st.Service(now, 1.0, 1.0), 0.001)
}

func TestLAS_FactorDecay_AppliesPerElapsedSecond(t *testing.T) {
	st := &lasState{attainedService: 100}
	now := time.Now()
	st.decayAnchor = now.Add(-2 * time.Second)

	// factor 0.5/s over 2 s: 100 * 0.5^2 = 25.
	assert.InDelta(t, 25.0, st.Service(now, 0, 0.5), 0.001)
}

// TestLAS_AddService_AppliesPendingDecayFirst pins that a completion does not forfeit the decay
// accrued during the preceding idle window: the pending decay is folded in before the new cost is
// accumulated.
func TestLAS_AddService_AppliesPendingDecayFirst(t *testing.T) {
	st := &lasState{attainedService: 100}
	now := time.Now()
	st.decayAnchor = now.Add(-1 * time.Second) // one half-life ago

	got := st.AddService(10, now, 1.0, 1.0)

	assert.InDelta(t, 60.0, got, 0.001, "100 must halve to 50 before the +10 lands")
}

func TestLAS_EvictProgram_DropsState(t *testing.T) {
	s := &LASStrategy{}
	now := time.Now()
	s.getOrCreateState("alpha").AddService(100, now, 0, 1.0)

	s.EvictProgram("alpha")

	// A subsequent getOrCreateState returns a fresh zero entry.
	assert.Equal(t, 0.0, s.getOrCreateState("alpha").Service(now, 0, 1.0))
}

func TestRangeNormalize(t *testing.T) {
	assert.Equal(t, 0.5, rangeNormalize(5, 10, 10), "min == max returns 0.5")
	assert.Equal(t, 0.0, rangeNormalize(0, 0, 10))
	assert.Equal(t, 1.0, rangeNormalize(10, 0, 10))
	assert.InDelta(t, 0.25, rangeNormalize(2.5, 0, 10), 0.001)
	assert.True(t, math.IsNaN(rangeNormalize(0, 1, 1)) || rangeNormalize(0, 1, 1) == 0.5)
}
