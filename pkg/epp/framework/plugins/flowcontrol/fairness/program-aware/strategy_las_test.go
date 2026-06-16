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
		q.PeekHeadV = &fwkfcmocks.MockQueueItemAccessor{EnqueueTimeV: headEnqueue}
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
	s.getOrCreateState("alpha").AddService(1000)
	s.getOrCreateState("beta").AddService(10)

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

func TestLAS_Pick_DecaysInactiveService(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0, decayFactor: 0.5}
	s.getOrCreateState("idle").AddService(100)

	// Empty queue and no in-flight: decay applies.
	queues := map[string]QueueInfo{
		"idle": {Queue: makeQueue("idle", 0, time.Time{}), Metrics: &ProgramMetrics{}, Len: 0},
	}
	s.Pick(0, queues)

	assert.InDelta(t, 50.0, s.getOrCreateState("idle").Service(), 0.001)
}

func TestLAS_Pick_NoDecayWhenInFlight(t *testing.T) {
	s := &LASStrategy{weightService: 1.0, weightHeadWait: 0.0, decayFactor: 0.5}
	s.getOrCreateState("busy").AddService(100)

	m := &ProgramMetrics{}
	m.RecordDispatched(time.Time{}) // inFlight = 1
	queues := map[string]QueueInfo{
		"busy": {Queue: makeQueue("busy", 0, time.Time{}), Metrics: m, Len: 0},
	}
	s.Pick(0, queues)

	assert.Equal(t, 100.0, s.getOrCreateState("busy").Service(), "in-flight gates decay")
}

func TestLAS_OnCompleted_AccumulatesWeightedCost(t *testing.T) {
	s := &LASStrategy{}
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	resp := &fwkrc.Response{EndOfStream: true}
	resp.Usage.PromptTokens = 100
	resp.Usage.CompletionTokens = 50

	s.OnCompleted(nil, req, resp)

	// cost = 1*100 + 2*50 = 200
	assert.Equal(t, 200.0, s.getOrCreateState("alpha").Service())
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

	st.Decay(now, 1.0, 0)

	assert.InDelta(t, 50.0, st.Service(), 0.001)
}

func TestLAS_FactorDecay_AppliesPerCall(t *testing.T) {
	st := &lasState{attainedService: 100}
	st.Decay(time.Now(), 0, 0.5)
	assert.Equal(t, 50.0, st.Service())
	st.Decay(time.Now(), 0, 0.5)
	assert.Equal(t, 25.0, st.Service())
}

func TestLAS_EvictProgram_DropsState(t *testing.T) {
	s := &LASStrategy{}
	s.getOrCreateState("alpha").AddService(100)

	s.EvictProgram("alpha")

	// A subsequent getOrCreateState returns a fresh zero entry.
	assert.Equal(t, 0.0, s.getOrCreateState("alpha").Service())
}

func TestRangeNormalize(t *testing.T) {
	assert.Equal(t, 0.5, rangeNormalize(5, 10, 10), "min == max returns 0.5")
	assert.Equal(t, 0.0, rangeNormalize(0, 0, 10))
	assert.Equal(t, 1.0, rangeNormalize(10, 0, 10))
	assert.InDelta(t, 0.25, rangeNormalize(2.5, 0, 10), 0.001)
	assert.True(t, math.IsNaN(rangeNormalize(0, 1, 1)) || rangeNormalize(0, 1, 1) == 0.5)
}
