package programaware

import (
	"context"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	requesthandling "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Factory tests ---

func TestFactory(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test-instance", nil, nil)
	require.NoError(t, err)
	require.NotNil(t, p)

	assert.Equal(t, ProgramAwarePluginType, p.TypedName().Type)
	assert.Equal(t, "test-instance", p.TypedName().Name)
}

// --- ProgramMetrics tests ---

func TestProgramMetrics_AverageWaitTime(t *testing.T) {
	m := &ProgramMetrics{}

	assert.Equal(t, 0.0, m.AverageWaitTime(), "no data → 0")
	assert.Equal(t, int64(0), m.WaitCount())

	m.RecordWaitTime(100)
	assert.InDelta(t, 100.0, m.AverageWaitTime(), 0.01)

	m.RecordWaitTime(200)
	// (100 + 200) / 2 = 150
	assert.InDelta(t, 150.0, m.AverageWaitTime(), 0.01)

	m.RecordWaitTime(50)
	// (100 + 200 + 50) / 3 = 116.67
	assert.InDelta(t, 116.67, m.AverageWaitTime(), 0.01)
	assert.Equal(t, int64(3), m.WaitCount())
}

func TestProgramMetrics_Counters(t *testing.T) {
	m := &ProgramMetrics{}

	m.IncrementRequests()
	m.IncrementRequests()
	m.IncrementDispatched()
	m.RecordTokens(100, 50)
	m.RecordTokens(200, 75)

	assert.Equal(t, int64(2), m.TotalRequests())
	assert.Equal(t, int64(1), m.DispatchedCount())
}

// --- Pick tests ---

func TestPick_NilBand(t *testing.T) {
	p := &ProgramAwarePlugin{}
	queue, err := p.Pick(context.Background(), nil)
	assert.NoError(t, err)
	assert.Nil(t, queue)
}

func TestPick_AllQueuesEmpty(t *testing.T) {
	p := &ProgramAwarePlugin{}

	band := &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(&fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "prog-a"}})
			cb(&fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "prog-b"}})
		},
	}

	queue, err := p.Pick(context.Background(), band)
	assert.NoError(t, err)
	assert.Nil(t, queue)
}

func TestPick_SingleNonEmptyQueue(t *testing.T) {
	p := &ProgramAwarePlugin{}

	queueA := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     3,
		FlowKeyV: flowcontrol.FlowKey{ID: "prog-a"},
		PeekHeadV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV: time.Now().Add(-2 * time.Second),
		},
	}

	band := &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(queueA)
			cb(&fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "prog-b"}})
		},
	}

	queue, err := p.Pick(context.Background(), band)
	assert.NoError(t, err)
	assert.Equal(t, queueA, queue)
}

func TestPick_RecordsEnqueueTime(t *testing.T) {
	p := &ProgramAwarePlugin{}

	enqueueTime := time.Now().Add(-500 * time.Millisecond)
	request := &fwksched.InferenceRequest{RequestID: "req-123"}
	queueA := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: "prog-a"},
		PeekHeadV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV: enqueueTime,
			OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{
				IDV:               "req-123",
				InferenceRequestV: request,
			},
		},
	}

	band := &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(queueA)
		},
	}

	queue, err := p.Pick(context.Background(), band)
	assert.NoError(t, err)
	assert.Equal(t, queueA, queue)

	// Verify Pick() stored the enqueue time on the request's own attribute store.
	storedTime, ok := fwksched.ReadRequestAttribute[time.Time](request, enqueueTimeAttributeKey)
	require.True(t, ok, "Pick should stash enqueue time on the request")
	assert.Equal(t, enqueueTime, storedTime, "stored time should be the item's enqueue time")
}

// --- Produce tests ---

func TestProduce_UpdatesMetrics(t *testing.T) {
	p := &ProgramAwarePlugin{}

	request := &fwksched.InferenceRequest{
		RequestID:  "req-1",
		FairnessID: "prog-a",
	}

	err := p.Produce(context.Background(), request, nil)
	assert.NoError(t, err)

	// Check metrics were created and incremented.
	metricsRaw, ok := p.programMetrics.Load("prog-a")
	require.True(t, ok)
	metrics := metricsRaw.(*ProgramMetrics)
	assert.Equal(t, int64(1), metrics.TotalRequests())
}

func TestProduce_NoFairnessID(t *testing.T) {
	p := &ProgramAwarePlugin{}

	request := &fwksched.InferenceRequest{
		RequestID: "req-1",
	}

	err := p.Produce(context.Background(), request, nil)
	assert.NoError(t, err)

	// No metrics should be created.
	_, ok := p.programMetrics.Load("")
	assert.False(t, ok)
}

// --- PreRequest tests ---

func TestPreRequest_RecordsWaitTime(t *testing.T) {
	p := &ProgramAwarePlugin{}

	p.programMetrics.Store("prog-a", &ProgramMetrics{})

	request := &fwksched.InferenceRequest{
		RequestID:  "req-1",
		FairnessID: "prog-a",
	}
	// Simulate Pick() having stashed the enqueue time 50ms ago on the request.
	request.PutAttribute(enqueueTimeAttributeKey, time.Now().Add(-50*time.Millisecond))

	p.PreRequest(context.Background(), request, nil)

	metricsRaw, _ := p.programMetrics.Load("prog-a")
	metrics := metricsRaw.(*ProgramMetrics)
	assert.Equal(t, int64(1), metrics.DispatchedCount())
	assert.Greater(t, metrics.AverageWaitTime(), 0.0)
}

// --- ResponseComplete tests ---

func TestResponseComplete_RecordsTokens(t *testing.T) {
	p := &ProgramAwarePlugin{}
	p.programMetrics.Store("prog-a", &ProgramMetrics{})

	request := &fwksched.InferenceRequest{
		RequestID:  "req-1",
		FairnessID: "prog-a",
	}
	response := &fwkrc.Response{
		EndOfStream: true,
		Usage: requesthandling.Usage{
			PromptTokens:     100,
			CompletionTokens: 50,
		},
	}

	p.ResponseBody(context.Background(), request, response, &datalayer.EndpointMetadata{})

	metricsRaw, _ := p.programMetrics.Load("prog-a")
	metrics := metricsRaw.(*ProgramMetrics)

	// EWMA token cost should be recorded: 100*1 + 50*2 = 200 weighted tokens.
	assert.Greater(t, metrics.AverageTokens(), 0.0, "token usage should be recorded")
}

func TestResponseBody_IntermediateChunks_AreNoOp(t *testing.T) {
	// Streaming responses fire ResponseBody once per chunk; only the final
	// chunk (EndOfStream=true) should perform terminal-state work. Verifies
	// InFlight is decremented exactly once and tokens are recorded once.
	p := &ProgramAwarePlugin{}
	m := &ProgramMetrics{}
	p.programMetrics.Store("prog-a", m)
	m.IncrementInFlight() // simulates PreRequest

	request := &fwksched.InferenceRequest{RequestID: "req-1", FairnessID: "prog-a"}

	// Five intermediate chunks — must be no-ops.
	for range 5 {
		p.ResponseBody(context.Background(), request, &fwkrc.Response{EndOfStream: false}, &datalayer.EndpointMetadata{})
	}
	assert.Equal(t, int64(1), m.InFlight(), "intermediate chunks must not decrement InFlight")
	assert.Equal(t, 0.0, m.AverageTokens(), "intermediate chunks must not record tokens")

	// Final chunk fires the terminal hook exactly once.
	finalResp := &fwkrc.Response{
		EndOfStream: true,
		Usage:       requesthandling.Usage{PromptTokens: 100, CompletionTokens: 50},
	}
	p.ResponseBody(context.Background(), request, finalResp, &datalayer.EndpointMetadata{})
	assert.Equal(t, int64(0), m.InFlight())
	assert.Greater(t, m.AverageTokens(), 0.0, "terminal chunk records token cost")
}

func TestResponseComplete_NilResponse_NoOp(t *testing.T) {
	// A nil response must not run terminal work — the final-chunk hook is
	// the only place tokens are recorded and InFlight is decremented.
	p := &ProgramAwarePlugin{}
	m := &ProgramMetrics{}
	p.programMetrics.Store("prog-a", m)
	m.IncrementInFlight()

	request := &fwksched.InferenceRequest{RequestID: "req-1", FairnessID: "prog-a"}

	p.ResponseBody(context.Background(), request, nil, nil)
	assert.Equal(t, int64(1), m.InFlight(), "nil response must not decrement InFlight")
	assert.Equal(t, 0.0, m.AverageTokens(), "nil response must not record tokens")
}

// --- rangeNormalize tests ---

func TestRangeNormalize(t *testing.T) {
	assert.InDelta(t, 0.0, rangeNormalize(0, 0, 100), 0.001)
	assert.InDelta(t, 0.5, rangeNormalize(50, 0, 100), 0.001)
	assert.InDelta(t, 1.0, rangeNormalize(100, 0, 100), 0.001)
	assert.InDelta(t, 0.5, rangeNormalize(42, 42, 42), 0.001, "min==max returns 0.5")
	assert.InDelta(t, 0.5, rangeNormalize(-10, -20, 0), 0.001, "works with negative range")
	assert.InDelta(t, 0.0, rangeNormalize(-20, -20, 0), 0.001, "min of negative range")
	assert.InDelta(t, 1.0, rangeNormalize(0, -20, 0), 0.001, "max of negative range")
}

// --- Produces / Consumes tests ---

func TestProducesConsumes(t *testing.T) {
	p := &ProgramAwarePlugin{}
	assert.Empty(t, p.Produces())
	assert.Empty(t, p.Consumes())
}

// --- Full lifecycle integration test ---

func TestFullLifecycle(t *testing.T) {
	p := &ProgramAwarePlugin{name: "test"}

	programID := "prog-integration"
	request := &fwksched.InferenceRequest{
		RequestID:  "req-lifecycle",
		FairnessID: programID,
	}

	// 0. Simulate Pick() stashing the enqueue time on the request (flow
	//    control layer). In production this happens when the request is
	//    dispatched from the queue.
	request.PutAttribute(enqueueTimeAttributeKey, time.Now().Add(-20*time.Millisecond))

	// 1. PrepareData (runs after flow control dispatch)
	err := p.Produce(context.Background(), request, nil)
	require.NoError(t, err)

	// Verify metrics created.
	metricsRaw, ok := p.programMetrics.Load(programID)
	require.True(t, ok)
	metrics := metricsRaw.(*ProgramMetrics)
	assert.Equal(t, int64(1), metrics.TotalRequests())
	assert.Equal(t, int64(0), metrics.DispatchedCount())

	// 2. PreRequest — computes wait time from the request's enqueue-time attribute
	p.PreRequest(context.Background(), request, nil)
	assert.Equal(t, int64(1), metrics.DispatchedCount())
	assert.Greater(t, metrics.AverageWaitTime(), 0.0, "wait time should reflect queue residence time")

	// 3. ResponseComplete
	response := &fwkrc.Response{Headers: map[string]string{}, EndOfStream: true}
	response.Usage = requesthandling.Usage{PromptTokens: 42, CompletionTokens: 17}
	p.ResponseBody(context.Background(), request, response, &datalayer.EndpointMetadata{})
	// 42 input + 17 output → weighted cost 42 + 34 = 76 tokens.
	assert.InDelta(t, 76.0, metrics.AverageTokens(), 0.01)
}

// --- fairness index tests (wait-time-based) ---

func TestComputeFairnessIndex_EqualWaitTime(t *testing.T) {
	p := &ProgramAwarePlugin{}

	mA := &ProgramMetrics{}
	mA.RecordWaitTime(100)
	mA.RecordWaitTime(100)
	p.programMetrics.Store("prog-a", mA)

	mB := &ProgramMetrics{}
	mB.RecordWaitTime(100)
	mB.RecordWaitTime(100)
	p.programMetrics.Store("prog-b", mB)

	assert.InDelta(t, 1.0, p.computeFairnessIndex(), 0.001, "equal wait time → perfect fairness")
}

func TestComputeFairnessIndex_SkewedWaitTime(t *testing.T) {
	p := &ProgramAwarePlugin{}

	mA := &ProgramMetrics{}
	mA.RecordWaitTime(1000)
	p.programMetrics.Store("prog-a", mA)

	mB := &ProgramMetrics{}
	mB.RecordWaitTime(100)
	p.programMetrics.Store("prog-b", mB)

	idx := p.computeFairnessIndex()
	assert.Less(t, idx, 1.0, "skewed wait should produce index < 1")
	// J = (1000+100)^2 / (2 * (1000^2 + 100^2)) ≈ 0.599
	assert.InDelta(t, 0.599, idx, 0.01)
}

func TestComputeFairnessIndex_SingleProgram(t *testing.T) {
	p := &ProgramAwarePlugin{}

	m := &ProgramMetrics{}
	m.RecordWaitTime(500)
	p.programMetrics.Store("prog-a", m)

	assert.InDelta(t, 1.0, p.computeFairnessIndex(), 0.001, "single program → trivially fair")
}

func TestComputeFairnessIndex_NoWaitData(t *testing.T) {
	p := &ProgramAwarePlugin{}

	// Programs exist but have no wait observations yet.
	p.programMetrics.Store("prog-a", &ProgramMetrics{})
	p.programMetrics.Store("prog-b", &ProgramMetrics{})

	assert.InDelta(t, 1.0, p.computeFairnessIndex(), 0.001, "no wait data → 1.0")
}

// --- Two-pass scoring tests ---

func TestPick_AllIdenticalMetrics(t *testing.T) {
	// When all queues have identical metrics, Pick should still return a valid queue.
	p := &ProgramAwarePlugin{}

	now := time.Now()
	queueA := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: "prog-a"},
		PeekHeadV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     now,
			OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{IDV: "a-req"},
		},
	}
	queueB := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: "prog-b"},
		PeekHeadV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     now,
			OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{IDV: "b-req"},
		},
	}

	band := &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(queueA)
			cb(queueB)
		},
	}

	queue, err := p.Pick(context.Background(), band)
	assert.NoError(t, err)
	assert.NotNil(t, queue, "should select a queue even when all metrics are identical")
}

// --- Eviction tests ---

func TestEvictIdle_RemovesIdleEntries(t *testing.T) {
	p := &ProgramAwarePlugin{}

	idle := &ProgramMetrics{}
	idle.RecordServiceRate(100, time.Now().Add(-1*time.Hour))
	p.programMetrics.Store("idle-prog", idle)

	recent := &ProgramMetrics{}
	recent.RecordServiceRate(100, time.Now())
	p.programMetrics.Store("recent-prog", recent)

	p.evictIdle(time.Minute)

	_, idleStillThere := p.programMetrics.Load("idle-prog")
	_, recentStillThere := p.programMetrics.Load("recent-prog")
	assert.False(t, idleStillThere, "program past TTL should be evicted")
	assert.True(t, recentStillThere, "program inside TTL must be kept")
}

func TestEvictIdle_KeepsInFlight(t *testing.T) {
	p := &ProgramAwarePlugin{}

	m := &ProgramMetrics{}
	m.RecordServiceRate(100, time.Now().Add(-1*time.Hour))
	m.IncrementInFlight()
	p.programMetrics.Store("busy-prog", m)

	p.evictIdle(time.Minute)

	_, ok := p.programMetrics.Load("busy-prog")
	assert.True(t, ok, "program with in-flight requests must not be evicted regardless of TTL")
}

func TestEvictIdle_KeepsZeroCompletion(t *testing.T) {
	p := &ProgramAwarePlugin{}

	// Fresh entry with no completions yet — e.g. queued but not dispatched.
	p.programMetrics.Store("new-prog", &ProgramMetrics{})

	p.evictIdle(time.Nanosecond)

	_, ok := p.programMetrics.Load("new-prog")
	assert.True(t, ok, "program with no completion time must not be evicted")
}
