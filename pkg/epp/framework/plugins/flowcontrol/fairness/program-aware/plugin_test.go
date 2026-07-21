package programaware

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func decoder(s string) *json.Decoder { return json.NewDecoder(strings.NewReader(s)) }

func TestFactory_DefaultConfig(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", nil, nil)
	require.NoError(t, err)
	require.NotNil(t, p)
}

func TestFactory_LASConfig(t *testing.T) {
	cfg := `{"strategy":"las","lasWeightService":0.7,"lasWeightHeadWait":0.3,"lasHalfLifeSeconds":60}`
	p, err := ProgramAwarePluginFactory("test", decoder(cfg), nil)
	require.NoError(t, err)
	require.NotNil(t, p)
}

func TestFactory_UnknownStrategy(t *testing.T) {
	_, err := ProgramAwarePluginFactory("test", decoder(`{"strategy":"wfq"}`), nil)
	require.Error(t, err)
}

func TestFactory_InvalidConfig(t *testing.T) {
	cases := map[string]string{
		"negative ttl":       `{"evictionTtlSeconds":-1}`,
		"zero sweep":         `{"evictionSweepSeconds":0}`,
		"negative weight":    `{"lasWeightService":-0.1}`,
		"decay factor > 1":   `{"lasDecayFactor":1.5}`,
		"decay factor 0":     `{"lasDecayFactor":0}`,
		"negative half life": `{"lasHalfLifeSeconds":-1}`,
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ProgramAwarePluginFactory("test", decoder(cfg), nil)
			require.Error(t, err)
		})
	}
}

func TestPick_NilBand(t *testing.T) {
	p := &ProgramAwarePlugin{}
	got, err := p.Pick(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestPick_AllQueuesEmpty(t *testing.T) {
	band := &fwkfcmocks.MockPriorityBandAccessor{
		PriorityV: 0,
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(&fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "p1"}})
			cb(&fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "p2"}})
		},
	}
	p := &ProgramAwarePlugin{}
	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestPick_SingleNonEmptyQueue_StashesEnqueueTime(t *testing.T) {
	enqueue := time.Now().Add(-100 * time.Millisecond)
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	item := &fwkfcmocks.MockQueueItemAccessor{
		EnqueueTimeV:     enqueue,
		OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{InferenceRequestV: req},
	}
	queue := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: "alpha"},
		PeekV:    item,
	}
	band := &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) { cb(queue) },
	}

	p := &ProgramAwarePlugin{}
	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Equal(t, queue, got)

	stashed, ok := fwksched.ReadRequestAttribute[time.Time](req, enqueueTimeAttributeKey)
	require.True(t, ok)
	assert.Equal(t, enqueue, stashed)
}

func TestPreRequest_RecordsDispatchAndWait(t *testing.T) {
	enqueue := time.Now().Add(-50 * time.Millisecond)
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	req.PutAttribute(enqueueTimeAttributeKey, enqueue)

	p := &ProgramAwarePlugin{}
	p.PreRequest(context.Background(), req, nil)

	m := p.getOrCreateMetrics("alpha")
	assert.Equal(t, int64(1), m.DispatchedCount())
	assert.Equal(t, int64(1), m.InFlight())
	assert.Equal(t, int64(1), m.WaitCount())
	assert.Greater(t, m.AverageWaitTime(), 0.0)
}

func TestPreRequest_NoEnqueueAttribute_StillDispatches(t *testing.T) {
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	p := &ProgramAwarePlugin{}
	p.PreRequest(context.Background(), req, nil)

	m := p.getOrCreateMetrics("alpha")
	assert.Equal(t, int64(1), m.DispatchedCount())
	assert.Equal(t, int64(1), m.InFlight())
	assert.Equal(t, int64(0), m.WaitCount())
}

func TestPreRequest_NoFairnessID_FallsBackToDefault(t *testing.T) {
	req := &fwksched.InferenceRequest{}
	p := &ProgramAwarePlugin{}
	p.PreRequest(context.Background(), req, nil)

	got, ok := p.programMetrics.Load(metadata.DefaultFairnessID)
	require.True(t, ok, "default fairness ID entry should be created")
	m, ok := got.(*ProgramMetrics)
	require.True(t, ok)
	assert.Equal(t, int64(1), m.DispatchedCount())
}

func TestResponseBody_FinalChunkOnly(t *testing.T) {
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	seedTime := m.LastCompletionTime()
	m.RecordDispatched(time.Time{})

	// Intermediate chunk: in-flight unchanged, completion time unchanged.
	p.ResponseBody(context.Background(), req, &fwkrc.Response{EndOfStream: false}, nil)
	assert.Equal(t, int64(1), m.InFlight())
	assert.Equal(t, seedTime, m.LastCompletionTime())

	// Final chunk: completion advanced, in-flight decremented.
	time.Sleep(time.Millisecond)
	p.ResponseBody(context.Background(), req, &fwkrc.Response{EndOfStream: true}, nil)
	assert.Equal(t, int64(0), m.InFlight())
	assert.True(t, m.LastCompletionTime().After(seedTime))
}

func TestResponseBody_NilSafe(t *testing.T) {
	p := &ProgramAwarePlugin{}
	p.ResponseBody(context.Background(), nil, &fwkrc.Response{EndOfStream: true}, nil)
	p.ResponseBody(context.Background(), &fwksched.InferenceRequest{}, nil, nil)
}

func TestEvictIdle_RemovesIdle(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	m.RecordDispatched(time.Time{})
	m.RecordCompletion(time.Now().Add(-10 * time.Second))

	p.evictIdle(time.Second)

	_, exists := p.programMetrics.Load("alpha")
	assert.False(t, exists)
}

func TestEvictIdle_KeepsInFlight(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	m.RecordDispatched(time.Time{}) // inFlight = 1
	// Force lastCompletionTime old; in-flight should still gate eviction.
	m.mu.Lock()
	m.lastCompletionTime = time.Now().Add(-10 * time.Second)
	m.mu.Unlock()

	p.evictIdle(time.Second)

	_, exists := p.programMetrics.Load("alpha")
	assert.True(t, exists)
}

func TestEvictIdle_KeepsRecent(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	m.RecordDispatched(time.Time{})
	m.RecordCompletion(time.Now())

	p.evictIdle(time.Hour)

	_, exists := p.programMetrics.Load("alpha")
	assert.True(t, exists)
}

func TestEvictIdle_EvictsNeverCompletedAfterTTL(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	// Force the seed time into the past so the TTL gate trips.
	m.mu.Lock()
	m.lastCompletionTime = time.Now().Add(-10 * time.Second)
	m.mu.Unlock()

	p.evictIdle(time.Second)

	_, exists := p.programMetrics.Load("alpha")
	assert.False(t, exists)
}

func TestComputeFairnessIndex_EqualWaits(t *testing.T) {
	p := &ProgramAwarePlugin{}
	for _, id := range []string{"a", "b", "c"} {
		m := p.getOrCreateMetrics(id)
		m.RecordDispatched(time.Now().Add(-100 * time.Millisecond))
	}
	got := p.computeFairnessIndex()
	assert.InDelta(t, 1.0, got, 0.05)
}

func TestComputeFairnessIndex_SingleProgram(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("a")
	m.RecordDispatched(time.Now().Add(-50 * time.Millisecond))
	assert.Equal(t, 1.0, p.computeFairnessIndex())
}

func TestComputeFairnessIndex_NoData(t *testing.T) {
	p := &ProgramAwarePlugin{}
	assert.Equal(t, 1.0, p.computeFairnessIndex())
}

func TestComputeFairnessIndex_SkewedWaits(t *testing.T) {
	p := &ProgramAwarePlugin{}
	a := p.getOrCreateMetrics("a")
	b := p.getOrCreateMetrics("b")
	a.RecordDispatched(time.Now().Add(-10 * time.Millisecond))
	b.RecordDispatched(time.Now().Add(-1000 * time.Millisecond))
	got := p.computeFairnessIndex()
	assert.Less(t, got, 0.9, "skewed waits should produce sub-1.0 fairness")
}

func TestGetOrCreateMetrics_Idempotent(t *testing.T) {
	p := &ProgramAwarePlugin{}
	a := p.getOrCreateMetrics("alpha")
	b := p.getOrCreateMetrics("alpha")
	assert.Same(t, a, b)
}

func TestDumpState(t *testing.T) {
	p := &ProgramAwarePlugin{}
	p.getOrCreateMetrics("prog-a").RecordDispatched(time.Now().Add(-10 * time.Millisecond))
	p.getOrCreateMetrics("prog-b").RecordDispatched(time.Now().Add(-20 * time.Millisecond))

	payload, err := p.DumpState()
	require.NoError(t, err)

	var state fairnessDumpState
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Equal(t, 2, state.TotalPrograms)
	assert.Equal(t, int64(2), state.TotalInFlight)
	assert.GreaterOrEqual(t, state.FairnessIndex, 0.0)
	assert.LessOrEqual(t, state.FairnessIndex, 1.0)
}

func TestDumpStateOmitsProgramIDs(t *testing.T) {
	p := &ProgramAwarePlugin{}
	// Program IDs come from a user-controlled header and must never be dumped.
	p.getOrCreateMetrics("secret-tenant-xyz").RecordDispatched(time.Now().Add(-5 * time.Millisecond))

	payload, err := p.DumpState()
	require.NoError(t, err)
	assert.NotContains(t, string(payload), "secret-tenant-xyz")
}

func TestDumpStateEmpty(t *testing.T) {
	p := &ProgramAwarePlugin{}

	payload, err := p.DumpState()
	require.NoError(t, err)
	assert.True(t, json.Valid(payload))

	var state fairnessDumpState
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Equal(t, 0, state.TotalPrograms)
	assert.Equal(t, int64(0), state.TotalInFlight)
	// With no programs the policy is trivially fair.
	assert.Equal(t, 1.0, state.FairnessIndex)
}
