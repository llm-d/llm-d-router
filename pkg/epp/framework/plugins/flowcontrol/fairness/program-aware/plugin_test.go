package programaware

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
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

func newDumpPlugin(t *testing.T) *ProgramAwarePlugin {
	t.Helper()
	s, err := newStrategy(DefaultConfig())
	require.NoError(t, err)
	return &ProgramAwarePlugin{strategy: s}
}

// countingLookupStrategy counts serviceLookup calls, pinning that DumpState
// resolves attained service only for rows retained after the cap.
type countingLookupStrategy struct {
	lookups int
}

func (c *countingLookupStrategy) Name() string { return "counting" }
func (c *countingLookupStrategy) Pick(int, map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	return nil
}
func (c *countingLookupStrategy) OnPreRequest(*ProgramMetrics, *fwksched.InferenceRequest) {}
func (c *countingLookupStrategy) OnCompleted(*ProgramMetrics, *fwksched.InferenceRequest, *fwkrc.Response) {
}
func (c *countingLookupStrategy) EvictProgram(string)                {}
func (c *countingLookupStrategy) Collectors() []prometheus.Collector { return nil }
func (c *countingLookupStrategy) ServiceForProgram(string) float64 {
	c.lookups++
	return 0
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
	assert.Equal(t, "las", state.Strategy)

	// #1839 consumers rely on these aggregate keys; the per-program superset
	// must keep emitting them.
	assert.Contains(t, string(payload), `"totalInFlight"`)
	assert.Contains(t, string(payload), `"fairnessIndex"`)
}

// TestDumpStateHashesProgramIDs guards the #1839 decision to keep raw,
// user-controlled program IDs out of the debug dump. The per-program list
// hashes each ID (sanitizeProgramID) so per-program granularity is retained
// without echoing raw tenant strings. Whether to expose raw IDs instead is a
// maintainer (liu-cong) decision; until then the hash default holds.
func TestDumpStateHashesProgramIDs(t *testing.T) {
	const raw = "secret-tenant-xyz"
	p := newDumpPlugin(t)
	p.getOrCreateMetrics(raw).RecordDispatched(time.Now().Add(-5 * time.Millisecond))

	payload, err := p.DumpState()
	require.NoError(t, err)
	assert.NotContains(t, string(payload), raw)
	assert.Contains(t, string(payload), sanitizeProgramID(raw))
}

func TestDumpStateEmpty(t *testing.T) {
	p := &ProgramAwarePlugin{}

	payload, err := p.DumpState()
	require.NoError(t, err)
	assert.True(t, json.Valid(payload))
	// Empty list must marshal as [] not null.
	assert.Contains(t, string(payload), `"programs":[]`)

	var state fairnessDumpState
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Equal(t, 0, state.TotalPrograms)
	assert.Equal(t, int64(0), state.TotalInFlight)
	// With no programs the policy is trivially fair.
	assert.Equal(t, 1.0, state.FairnessIndex)
	assert.Empty(t, state.Programs)
	assert.Equal(t, maxDebugDumpPrograms, state.MaxPrograms)
	assert.False(t, state.Truncated)
}

func TestDumpStatePrograms(t *testing.T) {
	t.Run("cap and truncation", func(t *testing.T) {
		p := newDumpPlugin(t)
		for i := range maxDebugDumpPrograms + 5 {
			m := p.getOrCreateMetrics(fmt.Sprintf("prog-%04d", i))
			// Higher indices carry more in-flight so the retained head is
			// deterministic under the in-flight-desc sort.
			for j := 0; j <= i; j++ {
				m.RecordDispatched(time.Time{})
			}
		}

		var state fairnessDumpState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		assert.Len(t, state.Programs, maxDebugDumpPrograms)
		assert.Equal(t, maxDebugDumpPrograms+5, state.TotalPrograms)
		assert.True(t, state.Truncated)
		assert.Equal(t, maxDebugDumpPrograms, state.MaxPrograms)
		// The retained rows are the top-K by in-flight, head first: prog-1004
		// carries the most in-flight, prog-0005 is the last to make the cap.
		assert.Equal(t, sanitizeProgramID(fmt.Sprintf("prog-%04d", maxDebugDumpPrograms+4)), state.Programs[0].ProgramID)
		assert.Equal(t, sanitizeProgramID("prog-0005"), state.Programs[maxDebugDumpPrograms-1].ProgramID)
	})

	t.Run("service lookups only for retained rows", func(t *testing.T) {
		s := &countingLookupStrategy{}
		p := &ProgramAwarePlugin{strategy: s}
		for i := range maxDebugDumpPrograms + 5 {
			p.getOrCreateMetrics(fmt.Sprintf("prog-%04d", i)).RecordDispatched(time.Time{})
		}

		_, err := p.DumpState()
		require.NoError(t, err)
		assert.Equal(t, maxDebugDumpPrograms, s.lookups)
	})

	t.Run("sort by in-flight then dispatched", func(t *testing.T) {
		p := newDumpPlugin(t)
		// hi: 2 in-flight; mid: 1 in-flight, 3 dispatched; lo: 1 in-flight, 1 dispatched.
		hi := &ProgramMetrics{}
		hi.RecordDispatched(time.Time{})
		hi.RecordDispatched(time.Time{})
		p.programMetrics.Store("hi", hi)
		mid := &ProgramMetrics{}
		mid.RecordDispatched(time.Time{})
		mid.RecordDispatched(time.Time{})
		mid.RecordDispatched(time.Time{})
		mid.RecordCompletion(time.Now()) // dispatched 3, in-flight 2 -> bump dispatched, drop in-flight
		mid.RecordCompletion(time.Now()) // in-flight 1
		p.programMetrics.Store("mid", mid)
		lo := &ProgramMetrics{}
		lo.RecordDispatched(time.Time{})
		p.programMetrics.Store("lo", lo)

		var state fairnessDumpState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		require.Len(t, state.Programs, 3)
		assert.Equal(t, sanitizeProgramID("hi"), state.Programs[0].ProgramID)  // 2 in-flight
		assert.Equal(t, sanitizeProgramID("mid"), state.Programs[1].ProgramID) // 1 in-flight, 3 dispatched
		assert.Equal(t, sanitizeProgramID("lo"), state.Programs[2].ProgramID)  // 1 in-flight, 1 dispatched
	})

	t.Run("completed reports service and completion time", func(t *testing.T) {
		p := newDumpPlugin(t)
		req := &fwksched.InferenceRequest{FairnessID: "alpha"}
		req.PutAttribute(enqueueTimeAttributeKey, time.Now().Add(-50*time.Millisecond))
		p.PreRequest(context.Background(), req, nil)
		p.ResponseBody(context.Background(), req, &fwkrc.Response{
			EndOfStream: true,
			Usage:       fwkrh.Usage{PromptTokens: 10, CompletionTokens: 5},
		}, nil)

		var state fairnessDumpState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		require.Len(t, state.Programs, 1)
		ps := state.Programs[0]
		assert.Equal(t, sanitizeProgramID("alpha"), ps.ProgramID)
		assert.Equal(t, int64(1), ps.DispatchedCount)
		assert.Equal(t, int64(0), ps.InFlight)
		assert.Equal(t, int64(1), ps.WaitCount)
		assert.Equal(t, float64(1*10+2*5), ps.AttainedService)
		assert.NotEmpty(t, ps.LastCompletionTime)
	})

	t.Run("in-flight reports no completion time", func(t *testing.T) {
		p := newDumpPlugin(t)
		req := &fwksched.InferenceRequest{FairnessID: "beta"}
		p.PreRequest(context.Background(), req, nil)

		var state fairnessDumpState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		require.Len(t, state.Programs, 1)
		assert.Equal(t, int64(1), state.Programs[0].DispatchedCount)
		assert.Equal(t, int64(1), state.Programs[0].InFlight)
		assert.Equal(t, float64(0), state.Programs[0].AttainedService)
		assert.Empty(t, state.Programs[0].LastCompletionTime)
	})

	t.Run("program absent from service snapshot reports zero service", func(t *testing.T) {
		// A program written straight into programMetrics never entered the LAS
		// strategy's state map, so ServiceForProgram returns 0 for it. It must
		// still appear in the dump with attainedService 0.
		p := newDumpPlugin(t)
		m := &ProgramMetrics{}
		m.RecordDispatched(time.Time{})
		p.programMetrics.Store("orphan", m)

		var state fairnessDumpState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		require.Len(t, state.Programs, 1)
		assert.Equal(t, sanitizeProgramID("orphan"), state.Programs[0].ProgramID)
		assert.Equal(t, float64(0), state.Programs[0].AttainedService)
	})
}

func TestDumpStateConcurrentWithDispatch(t *testing.T) {
	p := newDumpPlugin(t)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := range 100 {
			m := p.getOrCreateMetrics(fmt.Sprintf("prog-%03d", i))
			m.RecordDispatched(time.Time{})
			m.RecordCompletion(time.Now())
		}
	}()
	go func() {
		defer wg.Done()
		for range 100 {
			if _, err := p.DumpState(); err != nil {
				t.Errorf("DumpState returned error: %v", err)
			}
		}
	}()
	wg.Wait()
}
