/*
Copyright 2025 The Kubernetes Authors.

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

package vtc

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// newMockQueue builds a MockFlowQueueAccessor with the given byte size and non-zero Len.
// defaultByteSize is 400 bytes, which yields 100 estimated input tokens (400/4) and a cost of
// 1.0×100 + 3.0×128 = 484 under default weights.
const defaultByteSize uint64 = 400

func newMockQueue(id string) *fwkfcmocks.MockFlowQueueAccessor {
	req := fwkfcmocks.NewMockFlowControlRequest(defaultByteSize, id, flowcontrol.FlowKey{ID: id})
	item := &fwkfcmocks.MockQueueItemAccessor{OriginalRequestV: req}
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:      1,
		ByteSizeV: defaultByteSize,
		FlowKeyV:  flowcontrol.FlowKey{ID: id},
		PeekV:     item,
	}
}

// newMockQueueWithInferenceRequest creates a queue with a request that has a populated
// InferenceRequest, allowing token estimation heuristics to be tested.
func newMockQueueWithInferenceRequest(id string, ir *fwksched.InferenceRequest) *fwkfcmocks.MockFlowQueueAccessor {
	req := fwkfcmocks.NewMockFlowControlRequest(0, id, flowcontrol.FlowKey{ID: id})
	req.InferenceRequestV = ir
	item := &fwkfcmocks.MockQueueItemAccessor{OriginalRequestV: req}
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: id},
		PeekV:    item,
	}
}

// buildBand constructs a MockPriorityBandAccessor from a map of flow ID → queue.
func buildBand(state any, queues map[string]*fwkfcmocks.MockFlowQueueAccessor) *fwkfcmocks.MockPriorityBandAccessor {
	keys := make([]flowcontrol.FlowKey, 0, len(queues))
	for id := range queues {
		keys = append(keys, flowcontrol.FlowKey{ID: id})
	}
	return &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		FlowKeysFunc: func() []flowcontrol.FlowKey { return keys },
		QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
			return queues[id]
		},
	}
}

func TestVTC_TypedName(t *testing.T) {
	t.Parallel()

	p, err := newVTC("my-vtc", nil)
	require.NoError(t, err)
	assert.Equal(t, VTCFairnessPolicyType, p.TypedName().Type)
	assert.Equal(t, "my-vtc", p.TypedName().Name)
}

func TestVTC_TypedName_DefaultsToType(t *testing.T) {
	t.Parallel()

	p, err := newVTC("", nil)
	require.NoError(t, err)
	assert.Equal(t, VTCFairnessPolicyType, p.TypedName().Name)
}

func TestVTC_Factory(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		params    json.RawMessage
		expectErr bool
	}{
		{name: "nil params uses defaults", params: nil},
		{name: "empty params uses defaults", params: json.RawMessage{}},
		{
			name:   "valid params",
			params: json.RawMessage(`{"inputTokenWeight": 2.0, "outputTokenWeight": 5.0}`),
		},
		{
			name:   "zero inputTokenWeight gets corrected to 1.0",
			params: json.RawMessage(`{"inputTokenWeight": 0}`),
		},
		{
			name:   "zero outputTokenWeight gets corrected to 3.0",
			params: json.RawMessage(`{"outputTokenWeight": 0}`),
		},
		{
			name:      "invalid JSON",
			params:    json.RawMessage(`{invalid`),
			expectErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var decoder *json.Decoder
			if len(tc.params) > 0 {
				decoder = json.NewDecoder(bytes.NewReader(tc.params))
			}
			plugin, err := VTCFairnessPolicyFactory("test", decoder, nil)
			if tc.expectErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, plugin)
		})
	}
}

func TestVTC_Pick_NilBand(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)

	got, err := p.Pick(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestVTC_Pick_EmptyBand(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	band := &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		FlowKeysFunc: func() []flowcontrol.FlowKey { return nil },
	}

	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestVTC_Pick_AllEmptyQueues(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	emptyQueue := &fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "a"}}
	band := buildBand(state, map[string]*fwkfcmocks.MockFlowQueueAccessor{"a": emptyQueue})

	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestVTC_Pick_SingleQueue(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	q := newMockQueue("tenant-a")
	band := buildBand(state, map[string]*fwkfcmocks.MockFlowQueueAccessor{"tenant-a": q})

	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "tenant-a", got.FlowKey().ID)
}

func TestVTC_Pick_LowerCounterWins(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background()).(*vtcBandState)

	// Pre-set counters: flow-b has a lower counter than flow-a.
	state.counters["flow-a"] = 500
	state.counters["flow-b"] = 100

	qA := newMockQueue("flow-a")
	qB := newMockQueue("flow-b")
	band := buildBand(state, map[string]*fwkfcmocks.MockFlowQueueAccessor{
		"flow-a": qA,
		"flow-b": qB,
	})

	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "flow-b", got.FlowKey().ID, "flow-b has lower counter and should be selected")
}

func TestVTC_Pick_CounterAdvancesAfterDispatch(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background()).(*vtcBandState)

	qA := newMockQueue("flow-a")
	qB := newMockQueue("flow-b")
	band := buildBand(state, map[string]*fwkfcmocks.MockFlowQueueAccessor{
		"flow-a": qA,
		"flow-b": qB,
	})

	// First dispatch: both counters start at 0; lexicographically, flow-a sorts before flow-b.
	first, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, first)
	firstID := first.FlowKey().ID

	// The counter for the first winner must now be higher than the other.
	otherID := "flow-b"
	if firstID == "flow-b" {
		otherID = "flow-a"
	}
	assert.Greater(t, state.counters[firstID], state.counters[otherID],
		"winner's counter must be advanced after dispatch")
}

func TestVTC_Pick_CounterLift_OnRejoin(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background()).(*vtcBandState)

	// Simulate an existing flow with a high counter.
	state.counters["flow-a"] = 1000

	// flow-b is new (rejoining); it should get its counter lifted to 1000 (activeMin), not 0.
	qA := newMockQueue("flow-a")
	qB := newMockQueue("flow-b")
	band := buildBand(state, map[string]*fwkfcmocks.MockFlowQueueAccessor{
		"flow-a": qA,
		"flow-b": qB,
	})

	_, err = p.Pick(context.Background(), band)
	require.NoError(t, err)

	// After the pick, flow-b's initial counter should have been set to 1000 (the activeMin),
	// not 0, so it could still be selected (it ties with flow-a at 1000).
	// The winner is the one with the lower counter; both started at 1000 so one will now be higher.
	assert.GreaterOrEqual(t, state.counters["flow-b"], float64(1000),
		"rejoining flow should be lifted to the active minimum counter")
}

func TestVTC_Pick_StaleCounterPruned(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background()).(*vtcBandState)

	// Pre-seed a stale counter for flow-gone (not in the band).
	state.counters["flow-gone"] = 9999

	qA := newMockQueue("flow-a")
	band := buildBand(state, map[string]*fwkfcmocks.MockFlowQueueAccessor{"flow-a": qA})

	_, err = p.Pick(context.Background(), band)
	require.NoError(t, err)

	_, exists := state.counters["flow-gone"]
	assert.False(t, exists, "stale counter for absent flow should be pruned")
}

func TestVTC_Pick_CounterNormalization(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background()).(*vtcBandState)

	// Set counters above the normalization threshold.
	state.counters["flow-a"] = normalizationThreshold + 1e6
	state.counters["flow-b"] = normalizationThreshold + 2e6

	qA := newMockQueue("flow-a")
	qB := newMockQueue("flow-b")
	band := buildBand(state, map[string]*fwkfcmocks.MockFlowQueueAccessor{
		"flow-a": qA,
		"flow-b": qB,
	})

	_, err = p.Pick(context.Background(), band)
	require.NoError(t, err)

	for id, c := range state.counters {
		assert.LessOrEqual(t, c, normalizationThreshold,
			"counter for %s should be normalized below threshold, got %f", id, c)
	}
}

func TestVTC_Pick_WrongStateType(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)

	band := &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: "not-a-vtcBandState",
		FlowKeysFunc: func() []flowcontrol.FlowKey {
			return []flowcontrol.FlowKey{{ID: "a"}}
		},
	}

	_, err = p.Pick(context.Background(), band)
	require.Error(t, err)
}

func TestVTC_EstimateInputTokens_Cascade(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	t.Run("uses pre-tokenized prompt token IDs when available", func(t *testing.T) {
		t.Parallel()
		ir := &fwksched.InferenceRequest{
			Body: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{TokenIDs: make([]uint32, 200)},
				},
			},
		}
		q := newMockQueueWithInferenceRequest("a", ir)
		band := buildBand(state, map[string]*fwkfcmocks.MockFlowQueueAccessor{"a": q})

		_, pickErr := p.Pick(context.Background(), band)
		require.NoError(t, pickErr)
		// cost = inputWeight×200 + outputWeight×128 = 1.0×200 + 3.0×128 = 584
		assert.InDelta(t, 584.0, state.(*vtcBandState).counters["a"], 0.01)
	})

	t.Run("falls back to TokenizedPrompt length", func(t *testing.T) {
		t.Parallel()
		state2 := p.NewState(context.Background())
		ir := &fwksched.InferenceRequest{
			Body: &fwkrh.InferenceRequestBody{
				TokenizedPrompt: &fwksched.TokenizedPrompt{
					PerPromptTokens: [][]uint32{make([]uint32, 50)},
				},
			},
		}
		q := newMockQueueWithInferenceRequest("b", ir)
		band := buildBand(state2, map[string]*fwkfcmocks.MockFlowQueueAccessor{"b": q})

		_, pickErr := p.Pick(context.Background(), band)
		require.NoError(t, pickErr)
		// cost = 1.0×50 + 3.0×128 = 434
		assert.InDelta(t, 434.0, state2.(*vtcBandState).counters["b"], 0.01)
	})

	t.Run("falls back to ByteSize/4", func(t *testing.T) {
		t.Parallel()
		state3 := p.NewState(context.Background())
		// 400 bytes → 100 input tokens
		q := newMockQueue("c")
		band := buildBand(state3, map[string]*fwkfcmocks.MockFlowQueueAccessor{"c": q})

		_, pickErr := p.Pick(context.Background(), band)
		require.NoError(t, pickErr)
		// cost = 1.0×100 + 3.0×128 = 484
		assert.InDelta(t, 484.0, state3.(*vtcBandState).counters["c"], 0.01)
	})
}

func TestVTC_DefaultWeights(t *testing.T) {
	t.Parallel()

	p, err := newVTC("vtc", nil)
	require.NoError(t, err)
	assert.Equal(t, 1.0, p.inputTokenWeight)
	assert.Equal(t, 3.0, p.outputTokenWeight)
}

func TestVTC_CustomWeights(t *testing.T) {
	t.Parallel()

	params := json.RawMessage(`{"inputTokenWeight": 2.0, "outputTokenWeight": 5.0}`)
	p, err := newVTC("vtc", params)
	require.NoError(t, err)
	assert.Equal(t, 2.0, p.inputTokenWeight)
	assert.Equal(t, 5.0, p.outputTokenWeight)
}

func TestNormalizeCounters_BelowThreshold(t *testing.T) {
	t.Parallel()
	s := &vtcBandState{counters: map[string]float64{"a": 100, "b": 200}}
	normalizeCounters(s)
	assert.Equal(t, 100.0, s.counters["a"])
	assert.Equal(t, 200.0, s.counters["b"])
}

func TestNormalizeCounters_AboveThreshold(t *testing.T) {
	t.Parallel()
	s := &vtcBandState{counters: map[string]float64{
		"a": normalizationThreshold + 500,
		"b": normalizationThreshold + 1500,
	}}
	normalizeCounters(s)
	assert.Equal(t, 0.0, s.counters["a"], "min counter should be subtracted to zero")
	assert.Equal(t, 1000.0, s.counters["b"], "relative difference must be preserved")
}
