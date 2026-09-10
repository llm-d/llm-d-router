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

package mocks

import (
	"context"
	"errors"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// MockFlowControlRequest provides a mock implementation of the FlowControlRequest interface.
type MockFlowControlRequest struct {
	FlowKeyV             flowcontrol.FlowKey
	ByteSizeV            uint64
	InferenceRequestV    *scheduling.InferenceRequest
	ReceivedTimestampV   time.Time
	InitialEffectiveTTLV time.Duration
	IDV                  string
	MetadataV            map[string]any
	InferencePoolNameV   string
	ModelNameV           string
	TargetModelNameV     string
}

// MockRequestOption is a functional option for configuring a MockFlowControlRequest.
type MockRequestOption func(*MockFlowControlRequest)

// NewMockFlowControlRequest creates a new MockFlowControlRequest instance with optional configuration.
func NewMockFlowControlRequest(
	byteSize uint64,
	id string,
	key flowcontrol.FlowKey,
	opts ...MockRequestOption,
) *MockFlowControlRequest {
	m := &MockFlowControlRequest{
		ByteSizeV: byteSize,
		IDV:       id,
		FlowKeyV:  key,
		MetadataV: make(map[string]any),
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

func (m *MockFlowControlRequest) FlowKey() flowcontrol.FlowKey { return m.FlowKeyV }
func (m *MockFlowControlRequest) ByteSize() uint64             { return m.ByteSizeV }
func (m *MockFlowControlRequest) InferenceRequest() *scheduling.InferenceRequest {
	return m.InferenceRequestV
}
func (m *MockFlowControlRequest) ReceivedTimestamp() time.Time       { return m.ReceivedTimestampV }
func (m *MockFlowControlRequest) InitialEffectiveTTL() time.Duration { return m.InitialEffectiveTTLV }
func (m *MockFlowControlRequest) ID() string                         { return m.IDV }
func (m *MockFlowControlRequest) GetMetadata() map[string]any        { return m.MetadataV }
func (m *MockFlowControlRequest) InferencePoolName() string          { return m.InferencePoolNameV }
func (m *MockFlowControlRequest) ModelName() string                  { return m.ModelNameV }
func (m *MockFlowControlRequest) TargetModelName() string            { return m.TargetModelNameV }

var _ flowcontrol.FlowControlRequest = &MockFlowControlRequest{}

// MockQueueItemHandle provides a mock implementation of the QueueItemHandle interface.
type MockQueueItemHandle struct {
	RawHandle      any
	IsInvalidatedV bool
}

func (m *MockQueueItemHandle) Handle() any         { return m.RawHandle }
func (m *MockQueueItemHandle) Invalidate()         { m.IsInvalidatedV = true }
func (m *MockQueueItemHandle) IsInvalidated() bool { return m.IsInvalidatedV }

var _ flowcontrol.QueueItemHandle = &MockQueueItemHandle{}

// MockQueueItemAccessor provides a mock implementation of the QueueItemAccessor interface.
type MockQueueItemAccessor struct {
	EnqueueTimeV     time.Time
	EffectiveTTLV    time.Duration
	OriginalRequestV flowcontrol.FlowControlRequest
	HandleV          flowcontrol.QueueItemHandle
}

func (m *MockQueueItemAccessor) EnqueueTime() time.Time      { return m.EnqueueTimeV }
func (m *MockQueueItemAccessor) EffectiveTTL() time.Duration { return m.EffectiveTTLV }

func (m *MockQueueItemAccessor) OriginalRequest() flowcontrol.FlowControlRequest {
	if m.OriginalRequestV == nil {
		return &MockFlowControlRequest{}
	}
	return m.OriginalRequestV
}

func (m *MockQueueItemAccessor) Handle() flowcontrol.QueueItemHandle          { return m.HandleV }
func (m *MockQueueItemAccessor) SetHandle(handle flowcontrol.QueueItemHandle) { m.HandleV = handle }

var _ flowcontrol.QueueItemAccessor = &MockQueueItemAccessor{}

// NewMockQueueItemAccessor is a constructor for MockQueueItemAccessor that initializes the mock with a default
// MockFlowControlRequest and MockQueueItemHandle to prevent nil pointer dereferences in tests.
// It accepts MockRequestOptions to configure the underlying request.
func NewMockQueueItemAccessor(
	byteSize uint64,
	reqID string,
	key flowcontrol.FlowKey,
	opts ...MockRequestOption,
) *MockQueueItemAccessor {
	return &MockQueueItemAccessor{
		EnqueueTimeV: time.Now(),
		OriginalRequestV: NewMockFlowControlRequest(
			byteSize,
			reqID,
			key,
			opts...,
		),
		HandleV: &MockQueueItemHandle{},
	}
}

// MockFlowQueueAccessor is a simple stub mock for the FlowQueueAccessor interface.
// It is used for tests that require static, predictable return values from a queue accessor.
// For complex, stateful queue behavior, use the mock in ../../contracts/mocks.MockManagedQueue.
type MockFlowQueueAccessor struct {
	LenV            int
	ByteSizeV       uint64
	PeekV           flowcontrol.QueueItemAccessor
	FlowKeyV        flowcontrol.FlowKey
	OrderingPolicyV flowcontrol.OrderingPolicy
}

func (m *MockFlowQueueAccessor) Len() int                                   { return m.LenV }
func (m *MockFlowQueueAccessor) ByteSize() uint64                           { return m.ByteSizeV }
func (m *MockFlowQueueAccessor) OrderingPolicy() flowcontrol.OrderingPolicy { return m.OrderingPolicyV }
func (m *MockFlowQueueAccessor) FlowKey() flowcontrol.FlowKey               { return m.FlowKeyV }

func (m *MockFlowQueueAccessor) Peek() flowcontrol.QueueItemAccessor {
	return m.PeekV
}

var _ flowcontrol.FlowQueueAccessor = &MockFlowQueueAccessor{}

// MockPriorityBandAccessor is a behavioral mock for the PriorityBandAccessor interface.
// Simple accessors are configured with public value fields (e.g., PriorityV).
// Complex methods with logic are configured with function fields (e.g., IterateQueuesFunc).
//
// Convention: Fields suffixed with 'V' (e.g., PriorityV) are static Value return fields.
// This avoids collision with the interface method of the same name.
type MockPriorityBandAccessor struct {
	PriorityV         int
	PolicyStateV      any
	FlowKeysFunc      func() []flowcontrol.FlowKey
	QueueFunc         func(flowID string) flowcontrol.FlowQueueAccessor
	IterateQueuesFunc func(callback func(flow flowcontrol.FlowQueueAccessor) (keepIterating bool))
}

func (m *MockPriorityBandAccessor) Priority() int    { return m.PriorityV }
func (m *MockPriorityBandAccessor) PolicyState() any { return m.PolicyStateV }

func (m *MockPriorityBandAccessor) FlowKeys() []flowcontrol.FlowKey {
	if m.FlowKeysFunc != nil {
		return m.FlowKeysFunc()
	}
	return nil
}

func (m *MockPriorityBandAccessor) Queue(id string) flowcontrol.FlowQueueAccessor {
	if m.QueueFunc != nil {
		return m.QueueFunc(id)
	}
	return nil
}

func (m *MockPriorityBandAccessor) IterateQueues(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
	if m.IterateQueuesFunc != nil {
		m.IterateQueuesFunc(callback)
	}
}

var _ flowcontrol.PriorityBandAccessor = &MockPriorityBandAccessor{}

// MockOrderingPolicy is a behavioral mock for the OrderingPolicy interface.
// Simple accessors are configured with public value fields (e.g., TypedNameV).
// Complex methods with logic are configured with function fields (e.g., LessFunc).
type MockOrderingPolicy struct {
	TypedNameV plugin.TypedName
	LessFunc   func(a, b flowcontrol.QueueItemAccessor) bool
}

func (m *MockOrderingPolicy) TypedName() plugin.TypedName { return m.TypedNameV }

func (m *MockOrderingPolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
	if m.LessFunc != nil {
		return m.LessFunc(a, b)
	}
	return false
}

var _ flowcontrol.OrderingPolicy = &MockOrderingPolicy{}

// MockScoringOrderingPolicy is a behavioral mock for the ScoringOrderingPolicy interface.
//
// ScoreFunc is supplied by the caller and reports an item's score; the mock holds no scores of its own.
// A test drives it from caller-owned state (e.g. a map it mutates to model a queued item's key drifting);
// a benchmark supplies a lock-free O(1) func so the measured cost is the queue's rebuild, not the policy.
//
// Generation is an atomic counter advanced by Bump, standing in for a real policy's throttled latch:
// changing an item's score without bumping models drift the queue has not been told about yet. It is
// atomic rather than plain so concurrent Bump and Generation (a peeking goroutine) do not race.
//
// ScoreCalls counts Score invocations so a test can assert the queue recomputes exactly once per item per
// generation change, and not at all without one.
type MockScoringOrderingPolicy struct {
	TypedNameV plugin.TypedName

	// ScoreFunc reports the item's score. Required; Score panics if it is nil.
	ScoreFunc func(item flowcontrol.QueueItemAccessor) float64

	generation atomic.Uint64
	scoreCalls atomic.Int64
}

// NewMockScoringOrderingPolicy creates a scoring policy mock with the given score function.
func NewMockScoringOrderingPolicy(name string, scoreFunc func(item flowcontrol.QueueItemAccessor) float64) *MockScoringOrderingPolicy {
	return &MockScoringOrderingPolicy{
		TypedNameV: plugin.TypedName{Type: "mock-scoring-ordering-policy", Name: name},
		ScoreFunc:  scoreFunc,
	}
}

func (m *MockScoringOrderingPolicy) TypedName() plugin.TypedName { return m.TypedNameV }

// Bump advances the generation, signalling to every queue ordered by this instance that its cached scores
// are stale.
func (m *MockScoringOrderingPolicy) Bump() { m.generation.Add(1) }

// ScoreCalls returns the number of times Score has been called since construction.
func (m *MockScoringOrderingPolicy) ScoreCalls() int64 { return m.scoreCalls.Load() }

func (m *MockScoringOrderingPolicy) Score(item flowcontrol.QueueItemAccessor) float64 {
	m.scoreCalls.Add(1)
	return m.ScoreFunc(item)
}

func (m *MockScoringOrderingPolicy) Generation() uint64 { return m.generation.Load() }

// Less delegates to CompareByScore, as every conforming scoring policy must, so that tests exercise the
// same relationship between the live and cached comparison paths that a real policy has.
func (m *MockScoringOrderingPolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
	return flowcontrol.CompareByScore(m, a, b)
}

var _ flowcontrol.ScoringOrderingPolicy = &MockScoringOrderingPolicy{}

// MockFairnessPolicy is a behavioral mock for the FairnessPolicy interface.
// Simple accessors are configured with public value fields (e.g., NameV).
// Complex methods with logic are configured with function fields (e.g., PickFunc).
type MockFairnessPolicy struct {
	TypedNameV   plugin.TypedName
	NewStateFunc func(ctx context.Context) any
	PickFunc     func(ctx context.Context, flowGroup flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error)
}

func (m *MockFairnessPolicy) TypedName() plugin.TypedName { return m.TypedNameV }

func (m *MockFairnessPolicy) NewState(ctx context.Context) any {
	if m.NewStateFunc != nil {
		return m.NewStateFunc(ctx)
	}
	return nil
}

func (m *MockFairnessPolicy) Pick(ctx context.Context, flowGroup flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error) {
	if m.PickFunc != nil {
		return m.PickFunc(ctx, flowGroup)
	}
	return nil, errors.New("sentinel nothing to pick")
}

var _ flowcontrol.FairnessPolicy = &MockFairnessPolicy{}

// MockSaturationDetector is a behavioral mock for the SaturationDetector interface.
type MockSaturationDetector struct {
	TypedNameV   plugin.TypedName
	IsSaturatedV bool
	SaturationV  float64
}

func (m *MockSaturationDetector) TypedName() plugin.TypedName { return m.TypedNameV }
func (m *MockSaturationDetector) IsSaturated() bool           { return m.IsSaturatedV }
func (m *MockSaturationDetector) Saturation(ctx context.Context, endpoints []datalayer.Endpoint) float64 {
	return m.SaturationV
}

func (m *MockSaturationDetector) LastCheckTime() time.Time { return time.Time{} }

var _ flowcontrol.SaturationDetector = &MockSaturationDetector{}
