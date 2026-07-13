/*
Copyright 2026 The Kubernetes Authors.

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

package statestore

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// fakeLocalFlowControlState is a hand-rolled FlowControlState standing in for
// the existing local queue admission (localFlowControlState), so these tests
// exercise localFallbackConcurrencyState's composition logic in isolation.
type fakeLocalFlowControlState struct {
	admitOutcome FlowControlOutcome
	admitErr     error
	admitCalls   int
}

func (f *fakeLocalFlowControlState) Admit(_ context.Context, _ flowcontrol.FlowControlRequest) (FlowControlOutcome, error) {
	f.admitCalls++
	return f.admitOutcome, f.admitErr
}
func (f *fakeLocalFlowControlState) Release(_ context.Context, _ string, _ FlowControlKey) error {
	return nil
}

var _ FlowControlState = (*fakeLocalFlowControlState)(nil)

// fakeConcurrencyState is a hand-rolled ConcurrencyState standing in for the
// gRPC-backed remoteConcurrencyState.
type fakeConcurrencyState struct {
	admitOutcome FlowControlOutcome
	admitErr     error
	releaseErr   error
	admitCalls   int
	releaseCalls int
}

func (f *fakeConcurrencyState) Admit(_ context.Context, _ FlowControlKey, _ string) (FlowControlOutcome, error) {
	f.admitCalls++
	return f.admitOutcome, f.admitErr
}
func (f *fakeConcurrencyState) Release(_ context.Context, _ FlowControlKey, _ string) error {
	f.releaseCalls++
	return f.releaseErr
}

var _ ConcurrencyState = (*fakeConcurrencyState)(nil)

func TestLocalFallbackConcurrencyState_LocalRejectionShortCircuits(t *testing.T) {
	t.Parallel()
	local := &fakeLocalFlowControlState{admitOutcome: FlowControlOutcomeRejected}
	remote := &fakeConcurrencyState{admitOutcome: FlowControlOutcomeAdmitted}
	s := NewLocalFallbackFlowControlState(local, remote, time.Second, 0)

	outcome, err := s.Admit(context.Background(), &stubFlowControlRequest{id: "req-1"})

	require.NoError(t, err)
	require.Equal(t, FlowControlOutcomeRejected, outcome)
	require.Zero(t, remote.admitCalls, "remote concurrency check must not run when the local queue already rejected")

	// A request that was never admitted must not be released.
	require.NoError(t, s.Release(context.Background(), "req-1", FlowControlKey{}))
	require.Zero(t, remote.releaseCalls)
}

func TestLocalFallbackConcurrencyState_RemoteAdmitThenRelease(t *testing.T) {
	t.Parallel()
	local := &fakeLocalFlowControlState{admitOutcome: FlowControlOutcomeAdmitted}
	remote := &fakeConcurrencyState{admitOutcome: FlowControlOutcomeAdmitted}
	s := NewLocalFallbackFlowControlState(local, remote, time.Second, 0)

	outcome, err := s.Admit(context.Background(), &stubFlowControlRequest{id: "req-1"})
	require.NoError(t, err)
	require.Equal(t, FlowControlOutcomeAdmitted, outcome)

	require.NoError(t, s.Release(context.Background(), "req-1", FlowControlKey{ID: "tenant-a"}))
	require.Equal(t, 1, remote.releaseCalls)
}

func TestLocalFallbackConcurrencyState_RemoteRejectsAfterLocalAdmits(t *testing.T) {
	t.Parallel()
	local := &fakeLocalFlowControlState{admitOutcome: FlowControlOutcomeAdmitted}
	remote := &fakeConcurrencyState{admitOutcome: FlowControlOutcomeRejected}
	s := NewLocalFallbackFlowControlState(local, remote, time.Second, 0)

	outcome, err := s.Admit(context.Background(), &stubFlowControlRequest{id: "req-1"})

	require.NoError(t, err)
	require.Equal(t, FlowControlOutcomeRejected, outcome)

	// No lease was actually held (remote rejected), so Release is a no-op.
	require.NoError(t, s.Release(context.Background(), "req-1", FlowControlKey{}))
	require.Zero(t, remote.releaseCalls)
}

func TestLocalFallbackConcurrencyState_RemoteFailureDegradesWithinLocalCap(t *testing.T) {
	t.Parallel()
	local := &fakeLocalFlowControlState{admitOutcome: FlowControlOutcomeAdmitted}
	remote := &fakeConcurrencyState{admitErr: errors.New("unavailable")}
	s := NewLocalFallbackFlowControlState(local, remote, time.Second, 1)

	outcome, err := s.Admit(context.Background(), &stubFlowControlRequest{id: "req-1"})
	require.NoError(t, err)
	require.Equal(t, FlowControlOutcomeDegraded, outcome)

	// Local cap is 1 and already in use: a second concurrent request must be rejected.
	outcome, err = s.Admit(context.Background(), &stubFlowControlRequest{id: "req-2"})
	require.NoError(t, err)
	require.Equal(t, FlowControlOutcomeRejected, outcome)

	// Releasing the first frees the local-fallback slot for a third.
	require.NoError(t, s.Release(context.Background(), "req-1", FlowControlKey{}))
	require.Zero(t, remote.releaseCalls, "a degraded (local-fallback) lease must not call remote Release")

	outcome, err = s.Admit(context.Background(), &stubFlowControlRequest{id: "req-3"})
	require.NoError(t, err)
	require.Equal(t, FlowControlOutcomeDegraded, outcome)
}

func TestLocalFallbackConcurrencyState_RemoteFailureWithNoLocalCapAlwaysDegrades(t *testing.T) {
	t.Parallel()
	local := &fakeLocalFlowControlState{admitOutcome: FlowControlOutcomeAdmitted}
	remote := &fakeConcurrencyState{admitErr: errors.New("unavailable")}
	s := NewLocalFallbackFlowControlState(local, remote, time.Second, 0)

	for i := 0; i < 5; i++ {
		outcome, err := s.Admit(context.Background(), &stubFlowControlRequest{id: "req"})
		require.NoError(t, err)
		require.Equal(t, FlowControlOutcomeDegraded, outcome)
	}
}
