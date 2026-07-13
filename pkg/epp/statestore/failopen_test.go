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
)

// fakeInflightState is a hand-rolled InflightState for exercising
// failOpenInflightState's composition without a real gRPC server.
type fakeInflightState struct {
	snapshotBatch      map[string]InflightSnapshot // nil means "simulate failure"
	reserveErr         error
	releaseErr         error
	reserveCalls       int
	releaseCalls       int
	deleteEndpointArgs []string
}

func (f *fakeInflightState) GetInflightSnapshot(ctx context.Context, endpointID string) InflightSnapshot {
	return f.GetInflightSnapshotBatch(ctx, []string{endpointID})[endpointID]
}
func (f *fakeInflightState) GetInflightSnapshotBatch(_ context.Context, _ []string) map[string]InflightSnapshot {
	return f.snapshotBatch
}
func (f *fakeInflightState) ReserveInflight(_ context.Context, _, _ string, _ int64) error {
	f.reserveCalls++
	return f.reserveErr
}
func (f *fakeInflightState) ReleaseInflight(_ context.Context, _, _ string, _ int64) error {
	f.releaseCalls++
	return f.releaseErr
}
func (f *fakeInflightState) DeleteEndpoint(_ context.Context, endpointID string) {
	f.deleteEndpointArgs = append(f.deleteEndpointArgs, endpointID)
}

var _ InflightState = (*fakeInflightState)(nil)

func TestFailOpenInflightState_PrefersRemoteOnSuccess(t *testing.T) {
	t.Parallel()
	remote := &fakeInflightState{snapshotBatch: map[string]InflightSnapshot{"ep-1": {Requests: 5, Tokens: 500}}}
	local := &fakeInflightState{snapshotBatch: map[string]InflightSnapshot{"ep-1": {Requests: 1, Tokens: 1}}}
	s := NewFailOpenInflightState(remote, local, time.Second)

	snap := s.GetInflightSnapshot(context.Background(), "ep-1")

	require.Equal(t, InflightSnapshot{Requests: 5, Tokens: 500}, snap)
}

func TestFailOpenInflightState_FallsBackToLocalOnRemoteFailure(t *testing.T) {
	t.Parallel()
	remote := &fakeInflightState{snapshotBatch: nil} // simulates remote error/timeout
	local := &fakeInflightState{snapshotBatch: map[string]InflightSnapshot{"ep-1": {Requests: 1, Tokens: 1}}}
	s := NewFailOpenInflightState(remote, local, time.Second)

	snap := s.GetInflightSnapshot(context.Background(), "ep-1")

	require.Equal(t, InflightSnapshot{Requests: 1, Tokens: 1}, snap)
}

func TestFailOpenInflightState_ReserveCallsBothLocalAndRemote(t *testing.T) {
	t.Parallel()
	remote := &fakeInflightState{}
	local := &fakeInflightState{}
	s := NewFailOpenInflightState(remote, local, time.Second)

	err := s.ReserveInflight(context.Background(), "req-1", "ep-1", 100)

	require.NoError(t, err)
	require.Equal(t, 1, local.reserveCalls)
	require.Equal(t, 1, remote.reserveCalls)
}

func TestFailOpenInflightState_ReserveRemoteFailureIsBestEffort(t *testing.T) {
	t.Parallel()
	remote := &fakeInflightState{reserveErr: errors.New("unavailable")}
	local := &fakeInflightState{}
	s := NewFailOpenInflightState(remote, local, time.Second)

	err := s.ReserveInflight(context.Background(), "req-1", "ep-1", 100)

	require.NoError(t, err, "a failed remote push must degrade silently, not fail the request")
	require.Equal(t, 1, local.reserveCalls)
}

// fakePrefixState mirrors fakeInflightState for PrefixState.
type fakePrefixState struct {
	matchBatch  map[uint64][]string // nil means "simulate failure"
	commitErr   error
	commitCalls int
}

func (f *fakePrefixState) GetPrefixMatch(ctx context.Context, hash uint64) []string {
	return f.GetPrefixMatchBatch(ctx, []uint64{hash})[hash]
}
func (f *fakePrefixState) GetPrefixMatchBatch(_ context.Context, _ []uint64) map[uint64][]string {
	return f.matchBatch
}
func (f *fakePrefixState) CommitPrefix(_ context.Context, _, _ string, _ []uint64) error {
	f.commitCalls++
	return f.commitErr
}
func (f *fakePrefixState) RemoveEndpoint(_ context.Context, _ string) {}

var _ PrefixState = (*fakePrefixState)(nil)

func TestFailOpenPrefixState_FallsBackToLocalOnRemoteFailure(t *testing.T) {
	t.Parallel()
	remote := &fakePrefixState{matchBatch: nil}
	local := &fakePrefixState{matchBatch: map[uint64][]string{42: {"ep-1"}}}
	s := NewFailOpenPrefixState(remote, local, time.Second)

	require.Equal(t, []string{"ep-1"}, s.GetPrefixMatch(context.Background(), 42))
}
