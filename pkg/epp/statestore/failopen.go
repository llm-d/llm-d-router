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
	"time"
)

// asyncRemoteWrite fires a best-effort remote write in the background instead
// of blocking the caller on it.
//
// This is safe specifically because these writes are best-effort by design:
// the Local write they sit alongside is already a documented no-op (the
// producer's own PreRequest/ResponseBody hooks are the actual mutation of
// record), and no caller consumes this call's outcome — unlike the read path
// (GetInflightSnapshotBatch/GetPrefixMatchBatch), whose result IS read by the
// current request and therefore MUST stay synchronous.
//
// The goroutine uses context.Background(), not the caller's ctx: a fire-and-
// forget call must not be cancelled just because the request that triggered
// it finished and its context was torn down before the remote write lands.
func asyncRemoteWrite(timeout time.Duration, fallbackLabel string, fn func(ctx context.Context) error) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if err := fn(ctx); err != nil {
			recordRemoteFallback(fallbackLabel)
		}
	}()
}

// asyncRemoteWriteNoError is asyncRemoteWrite's counterpart for the two calls
// (DeleteEndpoint/RemoveEndpoint) whose signatures don't surface an error to
// begin with (pre-existing design: best-effort cleanup, outcome was already
// discarded even when synchronous).
func asyncRemoteWriteNoError(timeout time.Duration, fn func(ctx context.Context)) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		fn(ctx)
	}()
}

// failOpenInflightState decorates a remote InflightState with the FailOpen
// degradation strategy: reads prefer the remote (global) view within a short
// timeout, falling back to the local shadow on any error or timeout; writes
// apply to the local shadow (a no-op — the producer's own hooks already
// mutate it directly, see store.go) synchronously, and push to the remote
// store asynchronously (see asyncRemoteWrite) — the remote push's outcome
// affects only future requests' reads, never this one, so there is no
// correctness reason to hold the current request's critical path on it.
type failOpenInflightState struct {
	remote  InflightState
	local   InflightState
	timeout time.Duration
}

// NewFailOpenInflightState returns an InflightState that prefers remote for
// reads (bounded by timeout, falling back to local) and pushes writes to both.
func NewFailOpenInflightState(remote, local InflightState, timeout time.Duration) InflightState {
	return &failOpenInflightState{remote: remote, local: local, timeout: timeout}
}

func (s *failOpenInflightState) GetInflightSnapshot(ctx context.Context, endpointID string) InflightSnapshot {
	return s.GetInflightSnapshotBatch(ctx, []string{endpointID})[endpointID]
}

func (s *failOpenInflightState) GetInflightSnapshotBatch(ctx context.Context, endpointIDs []string) map[string]InflightSnapshot {
	recordRemoteCall("inflight_read")
	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if snap := s.remote.GetInflightSnapshotBatch(rctx, endpointIDs); snap != nil {
		return snap
	}
	recordRemoteFallback("inflight_read")
	return s.local.GetInflightSnapshotBatch(ctx, endpointIDs)
}

func (s *failOpenInflightState) ReserveInflight(ctx context.Context, requestID, endpointID string, estimatedTokens int64) error {
	// Local first: the producer's own hooks already mutate the local shadow
	// directly, so this call is a documented no-op (see store.go). Kept for
	// interface symmetry, not because it does anything here.
	_ = s.local.ReserveInflight(ctx, requestID, endpointID, estimatedTokens)

	recordRemoteCall("inflight_reserve")
	asyncRemoteWrite(s.timeout, "inflight_reserve", func(rctx context.Context) error {
		return s.remote.ReserveInflight(rctx, requestID, endpointID, estimatedTokens)
	})
	return nil
}

func (s *failOpenInflightState) ReleaseInflight(ctx context.Context, requestID, endpointID string, estimatedTokens int64) error {
	_ = s.local.ReleaseInflight(ctx, requestID, endpointID, estimatedTokens)

	recordRemoteCall("inflight_release")
	asyncRemoteWrite(s.timeout, "inflight_release", func(rctx context.Context) error {
		return s.remote.ReleaseInflight(rctx, requestID, endpointID, estimatedTokens)
	})
	return nil
}

func (s *failOpenInflightState) DeleteEndpoint(ctx context.Context, endpointID string) {
	s.local.DeleteEndpoint(ctx, endpointID)
	recordRemoteCall("inflight_delete_endpoint")
	asyncRemoteWriteNoError(s.timeout, func(rctx context.Context) {
		s.remote.DeleteEndpoint(rctx, endpointID)
	})
}

// failOpenPrefixState is PrefixState's counterpart to failOpenInflightState.
type failOpenPrefixState struct {
	remote  PrefixState
	local   PrefixState
	timeout time.Duration
}

// NewFailOpenPrefixState returns a PrefixState that prefers remote for reads
// (bounded by timeout, falling back to local) and pushes writes to both.
func NewFailOpenPrefixState(remote, local PrefixState, timeout time.Duration) PrefixState {
	return &failOpenPrefixState{remote: remote, local: local, timeout: timeout}
}

func (s *failOpenPrefixState) GetPrefixMatch(ctx context.Context, hash uint64) []string {
	return s.GetPrefixMatchBatch(ctx, []uint64{hash})[hash]
}

func (s *failOpenPrefixState) GetPrefixMatchBatch(ctx context.Context, hashes []uint64) map[uint64][]string {
	recordRemoteCall("prefix_read")
	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if matches := s.remote.GetPrefixMatchBatch(rctx, hashes); matches != nil {
		return matches
	}
	recordRemoteFallback("prefix_read")
	return s.local.GetPrefixMatchBatch(ctx, hashes)
}

func (s *failOpenPrefixState) CommitPrefix(ctx context.Context, requestID, endpointID string, hashes []uint64) error {
	_ = s.local.CommitPrefix(ctx, requestID, endpointID, hashes)

	recordRemoteCall("prefix_commit")
	asyncRemoteWrite(s.timeout, "prefix_commit", func(rctx context.Context) error {
		return s.remote.CommitPrefix(rctx, requestID, endpointID, hashes)
	})
	return nil
}

func (s *failOpenPrefixState) RemoveEndpoint(ctx context.Context, endpointID string) {
	s.local.RemoveEndpoint(ctx, endpointID)
	recordRemoteCall("prefix_remove_endpoint")
	asyncRemoteWriteNoError(s.timeout, func(rctx context.Context) {
		s.remote.RemoveEndpoint(rctx, endpointID)
	})
}
