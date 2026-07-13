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

// failOpenInflightState decorates a remote InflightState with the RFC's
// FailOpen degradation strategy: reads prefer the remote (global) view within
// a short timeout, falling back to the local shadow on any error or timeout;
// writes apply to the local shadow (an existing Phase-1 no-op — the
// producer's own hooks already mutate it directly) and, best-effort, to the
// remote store.
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

	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if err := s.remote.ReserveInflight(rctx, requestID, endpointID, estimatedTokens); err != nil {
		recordRemoteFallback("inflight_reserve")
		return nil // best-effort: a failed remote push degrades silently, per FailOpen.
	}
	return nil
}

func (s *failOpenInflightState) ReleaseInflight(ctx context.Context, requestID, endpointID string, estimatedTokens int64) error {
	_ = s.local.ReleaseInflight(ctx, requestID, endpointID, estimatedTokens)

	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if err := s.remote.ReleaseInflight(rctx, requestID, endpointID, estimatedTokens); err != nil {
		recordRemoteFallback("inflight_release")
	}
	return nil
}

func (s *failOpenInflightState) DeleteEndpoint(ctx context.Context, endpointID string) {
	s.local.DeleteEndpoint(ctx, endpointID)
	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	s.remote.DeleteEndpoint(rctx, endpointID)
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

	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if err := s.remote.CommitPrefix(rctx, requestID, endpointID, hashes); err != nil {
		recordRemoteFallback("prefix_commit")
	}
	return nil
}

func (s *failOpenPrefixState) RemoveEndpoint(ctx context.Context, endpointID string) {
	s.local.RemoveEndpoint(ctx, endpointID)
	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	s.remote.RemoveEndpoint(rctx, endpointID)
}
