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

// Package statestore is the abstraction over the three categories of shared
// scheduling state that must remain consistent across EPP replicas: inflight
// request/token counters, prefix cache index, and flow control admission. See
// RFC #1593 ("Shared Scheduling State for Horizontally Scalable EPP
// Deployments") for the design rationale.
//
// Design principles:
//   - EPP stable operation first: all failures degrade rather than interrupt.
//   - Zero-change deployment: default behavior is unchanged.
//   - Minimal dependencies: no external storage; shared state lives in the same binary.
//   - Local shadow always runs: immediate fallback when remote fails.
//   - Scheduling modules remain unaware: scorer, filter, and picker require no changes.
//
// Three implementations exist: Local (in-process, wraps the existing
// per-replica trackers so it's behaviorally equivalent to classic mode),
// Remote (remote_client.go, forwards to a stateful EPP over the internal gRPC
// State API in pkg/epp/statestore/stateapi), and FailOpen/LocalFallback
// (failopen.go, concurrency_lease.go), which decorate Remote to prefer its
// reads/admission decisions within a timeout and fall back to Local on
// failure or timeout.
//
// Every write method on the three sub-interfaces below is a no-op in the
// Local implementation: the corresponding producer's own PreRequest/
// ResponseBody hooks already perform that mutation as a byproduct of routing
// decisions. Only the read methods (GetInflightSnapshot, GetPrefixMatch) and
// flow control's Admit are real pass-throughs in Local mode. In the Remote/
// FailOpen configuration, write methods additionally push a best-effort
// update to the stateful EPP alongside the unchanged local mutation; see
// failopen.go for the composition.
package statestore

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// StateStore is the unified abstraction over the three categories of shared
// scheduling state. It composes the per-category interfaces so that a single
// implementation (Local or Remote) can serve all scheduling state access.
//
// Conformance: Implementations MUST be goroutine-safe.
type StateStore interface {
	InflightState
	PrefixState
	FlowControlState
}

// InflightState provides read/write access to inflight request and token
// counters per endpoint. These counters back load-aware routing.
//
// In the Local provider this wraps the atomic counters in
// inflightload/producer.go. In the Remote provider it forwards to the
// stateful EPP via gRPC.
//
// Degradation strategy: FailOpen. The local shadow is a natural byproduct of
// routing decisions because the ResponseBodyProcessor hook must run locally.
type InflightState interface {
	// GetInflightSnapshot returns the inflight request and token counts for the
	// given endpoint. It is pulled on demand per candidate endpoint during
	// request data preparation; the stateful EPP does not actively push.
	//
	// Callers on the request hot path MUST use GetInflightSnapshotBatch instead
	// of looping this per endpoint when the underlying implementation may be
	// remote (a per-endpoint RPC loop turns one request into N round trips).
	// This single-endpoint form exists for interface completeness and tests.
	GetInflightSnapshot(ctx context.Context, endpointID string) InflightSnapshot

	// GetInflightSnapshotBatch returns snapshots for multiple endpoints in a
	// single call. The Local implementation loops in-process (cheap, no I/O);
	// the Remote implementation issues one batched gRPC call. This is the
	// method producers must call from Produce (which runs once per request and
	// has a context) rather than from any per-endpoint, per-scorer lazy
	// attribute accessor.
	GetInflightSnapshotBatch(ctx context.Context, endpointIDs []string) map[string]InflightSnapshot

	// ReserveInflight is a Local no-op: inflightload's own PreRequest hook
	// already increments requestTracker/tokenTracker directly (with
	// PluginState-backed, endpoint-flap-safe release bookkeeping) as a
	// byproduct of the routing decision, so duplicating that here would double
	// count. Remote/FailOpen additionally push a best-effort update to the
	// stateful EPP after the local mutation.
	ReserveInflight(ctx context.Context, requestID, endpointID string, estimatedTokens int64) error

	// ReleaseInflight is a Local no-op; see ReserveInflight.
	ReleaseInflight(ctx context.Context, requestID, endpointID string, estimatedTokens int64) error

	// DeleteEndpoint is a Local no-op; inflightload's own endpoint-delete
	// handling already calls DeleteEndpoint on its trackers directly.
	DeleteEndpoint(ctx context.Context, endpointID string)
}

// PrefixState provides read/write access to the prefix cache index used by
// prefix-aware routing.
//
// In the Local provider this wraps the LRU index in approximateprefix. In the
// Remote provider it forwards prefix score/commit operations to stateful EPP.
//
// Degradation strategy: FailOpen. The local shadow auto-updates after each
// routing decision.
type PrefixState interface {
	// GetPrefixMatch returns the set of endpoint IDs that have the given prefix
	// hash cached. Used during request data preparation to populate per-endpoint
	// prefix cache match info.
	//
	// A single request's prompt can carry many block hashes; callers on the
	// hot path MUST use GetPrefixMatchBatch instead of looping this per hash
	// when the underlying implementation may be remote, for the same reason as
	// InflightState.GetInflightSnapshot.
	GetPrefixMatch(ctx context.Context, hash uint64) []string

	// GetPrefixMatchBatch returns matches for multiple prefix hashes in a
	// single call. See GetInflightSnapshotBatch for the same rationale.
	GetPrefixMatchBatch(ctx context.Context, hashes []uint64) map[uint64][]string

	// CommitPrefix is a Local no-op: approximateprefix's own PreRequest hook
	// already commits matched hashes to its indexer directly as a byproduct of
	// the routing decision, so duplicating that here would be redundant.
	CommitPrefix(ctx context.Context, requestID, endpointID string, hashes []uint64) error

	// RemoveEndpoint is a Local no-op; approximateprefix's own endpoint-delete
	// handling already calls RemovePod on its indexer directly.
	RemoveEndpoint(ctx context.Context, endpointID string)
}

// FlowControlState provides admit/release access to flow control quotas.
//
// In the Local provider this wraps flowcontrol/controller.FlowController. In
// the Remote provider it forwards admit/release operations to stateful EPP
// via gRPC.
//
// Degradation strategy: LocalFallback. A dual-quota design is used:
// globalMaxConcurrency (remote authority, precise) and localMaxConcurrency
// (local shadow). Note: unlike Inflight/Prefix, Admit is not a no-op — it IS
// the admission decision, not downstream bookkeeping.
type FlowControlState interface {
	// Admit submits a request to flow control and blocks until it reaches a
	// terminal outcome.
	Admit(ctx context.Context, req flowcontrol.FlowControlRequest) (FlowControlOutcome, error)

	// Release is a Local no-op. FlowController's only exported method is
	// EnqueueAndWait; finalization (dispatch, TTL/cancellation eviction,
	// cleanup) happens entirely inside that blocking call, with no separate
	// release call to invoke afterward. Note this also means the existing
	// FlowRegistry counts queue occupancy, not concurrent-execution leases;
	// true admit-reserve-execute-release concurrency limiting needs the
	// separate ConcurrencyState abstraction (concurrency_lease.go).
	Release(ctx context.Context, requestID string, flowKey FlowControlKey) error
}

// InflightSnapshot is a point-in-time view of inflight load for one endpoint.
type InflightSnapshot struct {
	// Requests is the number of inflight requests currently routed to the endpoint.
	Requests int64
	// Tokens is the estimated total inflight tokens for the endpoint.
	Tokens int64
}

// FlowControlKey identifies a flow control instance, mirroring
// flowcontrol.FlowKey but kept as a plain value type in this package so the
// Release signature does not require callers to import the flowcontrol package.
type FlowControlKey struct {
	// ID is the logical grouping identifier for a flow (e.g. tenant or model).
	ID string
	// Priority is the numerical priority level for this flow instance.
	Priority int
}

// FlowControlOutcome describes the result of a flow control admission attempt.
type FlowControlOutcome int

const (
	// FlowControlOutcomeAdmitted indicates the request was admitted.
	FlowControlOutcomeAdmitted FlowControlOutcome = iota
	// FlowControlOutcomeRejected indicates the request was rejected due to
	// capacity limits.
	FlowControlOutcomeRejected
	// FlowControlOutcomeDegraded indicates the remote authority was
	// unavailable and the request was admitted using the local shadow quota.
	FlowControlOutcomeDegraded
)
