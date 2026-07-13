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

// Package stateapi implements the server side of the internal State API a
// "stateful" EPP replica exposes so "stateless" EPP replicas can read/write
// shared scheduling state over gRPC.
//
// This is a feasibility-spike implementation of RFC #1593 Phase 2. The
// stateful EPP process never runs a real Director/Scheduler for actual
// requests, so its storage is a freestanding structure — not a wrapper around
// an in-process producer instance, unlike pkg/epp/statestore's Local provider
// (which wraps a specific producer, appropriate for a stateless replica's own
// classic-equivalent local shadow).
//
// Storage is deliberately behind the Store interface so a future Redis/PVC
// backend is a drop-in replacement; only memoryStore (in-memory, unbounded for
// the run's duration) is implemented for this spike.
package stateapi

// InflightSnapshot is a point-in-time view of inflight load for one endpoint.
type InflightSnapshot struct {
	Requests int64
	Tokens   int64
}

// InflightStore holds fleet-wide inflight request/token counters, aggregated
// from Reserve/Release calls made by every stateless EPP replica.
//
// Conformance: implementations MUST be goroutine-safe and MUST treat
// Reserve/Release as idempotent per requestID+endpointID pair, since a
// FailOpen client may retry after an ambiguous (timed out but possibly
// applied) call.
type InflightStore interface {
	// SnapshotBatch returns the current counters for the given endpoints.
	// Endpoints with no recorded state are omitted from the result.
	SnapshotBatch(endpointIDs []string) map[string]InflightSnapshot
	// Reserve records that requestID has reserved estimatedTokens on endpointID.
	// A second Reserve call with the same requestID+endpointID is a no-op.
	Reserve(requestID, endpointID string, estimatedTokens int64)
	// Release undoes a prior Reserve. A Release with no matching prior Reserve
	// (already released, or never reserved) is a no-op.
	Release(requestID, endpointID string, estimatedTokens int64)
	// DeleteEndpoint removes all inflight state for an endpoint.
	DeleteEndpoint(endpointID string)
}

// PrefixStore holds the fleet-wide approximate prefix cache index. Unlike
// pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix's
// indexer, this has no per-endpoint LRU eviction — acceptable for a bounded
// load-test run, not representative of unbounded steady-state memory.
//
// Conformance: implementations MUST be goroutine-safe.
type PrefixStore interface {
	// Match returns the endpoint IDs known to have the given prefix hash cached.
	Match(hash uint64) []string
	// Commit records that hashes are now cached on endpointID.
	Commit(requestID, endpointID string, hashes []uint64)
	// RemoveEndpoint removes all prefix state for an endpoint.
	RemoveEndpoint(endpointID string)
}

// FlowKey identifies a flow control instance (mirrors
// pkg/epp/statestore.FlowControlKey, kept as a plain value type here to avoid
// a package dependency in either direction).
type FlowKey struct {
	ID       string
	Priority int
}

// ConcurrencyOutcome describes the result of a concurrency-lease admission attempt.
type ConcurrencyOutcome int

const (
	// ConcurrencyOutcomeAdmitted indicates the lease was granted.
	ConcurrencyOutcomeAdmitted ConcurrencyOutcome = iota
	// ConcurrencyOutcomeRejected indicates globalMaxConcurrency was reached for the flow key.
	ConcurrencyOutcomeRejected
)

// ConcurrencyStore implements a fleet-wide concurrency lease: a primitive
// that does not exist anywhere else in this codebase today. The existing
// FlowRegistry (pkg/epp/flowcontrol/registry) counts queue occupancy — items
// are dequeued and finalized the instant they are dispatched, not held for the
// duration of execution — so it cannot serve as a remote concurrency cap
// without this separate primitive.
//
// Conformance: implementations MUST be goroutine-safe and MUST reclaim leases
// that are never released (a stateless replica crash, a dropped Release call)
// via a TTL sweep — an unreclaimed lease permanently shrinks effective
// capacity, which would corrupt any throughput measurement taken against this
// store over time.
type ConcurrencyStore interface {
	// Admit attempts to acquire a concurrency lease for requestID under key.
	// A second Admit call with the same requestID is idempotent: it returns the
	// original outcome without acquiring a second lease.
	Admit(requestID string, key FlowKey) ConcurrencyOutcome
	// ReleaseConcurrency releases a previously admitted lease. A call with no
	// matching prior admitted lease (already released, expired, or never
	// admitted) is a no-op. Named distinctly from InflightStore.Release so
	// Store (which embeds both) has no ambiguous selector.
	ReleaseConcurrency(requestID string, key FlowKey)
}

// Store is the unified storage abstraction backing the State API server.
type Store interface {
	InflightStore
	PrefixStore
	ConcurrencyStore
}
