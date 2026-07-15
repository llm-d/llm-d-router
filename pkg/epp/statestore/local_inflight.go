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

import "context"

// InflightBackend is the minimal adapter interface that the Local inflight
// provider requires from the existing in-process inflight load implementation
// (inflightload.InFlightLoadProducer). It matches that type's existing exported
// methods exactly, so InFlightLoadProducer satisfies it with no wrapper code.
//
// Conformance: Implementations MUST be goroutine-safe.
type InflightBackend interface {
	// GetRequests returns the inflight request count for an endpoint.
	GetRequests(endpointID string) int64
	// GetTokens returns the inflight token count for an endpoint.
	GetTokens(endpointID string) int64
	// DeleteEndpoint removes all inflight state for an endpoint.
	DeleteEndpoint(endpointID string)
}

// localInflightState wraps an InflightBackend to satisfy InflightState. Its
// write methods (ReserveInflight/ReleaseInflight) are no-ops: the backend's
// own PreRequest/ResponseBody hooks already mutate its counters directly as a
// byproduct of routing decisions, so calling through here as well would
// double count. See the InflightState doc comments for the full rationale.
type localInflightState struct {
	backend InflightBackend
}

// NewLocalInflightState returns an InflightState backed by the given
// InflightBackend (typically an *inflightload.InFlightLoadProducer).
func NewLocalInflightState(backend InflightBackend) InflightState {
	return &localInflightState{backend: backend}
}

func (s *localInflightState) GetInflightSnapshot(_ context.Context, endpointID string) InflightSnapshot {
	return InflightSnapshot{
		Requests: s.backend.GetRequests(endpointID),
		Tokens:   s.backend.GetTokens(endpointID),
	}
}

func (s *localInflightState) GetInflightSnapshotBatch(_ context.Context, endpointIDs []string) map[string]InflightSnapshot {
	result := make(map[string]InflightSnapshot, len(endpointIDs))
	for _, id := range endpointIDs {
		result[id] = InflightSnapshot{Requests: s.backend.GetRequests(id), Tokens: s.backend.GetTokens(id)}
	}
	return result
}

func (s *localInflightState) ReserveInflight(_ context.Context, _, _ string, _ int64) error {
	return nil
}

func (s *localInflightState) ReleaseInflight(_ context.Context, _, _ string, _ int64) error {
	return nil
}

func (s *localInflightState) DeleteEndpoint(_ context.Context, _ string) {
}
