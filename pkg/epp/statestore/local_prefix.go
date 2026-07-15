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

// PrefixBackend is the minimal adapter interface that the Local prefix
// provider requires from a prefix cache producer. approximateprefix's
// dataProducer implements this via the exported methods added in
// state_export.go (GetPrefixMatch/CommitPrefix/RemovePrefixEndpoint).
//
// Conformance: Implementations MUST be goroutine-safe.
type PrefixBackend interface {
	// GetMatch returns the endpoint IDs that have the given prefix hash cached.
	GetMatch(hash uint64) []string
	// Commit records that the given hashes are cached on the endpoint.
	Commit(endpointID string, hashes []uint64)
	// RemoveEndpoint removes all prefix state for an endpoint.
	RemoveEndpoint(endpointID string)
}

// localPrefixState wraps a PrefixBackend to satisfy PrefixState. Its write
// method (CommitPrefix) is a no-op for the same reason as
// localInflightState's Reserve/Release: the backend's own PreRequest hook
// already commits to its indexer directly as a byproduct of routing
// decisions.
type localPrefixState struct {
	backend PrefixBackend
}

// NewLocalPrefixState returns a PrefixState backed by the given PrefixBackend.
func NewLocalPrefixState(backend PrefixBackend) PrefixState {
	return &localPrefixState{backend: backend}
}

func (s *localPrefixState) GetPrefixMatch(_ context.Context, hash uint64) []string {
	return s.backend.GetMatch(hash)
}

func (s *localPrefixState) GetPrefixMatchBatch(_ context.Context, hashes []uint64) map[uint64][]string {
	result := make(map[uint64][]string, len(hashes))
	for _, hash := range hashes {
		result[hash] = s.backend.GetMatch(hash)
	}
	return result
}

func (s *localPrefixState) CommitPrefix(_ context.Context, _, _ string, _ []uint64) error {
	return nil
}

func (s *localPrefixState) RemoveEndpoint(_ context.Context, _ string) {
}
