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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubInflightBackend struct {
	requests        map[string]int64
	tokens          map[string]int64
	deletedEndpoint string
}

func (b *stubInflightBackend) GetRequests(endpointID string) int64 { return b.requests[endpointID] }
func (b *stubInflightBackend) GetTokens(endpointID string) int64   { return b.tokens[endpointID] }
func (b *stubInflightBackend) DeleteEndpoint(endpointID string)    { b.deletedEndpoint = endpointID }

func TestLocalInflightState_GetInflightSnapshot(t *testing.T) {
	t.Parallel()
	backend := &stubInflightBackend{
		requests: map[string]int64{"ep-1": 3},
		tokens:   map[string]int64{"ep-1": 500},
	}
	state := NewLocalInflightState(backend)

	snap := state.GetInflightSnapshot(context.Background(), "ep-1")

	require.Equal(t, int64(3), snap.Requests)
	require.Equal(t, int64(500), snap.Tokens)
}

func TestLocalInflightState_GetInflightSnapshot_UnknownEndpoint(t *testing.T) {
	t.Parallel()
	backend := &stubInflightBackend{requests: map[string]int64{}, tokens: map[string]int64{}}
	state := NewLocalInflightState(backend)

	snap := state.GetInflightSnapshot(context.Background(), "unknown")

	require.Equal(t, InflightSnapshot{}, snap)
}

func TestLocalInflightState_WritesAreNoOps(t *testing.T) {
	t.Parallel()
	backend := &stubInflightBackend{
		requests: map[string]int64{"ep-1": 3},
		tokens:   map[string]int64{"ep-1": 500},
	}
	state := NewLocalInflightState(backend)
	ctx := context.Background()

	assert.NoError(t, state.ReserveInflight(ctx, "req-1", "ep-1", 100))
	assert.NoError(t, state.ReleaseInflight(ctx, "req-1", "ep-1", 100))
	state.DeleteEndpoint(ctx, "ep-1")

	// The backend must be untouched: mutation stays in the producer's own
	// PreRequest/ResponseBody hooks, not in the statestore layer, in Phase 1.
	assert.Equal(t, int64(3), backend.requests["ep-1"])
	assert.Equal(t, int64(500), backend.tokens["ep-1"])
	assert.Empty(t, backend.deletedEndpoint)
}
