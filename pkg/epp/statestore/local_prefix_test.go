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

type stubPrefixBackend struct {
	matches         map[uint64][]string
	committed       bool
	removedEndpoint string
}

func (b *stubPrefixBackend) GetMatch(hash uint64) []string { return b.matches[hash] }
func (b *stubPrefixBackend) Commit(_ string, _ []uint64)   { b.committed = true }
func (b *stubPrefixBackend) RemoveEndpoint(endpointID string) {
	b.removedEndpoint = endpointID
}

func TestLocalPrefixState_GetPrefixMatch(t *testing.T) {
	t.Parallel()
	backend := &stubPrefixBackend{matches: map[uint64][]string{42: {"ep-1", "ep-2"}}}
	state := NewLocalPrefixState(backend)

	got := state.GetPrefixMatch(context.Background(), 42)

	require.ElementsMatch(t, []string{"ep-1", "ep-2"}, got)
}

func TestLocalPrefixState_WritesAreNoOps(t *testing.T) {
	t.Parallel()
	backend := &stubPrefixBackend{matches: map[uint64][]string{}}
	state := NewLocalPrefixState(backend)
	ctx := context.Background()

	assert.NoError(t, state.CommitPrefix(ctx, "req-1", "ep-1", []uint64{1, 2, 3}))
	state.RemoveEndpoint(ctx, "ep-1")

	// The backend must be untouched: approximateprefix's own PreRequest hook
	// already commits to the indexer directly, in Phase 1.
	assert.False(t, backend.committed)
	assert.Empty(t, backend.removedEndpoint)
}
