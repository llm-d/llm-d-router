/*
Copyright 2026 The llm-d Authors.

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

package kvblock

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestInMemoryIndexEvictKeepsEntryAddedConcurrently drives, for each public
// eviction entrypoint, the interleaving where Add has fetched a key's PodCache
// and the eviction empties that cache before Add writes to it. The test holds
// m.mu as Add does, so the eviction must not remove the key until the write lands.
func TestInMemoryIndexEvictKeepsEntryAddedConcurrently(t *testing.T) {
	const (
		engineKey  BlockHash = 101
		requestKey BlockHash = 1
	)
	podA := PodEntry{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}
	podB := PodEntry{PodIdentifier: "10.0.0.2:8000", DeviceTier: "gpu"}

	tests := []struct {
		name  string
		evict func(context.Context, *InMemoryIndex) error
	}{
		{
			name: "Evict by request key",
			evict: func(ctx context.Context, idx *InMemoryIndex) error {
				return idx.Evict(ctx, RequestKey, []BlockHash{requestKey}, []PodEntry{podA})
			},
		},
		{
			name: "Evict by engine key",
			evict: func(ctx context.Context, idx *InMemoryIndex) error {
				return idx.Evict(ctx, EngineKey, []BlockHash{engineKey}, []PodEntry{podA})
			},
		},
		{
			name: "Clear",
			evict: func(ctx context.Context, idx *InMemoryIndex) error {
				return idx.Clear(ctx, podA.PodIdentifier)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()
			idx, err := NewInMemoryIndex(nil)
			require.NoError(t, err)
			require.NoError(t, idx.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{podA}))

			recordsB, err := idx.internRecords([]PodEntry{podB})
			require.NoError(t, err)

			// Add's critical section: the key's PodCache has been fetched.
			idx.mu.Lock()
			podCache, found := idx.data.Get(requestKey)
			require.True(t, found)

			evicted := make(chan error, 1)
			go func() { evicted <- tt.evict(ctx, idx) }()

			require.Eventually(t, func() bool { return podCache.size() == 0 }, time.Second, time.Millisecond)
			// Give an eviction that does not wait for m.mu the chance to remove the key.
			var evictErr error
			done := false
			select {
			case evictErr = <-evicted:
				done = true
			case <-time.After(50 * time.Millisecond):
			}

			podCache.addAll(recordsB)
			idx.mu.Unlock()
			if !done {
				evictErr = <-evicted
			}
			require.NoError(t, evictErr)

			got, err := idx.Lookup(ctx, []BlockHash{requestKey}, nil)
			require.NoError(t, err)
			assert.Equal(t, []PodEntry{podB}, got[requestKey])
		})
	}
}
