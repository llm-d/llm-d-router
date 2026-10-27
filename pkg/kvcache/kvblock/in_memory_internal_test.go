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
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestInMemoryIndexEvictKeepsEntryAddedConcurrently drives the interleaving
// where Add has fetched a key's PodCache and an eviction empties that cache
// before Add writes to it. The test holds m.mu as Add does, so the eviction
// must not remove the key until the write lands.
func TestInMemoryIndexEvictKeepsEntryAddedConcurrently(t *testing.T) {
	ctx := t.Context()
	idx, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	const key BlockHash = 1
	podA := PodEntry{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}
	podB := PodEntry{PodIdentifier: "10.0.0.2:8000", DeviceTier: "gpu"}
	require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{podA}))

	recordsB, err := idx.internRecords([]PodEntry{podB})
	require.NoError(t, err)

	// Add's critical section: the key's PodCache has been fetched.
	idx.mu.Lock()
	podCache, found := idx.data.Get(key)
	require.True(t, found)

	evicted := make(chan struct{})
	go func() {
		defer close(evicted)
		idx.evictPodsFromRequestKey(key, EmptyBlockHash, []PodEntry{podA}, logr.Discard())
	}()

	require.Eventually(t, func() bool { return podCache.size() == 0 }, time.Second, time.Millisecond)
	// Give an eviction that does not wait for m.mu the chance to remove the key.
	select {
	case <-evicted:
	case <-time.After(50 * time.Millisecond):
	}

	podCache.addAll(recordsB)
	idx.mu.Unlock()
	<-evicted

	got, err := idx.Lookup(ctx, []BlockHash{key}, nil)
	require.NoError(t, err)
	assert.Equal(t, []PodEntry{podB}, got[key])
}
