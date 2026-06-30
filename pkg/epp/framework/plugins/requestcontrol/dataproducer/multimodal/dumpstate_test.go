/*
Copyright 2025 The Kubernetes Authors.

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

package multimodal

import (
	"encoding/json"
	"fmt"
	"testing"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func podCache(t *testing.T, size int, hashes ...string) *lru.Cache[string, struct{}] {
	t.Helper()
	c, err := lru.New[string, struct{}](size)
	require.NoError(t, err)
	for _, h := range hashes {
		c.Add(h, struct{}{})
	}
	return c
}

func decodeMMDump(t *testing.T, p *Producer) encoderCacheState {
	t.Helper()
	raw, err := p.DumpState()
	require.NoError(t, err)
	var state encoderCacheState
	require.NoError(t, json.Unmarshal(raw, &state))
	return state
}

func TestDumpState_EmptyReportsConfigOnly(t *testing.T) {
	p := &Producer{cacheSize: 256, caches: map[string]*lru.Cache[string, struct{}]{}}

	state := decodeMMDump(t, p)

	assert.Equal(t, 256, state.CacheSizePerPod)
	assert.Equal(t, 0, state.TotalPods)
	assert.Equal(t, 0, state.TotalEntries)
	assert.Empty(t, state.Pods)
	assert.False(t, state.Truncated)
	assert.Equal(t, maxDebugDumpPods, state.MaxPods)
}

func TestDumpState_SummarizesPerPodCounts(t *testing.T) {
	p := &Producer{
		cacheSize: 256,
		caches: map[string]*lru.Cache[string, struct{}]{
			"ns/pod-a": podCache(t, 256, "h1", "h2", "h3"),
			"ns/pod-b": podCache(t, 256, "h1", "h2"),
		},
	}

	state := decodeMMDump(t, p)

	assert.Equal(t, 2, state.TotalPods)
	assert.Equal(t, 5, state.TotalEntries)
	assert.False(t, state.Truncated)

	// Sorted busiest-first; counts only, no item hashes.
	require.Len(t, state.Pods, 2)
	assert.Equal(t, podCacheState{Pod: "ns/pod-a", Entries: 3}, state.Pods[0])
	assert.Equal(t, podCacheState{Pod: "ns/pod-b", Entries: 2}, state.Pods[1])
}

func TestDumpState_CapsAndFlagsTruncation(t *testing.T) {
	const total = maxDebugDumpPods + 50
	caches := make(map[string]*lru.Cache[string, struct{}], total)
	for i := 0; i < total; i++ {
		caches[fmt.Sprintf("ns/pod-%04d", i)] = podCache(t, 8, "h1")
	}
	p := &Producer{cacheSize: 8, caches: caches}

	state := decodeMMDump(t, p)

	assert.Equal(t, total, state.TotalPods, "total reflects the true count, not the capped sample")
	assert.Len(t, state.Pods, maxDebugDumpPods)
	assert.True(t, state.Truncated)
	assert.Equal(t, "ns/pod-0000", state.Pods[0].Pod)
}

func TestProducer_ImplementsStateDumper(t *testing.T) {
	require.Implements(t, (*interface {
		DumpState() (json.RawMessage, error)
	})(nil), &Producer{})
}
