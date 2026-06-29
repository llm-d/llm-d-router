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

package preciseprefixcache

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newSpeculativeCache() *ttlcache.Cache[string, *speculativeEntries] {
	return ttlcache.New[string, *speculativeEntries](
		ttlcache.WithTTL[string, *speculativeEntries](time.Hour),
	)
}

func decodeDump(t *testing.T, p *Producer) speculativeCacheState {
	t.Helper()
	raw, err := p.DumpState()
	require.NoError(t, err)
	var state speculativeCacheState
	require.NoError(t, json.Unmarshal(raw, &state))
	return state
}

func TestDumpState_NilCacheReportsConfigOnly(t *testing.T) {
	p := &Producer{speculativeEnabled: false, blockSizeTokens: 16}

	state := decodeDump(t, p)

	assert.False(t, state.SpeculativeEnabled)
	assert.Equal(t, 16, state.BlockSizeTokens)
	assert.Equal(t, 0, state.TotalEntries)
	assert.Empty(t, state.Entries)
	assert.False(t, state.Truncated)
	assert.Equal(t, maxDebugDumpSpeculativeEntries, state.MaxEntries)
}

func TestDumpState_SummarizesEntriesAsCounts(t *testing.T) {
	cache := newSpeculativeCache()
	cache.Set("req-a", &speculativeEntries{
		perPromptKeys: [][]kvblock.BlockHash{{1, 2, 3}, {4}},
		podEntries:    []kvblock.PodEntry{{}},
	}, ttlcache.DefaultTTL)
	cache.Set("req-b", &speculativeEntries{
		perPromptKeys: [][]kvblock.BlockHash{{5, 6}},
		podEntries:    []kvblock.PodEntry{{}, {}},
	}, ttlcache.DefaultTTL)

	p := &Producer{
		speculativeEnabled: true,
		speculativeTTL:     30 * time.Second,
		blockSizeTokens:    32,
		speculativeCache:   cache,
	}

	state := decodeDump(t, p)

	assert.True(t, state.SpeculativeEnabled)
	assert.Equal(t, 30.0, state.SpeculativeTTLSeconds)
	assert.Equal(t, 32, state.BlockSizeTokens)
	assert.Equal(t, 2, state.TotalEntries)
	assert.False(t, state.Truncated)

	// Entries are sorted by request ID, and carry counts only (no block-key hashes).
	require.Len(t, state.Entries, 2)
	assert.Equal(t, speculativeEntryState{RequestID: "req-a", Prompts: 2, BlockKeys: 4, PodEntries: 1}, state.Entries[0])
	assert.Equal(t, speculativeEntryState{RequestID: "req-b", Prompts: 1, BlockKeys: 2, PodEntries: 2}, state.Entries[1])
}

func TestDumpState_CapsAndFlagsTruncation(t *testing.T) {
	cache := newSpeculativeCache()
	const total = maxDebugDumpSpeculativeEntries + 50
	for i := 0; i < total; i++ {
		cache.Set(fmt.Sprintf("req-%04d", i), &speculativeEntries{
			perPromptKeys: [][]kvblock.BlockHash{{kvblock.BlockHash(i)}},
		}, ttlcache.DefaultTTL)
	}

	p := &Producer{speculativeEnabled: true, speculativeCache: cache}
	state := decodeDump(t, p)

	assert.Equal(t, total, state.TotalEntries, "total reflects the true count, not the capped sample")
	assert.Len(t, state.Entries, maxDebugDumpSpeculativeEntries)
	assert.True(t, state.Truncated)
	// Sorted, so the retained sample is the lexicographically-first cap.
	assert.Equal(t, "req-0000", state.Entries[0].RequestID)
}

func TestProducer_ImplementsStateDumper(t *testing.T) {
	require.Implements(t, (*interface {
		DumpState() (json.RawMessage, error)
	})(nil), &Producer{})
}
