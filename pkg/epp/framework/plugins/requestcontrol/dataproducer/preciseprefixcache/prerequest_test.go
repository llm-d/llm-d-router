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

package preciseprefixcache

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

// addCall captures one fakeKVBlockIndex.Add invocation.
type addCall struct {
	keys    []kvblock.BlockHash
	entries []kvblock.PodEntry
}

// evictCall captures one fakeKVBlockIndex.Evict invocation.
type evictCall struct {
	keyType kvblock.KeyType
	keys    []kvblock.BlockHash
	entries []kvblock.PodEntry
}

func newProducerForPreRequest(ctx context.Context, speculativeEnabled bool, idx *fakeKVBlockIndex) *Producer {
	cache := ttlcache.New[string, *speculativeEntries](
		ttlcache.WithTTL[string, *speculativeEntries](time.Minute),
	)
	return &Producer{
		typedName:          plugin.TypedName{Type: PluginType, Name: "test"},
		kvCacheIndexer:     &fakeKVCacheIndexer{index: idx},
		speculativeCache:   cache,
		speculativeTTL:     time.Minute,
		speculativeEnabled: speculativeEnabled,
		pluginState:        plugin.NewPluginState(ctx),
	}
}

func primaryOnly(name string, endpoint scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: name,
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			name: {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

// speculativeEnabled=true with populated block keys: index.Add called once
// with the primary pod identifier, and a speculative cache entry is created.
func TestPreRequest_SeedsSpeculativeForPrimary(t *testing.T) {
	ctx := utils.NewTestContext(t)

	var calls []addCall
	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			calls = append(calls, addCall{keys: keys, entries: entries})
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	blockKeys := []kvblock.BlockHash{0xAA, 0xBB}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-1"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: [][]kvblock.BlockHash{blockKeys}})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	require.Len(t, calls, 1)
	assert.Equal(t, blockKeys, calls[0].keys)
	require.Len(t, calls[0].entries, 1)
	assert.Equal(t, "10.0.0.1:8080", calls[0].entries[0].PodIdentifier)
	assert.True(t, calls[0].entries[0].Speculative)

	cached := p.speculativeCache.Get(req.RequestID)
	require.NotNil(t, cached)
	assert.Equal(t, [][]kvblock.BlockHash{blockKeys}, cached.Value().perPromptKeys)
	require.Len(t, cached.Value().podEntries, 1)
	assert.Equal(t, "10.0.0.1:8080", cached.Value().podEntries[0].PodIdentifier)
}

// speculativeEnabled=true with empty blockKeys: PreRequest must not call
// index.Add and must not create a cache entry.
func TestPreRequest_EmptyBlockKeys_NoAdd(t *testing.T) {
	ctx := utils.NewTestContext(t)

	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, _ []kvblock.BlockHash, _ []kvblock.PodEntry) error {
			t.Fatalf("index.Add must not be called when blockKeys are empty")
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	req := &scheduling.InferenceRequest{RequestID: "req-pre-empty"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: nil})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	assert.Nil(t, p.speculativeCache.Get(req.RequestID))
}

// P/D prefill profile: index.Add called twice (primary + prefill), and the
// cache entry tracks both pod identifiers.
func TestPreRequest_PrefillProfile_SeedsBoth(t *testing.T) {
	ctx := utils.NewTestContext(t)

	var calls []addCall
	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			calls = append(calls, addCall{keys: keys, entries: entries})
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	blockKeys := []kvblock.BlockHash{0xCC}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-pd"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: [][]kvblock.BlockHash{blockKeys}})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":                   {TargetEndpoints: []scheduling.Endpoint{testEndpoints[0]}},
			experimentalPrefillProfile: {TargetEndpoints: []scheduling.Endpoint{testEndpoints[1]}},
		},
	}
	_ = p.PreRequest(ctx, req, result)

	require.Len(t, calls, 2)
	assert.Equal(t, "10.0.0.1:8080", calls[0].entries[0].PodIdentifier)
	assert.Equal(t, "10.0.0.2:8080", calls[1].entries[0].PodIdentifier)

	cached := p.speculativeCache.Get(req.RequestID)
	require.NotNil(t, cached)
	require.Len(t, cached.Value().podEntries, 2)
	assert.Equal(t, "10.0.0.1:8080", cached.Value().podEntries[0].PodIdentifier)
	assert.Equal(t, "10.0.0.2:8080", cached.Value().podEntries[1].PodIdentifier)
}

// speculativeEnabled=false: early return — no index writes, no cache entry,
// and PluginState is left untouched.
func TestPreRequest_SpeculativeDisabled_NoOp(t *testing.T) {
	ctx := utils.NewTestContext(t)

	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, _ []kvblock.BlockHash, _ []kvblock.PodEntry) error {
			t.Fatalf("index.Add must not be called when speculative indexing is disabled")
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, false, idx)

	req := &scheduling.InferenceRequest{RequestID: "req-pre-off"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey,
		&blockKeysState{perPromptKeys: [][]kvblock.BlockHash{{0xDD}}})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	assert.Nil(t, p.speculativeCache.Get(req.RequestID))
}

// TTL expiry of a speculative entry evicts every key of every prompt in the
// request, with RequestKey semantics and the seeded pod entries.
func TestSpeculativeTTLExpiry_EvictsAllPromptKeys(t *testing.T) {
	ctx, cancel := context.WithCancel(utils.NewTestContext(t))
	t.Cleanup(cancel)

	var mu sync.Mutex
	var evictions []evictCall
	idx := &fakeKVBlockIndex{
		evictFn: func(_ context.Context, keyType kvblock.KeyType, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			mu.Lock()
			defer mu.Unlock()
			evictions = append(evictions, evictCall{keyType: keyType, keys: keys, entries: entries})
			return nil
		},
	}

	cache, ttl, err := buildSpeculativeCache(ctx, PluginConfig{
		SpeculativeIndexing: true,
		SpeculativeTTL:      "20ms",
	}, idx)
	require.NoError(t, err)
	require.Equal(t, 20*time.Millisecond, ttl)

	perPrompt := [][]kvblock.BlockHash{{0xA1, 0xA2}, {0xB1}}
	pods := []kvblock.PodEntry{{PodIdentifier: "10.0.0.1:8080", Speculative: true}}
	cache.Set("req-ttl", &speculativeEntries{perPromptKeys: perPrompt, podEntries: pods}, ttl)

	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(evictions) == 1
	}, 2*time.Second, time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, evictions, 1)
	assert.Equal(t, []kvblock.BlockHash{0xA1, 0xA2, 0xB1}, evictions[0].keys)
	assert.Equal(t, kvblock.RequestKey, evictions[0].keyType)
	assert.Equal(t, pods, evictions[0].entries)
}
