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

package kvevents

import (
	"context"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
)

type snapshotRankAdapter struct{}

func (snapshotRankAdapter) ParseMessage(*RawMessage) (string, string, EventBatch, error) {
	return "", "", EventBatch{}, nil
}

func TestSnapshotManagerMatchesOnlyActiveGenerations(t *testing.T) {
	ctx := context.Background()
	tokens, err := kvblock.NewChunkedTokenDatabase(nil)
	require.NoError(t, err)
	cfg, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	matcher, err := kvcache.NewKVCacheIndexer(ctx, cfg, tokens)
	require.NoError(t, err)
	index, err := kvblock.NewInMemoryIndex(cfg.KVBlockIndexConfig.InMemoryConfig)
	require.NoError(t, err)
	key := kvblock.BlockHash(1)
	require.NoError(t, index.Add(ctx, nil, []kvblock.BlockHash{key}, []kvblock.PodEntry{
		{PodIdentifier: "ready-generation", DeviceTier: "gpu"},
		{PodIdentifier: "hidden-generation", DeviceTier: "gpu"},
	}))
	now := time.Now().UnixNano()
	ready := &snapshotSubscriber{sourceEndpoint: "pod-ready", activeID: "ready-generation"}
	ready.lastReceive.Store(now)
	recovering := &snapshotSubscriber{sourceEndpoint: "pod-recovering"}
	recovering.lastReceive.Store(now)
	stale := &snapshotSubscriber{sourceEndpoint: "pod-stale", activeID: "stale-generation"}
	stale.lastReceive.Store(time.Now().Add(-2 * heartbeatTimeout).UnixNano())
	manager := &SnapshotManager{
		matcher: matcher.WithIndex(index),
		index:   index,
		subscribers: map[string]*snapshotSubscriber{
			"ready":      ready,
			"recovering": recovering,
			"stale":      stale,
		},
	}

	matches, err := manager.MatchBlockKeys(ctx, []kvblock.BlockHash{key}, sets.Set[string]{})
	require.NoError(t, err)
	require.Equal(t, map[string]kvcache.PodMatch{
		"pod-ready": {
			WeightedScore:   1,
			MatchedBlocks:   1,
			ConfirmedBlocks: 1,
			BlocksByTier:    map[string]int{"gpu": 1},
		},
	}, matches)
	require.Equal(t, SnapshotStatus{Registered: 3, Ready: 1, Recovering: 1, Stale: 1}, manager.Status())
	ids, _ := manager.GetReadySubscribers()
	require.Equal(t, []string{"ready"}, ids)
}

func (snapshotRankAdapter) ShardingKey(*RawMessage) string { return "" }

func TestSnapshotSubscriberOffsetsPortByLiveRank(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cfg := DefaultConfig()
	cfg.PodDiscoveryConfig.SocketPort = 5557
	cfg.SnapshotPort = 6000
	tokens, err := kvblock.NewChunkedTokenDatabase(nil)
	require.NoError(t, err)
	manager, err := NewSnapshotManager(cfg, nil, tokens, snapshotRankAdapter{}, nil)
	require.NoError(t, err)

	require.NoError(t, manager.EnsureSubscriber(ctx, "pod-rank-3", "pod:8003", "tcp://127.0.0.1:5560", "", "kv@", true))
	manager.mu.RLock()
	require.Equal(t, "tcp://127.0.0.1:6003", manager.subscribers["pod-rank-3"].snapshotEndpoint)
	manager.mu.RUnlock()

	cancel()
	manager.Shutdown(context.Background())
}
