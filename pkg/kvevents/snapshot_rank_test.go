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

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/require"
)

type snapshotRankAdapter struct{}

func (snapshotRankAdapter) ParseMessage(*RawMessage) (string, string, EventBatch, error) {
	return "", "", EventBatch{}, nil
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
