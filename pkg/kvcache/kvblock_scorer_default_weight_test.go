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

package kvcache_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

func TestLongestPrefixScorerDefaultWeight(t *testing.T) {
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002})
	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu"}, {PodIdentifier: podB, DeviceTier: "unknown-tier"}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu"}, {PodIdentifier: podB, DeviceTier: "unknown-tier"}},
	}

	tests := []struct {
		name          string
		defaultWeight float64
		wantB         float64
	}{
		{name: "configured", defaultWeight: 0.25, wantB: 0.5},
		{name: "default keeps unknown tiers at full weight", defaultWeight: kvcache.DefaultKVBlockScorerConfig().DefaultWeight, wantB: 2.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := kvcache.DefaultKVBlockScorerConfig()
			cfg.DefaultWeight = tt.defaultWeight
			scorer, err := kvcache.NewKVBlockScorer(cfg)
			require.NoError(t, err)

			scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
			require.NoError(t, err)
			assert.InDelta(t, 2.0, scored[podA], 0.0001)
			assert.InDelta(t, tt.wantB, scored[podB], 0.0001)
		})
	}
}

func TestDefaultKVCacheBackendConfigIncludesLMCacheL1(t *testing.T) {
	cfg := kvcache.DefaultKVBlockScorerConfig()
	scorer, err := kvcache.NewKVBlockScorer(cfg)
	require.NoError(t, err)

	keys := int64KeysToKVBlockKeys([]uint64{1001})
	scored, err := scorer.Score(context.Background(), keys, map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "lmcache-l1"}},
	})
	require.NoError(t, err)
	assert.InDelta(t, 0.8, scored[podA], 0.0001)
}
