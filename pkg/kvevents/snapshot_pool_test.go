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

func TestSnapshotOffloadPreservesEveryCanonicalBlock(t *testing.T) {
	ctx := context.Background()
	pool, index, tokens := newTestPool(t, 4)
	pool.strict = true
	prompt := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	require.NoError(t, pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101}, Tokens: prompt, BlockSize: 8, DeviceTier: "GPU"},
		&BlockStoredEvent{BlockHashes: []uint64{101}, BlockSize: 8, DeviceTier: "CPU"},
		&AllBlocksClearedEvent{},
	}}, "pod", "model"))
	keys, err := tokens.TokensToKVBlockKeys(0, prompt, "model", nil)
	require.NoError(t, err)
	entries, err := index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	for _, key := range keys {
		require.Equal(t, []kvblock.PodEntry{{PodIdentifier: "pod", DeviceTier: "cpu"}}, entries[key])
	}
}
