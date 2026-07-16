/*
Copyright 2025 The llm-d Authors.

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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// TestCostAwareKeyIndexBoundedUnderEviction verifies keyIndex does not grow with
// every key ever added. keyIndex exists only so Clear can enumerate keys (ristretto
// has no iteration); without pruning it would defeat the cost cache's memory bound.
// The OnEvict/OnReject callback prunes keyIndex whenever ristretto drops an entry,
// so after adding N keys to a cache sized for ~2, keyIndex tracks the live set, not N.
func TestCostAwareKeyIndexBoundedUnderEviction(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	// Size the cache to hold only a couple of single-entry keys.
	probe := &CostPodCache{}
	probe.Add(PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"})
	oneKeyCost := probe.CalculateByteSize(BlockHash(1).String())

	cfg := DefaultCostAwareMemoryIndexConfig()
	cfg.Size = fmt.Sprintf("%d", 2*oneKeyCost) // room for ~2 keys

	index, err := NewCostAwareMemoryIndex(cfg)
	require.NoError(t, err)

	const n = 1000
	for i := uint64(0); i < n; i++ {
		key := BlockHash(i + 1)
		require.NoError(t, index.Add(ctx,
			[]BlockHash{key}, []BlockHash{key},
			[]PodEntry{{PodIdentifier: "pod", DeviceTier: "gpu"}}))
	}

	index.keyIndexMu.Lock()
	got := len(index.keyIndex)
	index.keyIndexMu.Unlock()

	t.Logf("keyIndex size after %d adds into a ~2-key cache: %d", n, got)
	// Must stay near the live cost-cache set, not grow with every Add.
	assert.Less(t, got, n/10, "keyIndex should be pruned on eviction/rejection, not retain every key")
}
