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

package kvblock_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// newSampledIndex builds each backend with the given sampling modulus.
func newSampledIndex(t *testing.T, modulus int) map[string]kvblock.Index {
	t.Helper()
	inMemory, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{
		Size:          1 << 12,
		PodCacheSize:  256,
		SampleModulus: modulus,
	})
	require.NoError(t, err)
	dense, err := kvblock.NewDenseMemoryIndex(&kvblock.DenseMemoryIndexConfig{
		Size:          "1Mi",
		PodCacheSize:  15,
		SampleModulus: modulus,
	})
	require.NoError(t, err)
	return map[string]kvblock.Index{"in-memory": inMemory, "dense": dense}
}

// TestSampleModulusStoresSampledKeysOnly checks that Add keeps only the
// request keys passing the sampling test, in both sampling backends.
func TestSampleModulusStoresSampledKeysOnly(t *testing.T) {
	entries := []kvblock.PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}
	keys := []kvblock.BlockHash{4, 5, 8, 9, 12} // sampled under modulus 4: 4, 8, 12

	for name, index := range newSampledIndex(t, 4) {
		t.Run(name, func(t *testing.T) {
			require.NoError(t, index.Add(t.Context(), nil, keys, entries))

			one := make([]kvblock.BlockHash, 1)
			for _, key := range keys {
				one[0] = key
				res, err := index.Lookup(t.Context(), one, nil)
				require.NoError(t, err)
				if kvblock.SampledKey(key, 4) {
					assert.NotEmpty(t, res[key], "sampled key %d stored", key)
				} else {
					assert.Empty(t, res[key], "unsampled key %d not stored", key)
				}
			}
		})
	}
}

// TestSampleModulusKeepsEngineMappings checks that the engine-key map is not
// sampled: a pair whose request key is unsampled still resolves.
func TestSampleModulusKeepsEngineMappings(t *testing.T) {
	entries := []kvblock.PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}
	for name, index := range newSampledIndex(t, 4) {
		t.Run(name, func(t *testing.T) {
			engineKey, requestKey := kvblock.BlockHash(100), kvblock.BlockHash(5)
			require.NoError(t, index.Add(t.Context(), []kvblock.BlockHash{engineKey}, []kvblock.BlockHash{requestKey}, entries))

			got, err := index.GetRequestKey(t.Context(), engineKey)
			require.NoError(t, err)
			assert.Equal(t, requestKey, got)
		})
	}
}

// TestSampleModulusValidation checks that a non-power-of-two modulus is
// rejected at construction.
func TestSampleModulusValidation(t *testing.T) {
	_, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 12, SampleModulus: 3})
	assert.Error(t, err)
	_, err = kvblock.NewDenseMemoryIndex(&kvblock.DenseMemoryIndexConfig{Size: "1Mi", SampleModulus: 3})
	assert.Error(t, err)
}

func TestConfigSampleModulus(t *testing.T) {
	assert.Zero(t, kvblock.ConfigSampleModulus(nil))
	assert.Zero(t, kvblock.ConfigSampleModulus(&kvblock.IndexConfig{}))
	assert.Equal(t, 16, kvblock.ConfigSampleModulus(&kvblock.IndexConfig{
		InMemoryConfig: &kvblock.InMemoryIndexConfig{SampleModulus: 16},
	}))
	assert.Equal(t, 16, kvblock.ConfigSampleModulus(&kvblock.IndexConfig{
		DenseMemoryConfig: &kvblock.DenseMemoryIndexConfig{SampleModulus: 16},
	}))
}

// TestSampleModulusEvictKeepsUnsampledMapping pins the engine-key evict path
// under sampling: an unsampled request key has no entry state to check, so
// the first eviction by any pod must not drop the mapping that other pods
// still need for parent resolution.
func TestSampleModulusEvictKeepsUnsampledMapping(t *testing.T) {
	const modulus = 8
	for name, idx := range newSampledIndex(t, modulus) {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			engineKey := kvblock.BlockHash(9001)
			pod := kvblock.PodEntry{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}
			requestKey := kvblock.BlockHash(2) // 2 & 7 != 0: never sampled
			require.False(t, kvblock.SampledKey(requestKey, modulus))

			require.NoError(t, idx.Add(ctx, []kvblock.BlockHash{engineKey}, []kvblock.BlockHash{requestKey}, []kvblock.PodEntry{pod}))
			require.NoError(t, idx.Evict(ctx, kvblock.EngineKey, []kvblock.BlockHash{engineKey}, []kvblock.PodEntry{pod}))

			rk, err := idx.GetRequestKey(ctx, engineKey)
			require.NoError(t, err, "eviction by one pod dropped the mapping for pods that still hold the block")
			require.Equal(t, requestKey, rk)
		})
	}
}

// TestSampleModulusEvictRemovesSampledMapping guards the fix's boundary: a
// chain of sampled keys whose entries are all gone still drops its mapping.
func TestSampleModulusEvictRemovesSampledMapping(t *testing.T) {
	const modulus = 8
	for name, idx := range newSampledIndex(t, modulus) {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			engineKey := kvblock.BlockHash(9001)
			pod := kvblock.PodEntry{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}
			requestKey := kvblock.BlockHash(8) // 8 & 7 == 0: sampled
			require.True(t, kvblock.SampledKey(requestKey, modulus))

			require.NoError(t, idx.Add(ctx, []kvblock.BlockHash{engineKey}, []kvblock.BlockHash{requestKey}, []kvblock.PodEntry{pod}))
			require.NoError(t, idx.Evict(ctx, kvblock.EngineKey, []kvblock.BlockHash{engineKey}, []kvblock.PodEntry{pod}))

			_, err := idx.GetRequestKey(ctx, engineKey)
			require.Error(t, err, "an all-sampled empty chain must still drop its mapping")
		})
	}
}
