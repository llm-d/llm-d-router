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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEngineToRequestMapping_OneToOne(t *testing.T) {
	engineKeys := []BlockHash{100, 200, 300, 400}
	requestKeys := []BlockHash{1000, 2000, 3000, 4000}

	mappings := engineToRequestMapping(engineKeys, requestKeys)

	require.Len(t, mappings, 4)
	for i, ek := range engineKeys {
		rks, ok := mappings[ek]
		require.True(t, ok, "engine key %d should be present", ek)
		assert.Equal(t, []BlockHash{requestKeys[i]}, rks)
	}
}

func TestEngineToRequestMapping_ManyToOne(t *testing.T) {
	engineKeys := []BlockHash{10, 11, 12, 13}
	requestKeys := []BlockHash{9000}

	mappings := engineToRequestMapping(engineKeys, requestKeys)

	require.Len(t, mappings, 4)
	for _, ek := range engineKeys {
		rks, ok := mappings[ek]
		require.True(t, ok, "engine key %d should be present", ek)
		assert.Equal(t, []BlockHash{9000}, rks, "each engine key should map to the single request key")
	}
}

func TestEngineToRequestMapping_OneToMany(t *testing.T) {
	engineKeys := []BlockHash{100}
	requestKeys := []BlockHash{1000, 2000, 3000, 4000}

	mappings := engineToRequestMapping(engineKeys, requestKeys)

	require.Len(t, mappings, 1)
	rks := mappings[BlockHash(100)]
	assert.Equal(t, []BlockHash{1000, 2000, 3000, 4000}, rks,
		"single engine key should map to all request keys in order")
}

func TestEngineToRequestMapping_EmptySlices(t *testing.T) {
	mappings := engineToRequestMapping(nil, nil)
	assert.Empty(t, mappings)

	mappings = engineToRequestMapping([]BlockHash{}, []BlockHash{1})
	assert.Empty(t, mappings)

	mappings = engineToRequestMapping([]BlockHash{1}, []BlockHash{})
	assert.Empty(t, mappings)
}

func TestEngineToRequestMapping_ScoreOrder(t *testing.T) {
	engineKeys := []BlockHash{100}
	requestKeys := []BlockHash{1000, 2000, 3000}

	mappings := engineToRequestMapping(engineKeys, requestKeys)

	rks := mappings[BlockHash(100)]
	require.Len(t, rks, 3)
	for j, rk := range rks {
		assert.Equal(t, requestKeys[j], rk,
			"relative index %d should correspond to requestKeys[%d]", j, j)
	}
}
