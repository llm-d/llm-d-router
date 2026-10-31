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

func TestInMemoryIndexEngineKeyLRUEvictionRemovesOneToManyReferences(t *testing.T) {
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 1})
	require.NoError(t, err)

	entry := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	require.NoError(t, index.Add(t.Context(), []BlockHash{1}, []BlockHash{11, 12}, []PodEntry{entry}))
	require.Contains(t, index.oneToManyEntries[11], BlockHash(1))

	require.NoError(t, index.Add(t.Context(), []BlockHash{2}, []BlockHash{21}, []PodEntry{entry}))

	assert.NotContains(t, index.oneToManyEntries, BlockHash(11))
	assert.NotContains(t, index.oneToManyEntries, BlockHash(12))
}
