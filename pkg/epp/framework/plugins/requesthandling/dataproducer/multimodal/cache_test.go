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

package multimodal

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

func TestPodCacheEvictsByEmbeddingWeight(t *testing.T) {
	cache := newPodCache(1024)

	for _, tc := range []struct {
		requestID string
		item      attrmm.MatchItem
	}{
		{requestID: "request-a", item: matchItem("a", 512)},
		{requestID: "request-b", item: matchItem("b", 64)},
		{requestID: "request-c", item: matchItem("c", 64)},
	} {
		require.True(t, cache.acquire(tc.requestID, tc.item))
		cache.release(tc.requestID, tc.item.Hash)
	}

	require.True(t, cache.acquire("request-d", matchItem("d", 512)))
	cache.release("request-d", "d")

	assert.False(t, cache.contains("a"))
	assert.True(t, cache.contains("b"))
	assert.True(t, cache.contains("c"))
	assert.True(t, cache.contains("d"))
	assert.Equal(t, 640, cache.used)
}

func TestPodCacheDoesNotEvictReferencedEntries(t *testing.T) {
	cache := newPodCache(640)
	require.True(t, cache.acquire("active", matchItem("a", 512)))

	require.True(t, cache.acquire("request-b", matchItem("b", 64)))
	cache.release("request-b", "b")
	require.True(t, cache.acquire("request-c", matchItem("c", 64)))
	cache.release("request-c", "c")

	require.True(t, cache.acquire("request-d", matchItem("d", 64)))

	assert.True(t, cache.contains("a"))
	assert.False(t, cache.contains("b"))
	assert.True(t, cache.contains("c"))
	assert.True(t, cache.contains("d"))
}

func TestPodCacheSharedReferencesBlockReclamation(t *testing.T) {
	cache := newPodCache(512)
	require.True(t, cache.acquire("request-1", matchItem("shared", 512)))
	cache.release("request-1", "shared")
	require.True(t, cache.acquire("request-2", matchItem("shared", 512)))
	require.True(t, cache.acquire("request-3", matchItem("shared", 512)))

	cache.release("request-2", "shared")
	assert.False(t, cache.commit(matchItem("replacement", 512)))
	assert.True(t, cache.contains("shared"))

	cache.release("request-3", "shared")
	assert.True(t, cache.commit(matchItem("replacement", 512)))
	assert.False(t, cache.contains("shared"))
	assert.True(t, cache.contains("replacement"))
}

func TestPodCacheHitRefreshesReclaimOrder(t *testing.T) {
	cache := newPodCache(2)
	require.True(t, cache.acquire("request-a", matchItem("a", 1)))
	cache.release("request-a", "a")
	require.True(t, cache.acquire("request-b", matchItem("b", 1)))
	cache.release("request-b", "b")

	require.True(t, cache.acquire("request-a-2", matchItem("a", 1)))
	cache.release("request-a-2", "a")
	require.True(t, cache.commit(matchItem("c", 1)))

	assert.True(t, cache.contains("a"))
	assert.False(t, cache.contains("b"))
	assert.True(t, cache.contains("c"))
}

func TestPodCacheNormalizesSizeAndRejectsOversizedItems(t *testing.T) {
	cache := newPodCache(2)

	require.True(t, cache.commit(matchItem("zero", 0)))
	assert.Equal(t, 1, cache.used)
	assert.False(t, cache.commit(matchItem("oversized", 3)))
	assert.False(t, cache.contains("oversized"))
}

func TestPodCacheReleaseIsIdempotent(t *testing.T) {
	cache := newPodCache(1)
	require.True(t, cache.acquire("request", matchItem("item", 1)))

	cache.release("request", "item")
	cache.release("request", "item")

	assert.Equal(t, 1, cache.freeable.Len())
	assert.Equal(t, 1, cache.used)
}

func matchItem(hash string, size int) attrmm.MatchItem {
	return attrmm.MatchItem{Hash: hash, Size: size, Modality: "image"}
}
