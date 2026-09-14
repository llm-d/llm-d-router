package kvblock

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"
)

func TestEngineScopeIsolationAndEviction(t *testing.T) {
	index, err := NewInMemoryIndex(DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	scopes := []EngineScope{
		{Endpoint: "a:8000"},
		{Endpoint: "b:8000"},
		{Endpoint: "a:8000", DataParallelRank: ptr.To(0)},
		{Endpoint: "a:8000", DataParallelRank: ptr.To(1)},
		{Endpoint: "a:8000", GroupIdx: ptr.To(0)},
		{Endpoint: "a:8000", GroupIdx: ptr.To(1)},
		{CacheNamespace: "model-v1", Endpoint: "a:8000"},
		{CacheNamespace: "model-v2", Endpoint: "a:8000"},
	}
	for _, scope := range scopes {
		require.NoError(t, index.Add(t.Context(), nil, scope.Keys([]uint64{42}), []PodEntry{{PodIdentifier: scope.Endpoint, DeviceTier: "gpu"}}))
	}
	require.NoError(t, index.Evict(t.Context(), scopes[0].Keys([]uint64{42})[0], RequestKey, []PodEntry{{PodIdentifier: scopes[0].Endpoint, DeviceTier: "gpu"}}))
	for i, scope := range scopes {
		keys := scope.Keys([]uint64{42})
		found, err := index.Lookup(t.Context(), keys, nil)
		require.NoError(t, err)
		if i == 0 {
			assert.Empty(t, found[keys[0]])
		} else {
			assert.Len(t, found[keys[0]], 1)
		}
	}
	require.NoError(t, index.Clear(t.Context(), "a:8000"))
	for _, scope := range scopes {
		keys := scope.Keys([]uint64{42})
		found, err := index.Lookup(t.Context(), keys, nil)
		require.NoError(t, err)
		if scope.Endpoint == "a:8000" {
			assert.Empty(t, found[keys[0]])
		} else {
			assert.Len(t, found[keys[0]], 1)
		}
	}
}

func TestEngineScopeNamespaceFraming(t *testing.T) {
	a := EngineScope{CacheNamespace: "a", Endpoint: "bc"}
	b := EngineScope{CacheNamespace: "ab", Endpoint: "c"}
	assert.NotEqual(t, a.Keys([]uint64{42}), b.Keys([]uint64{42}))
}
