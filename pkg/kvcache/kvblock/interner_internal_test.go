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
	"context"
	"fmt"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// An Add whose device tier would exceed the interner cap fails as a whole
// and leaves the index untouched: no key, no engine mapping, no ordinal for
// the batch's new pod. Known tiers keep working.
func TestAddRejectsTierCardinalityPastTheCap(t *testing.T) {
	ctx := log.IntoContext(context.Background(), logr.Discard())
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 1 << 16, PodCacheSize: 8})
	require.NoError(t, err)

	for i := 0; i < maxInternedTiers; i++ {
		entry := []PodEntry{{PodIdentifier: "pod-a", DeviceTier: fmt.Sprintf("tier-%d", i)}}
		require.NoError(t, index.Add(ctx, nil, []BlockHash{BlockHash(i + 1)}, entry))
	}
	podsBefore, tiersBefore := len(index.pods.ids), len(index.tiers.ids)

	overflow := []PodEntry{
		{PodIdentifier: "pod-new", DeviceTier: "tier-0"},
		{PodIdentifier: "pod-new", DeviceTier: "one-too-many"},
	}
	const engineKey, requestKey = BlockHash(1 << 40), BlockHash(1<<40 + 1)
	err = index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, overflow)
	require.ErrorIs(t, err, errIndexCardinality)
	_, found := index.data.Peek(requestKey)
	assert.False(t, found, "a rejected Add must not create the key")
	assert.False(t, index.engineToRequestKeys.Contains(engineKey), "a rejected Add must not map the engine key")
	assert.Equal(t, podsBefore, len(index.pods.ids), "a rejected Add must not consume a pod ordinal")
	assert.Equal(t, tiersBefore, len(index.tiers.ids), "a rejected Add must not consume a tier ordinal")

	require.NoError(t, index.Add(ctx, nil, []BlockHash{1 << 41}, []PodEntry{{PodIdentifier: "pod-b", DeviceTier: "tier-0"}}))
}
