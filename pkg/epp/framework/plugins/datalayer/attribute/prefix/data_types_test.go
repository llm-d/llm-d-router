/*
Copyright 2025 The Kubernetes Authors.

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

package prefix

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPrefixCacheMatchInfo_CachedBlockCountDefaultsToMatchBlocks(t *testing.T) {
	info := NewPrefixCacheMatchInfo(5, 10, 16)
	assert.Equal(t, 5, info.MatchBlocks())
	assert.Equal(t, 10, info.TotalBlocks())
	assert.Equal(t, 16, info.BlockSizeTokens())
	// Unset cachedBlockCount mirrors matchBlocks so existing producers and
	// consumers keep their current behavior.
	assert.Equal(t, 5, info.CachedBlockCount())
}

func TestPrefixCacheMatchInfo_WithCachedBlockCount(t *testing.T) {
	info := NewPrefixCacheMatchInfo(192, 256, 16).WithCachedBlockCount(240)
	// matchBlocks keeps the tier-weighted ranking value; cachedBlockCount holds
	// the unweighted literal count.
	assert.Equal(t, 192, info.MatchBlocks())
	assert.Equal(t, 240, info.CachedBlockCount())
}

func TestPrefixCacheMatchInfo_CloneCopiesCachedBlockCount(t *testing.T) {
	orig := NewPrefixCacheMatchInfo(192, 256, 16).WithCachedBlockCount(240)
	clone, ok := orig.Clone().(*PrefixCacheMatchInfo)
	require.True(t, ok)
	assert.Equal(t, orig.MatchBlocks(), clone.MatchBlocks())
	assert.Equal(t, orig.TotalBlocks(), clone.TotalBlocks())
	assert.Equal(t, orig.BlockSizeTokens(), clone.BlockSizeTokens())
	assert.Equal(t, orig.CachedBlockCount(), clone.CachedBlockCount())

	// Mutating the clone must not affect the original.
	clone.WithCachedBlockCount(1)
	assert.Equal(t, 240, orig.CachedBlockCount())
	assert.Equal(t, 1, clone.CachedBlockCount())
}
