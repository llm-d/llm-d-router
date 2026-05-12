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
)

func TestPrefixCacheMatchInfo_MatchBlocksUnweighted(t *testing.T) {
	t.Run("falls back to matchBlocks when unset", func(t *testing.T) {
		info := NewPrefixCacheMatchInfo(7, 10, 16)
		assert.Equal(t, 7, info.MatchBlocksUnweighted())
	})

	t.Run("explicit value is independent of matchBlocks", func(t *testing.T) {
		info := NewPrefixCacheMatchInfo(192, 256, 16).WithMatchBlocksUnweighted(240)
		assert.Equal(t, 192, info.MatchBlocks())
		assert.Equal(t, 240, info.MatchBlocksUnweighted())
	})
}

func TestPrefixCacheMatchInfo_Clone(t *testing.T) {
	orig := NewPrefixCacheMatchInfo(192, 256, 16).WithMatchBlocksUnweighted(240)
	cloned := orig.Clone().(*PrefixCacheMatchInfo)

	assert.Equal(t, 192, cloned.MatchBlocks())
	assert.Equal(t, 240, cloned.MatchBlocksUnweighted())

	cloned.WithMatchBlocksUnweighted(0)
	assert.Equal(t, 240, orig.MatchBlocksUnweighted(), "clone must not alias original")
}
