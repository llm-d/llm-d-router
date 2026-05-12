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
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	approxprefixconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/constants"
)

var PrefixCacheMatchInfoDataKey = plugin.NewDataKey("PrefixCacheMatchInfoDataKey", approxprefixconstants.ApproxPrefixCachePluginType)

// PrefixCacheMatchInfo carries per-endpoint prefix-cache match information.
//
// matchBlocks may be a device-weighted score (precise-prefix-cache) or a raw
// block count (approximate-prefix). Consumers translating blocks to tokens
// (e.g. P/D decider) must use MatchBlocksUnweighted. See issue #1047.
type PrefixCacheMatchInfo struct {
	matchBlocks           int
	matchBlocksUnweighted int
	totalBlocks           int
	blockSizeTokens       int
}

func NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, blockSizeTokens int) *PrefixCacheMatchInfo {
	return &PrefixCacheMatchInfo{
		matchBlocks:           matchBlocks,
		matchBlocksUnweighted: matchBlocks, // tier-agnostic default
		totalBlocks:           totalBlocks,
		blockSizeTokens:       blockSizeTokens,
	}
}

// WithMatchBlocksUnweighted sets the raw block-hit count, distinct from the
// weighted ranking score in matchBlocks. Used by tier-aware producers.
func (p *PrefixCacheMatchInfo) WithMatchBlocksUnweighted(n int) *PrefixCacheMatchInfo {
	p.matchBlocksUnweighted = n
	return p
}

// MatchBlocks returns the producer-supplied score. For tier-aware producers
// this is a device-weighted value, not a raw block count — use
// MatchBlocksUnweighted when you need the physical hit count.
func (p *PrefixCacheMatchInfo) MatchBlocks() int { return p.matchBlocks }

// MatchBlocksUnweighted returns the raw count of prefix blocks hit.
func (p *PrefixCacheMatchInfo) MatchBlocksUnweighted() int { return p.matchBlocksUnweighted }

func (p *PrefixCacheMatchInfo) TotalBlocks() int     { return p.totalBlocks }
func (p *PrefixCacheMatchInfo) BlockSizeTokens() int { return p.blockSizeTokens }

func (p *PrefixCacheMatchInfo) Clone() fwkdl.Cloneable {
	// All fields are value types (int), so a shallow struct copy is a full
	// deep copy. If a reference-type field (slice, map, pointer) is added
	// later, this must be updated to copy it explicitly.
	c := *p
	return &c
}
