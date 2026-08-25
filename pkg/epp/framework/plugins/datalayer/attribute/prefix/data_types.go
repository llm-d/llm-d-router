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
	"maps"

	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	approxprefixconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/constants"
	p2psourceconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/p2psource/constants"
)

var PrefixCacheMatchInfoDataKey = plugin.NewDataKey("PrefixCacheMatchInfoDataKey", approxprefixconstants.ApproxPrefixCachePluginType)

// ReusablePrefixTokensDataKey identifies the request-wide reusable prefix
// token floor published by a p2p-source-producer instance.
var ReusablePrefixTokensDataKey = plugin.NewDataKey("ReusablePrefixTokensDataKey", p2psourceconstants.P2PSourcePluginType)

// ReusablePrefixTokens is a floor on the prompt tokens every valid destination
// can reuse either from confirmed local cache or by pulling confirmed CPU-tier
// blocks from the sampled source.
type ReusablePrefixTokens int

// SpeculativeTierKey is the CachedBlocksByTier key for speculative index
// entries, which carry no engine-reported device tier.
const SpeculativeTierKey = "speculative"

type PrefixCacheMatchInfo struct {
	// matched prefix length in blocks. For the precise prefix cache this is the
	// device-tier-weighted longest-prefix score (e.g. RAM-tier blocks count as
	// less than 1.0), suitable for relative endpoint ranking.
	matchBlocks int
	// total length in blocks
	totalBlocks int
	// block length in tokens
	blockSizeTokens int
	// unweighted count of contiguous cached prefix blocks on the endpoint.
	// Unlike matchBlocks this is the literal number of cached blocks regardless
	// of device tier, so consumers that convert blocks to a token count (e.g.
	// the prefix-based PD decider) get an accurate cached-token figure rather
	// than a tier-attenuated one. Defaults to matchBlocks when not set.
	cachedBlockCount int
	// unweighted count of contiguous cached prefix blocks confirmed by the
	// engine. Speculative index entries are excluded, while confirmed blocks
	// can move between device tiers across the prefix.
	confirmedCachedBlockCount int
	// per device tier, the unweighted count of contiguous cached prefix blocks
	// the endpoint holds in that tier, from the first block until the first
	// block missing from that tier. A block held in several tiers counts once
	// per tier. Speculative index entries count under SpeculativeTierKey.
	// Nil when the producer supplies no tier data.
	cachedBlocksByTier map[string]int
	// optional multimodal block-match attribution
	mm *MMMatchInfo
}

type MMMatchInfo struct {
	MatchBlocks int
}

func NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, blockSizeTokens int) *PrefixCacheMatchInfo {
	return &PrefixCacheMatchInfo{
		matchBlocks:      matchBlocks,
		totalBlocks:      totalBlocks,
		blockSizeTokens:  blockSizeTokens,
		cachedBlockCount: matchBlocks,
	}
}

// WithCachedBlockCount sets the unweighted contiguous cached-block count and
// returns the receiver for chaining.
func (p *PrefixCacheMatchInfo) WithCachedBlockCount(cachedBlockCount int) *PrefixCacheMatchInfo {
	p.cachedBlockCount = cachedBlockCount
	return p
}

// WithMM attaches MM tracking. Call only for requests that carry MM content
// (MatchBlocks may be 0 on a miss); leave unset for text-only so MM() stays nil
// and consumers can tell "no MM" from "MM, zero match".
func (p *PrefixCacheMatchInfo) WithMM(mm MMMatchInfo) *PrefixCacheMatchInfo {
	p.mm = &mm
	return p
}

func (p *PrefixCacheMatchInfo) MatchBlocks() int     { return p.matchBlocks }
func (p *PrefixCacheMatchInfo) TotalBlocks() int     { return p.totalBlocks }
func (p *PrefixCacheMatchInfo) BlockSizeTokens() int { return p.blockSizeTokens }
func (p *PrefixCacheMatchInfo) MM() *MMMatchInfo     { return p.mm }

// CachedBlockCount returns the unweighted count of contiguous cached prefix
// blocks on the endpoint.
func (p *PrefixCacheMatchInfo) CachedBlockCount() int {
	return p.cachedBlockCount
}

// WithConfirmedCachedBlockCount sets the unweighted contiguous count of
// confirmed, non-speculative cached blocks and returns the receiver for
// chaining.
func (p *PrefixCacheMatchInfo) WithConfirmedCachedBlockCount(confirmedCachedBlockCount int) *PrefixCacheMatchInfo {
	p.confirmedCachedBlockCount = confirmedCachedBlockCount
	return p
}

// ConfirmedCachedBlockCount returns the unweighted contiguous count of
// confirmed, non-speculative cached blocks on the endpoint.
func (p *PrefixCacheMatchInfo) ConfirmedCachedBlockCount() int {
	return p.confirmedCachedBlockCount
}

// WithCachedBlocksByTier sets the per-device-tier contiguous cached-block
// counts and returns the receiver for chaining. Takes ownership of the map;
// the caller must not mutate it after the call.
func (p *PrefixCacheMatchInfo) WithCachedBlocksByTier(cachedBlocksByTier map[string]int) *PrefixCacheMatchInfo {
	p.cachedBlocksByTier = cachedBlocksByTier
	return p
}

// CachedBlocksByTier returns, per device tier, the unweighted count of
// contiguous cached prefix blocks the endpoint holds in that tier. Nil means
// the producer supplies no tier data. Callers must not mutate the map.
func (p *PrefixCacheMatchInfo) CachedBlocksByTier() map[string]int {
	return p.cachedBlocksByTier
}

func (p *PrefixCacheMatchInfo) Clone() fwkdl.Cloneable {
	clone := &PrefixCacheMatchInfo{
		matchBlocks:               p.matchBlocks,
		totalBlocks:               p.totalBlocks,
		blockSizeTokens:           p.blockSizeTokens,
		cachedBlockCount:          p.cachedBlockCount,
		confirmedCachedBlockCount: p.confirmedCachedBlockCount,
		cachedBlocksByTier:        maps.Clone(p.cachedBlocksByTier),
	}
	if p.mm != nil {
		clone.mm = ptr.To(*p.mm)
	}
	return clone
}
