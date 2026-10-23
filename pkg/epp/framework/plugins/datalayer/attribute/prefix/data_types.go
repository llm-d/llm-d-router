/*
Copyright 2025 The Kubernetes Authors.
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

package prefix

import (
	"encoding/binary"
	"maps"

	"github.com/cespare/xxhash/v2"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	approxprefixconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/constants"
	p2psourceconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/p2psource/constants"
)

var PrefixCacheMatchInfoDataKey = plugin.NewDataKey("PrefixCacheMatchInfoDataKey", approxprefixconstants.ApproxPrefixCachePluginType)

// PrefixHashSetDataKey carries the cross-replica recent-window of prefix block
// hashes this EPP believes an endpoint holds. Used by approx-prefix-cache-producer
// when a CrossReplicaSyncer is configured.
var PrefixHashSetDataKey = plugin.NewDataKey("PrefixHashSetDataKey", approxprefixconstants.ApproxPrefixCachePluginType)

// PrefixHashSet is a capped recent window of prompt hash sequences for one
// endpoint, published across EPP replicas and union-aggregated before matching.
// Each inner slice is one prompt's block hashes starting at block 0; prompts
// are ordered oldest to newest.
type PrefixHashSet struct {
	Prompts [][]uint64
	// digest caches ContentDigest for the Produce hot path. Not serialized.
	digest       uint64
	digestCached bool
}

// ContentDigest returns a content hash of Prompts. The value is cached on s
// because Prompts are not mutated after the set is published.
func (s *PrefixHashSet) ContentDigest() uint64 {
	if s == nil {
		return 0
	}
	if s.digestCached {
		return s.digest
	}
	s.digest = hashPrompts(s.Prompts)
	s.digestCached = true
	return s.digest
}

// hashPrompts is length-prefixed so adjacent equal hashes from different
// prompt boundaries cannot collide.
func hashPrompts(prompts [][]uint64) uint64 {
	h := xxhash.New()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(len(prompts)))
	_, _ = h.Write(buf[:])
	for _, p := range prompts {
		binary.LittleEndian.PutUint64(buf[:], uint64(len(p)))
		_, _ = h.Write(buf[:])
		for _, v := range p {
			binary.LittleEndian.PutUint64(buf[:], v)
			_, _ = h.Write(buf[:])
		}
	}
	return h.Sum64()
}

// Clone returns an independent copy of the hash set.
func (s *PrefixHashSet) Clone() fwkdl.Cloneable {
	if s == nil {
		return nil
	}
	out := &PrefixHashSet{Prompts: make([][]uint64, len(s.Prompts))}
	for i, p := range s.Prompts {
		cp := make([]uint64, len(p))
		copy(cp, p)
		out.Prompts[i] = cp
	}
	if s.digestCached {
		out.digest = s.digest
		out.digestCached = true
	}
	return out
}

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
	// can move between device tiers across the prefix. Defaults to
	// cachedBlockCount when not set.
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
	// MatchTokens is the MM feature tokens inside the matched prefix: each
	// feature counts the overlap of its token span with the matched prefix,
	// so a feature that starts or ends mid-block contributes only the tokens
	// it holds.
	MatchTokens int
}

func NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, blockSizeTokens int) *PrefixCacheMatchInfo {
	return &PrefixCacheMatchInfo{
		matchBlocks:               matchBlocks,
		totalBlocks:               totalBlocks,
		blockSizeTokens:           blockSizeTokens,
		cachedBlockCount:          matchBlocks,
		confirmedCachedBlockCount: matchBlocks,
	}
}

// WithCachedBlockCount sets the unweighted contiguous cached-block count and
// returns the receiver for chaining.
func (p *PrefixCacheMatchInfo) WithCachedBlockCount(cachedBlockCount int) *PrefixCacheMatchInfo {
	p.cachedBlockCount = cachedBlockCount
	p.confirmedCachedBlockCount = cachedBlockCount
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
	if p == nil {
		return nil
	}
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
