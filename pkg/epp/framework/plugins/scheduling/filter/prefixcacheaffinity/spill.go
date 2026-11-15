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

package prefixcacheaffinity

import (
	"cmp"
	"encoding/binary"
	"slices"

	"github.com/cespare/xxhash/v2"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// spill bounds a load override to the sticky endpoints plus at most
// MaxSpillEndpoints non-sticky ones. A non-sticky endpoint qualifies when its
// own TTFT estimate beats bestStickyTTFT by more than MaxTTFTPenaltyMs, the
// same test the gate applies to the best non-sticky endpoint. Qualifying
// endpoints are taken in rendezvous order for the request's cached prefix, so
// requests for one prefix spill to the same endpoints, and removing an
// endpoint changes only the spill sets that contained it. The result keeps the
// input order.
func (p *Plugin) spill(request *fwksched.InferenceRequest, endpoints, sticky, nonSticky []fwksched.Endpoint, bestStickyTTFT float64) []fwksched.Endpoint {
	key := p.spillKey(request, sticky)
	type candidate struct {
		id    fwkdl.ID
		score uint64
	}
	candidates := make([]candidate, 0, len(nonSticky))
	for _, ep := range nonSticky {
		if ttft, ok := p.endpointTTFT(ep); ok && bestStickyTTFT-ttft > p.config.MaxTTFTPenaltyMs {
			id := ep.GetMetadata().ID
			candidates = append(candidates, candidate{id: id, score: rendezvousScore(key, id)})
		}
	}
	slices.SortFunc(candidates, func(a, b candidate) int { return cmp.Compare(b.score, a.score) })

	keep := make(map[fwkdl.ID]bool, len(sticky)+p.config.MaxSpillEndpoints)
	for _, ep := range sticky {
		keep[ep.GetMetadata().ID] = true
	}
	for _, c := range candidates[:min(len(candidates), p.config.MaxSpillEndpoints)] {
		keep[c.id] = true
	}
	kept := make([]fwksched.Endpoint, 0, len(keep))
	for _, ep := range endpoints {
		if keep[ep.GetMetadata().ID] {
			kept = append(kept, ep)
		}
	}
	return kept
}

// spillKey hashes the first prompt's tokens up to the deepest cached prefix
// among the sticky endpoints, with the target model and cache salt. Without
// prompt tokens it hashes the ID of the sticky endpoint holding that prefix,
// the lowest ID on a tie.
func (p *Plugin) spillKey(request *fwksched.InferenceRequest, sticky []fwksched.Endpoint) uint64 {
	home, depth := sticky[0], p.cachedPrefixTokens(sticky[0])
	for _, ep := range sticky[1:] {
		n := p.cachedPrefixTokens(ep)
		if n > depth || (n == depth && ep.GetMetadata().ID.String() < home.GetMetadata().ID.String()) {
			home, depth = ep, n
		}
	}

	var tokens []uint32
	if request != nil && request.Body != nil && request.Body.TokenizedRequest != nil &&
		len(request.Body.TokenizedRequest.Prompts) > 0 {
		tokens = request.Body.TokenizedRequest.Prompts[0].TokenIDs
		tokens = tokens[:min(depth, len(tokens))]
	}
	var d xxhash.Digest
	d.Reset()
	if len(tokens) == 0 {
		_, _ = d.WriteString(home.GetMetadata().ID.String())
		return d.Sum64()
	}
	var buf [4096]byte
	for _, s := range []string{request.TargetModel, request.Body.TokenizedRequest.CacheSalt} {
		binary.LittleEndian.PutUint64(buf[:8], uint64(len(s)))
		_, _ = d.Write(buf[:8])
		_, _ = d.WriteString(s)
	}
	for len(tokens) > 0 {
		n := min(len(tokens), len(buf)/4)
		for i, t := range tokens[:n] {
			binary.LittleEndian.PutUint32(buf[4*i:], t)
		}
		_, _ = d.Write(buf[:4*n])
		tokens = tokens[n:]
	}
	return d.Sum64()
}

// cachedPrefixTokens returns the length in tokens of the contiguous prefix the
// endpoint holds for the request.
func (p *Plugin) cachedPrefixTokens(ep fwksched.Endpoint) int {
	if raw, ok := ep.Get(p.prefixMatchDataKey); ok {
		info := raw.(*attrprefix.PrefixCacheMatchInfo)
		return info.CachedBlockCount() * info.BlockSizeTokens()
	}
	return 0
}

// rendezvousScore ranks an endpoint for a key (highest random weight hashing).
func rendezvousScore(key uint64, id fwkdl.ID) uint64 {
	var d xxhash.Digest
	d.Reset()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], key)
	_, _ = d.Write(buf[:])
	_, _ = d.WriteString(id.String())
	return d.Sum64()
}
