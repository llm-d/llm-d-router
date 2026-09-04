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

package preciseprefixcache

import (
	"fmt"
	"slices"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// extractEndpointSet builds the "address:port" identifier set used to filter
// kvblock.Index lookups to candidate endpoints. Endpoints without metadata
// are skipped.
func extractEndpointSet(endpoints []scheduling.Endpoint) sets.Set[string] {
	endpointSet := sets.New[string]()
	for _, ep := range endpoints {
		if m := ep.GetMetadata(); m != nil {
			endpointSet.Insert(fmt.Sprintf("%s:%s", m.Address, m.Port))
		}
	}
	return endpointSet
}

// matchedBlocks returns the number of contiguous cached prefix blocks held by
// podID, counting from the first block until the first block the pod does not
// hold, together with the per-tier contiguous counts. Every held block counts
// once in the first return value regardless of device tier, and once per tier
// in the map, so each tier's count is at most the any-tier count. A tier's
// chain may end before the any-tier chain; that tier's count then stops
// growing while the first return value keeps advancing.
// Tiers are recorded as found in the index, except speculative entries, which
// count under attrprefix.SpeculativeTierKey: PreRequest inserts them before
// vLLM has reported placement, so they carry no device tier.
// The map is non-nil (possibly empty).
func matchedBlocks(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) (int, map[string]int) {
	var count int
	counts := map[string]int{}

	var chainTiers, tiers []string
	// tierChainEnded is set when chainTiers becomes empty. The intersection
	// can only shrink, so no tier counts change afterwards; only the any-tier
	// count keeps advancing. While set, per-block tier collection is skipped
	// and the entry scan stops at the pod's first entry.
	var tierChainEnded bool
	for _, key := range keys {
		var podHas bool
		tiers = tiers[:0]
		for _, e := range keyToPods[key] {
			if e.PodIdentifier != podID {
				continue
			}
			podHas = true
			if tierChainEnded {
				break
			}

			tier := e.DeviceTier
			if e.Speculative {
				tier = attrprefix.SpeculativeTierKey
			}
			if !slices.Contains(tiers, tier) {
				tiers = append(tiers, tier)
			}
		}
		if !podHas {
			break // any-tier chain broken: both counts stop
		}

		count++
		if tierChainEnded {
			continue
		}

		if chainTiers == nil {
			// First held block seeds the tier chain with a copy of tiers,
			// which is reused for every block.
			chainTiers = slices.Clone(tiers)
			for _, t := range chainTiers {
				counts[t]++
			}
			continue
		}

		n := 0
		for _, t := range chainTiers {
			if slices.Contains(tiers, t) {
				chainTiers[n] = t
				n++
			}
		}
		chainTiers = chainTiers[:n]
		if len(chainTiers) == 0 {
			tierChainEnded = true // tier counts stop; the any-tier count continues
			continue
		}
		for _, t := range chainTiers {
			counts[t]++
		}
	}
	return count, counts
}
