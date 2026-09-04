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

// extractEndpointSet builds the identifier set used to filter kvblock.Index
// lookups to candidate endpoints: each endpoint's "address:port" plus the
// node pseudo-pod of every node hosting a candidate. Endpoints without
// metadata are skipped.
func extractEndpointSet(endpoints []scheduling.Endpoint) sets.Set[string] {
	endpointSet := sets.New[string]()
	for _, ep := range endpoints {
		if m := ep.GetMetadata(); m != nil {
			endpointSet.Insert(fmt.Sprintf("%s:%s", m.Address, m.Port))
			if m.NodeName != "" {
				endpointSet.Insert(kvblock.NodePseudoPod(m.NodeName))
			}
		}
	}
	return endpointSet
}

// endpointsByNode groups candidate "address:port" identifiers by the node
// hosting them, and returns the flat list of all identifiers. Endpoints
// without metadata are skipped; endpoints without a node name appear only in
// the flat list.
func endpointsByNode(endpoints []scheduling.Endpoint) (map[string][]string, []string) {
	byNode := map[string][]string{}
	all := make([]string, 0, len(endpoints))
	for _, ep := range endpoints {
		m := ep.GetMetadata()
		if m == nil {
			continue
		}
		addr := fmt.Sprintf("%s:%s", m.Address, m.Port)
		all = append(all, addr)
		if m.NodeName != "" {
			byNode[m.NodeName] = append(byNode[m.NodeName], addr)
		}
	}
	return byNode, all
}

// matchedBlockCount returns the number of contiguous cached prefix blocks held
// by podID, counting from the first block until the first block the pod does
// not hold. This is the unweighted counterpart of the device-tier-weighted
// kvblock scorer: every cached block counts as one regardless of device tier,
// so a pod present at keys[0..n-1] yields n.
func matchedBlockCount(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) int {
	count := 0
	for _, key := range keys {
		if !slices.ContainsFunc(keyToPods[key], func(e kvblock.PodEntry) bool { return e.PodIdentifier == podID }) {
			break
		}
		count++
	}
	return count
}

// matchedBlockCountByTier returns, per device tier, the number of contiguous
// cached prefix blocks podID holds in that tier, counting from the first
// block until the first block the pod does not hold in that tier. A block
// held in several tiers counts once per tier, so each tier's count is at most
// matchedBlockCount for the same pod. Tiers are recorded as found in the
// index, except speculative entries, which count under
// attrprefix.SpeculativeTierKey: PreRequest inserts them before vLLM has
// reported placement, so they carry no device tier.
// Returns a non-nil (possibly empty) map.
func matchedBlockCountByTier(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) map[string]int {
	counts := map[string]int{}
	var alive sets.Set[string]
	for _, key := range keys {
		tiersAtKey := sets.New[string]()
		for _, e := range keyToPods[key] {
			if e.PodIdentifier == podID {
				if e.Speculative {
					tiersAtKey.Insert(attrprefix.SpeculativeTierKey)
				} else {
					tiersAtKey.Insert(e.DeviceTier)
				}
			}
		}
		if alive == nil {
			alive = tiersAtKey
		} else {
			alive = alive.Intersection(tiersAtKey)
		}
		if alive.Len() == 0 {
			break
		}
		for tier := range alive {
			counts[tier]++
		}
	}
	return counts
}
