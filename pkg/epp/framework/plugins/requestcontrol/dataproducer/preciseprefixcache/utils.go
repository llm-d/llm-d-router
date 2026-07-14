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

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
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

// speculativeTierKey is the CachedBlocksByTier key for speculative index
// entries: PreRequest inserts them before vLLM has reported placement, so
// they carry no device tier.
const speculativeTierKey = "speculative"

// matchedBlockCountByTier returns, per device tier, the number of contiguous
// cached prefix blocks podID holds in that tier, counting from the first
// block until the first block the pod does not hold in that tier. A block
// held in several tiers counts once per tier, so each tier's count is at most
// matchedBlockCount for the same pod. Tiers are recorded as found in the
// index, except speculative entries, which count under speculativeTierKey.
// Returns a non-nil (possibly empty) map.
func matchedBlockCountByTier(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) map[string]int {
	counts := map[string]int{}
	var alive sets.Set[string]
	for _, key := range keys {
		tiersAtKey := sets.New[string]()
		for _, e := range keyToPods[key] {
			if e.PodIdentifier == podID {
				if e.Speculative {
					tiersAtKey.Insert(speculativeTierKey)
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
