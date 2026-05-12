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

// computeUnweightedMatchBlocks returns the longest consecutive prefix match
// count per pod, ignoring device-tier weights. Mirrors the intersection loop
// in kvcache.LongestPrefixScorer but adds 1 per block instead of a weight.
// Pods absent from the returned map have 0 hits. The P/D decider needs this
// physical count to translate cached blocks into cached tokens; the weighted
// score returned by kvcache.KVBlockScorer is the wrong unit. See issue #1047.
func computeUnweightedMatchBlocks(
	blockKeys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) map[string]int {
	counts := make(map[string]int)
	if len(blockKeys) == 0 {
		return counts
	}

	active := make(map[string]struct{})
	for _, entry := range keyToPods[blockKeys[0]] {
		if _, seen := active[entry.PodIdentifier]; !seen {
			active[entry.PodIdentifier] = struct{}{}
			counts[entry.PodIdentifier] = 1
		}
	}

	present := make(map[string]struct{})
	for i := 1; i < len(blockKeys); i++ {
		if len(active) == 0 {
			break
		}
		clear(present)
		for _, entry := range keyToPods[blockKeys[i]] {
			present[entry.PodIdentifier] = struct{}{}
		}
		for pod := range active {
			if _, ok := present[pod]; ok {
				counts[pod]++
			} else {
				delete(active, pod)
			}
		}
	}
	return counts
}
