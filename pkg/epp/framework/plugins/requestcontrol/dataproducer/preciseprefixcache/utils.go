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
	"math"
	"slices"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
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

func calculateHitRatioStats(hitRatios []float64) (maxRatio float64, avgRatio float64, stdDevRatio float64) {
	if len(hitRatios) == 0 {
		return 0, 0, 0
	}

	sum := 0.0
	maxRatio = 0.0
	for _, hitRatio := range hitRatios {
		sum += hitRatio
		if hitRatio > maxRatio {
			maxRatio = hitRatio
		}
	}
	avgRatio = sum / float64(len(hitRatios))

	varianceSum := 0.0
	for _, hitRatio := range hitRatios {
		diff := hitRatio - avgRatio
		varianceSum += diff * diff
	}
	stdDevRatio = 0.0
	if len(hitRatios) > 1 {
		stdDevRatio = varianceSum / float64(len(hitRatios)-1)
		stdDevRatio = math.Sqrt(stdDevRatio)
	}

	// Round to two decimal places for consistency in metrics reporting
	avgRatio = math.Round(avgRatio*100) / 100
	stdDevRatio = math.Round(stdDevRatio*100) / 100

	return maxRatio, avgRatio, stdDevRatio
}
