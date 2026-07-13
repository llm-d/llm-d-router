/*
Copyright 2026 The Kubernetes Authors.

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

package burstprefix

import (
	"cmp"
	"slices"
	"strings"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
)

// totalBlocks returns the number of prefix blocks across all prompts.
func totalBlocks(hashes [][]prefixhash.BlockHash) int {
	total := 0
	for _, ph := range hashes {
		total += len(ph)
	}
	return total
}

// encodePrefix serializes the first n prefix blocks. Equal results imply an
// equal n-block leading prefix: each block hash chains its predecessor, so an
// equal leading hash sequence means an equal prompt prefix.
func encodePrefix(hashes [][]prefixhash.BlockHash, n int) string {
	var b strings.Builder
	var buf [8]byte
	count := 0
	for _, ph := range hashes {
		for _, h := range ph {
			if count >= n {
				return b.String()
			}
			prefixhash.PutBlockHash(buf[:], h)
			b.Write(buf[:])
			count++
		}
	}
	return b.String()
}

// groupKey identifies requests that share an identical prompt prefix. Requests
// with no prefix return "" and are never grouped.
func groupKey(hashes [][]prefixhash.BlockHash) string {
	n := totalBlocks(hashes)
	if n == 0 {
		return ""
	}
	return encodePrefix(hashes, n)
}

// sharedPrefixKeys returns the group keys whose requests share at least
// minColocateBlocks leading blocks with some other request in the batch. It is
// the affinity gate for non-identical requests: a lone request that overlaps no
// other keeps no affinity, while overlapping ones (e.g. one prompt contained in
// another) become placeable. Keys with fewer than minColocateBlocks blocks can
// never meet the threshold and are excluded.
func sharedPrefixKeys(groups map[string][]*entry, order []string, minColocateBlocks int) map[string]bool {
	if minColocateBlocks <= 0 {
		return nil
	}
	count := map[string]int{}
	keySig := map[string]string{}
	for _, key := range order {
		hashes := groups[key][0].hashes
		if totalBlocks(hashes) < minColocateBlocks {
			continue
		}
		s := encodePrefix(hashes, minColocateBlocks)
		keySig[key] = s
		count[s] += len(groups[key])
	}
	shared := map[string]bool{}
	for key, s := range keySig {
		if count[s] >= 2 {
			shared[key] = true
		}
	}
	return shared
}

// batchIndex records which replicas hold each prefix block, for matching later
// groups against groups already placed in this batch. Because a whole prompt
// prefix is added at once, holding block i implies holding blocks 0..i, so a
// greedy walk that stops at the first unheld block yields the longest contiguous
// shared prefix per replica (the approximateprefix matchLongestPrefix property).
type batchIndex struct {
	holders map[prefixhash.BlockHash]map[string]struct{}
}

func newBatchIndex() *batchIndex {
	return &batchIndex{holders: map[prefixhash.BlockHash]map[string]struct{}{}}
}

// add records every block of hashes as held by replica.
func (b *batchIndex) add(hashes [][]prefixhash.BlockHash, replica string) {
	for _, ph := range hashes {
		for _, h := range ph {
			s := b.holders[h]
			if s == nil {
				s = map[string]struct{}{}
				b.holders[h] = s
			}
			s[replica] = struct{}{}
		}
	}
}

// longestPrefix returns, per replica, the number of leading prefix blocks that
// replica already holds (summed across prompts) - the shared-prefix length.
func (b *batchIndex) longestPrefix(hashes [][]prefixhash.BlockHash) map[string]int {
	res := map[string]int{}
outer:
	for _, ph := range hashes {
		for _, h := range ph {
			holders := b.holders[h]
			if len(holders) == 0 {
				break outer
			}
			for name := range holders {
				res[name]++
			}
		}
	}
	return res
}

// assign steers each batched request to a replica so prompt-sharing requests
// co-locate. It groups requests by shared prefix, then places the groups jointly
// over the batch (longest-prefix first) so a shared prefix is prefilled once per
// replica rather than scattered. Identical groups are placed before prefix-sharing
// singletons so proven same-prompt co-location anchors the layout.
func assign(entries []*entry, k, minColocateBlocks int, byTokens bool) {
	groups := map[string][]*entry{}
	order := []string{}
	var replicas []fwksched.Endpoint
	for _, e := range entries {
		key := groupKey(e.hashes)
		if key == "" {
			continue
		}
		if _, ok := groups[key]; !ok {
			order = append(order, key)
		}
		groups[key] = append(groups[key], e)
		if replicas == nil {
			replicas = e.pods
		}
	}
	if len(replicas) == 0 {
		return
	}

	shared := sharedPrefixKeys(groups, order, minColocateBlocks)
	var groupUnits, singleUnits []string
	counts := map[string]int{} // prefix-block count per placed unit, computed once
	for _, key := range order {
		switch {
		case len(groups[key]) >= 2:
			groupUnits = append(groupUnits, key)
		case shared[key]:
			singleUnits = append(singleUnits, key)
		default:
			continue
		}
		counts[key] = totalBlocks(groups[key][0].hashes)
	}
	if len(groupUnits) == 0 && len(singleUnits) == 0 {
		return
	}

	// Longest-prefix first within each tier: a longer prefix seeds more blocks in
	// the index, so shorter units match against the richest set of placed blocks.
	sortByPrefixLen(groupUnits, counts)
	sortByPrefixLen(singleUnits, counts)

	// total and load carry one unit: request count, or prefix blocks under token
	// balancing. Token totals sum full per-unit block counts (a loose fair-share
	// target); the shared-prefix discount is applied to load at placement time.
	total := 0
	for key := range counts {
		if byTokens {
			total += counts[key]
		} else {
			total += len(groups[key])
		}
	}
	maxShare := (total + len(replicas) - 1) / len(replicas) // ceil: fair share per replica

	idx := newBatchIndex()
	load := map[string]int{} // per-replica load in the balancing unit (requests or blocks)
	// Groups first, then prefix-sharing singletons: same order as before, without
	// allocating a combined slice.
	for _, key := range groupUnits {
		placeGroup(groups[key], replicas, k, minColocateBlocks, maxShare, counts[key], idx, load, byTokens)
	}
	for _, key := range singleUnits {
		placeGroup(groups[key], replicas, k, minColocateBlocks, maxShare, counts[key], idx, load, byTokens)
	}
}

// sortByPrefixLen orders keys by descending prefix-block count (stable).
func sortByPrefixLen(keys []string, counts map[string]int) {
	slices.SortStableFunc(keys, func(a, b string) int {
		return cmp.Compare(counts[b], counts[a])
	})
}

// placeGroup places one unit (an identical group, or a single prefix-sharing
// request). The first (primary) replica is chosen with the inter-unit prefix
// preference bounded by the fair-share cap; remaining samples (when k caps the
// group) spill to least-loaded replicas. The unit's blocks are then recorded so
// later units can match against it.
func placeGroup(members []*entry, replicas []fwksched.Endpoint, k, minColocateBlocks, maxShare, cost int, idx *batchIndex, load map[string]int, byTokens bool) {
	if len(replicas) == 0 {
		return
	}
	hashes := members[0].hashes // identical group: any member represents it

	var matches map[string]int
	preferMatch := minColocateBlocks > 0
	// Token balancing also needs the shared-prefix length per replica: a replica
	// only prefills the blocks it does not already hold, so its load is charged the
	// remaining cost rather than the full prefix.
	if preferMatch || byTokens {
		matches = idx.longestPrefix(hashes)
	}

	perReplica := map[string]int{} // requests of this unit already on a replica
	i := 0
	for i < len(members) {
		target := pickReplica(replicas, perReplica, k, load, matches, minColocateBlocks, maxShare, preferMatch)
		name := target.GetMetadata().NamespacedName.String()

		run := len(members) - i
		if k != unlimitedPerReplica {
			capLeft := k - perReplica[name]
			if capLeft < 1 {
				capLeft = 1 // overflow: more members than k*replicas, place one and rebalance
			}
			if capLeft < run {
				run = capLeft
			}
		}
		for j := 0; j < run; j++ {
			members[i+j].assigned = target
		}
		firstTouch := perReplica[name] == 0
		perReplica[name] += run
		switch {
		case !byTokens:
			load[name] += run
		case firstTouch:
			// Charge the prefix once per replica this unit lands on, discounting the
			// leading blocks already held there (prefilled by an earlier request).
			inc := cost - matches[name]
			if inc < 0 {
				inc = 0
			}
			load[name] += inc
		}
		i += run
		preferMatch = false // only the primary replica gets the prefix preference
	}

	for _, m := range members {
		if m.assigned != nil {
			idx.add(hashes, m.assigned.GetMetadata().NamespacedName.String())
		}
	}
}

// pickReplica chooses the target replica for the next run of a group. When
// preferMatch is set it first tries the replica sharing the longest prefix (at
// least minColocateBlocks blocks) that still has per-group capacity and is below
// its fair share of the batch; otherwise, or when no such replica exists, it
// falls back to the least batch-loaded replica. The fair-share bound is what
// keeps many prefix-sharing groups from stampeding onto a single replica.
func pickReplica(replicas []fwksched.Endpoint, perReplica map[string]int, k int, load, matches map[string]int, minColocateBlocks, maxShare int, preferMatch bool) fwksched.Endpoint {
	if preferMatch {
		var best fwksched.Endpoint
		var bestName string
		bestMatch := 0
		for _, r := range replicas {
			name := r.GetMetadata().NamespacedName.String()
			if k != unlimitedPerReplica && perReplica[name] >= k {
				continue
			}
			if load[name] >= maxShare {
				continue // at fair share: co-locating here would unbalance the batch
			}
			m := matches[name]
			if m < minColocateBlocks {
				continue
			}
			if best == nil || m > bestMatch || (m == bestMatch && less(r, name, load, best, bestName)) {
				best, bestName, bestMatch = r, name, m
			}
		}
		if best != nil {
			return best
		}
	}
	return pickLeastLoaded(replicas, perReplica, k, load)
}

// pickLeastLoaded returns the least batch-loaded replica with capacity for this
// group, falling back to the overall least-loaded replica when all are at the
// cap.
func pickLeastLoaded(replicas []fwksched.Endpoint, perReplica map[string]int, k int, load map[string]int) fwksched.Endpoint {
	var best fwksched.Endpoint
	var bestName string
	for _, r := range replicas {
		name := r.GetMetadata().NamespacedName.String()
		if k != unlimitedPerReplica && perReplica[name] >= k {
			continue
		}
		if best == nil || less(r, name, load, best, bestName) {
			best, bestName = r, name
		}
	}
	if best != nil {
		return best
	}
	for _, r := range replicas {
		name := r.GetMetadata().NamespacedName.String()
		if best == nil || less(r, name, load, best, bestName) {
			best, bestName = r, name
		}
	}
	return best
}

// less reports whether replica a should be preferred over replica b: lower
// assigned load first, then fewer running requests, then lower name. The
// batch-local load (assigned requests, or prefix blocks under token balancing) is
// the primary signal; running requests only break ties within a batch and
// deliberately differ from the load-aware-scorer's WaitingQueueSize signal, since
// placement balances work already committed in this window rather than queue depth
// observed downstream.
func less(a fwksched.Endpoint, aName string, load map[string]int, b fwksched.Endpoint, bName string) bool {
	if load[aName] != load[bName] {
		return load[aName] < load[bName]
	}
	ra, rb := runningRequests(a), runningRequests(b)
	if ra != rb {
		return ra < rb
	}
	return aName < bName
}

// runningRequests returns the endpoint's running-request count, or 0 when
// metrics are unavailable.
func runningRequests(e fwksched.Endpoint) int {
	if m := e.GetMetrics(); m != nil {
		return m.RunningRequestsSize
	}
	return 0
}
