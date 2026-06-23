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
	"encoding/binary"
	"sort"
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

// groupKey identifies requests that share an identical prompt prefix. Because
// each block hash chains its predecessor, an identical leading hash sequence
// means an identical prompt prefix. Requests with no prefix return "" and are
// never grouped.
func groupKey(hashes [][]prefixhash.BlockHash) string {
	if totalBlocks(hashes) == 0 {
		return ""
	}
	var b strings.Builder
	var buf [8]byte
	for _, ph := range hashes {
		for _, h := range ph {
			binary.LittleEndian.PutUint64(buf[:], uint64(h))
			b.Write(buf[:])
		}
		b.WriteByte('|')
	}
	return b.String()
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
	for _, ph := range hashes {
		for _, h := range ph {
			holders := b.holders[h]
			if len(holders) == 0 {
				break
			}
			for name := range holders {
				res[name]++
			}
		}
	}
	return res
}

// assign steers each batched request to a replica so prompt-sharing samples
// co-locate. Pass 1 groups requests by identical prefix (samples of one prompt);
// only groups with more than one member receive an affinity (singletons have no
// reuse and are left for other scorers).
//
// Pass 2 places the groups jointly over the whole batch. Groups are placed
// longest-prefix first so they seed the batch index for shorter groups to match
// against. Each group prefers a replica that already holds a group sharing at
// least minColocateBlocks leading blocks AND is still below its fair share of the
// batch (total placed samples / replicas); this prefills a long shared prefix
// once across groups without letting many prefix-sharing groups stampede onto one
// replica. With no eligible match a group seeds the least-loaded replica. Within
// a group, samples fill one replica up to maxPerReplica (k) before spilling to
// the next least-loaded replica; k == -1 places the whole group on one replica.
// minColocateBlocks == 0 disables inter-group co-location (placement is purely
// load-balanced).
func assign(entries []*entry, k, minColocateBlocks int) {
	groups := map[string][]*entry{}
	order := []string{}
	for _, e := range entries {
		key := groupKey(e.hashes)
		if key == "" {
			continue
		}
		if _, ok := groups[key]; !ok {
			order = append(order, key)
		}
		groups[key] = append(groups[key], e)
	}

	// Placeable groups (more than one member) and the total samples they
	// contribute, which sets the per-replica fair-share cap.
	var placeKeys []string
	total := 0
	var replicas []fwksched.Endpoint
	for _, key := range order {
		members := groups[key]
		if len(members) < 2 {
			continue
		}
		placeKeys = append(placeKeys, key)
		total += len(members)
		if replicas == nil {
			replicas = members[0].pods
		}
	}
	if len(placeKeys) == 0 || len(replicas) == 0 {
		return
	}

	// Longest-prefix first: a longer group seeds more blocks in the index, so
	// shorter groups match against the richest set of already-placed prefixes.
	sort.SliceStable(placeKeys, func(i, j int) bool {
		return totalBlocks(groups[placeKeys[i]][0].hashes) > totalBlocks(groups[placeKeys[j]][0].hashes)
	})

	maxShare := (total + len(replicas) - 1) / len(replicas) // ceil: equal samples per replica

	idx := newBatchIndex()
	load := map[string]int{} // batch-wide samples assigned per replica
	for _, key := range placeKeys {
		placeGroup(groups[key], replicas, k, minColocateBlocks, maxShare, idx, load)
	}
}

// placeGroup places one identical-prompt group. The first (primary) replica is
// chosen with the inter-group prefix preference bounded by the fair-share cap;
// remaining samples (when k caps the group) spill to least-loaded replicas. The
// group's blocks are then recorded so later groups can match against it.
func placeGroup(members []*entry, replicas []fwksched.Endpoint, k, minColocateBlocks, maxShare int, idx *batchIndex, load map[string]int) {
	if len(replicas) == 0 {
		return
	}
	hashes := members[0].hashes // identical group: any member represents it
	matches := idx.longestPrefix(hashes)

	perReplica := map[string]int{} // samples of THIS group already on a replica
	preferMatch := minColocateBlocks > 0
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
		perReplica[name] += run
		load[name] += run
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
// cap. Ties break by running requests then by name for deterministic placement.
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

// less reports whether replica a should be preferred over replica b: fewer
// assigned samples first, then fewer running requests, then lower name.
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
