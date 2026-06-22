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

// assign steers each batched request to a replica so samples sharing a prompt
// co-locate. Requests are grouped by identical prefix; only groups with more
// than one member receive an affinity (singletons have no reuse to exploit and
// are left for other scorers). Within a group, samples fill one replica up to
// maxPerReplica (k) before spilling to the next least-loaded replica; k == -1
// places the whole group on one replica. Group placement is balanced by the
// batch-wide assignment count so distinct groups spread across replicas.
func assign(entries []*entry, k int) {
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

	// load is the batch-wide count of samples assigned per replica, used to
	// spread groups across replicas.
	load := map[string]int{}
	for _, key := range order {
		members := groups[key]
		if len(members) < 2 {
			continue
		}
		assignGroup(members, members[0].pods, k, load)
	}
}

// assignGroup places the members of one group across replicas honoring the
// per-replica cap k and updating the batch-wide load counter.
func assignGroup(members []*entry, replicas []fwksched.Endpoint, k int, load map[string]int) {
	if len(replicas) == 0 {
		return
	}
	perReplica := map[string]int{} // samples of THIS group already on a replica

	i := 0
	for i < len(members) {
		target := pickReplica(replicas, perReplica, k, load)
		name := target.GetMetadata().NamespacedName.String()

		// Fill this replica up to its remaining capacity before moving on, so a
		// group concentrates rather than scatters.
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
	}
}

// pickReplica returns the least batch-loaded replica that still has capacity for
// this group. When every replica is at the cap (more members than k*replicas)
// it falls back to the least-loaded replica ignoring the cap. Ties break by
// running requests then by name for deterministic placement.
func pickReplica(replicas []fwksched.Endpoint, perReplica map[string]int, k int, load map[string]int) fwksched.Endpoint {
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
	// All replicas at cap: ignore the cap and balance by load.
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
