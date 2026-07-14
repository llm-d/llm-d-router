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
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
)

func testEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes())
}

func assignedName(e *entry) string {
	if e.assigned == nil {
		return ""
	}
	return e.assigned.GetMetadata().NamespacedName.Name
}

// group builds n entries sharing one prompt prefix over the given replicas.
func group(n int, prefix []prefixhash.BlockHash, replicas []fwksched.Endpoint) []*entry {
	entries := make([]*entry, n)
	for i := range entries {
		entries[i] = &entry{hashes: [][]prefixhash.BlockHash{prefix}, pods: replicas}
	}
	return entries
}

// concat flattens groups into one entry slice in order.
func concat(groups ...[]*entry) []*entry {
	n := 0
	for _, g := range groups {
		n += len(g)
	}
	all := make([]*entry, 0, n)
	for _, g := range groups {
		all = append(all, g...)
	}
	return all
}

func counts(entries []*entry) map[string]int {
	c := map[string]int{}
	for _, e := range entries {
		c[assignedName(e)]++
	}
	return c
}

func TestAssign_UnlimitedColocatesWholeGroup(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4")}
	entries := group(8, []prefixhash.BlockHash{1, 2, 3}, replicas)

	assign(entries, unlimitedPerReplica, 0, false)

	for _, e := range entries {
		assert.Equal(t, "pod1", assignedName(e), "all samples of one group must co-locate when k=-1")
	}
}

func TestAssign_CapSpreadsEvenly(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4")}
	entries := group(8, []prefixhash.BlockHash{1, 2, 3}, replicas)

	assign(entries, 2, 0, false)

	assert.Equal(t, map[string]int{"pod1": 2, "pod2": 2, "pod3": 2, "pod4": 2}, counts(entries),
		"k=2 over 8 samples and 4 replicas must place 2 per replica")
}

func TestAssign_CapFillsBeforeSpilling(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4")}
	entries := group(8, []prefixhash.BlockHash{1, 2, 3}, replicas)

	assign(entries, 4, 0, false)

	// k=4 fills one replica to 4 before using the next; only two replicas used.
	assert.Equal(t, map[string]int{"pod1": 4, "pod2": 4}, counts(entries))
}

func TestAssign_SingletonGetsNoAffinity(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	entries := group(1, []prefixhash.BlockHash{1, 2, 3}, replicas)

	assign(entries, unlimitedPerReplica, 0, false)

	assert.Nil(t, entries[0].assigned, "a singleton group has no reuse and must not receive an affinity")
}

func TestAssign_EmptyPrefixGetsNoAffinity(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	entries := []*entry{
		{hashes: nil, pods: replicas},
		{hashes: nil, pods: replicas},
	}

	assign(entries, unlimitedPerReplica, 0, false)

	for _, e := range entries {
		assert.Nil(t, e.assigned, "requests with no prefix must not be grouped")
	}
}

func TestAssign_DistinctGroupsSpreadAcrossReplicas(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4")}
	groupA := group(8, []prefixhash.BlockHash{1, 2, 3}, replicas)
	groupB := group(8, []prefixhash.BlockHash{9, 9, 9}, replicas)
	entries := append(append([]*entry{}, groupA...), groupB...)

	assign(entries, unlimitedPerReplica, 0, false)

	// Each group co-locates, and the second group lands on a different, less
	// loaded replica than the first.
	assert.Equal(t, "pod1", assignedName(groupA[0]))
	assert.Equal(t, "pod2", assignedName(groupB[0]))
	for _, e := range groupA {
		assert.Equal(t, "pod1", assignedName(e))
	}
	for _, e := range groupB {
		assert.Equal(t, "pod2", assignedName(e))
	}
}

func TestAssign_SharedPrefixFamiliesColocateWithinFairShare(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	// Two prefix families of two groups each: famX groups share leading {1, 2},
	// famY groups share leading {7, 8}.
	x1 := group(2, []prefixhash.BlockHash{1, 2, 3}, replicas)
	x2 := group(2, []prefixhash.BlockHash{1, 2, 4}, replicas)
	y1 := group(2, []prefixhash.BlockHash{7, 8, 5}, replicas)
	y2 := group(2, []prefixhash.BlockHash{7, 8, 6}, replicas)
	entries := concat(x1, x2, y1, y2)

	assign(entries, unlimitedPerReplica, 2, false)

	// Each family co-locates onto one replica, the two families land on
	// different replicas, and the batch stays balanced (4 samples per replica).
	assert.Equal(t, assignedName(x1[0]), assignedName(x2[0]), "groups sharing >= minColocateBlocks leading blocks co-locate")
	assert.Equal(t, assignedName(y1[0]), assignedName(y2[0]), "groups sharing >= minColocateBlocks leading blocks co-locate")
	assert.NotEqual(t, assignedName(x1[0]), assignedName(y1[0]), "distinct families spread across replicas")
	assert.Equal(t, map[string]int{"pod1": 4, "pod2": 4}, counts(entries))
}

func TestAssign_FairShareSpreadsPrefixSharingGroups(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4")}
	// Four groups all sharing leading {1, 2}. Pure longest-match would pile them
	// onto one replica; the fair-share cap must spread them instead.
	g1 := group(2, []prefixhash.BlockHash{1, 2, 3}, replicas)
	g2 := group(2, []prefixhash.BlockHash{1, 2, 4}, replicas)
	g3 := group(2, []prefixhash.BlockHash{1, 2, 5}, replicas)
	g4 := group(2, []prefixhash.BlockHash{1, 2, 6}, replicas)
	entries := concat(g1, g2, g3, g4)

	assign(entries, unlimitedPerReplica, 2, false)

	assert.Equal(t, map[string]int{"pod1": 2, "pod2": 2, "pod3": 2, "pod4": 2}, counts(entries),
		"the fair-share cap must spread prefix-sharing groups, not stampede them onto one replica")
}

func TestAssign_SingletonAttachesToOverlappingGroup(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	// An identical group plus a single request that overlaps its prefix but is
	// not identical (shares leading {1,2} / {7,8}).
	gx := group(2, []prefixhash.BlockHash{1, 2, 3}, replicas)
	sx := group(1, []prefixhash.BlockHash{1, 2, 8}, replicas)
	gy := group(2, []prefixhash.BlockHash{7, 8, 5}, replicas)
	sy := group(1, []prefixhash.BlockHash{7, 8, 9}, replicas)
	entries := concat(gx, sx, gy, sy)

	assign(entries, unlimitedPerReplica, 2, false)

	// The identical group stays whole, and the overlapping singleton attaches to
	// the replica holding the group it shares a prefix with.
	assert.Equal(t, assignedName(gx[0]), assignedName(gx[1]), "an identical group is never broken")
	assert.Equal(t, assignedName(gx[0]), assignedName(sx[0]), "a prefix-sharing singleton attaches to the overlapping group")
	assert.Equal(t, assignedName(gy[0]), assignedName(sy[0]), "a prefix-sharing singleton attaches to the overlapping group")
	assert.Equal(t, map[string]int{"pod1": 3, "pod2": 3}, counts(entries))
}

func TestAssign_LoneSingletonGetsNoAffinity(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	g := group(2, []prefixhash.BlockHash{1, 2, 3}, replicas)
	lone := group(1, []prefixhash.BlockHash{5, 5, 5}, replicas) // overlaps nothing
	entries := concat(g, lone)

	assign(entries, unlimitedPerReplica, 2, false)

	assert.Nil(t, lone[0].assigned, "a request overlapping no other must keep no affinity")
	assert.Equal(t, assignedName(g[0]), assignedName(g[1]), "the identical group still co-locates")
}

func TestAssign_OverlappingSingletonsColocateWithoutGroups(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	// No identical duplicates at all: four distinct requests sharing leading
	// blocks {1,2} (the RAG / shared-system-prompt burst).
	s1 := group(1, []prefixhash.BlockHash{1, 2, 3}, replicas)
	s2 := group(1, []prefixhash.BlockHash{1, 2, 4}, replicas)
	s3 := group(1, []prefixhash.BlockHash{1, 2, 5}, replicas)
	s4 := group(1, []prefixhash.BlockHash{1, 2, 6}, replicas)
	entries := concat(s1, s2, s3, s4)

	assign(entries, unlimitedPerReplica, 2, false)

	for _, e := range entries {
		assert.NotNil(t, e.assigned, "overlapping singletons receive an affinity even without identical groups")
	}
	assert.Equal(t, map[string]int{"pod1": 2, "pod2": 2}, counts(entries),
		"overlapping singletons co-locate but stay balanced under the fair-share cap")
}

func TestAssign_SharedPrefixBelowThresholdDoesNotColocate(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	// Both families share only 2 leading blocks, below minColocateBlocks=3.
	x1 := group(2, []prefixhash.BlockHash{1, 2, 3}, replicas)
	x2 := group(2, []prefixhash.BlockHash{1, 2, 4}, replicas)
	y1 := group(2, []prefixhash.BlockHash{7, 8, 5}, replicas)
	y2 := group(2, []prefixhash.BlockHash{7, 8, 6}, replicas)
	entries := concat(x1, x2, y1, y2)

	assign(entries, unlimitedPerReplica, 3, false)

	// The shared prefix falls short of the threshold, so groups are placed purely
	// by load and a family is not kept together.
	assert.NotEqual(t, assignedName(x1[0]), assignedName(x2[0]),
		"a shared prefix below minColocateBlocks must not co-locate")
	assert.Equal(t, map[string]int{"pod1": 4, "pod2": 4}, counts(entries))
}

func TestAssign_SingleReplicaPlacesEverything(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1")}
	// Distinct groups that would normally spread across replicas; with a single
	// replica the fair-share and load logic has nowhere else to send them.
	groupA := group(4, []prefixhash.BlockHash{1, 2, 3}, replicas)
	groupB := group(4, []prefixhash.BlockHash{9, 9, 9}, replicas)
	entries := concat(groupA, groupB)

	assign(entries, unlimitedPerReplica, 2, false)

	for _, e := range entries {
		assert.Equal(t, "pod1", assignedName(e), "with a single replica every request must land on it")
	}
}

func TestAssign_BalanceByTokensSpreadsLargePrefix(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	// One long-prefix unit and two short-prefix units with distinct prefixes.
	// Balancing by request count puts a short unit on the same replica as the long
	// one (block-imbalanced); balancing by tokens keeps the long unit alone and
	// packs the short ones onto the other replica.
	long := group(2, []prefixhash.BlockHash{1, 2, 3, 4}, replicas)
	short1 := group(2, []prefixhash.BlockHash{5}, replicas)
	short2 := group(2, []prefixhash.BlockHash{6}, replicas)

	byReq := concat(long, short1, short2)
	assign(byReq, unlimitedPerReplica, 0, false)
	assert.Equal(t, assignedName(long[0]), assignedName(short2[0]),
		"balancing by requests colocates a short unit with the long-prefix unit")

	long = group(2, []prefixhash.BlockHash{1, 2, 3, 4}, replicas)
	short1 = group(2, []prefixhash.BlockHash{5}, replicas)
	short2 = group(2, []prefixhash.BlockHash{6}, replicas)
	byTok := concat(long, short1, short2)
	assign(byTok, unlimitedPerReplica, 0, true)
	assert.NotEqual(t, assignedName(long[0]), assignedName(short1[0]),
		"balancing by tokens keeps the long-prefix unit off the replica holding the short ones")
	assert.Equal(t, assignedName(short1[0]), assignedName(short2[0]),
		"the short units pack onto one replica under token balancing")
}

func TestAssign_BalanceByTokensDiscountsSharedPrefix(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	// Three units sharing a 9-block leading prefix. Charging each its full 10-block
	// prefix would push the third off the shared replica once the fair-share cap is
	// hit; discounting the already-prefilled 9 blocks keeps all three colocated so
	// the shared prefix is prefilled once.
	a1 := group(2, []prefixhash.BlockHash{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, replicas)
	a2 := group(2, []prefixhash.BlockHash{1, 2, 3, 4, 5, 6, 7, 8, 9, 11}, replicas)
	a3 := group(2, []prefixhash.BlockHash{1, 2, 3, 4, 5, 6, 7, 8, 9, 12}, replicas)
	entries := concat(a1, a2, a3)

	assign(entries, unlimitedPerReplica, 9, true)

	assert.Equal(t, assignedName(a1[0]), assignedName(a2[0]),
		"units sharing a prefix colocate when the shared prefill is not double-charged")
	assert.Equal(t, assignedName(a2[0]), assignedName(a3[0]),
		"the shared-prefix discount keeps the third unit on the shared replica")
}

func TestAssign_OverflowGuardBalancesBeyondCap(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	// One group of 8 with k=2 over 2 replicas: k*replicas = 4 < 8, so the
	// per-replica cap cannot hold the whole group. The capLeft overflow guard must
	// still place every member and keep the batch balanced.
	entries := group(8, []prefixhash.BlockHash{1, 2, 3}, replicas)

	assign(entries, 2, 0, false)

	for _, e := range entries {
		assert.NotNil(t, e.assigned, "the overflow guard must still assign every member")
	}
	assert.Equal(t, map[string]int{"pod1": 4, "pod2": 4}, counts(entries),
		"members beyond k*replicas must rebalance evenly rather than drop")
}
