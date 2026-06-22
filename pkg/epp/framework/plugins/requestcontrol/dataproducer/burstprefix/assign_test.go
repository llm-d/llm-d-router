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

	assign(entries, unlimitedPerReplica)

	for _, e := range entries {
		assert.Equal(t, "pod1", assignedName(e), "all samples of one group must co-locate when k=-1")
	}
}

func TestAssign_CapSpreadsEvenly(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4")}
	entries := group(8, []prefixhash.BlockHash{1, 2, 3}, replicas)

	assign(entries, 2)

	assert.Equal(t, map[string]int{"pod1": 2, "pod2": 2, "pod3": 2, "pod4": 2}, counts(entries),
		"k=2 over 8 samples and 4 replicas must place 2 per replica")
}

func TestAssign_CapFillsBeforeSpilling(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4")}
	entries := group(8, []prefixhash.BlockHash{1, 2, 3}, replicas)

	assign(entries, 4)

	// k=4 fills one replica to 4 before using the next; only two replicas used.
	assert.Equal(t, map[string]int{"pod1": 4, "pod2": 4}, counts(entries))
}

func TestAssign_SingletonGetsNoAffinity(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	entries := group(1, []prefixhash.BlockHash{1, 2, 3}, replicas)

	assign(entries, unlimitedPerReplica)

	assert.Nil(t, entries[0].assigned, "a singleton group has no reuse and must not receive an affinity")
}

func TestAssign_EmptyPrefixGetsNoAffinity(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	entries := []*entry{
		{hashes: nil, pods: replicas},
		{hashes: nil, pods: replicas},
	}

	assign(entries, unlimitedPerReplica)

	for _, e := range entries {
		assert.Nil(t, e.assigned, "requests with no prefix must not be grouped")
	}
}

func TestAssign_DistinctGroupsSpreadAcrossReplicas(t *testing.T) {
	replicas := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4")}
	groupA := group(8, []prefixhash.BlockHash{1, 2, 3}, replicas)
	groupB := group(8, []prefixhash.BlockHash{9, 9, 9}, replicas)
	entries := append(append([]*entry{}, groupA...), groupB...)

	assign(entries, unlimitedPerReplica)

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
