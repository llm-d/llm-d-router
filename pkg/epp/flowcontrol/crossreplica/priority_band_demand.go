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

package crossreplica

import (
	"strconv"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

const priorityBandDemandStateKeyPrefix = "flowcontrol-priorityband-demand:"

// bandOccupancy is the subset of the flow registry the priority band source reads.
type bandOccupancy interface {
	AllOrderedPriorityLevels() []int
	CapacitySnapshot(priority int) (contracts.CapacitySnapshot, error)
}

// bandDemand is the value each replica publishes for one priority band. Fields are exported for gob.
type bandDemand struct {
	Requests uint64
	Bytes    uint64
}

func (d *bandDemand) Clone() fwkdl.Cloneable {
	c := *d
	return &c
}

// newPriorityBandDemandSource shares the queued requests and bytes of every priority band, including
// idle ones so a drained backlog is visible to peers on the next cycle. Bands are identified by
// priority value, so the scope must identify one EPP deployment.
func newPriorityBandDemandSource(registry bandOccupancy, scope string) *crossReplicaSource {
	return &crossReplicaSource{
		spec: fwkdl.CrossReplicaSpec{
			StateKey: fwkdl.StateKey(priorityBandDemandStateKeyPrefix + scope),
			Read: func(id string) fwkdl.Cloneable {
				priority, err := strconv.Atoi(id)
				if err != nil {
					return &bandDemand{}
				}
				snapshot, err := registry.CapacitySnapshot(priority)
				if err != nil {
					return &bandDemand{}
				}
				return &bandDemand{Requests: snapshot.Band.Len, Bytes: snapshot.Band.ByteSize}
			},
			Aggregate: aggregateBandDemand,
		},
		ids: func() []string {
			priorities := registry.AllOrderedPriorityLevels()
			ids := make([]string, len(priorities))
			for i, priority := range priorities {
				ids[i] = strconv.Itoa(priority)
			}
			return ids
		},
	}
}

func aggregateBandDemand(values []any) any {
	total := &bandDemand{}
	for _, v := range values {
		if d, ok := v.(*bandDemand); ok {
			total.Requests += d.Requests
			total.Bytes += d.Bytes
		}
	}
	return total
}
