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
	"context"
	"strconv"
	"sync/atomic"
	"time"

	"k8s.io/utils/clock"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// peerDemandMaxAgeIntervals is how many publish intervals a view stays current without a refresh.
const peerDemandMaxAgeIntervals = 5

// peerDemandView caches the requests queued per priority band on peer replicas, counting the ones dispatched and
// not yet on an endpoint as queued. The publisher refreshes it after each publish, so the dispatch cycle reads it
// without calling the syncer.
type peerDemandView struct {
	syncer fwkdl.CrossReplicaSyncer
	// spec is the band demand spec with a zero local value, so Get returns the peer aggregate alone.
	spec     fwkdl.CrossReplicaSpec
	registry bandOccupancy
	scope    string
	clock    clock.PassiveClock
	interval time.Duration
	maxAge   time.Duration
	current  atomic.Pointer[peerDemandSnapshot]
}

type peerDemandSnapshot struct {
	at     time.Time
	queued map[int]uint64
}

var _ flowcontrol.PeerDemand = &peerDemandView{}

func newPeerDemandView(
	syncer fwkdl.CrossReplicaSyncer,
	source *crossReplicaSource,
	registry bandOccupancy,
	scope string,
	interval time.Duration,
	clk clock.PassiveClock,
) *peerDemandView {
	spec := source.spec
	spec.Read = func(string) fwkdl.Cloneable { return &bandDemand{} }
	return &peerDemandView{
		syncer:   syncer,
		spec:     spec,
		registry: registry,
		scope:    scope,
		clock:    clk,
		interval: interval,
		maxAge:   peerDemandMaxAgeIntervals * interval,
	}
}

// refresh reads the peer aggregate of every band. On a read error the previous snapshot is kept and ages out.
func (v *peerDemandView) refresh(ctx context.Context) error {
	priorities := v.registry.AllOrderedPriorityLevels()
	queued := make(map[int]uint64, len(priorities))
	for _, priority := range priorities {
		value, ok, err := v.syncer.Get(ctx, v.spec, strconv.Itoa(priority))
		if err != nil {
			return err
		}
		if d, isDemand := value.(*bandDemand); ok && isDemand {
			queued[priority] = d.Requests + d.Dispatching
		}
	}
	v.current.Store(&peerDemandSnapshot{at: v.clock.Now(), queued: queued})
	for priority, n := range queued {
		metrics.RecordFlowControlPeerQueuedRequests(strconv.Itoa(priority), v.scope, float64(n))
	}
	return nil
}

// QueuedRequests implements flowcontrol.PeerDemand.
func (v *peerDemandView) QueuedRequests(priority int) (uint64, bool) {
	s := v.current.Load()
	if s == nil || v.clock.Since(s.at) > v.maxAge {
		return 0, false
	}
	return s.queued[priority], true
}

// SyncInterval implements flowcontrol.PeerDemand.
func (v *peerDemandView) SyncInterval() time.Duration {
	return v.interval
}
