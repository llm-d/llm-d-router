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

package thunderagent

import (
	"context"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// Saturation refreshes the pod ledger from the live endpoint list, runs the
// pause sweep, and always returns 0.0.
//
// Returning 0 is deliberate: the flow controller's saturation gate is a
// band-level head-of-line block, and while it reports saturated nothing
// dispatches, including the turns of sessions already admitted. Those turns
// are what complete trajectories and free capacity, so blocking them can
// only stall the pool. The per-session gate lives in Pick instead, where an
// admitted session's turn can be told apart from a new one. This hook is
// used because it is the only flow-control callback that receives the
// endpoint list each dispatch cycle.
func (a *ThunderAgent) Saturation(ctx context.Context, endpoints []datalayer.Endpoint) float64 {
	now := time.Now()
	m := a.mgr

	m.mu.Lock()
	m.maintainLocked(now)
	endpointStates := make([]*endpointState, 0, len(endpoints))
	for _, endpoint := range endpoints {
		md := endpoint.GetMetadata()
		if md == nil {
			continue
		}
		endpointStates = append(endpointStates, m.ensureEndpointLocked(md.ID.String(), a.endpointCapacity(endpoint.GetMetrics()), now))
	}
	pausedNow := 0
	if a.pauseSweep == 0 || now.Sub(m.lastSweep) >= a.pauseSweep {
		m.lastSweep = now
		for _, p := range endpointStates {
			ceiling := p.capacity * a.utilThreshold
			if undecayed, _ := p.occupancy(now, m.halfLife); undecayed > ceiling {
				pausedNow += p.pauseSmallest(undecayed, ceiling)
			}
		}
	}
	// A pod the endpoint list has stopped reporting is gone: unbind its
	// sessions so their next turn re-enters as new, and drop the entry.
	for id, p := range m.endpoints {
		if now.Sub(p.updatedAt) <= endpointStaleAfter {
			continue
		}
		for sid, s := range p.sessions {
			s.endpoint = nil
			s.paused = false
			delete(p.sessions, sid)
		}
		delete(m.endpoints, id)
	}
	m.mu.Unlock()

	if pausedNow > 0 {
		a.metrics.pauses.Add(float64(pausedNow))
		log.FromContext(ctx).V(logutil.DEBUG).Info("thunderagent.pause_sweep", "paused", pausedNow)
	}
	return 0.0
}
