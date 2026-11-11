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
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

// endpointStaleAfter bounds how long an endpoint from another saturation stage
// remains in the admission view after Saturation stops reporting it.
const endpointStaleAfter = 5 * time.Second

type trackedEndpoint struct {
	ep        fwkdl.Endpoint
	capacity  float64
	stage     string
	updatedAt time.Time
}

func (a *ThunderAgent) endpointCapacity(m *fwkdl.Metrics) float64 {
	if m != nil && m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
		return float64(m.CacheBlockSize) * float64(m.CacheNumBlocks)
	}
	return a.fallbackCapacityTokens
}

// Saturation refreshes the live endpoint view and always returns 0.0.
//
// Returning 0 is deliberate: the flow controller's saturation gate is a
// band-level head-of-line block, and while it reports saturated nothing
// dispatches, including the turns of sessions already admitted. Those turns
// are what complete trajectories and free capacity, so blocking them can
// only stall the pool. The per-session gate lives in Pick instead, where an
// admitted session's turn can be told apart from a new one. This hook is
// used because it receives the live endpoint list each dispatch cycle.
func (a *ThunderAgent) Saturation(ctx context.Context, endpoints []fwkdl.Endpoint) float64 {
	now := a.clock.Now()
	stage := fwkfc.SaturationStageFromContext(ctx)
	seen := make(map[string]bool, len(endpoints))
	var prefillPods []string

	a.mu.Lock()
	for _, endpoint := range endpoints {
		if endpoint == nil {
			continue
		}
		md := endpoint.GetMetadata()
		if md == nil {
			continue
		}
		id := md.ID.String()
		seen[id] = true
		a.endpoints[id] = &trackedEndpoint{
			ep:        endpoint,
			capacity:  a.endpointCapacity(endpoint.GetMetrics()),
			stage:     stage,
			updatedAt: now,
		}
		if role := md.Labels[bylabel.RoleLabel]; !a.prefillSeen[id] && (role == bylabel.RolePrefill || role == bylabel.RoleEncodePrefill) {
			a.prefillSeen[id] = true
			prefillPods = append(prefillPods, id)
		}
	}
	// A pod absent from an unstaged call or from its own stage's call is gone;
	// entries from other stages time out after endpointStaleAfter.
	for id, entry := range a.endpoints {
		if seen[id] {
			continue
		}
		if stage == "" || entry.stage == stage || now.Sub(entry.updatedAt) > endpointStaleAfter {
			a.deleteEndpointLocked(id)
		}
	}
	a.mu.Unlock()

	for _, id := range prefillPods {
		log.FromContext(ctx).V(logutil.DEFAULT).Info("thunder-agent does not support prefill/decode disaggregation: prefill pods enter its admission pool", "pod", id)
	}
	return 0.0
}
