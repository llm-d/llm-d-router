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
	"sort"
	"time"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Saturation refreshes the per-pod fit view and always returns 0.0.
//
// The flow controller's saturation gate is a band-level head-of-line block:
// while it reports saturated, no request dispatches, including the turns of
// programs already admitted. Those turns are what complete trajectories and
// free capacity, and their footprint is already counted in the working set,
// so blocking them can only stall the pool. ThunderAgent therefore gates in
// Pick instead, where it can tell an admitted program's turn from a new
// program's first turn: admitted programs always dispatch, new programs
// dispatch only when their projected footprint fits a pod.
//
// This hook remains the plugin's per-cycle feed of the endpoint pool. Each
// call updates capacity and working set for the pods it was given, so
// per-stage calls (prefill, decode) each maintain their own pods. Pods no
// longer reported by any call age out of the fit view.
//
// Engine-level overload protection is not this detector's job: pair the
// scheduling profile with a utilization filter and the kv-cache and queue
// scorers.
func (a *ThunderAgent) Saturation(ctx context.Context, endpoints []datalayer.Endpoint) float64 {
	now := time.Now()

	t := a.table
	t.mu.Lock()
	loads := t.podLoads(now)

	var bound, unbound int
	for _, st := range t.programs {
		if st.podName == "" {
			unbound++
			continue
		}
		bound++
	}
	a.metrics.programs.WithLabelValues("bound").Set(float64(bound))
	a.metrics.programs.WithLabelValues("unbound").Set(float64(unbound))

	logger := log.FromContext(ctx)
	for _, endpoint := range endpoints {
		md := endpoint.GetMetadata()
		if md == nil {
			continue
		}
		pod := md.ID.String()
		capTokens, source := a.endpointCapacity(endpoint)
		load := loads[pod]
		tokens := load.tokens + a.bufferTokensPerProgram*float64(load.programs)

		if a.shedIdle > 0 && capTokens > 0 {
			if ceiling := capTokens * a.utilThreshold; tokens > ceiling {
				tokens = a.shedFromPodLocked(ctx, pod, tokens, ceiling, now)
			}
		}

		t.snapshot[pod] = &podSnapshot{capacity: capTokens, tokens: tokens, updatedAt: now}

		util := 1.0
		if capTokens > 0 {
			util = tokens / capTokens
		}
		a.metrics.podUtilization.WithLabelValues(pod).Set(util)
		a.metrics.setPodCapacity(pod, source, capTokens)
		logger.V(logutil.TRACE).Info("thunderagent fit view",
			"pod", pod, "util", util, "capacity", capTokens, "capacitySource", source)
	}
	for pod, snap := range t.snapshot {
		if now.Sub(snap.updatedAt) > snapshotStaleAfter {
			delete(t.snapshot, pod)
			a.metrics.forgetPod(pod)
		}
	}
	t.mu.Unlock()

	return 0.0
}

// shedFromPodLocked unbinds the pod's idle programs, smallest footprint
// first, until its working set is back under the fit ceiling, and returns
// the reduced working set. A shed program keeps its record (committed tokens,
// parent, estimator history) but stops counting against any pod and loses its
// REASONING bypass: its next turn re-enters admission as a new program, so
// the engine is free to evict its KV while it queues. Programs with a request
// in flight or idle for less than shedIdle are never shed. Callers must hold
// t.mu.
func (a *ThunderAgent) shedFromPodLocked(ctx context.Context, pod string, tokens, ceiling float64, now time.Time) float64 {
	t := a.table
	type candidate struct {
		id        string
		st        *program
		footprint float64
	}
	var candidates []candidate
	for id, st := range t.programs {
		if st.podName != pod || st.inflightTokens > 0 {
			continue
		}
		idleSince := st.lastResponseAt
		if idleSince.IsZero() {
			idleSince = st.lastActivity
		}
		if now.Sub(idleSince) < a.shedIdle {
			continue
		}
		candidates = append(candidates, candidate{
			id: id, st: st, footprint: t.programTokens(st, now) + a.bufferTokensPerProgram,
		})
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].footprint < candidates[j].footprint })

	logger := log.FromContext(ctx)
	shed := 0
	for _, c := range candidates {
		if tokens <= ceiling {
			break
		}
		c.st.podName = ""
		tokens -= c.footprint
		shed++
		t.shedsTotal++
		a.metrics.sheds.Inc()
	}
	if shed > 0 {
		logger.Info("thunderagent.shed", "pod", pod, "shed", shed,
			"working_set", tokens, "ceiling", ceiling)
	}
	return tokens
}
