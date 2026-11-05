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

package groupsched

// THE POD SIDE: the detector's answer (and the capacity it learns), the endpoint life cycle, and
// the startup gate that refuses to run without the plugins this one reads.

import (
	"context"
	"fmt"
	"sort"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

// Saturation answers the flow controller's pool gate: 0.0 (NOT saturated), always. It is also
// where capacity is learned, from the candidates it is handed — this runs about once a millisecond,
// so the path is built to touch neither the scheduler's mutex nor the logger once an engine is
// known (see learnFromCandidates).
func (r *GroupScheduler) Saturation(_ context.Context, candidates []fwkdl.Endpoint) float64 {
	r.learnFromCandidates(candidates)
	return 0
}

// learnFromCandidates takes capacity from the LIVE candidate set on Saturation's own path, which is
// the only place it now happens. It is written to stay off the scheduler's mutex: the learned and
// rejected sets decide first (capMu only), the metrics are read before any lock is taken, and a
// candidate with no geometry yet costs nothing.
func (r *GroupScheduler) learnFromCandidates(cands []fwkdl.Endpoint) {
	for _, e := range cands {
		if e == nil {
			continue
		}
		key := endpointKey(e.GetMetadata())
		if key == "" {
			continue
		}
		r.capMu.Lock()
		_, done := r.learned[key]
		_, bad := r.rejected[key]
		at := r.epoch[key] // the key's generation BEFORE the learn: see below
		r.capMu.Unlock()
		if done || bad {
			continue // nothing to do, and no lock taken
		}
		m := e.GetMetrics()
		if m == nil || m.CacheBlockSize <= 0 || m.CacheNumBlocks <= 0 {
			continue // the geometry has not arrived: don't wake the scheduler for it
		}
		// The one scheduler lock on this path, per engine, ONCE. capMu is NOT held across it, and the
		// scheduler is asked what it now holds BEFORE capMu is taken again.
		err := r.learnCapacity(key, m)
		learned := err == nil && r.sched.HasCapacityFor(key)
		if r.learnHook != nil {
			r.learnHook(key) // test-only seam: the window this race lives in
		}
		r.capMu.Lock()
		// DID THE ENDPOINT DIE WHILE WE WERE LEARNING IT? The epoch moves on a delete, so a moved
		// epoch means the group this learn just created belongs to an endpoint that is gone. It must
		// not be marked (the re-added endpoint has to learn again) and it must be TORN DOWN here,
		// because the delete's own teardown ran before the group existed — nothing else would.
		stale := r.epoch[key] != at
		switch {
		case stale:
			// no mark of either kind: the key returns to "never looked at"
		case err != nil:
			r.rejected[key] = struct{}{}
		case learned:
			r.learned[key] = struct{}{}
		}
		r.capMu.Unlock()
		if stale {
			// capMu is NOT held here: RemoveEngine takes the scheduler's own lock
			r.sched.RemoveEngine(key, r.now())
			r.logger.V(logutil.DEBUG).Info("group-scheduler: capacity learn discarded — the endpoint was deleted while it was being learned",
				"engine", key)
			continue
		}
		if err != nil {
			r.logger.V(logutil.DEFAULT).Info("group-scheduler: CAPACITY REJECTED from the live detector",
				"engine", r.sched.GroupLabel(key), "err", err.Error())
		}
	}
}

// learnCapacity takes ONE engine's measured capacity from a metrics snapshot, once per engine.
func (r *GroupScheduler) learnCapacity(engine string, m *fwkdl.Metrics) error {
	if m == nil || m.CacheBlockSize <= 0 || m.CacheNumBlocks <= 0 {
		return nil
	}
	if r.sched.HasCapacityFor(engine) {
		return nil
	}
	if err := r.sched.SetNumBlocksFor(engine, m.CacheBlockSize, m.CacheNumBlocks); err != nil {
		r.logBlockSizeMismatches()
		return err
	}
	r.logger.V(logutil.DEFAULT).Info("group-scheduler: capacity learned",
		"engine", r.sched.GroupLabel(engine),
		"cacheBlockSize", m.CacheBlockSize, "numGpuBlocks", m.CacheNumBlocks,
		"budgetBlocks", int(r.cfg.GroupBudgetFraction*float64(m.CacheNumBlocks)))
	return nil
}

// checkBlockSize is the loud mismatch failure, and nothing else: a request that lands on an
// engine whose measured block size disagrees with the configured one FAILS here rather than being
// costed in the wrong unit. It learns no capacity and returns nil unless the two disagree.
func (r *GroupScheduler) checkBlockSize(engine string, m *fwkdl.Metrics) error {
	if m == nil || m.CacheBlockSize <= 0 || m.CacheNumBlocks <= 0 {
		return nil
	}
	// An agreeing engine needs nothing from this call, and asking the scheduler about it would
	// create its group with no capacity — a state that should not exist.
	if m.CacheBlockSize == r.cfg.BlockSizeTokens {
		return nil
	}
	if r.sched.HasCapacityFor(engine) {
		return nil
	}
	if err := r.sched.SetNumBlocksFor(engine, m.CacheBlockSize, 0); err != nil {
		r.logBlockSizeMismatches()
		return err
	}
	return nil
}

// logBlockSizeMismatches names EVERY mismatched engine, not just the one that failed first: a
// single engine's misconfigured block size must not hide behind another's.
func (r *GroupScheduler) logBlockSizeMismatches() {
	for bad, berr := range r.sched.BlockSizeMismatches() {
		r.logger.V(logutil.DEFAULT).Info("group-scheduler: BLOCK SIZE MISMATCH — this engine's measured block size disagrees with the configured one, so its budget would be in the wrong unit; fix --block-size or blockSizeTokens",
			"engine", bad, "err", berr.Error())
	}
}

// Extract follows the endpoint lifecycle: an added or updated endpoint is REMEMBERED (the gauge
// reports it while its capacity is unknown) and a deleted one tears its group down. Capacity is
// not learned here — the event carries no metrics — but the block-size mismatch is still loud.
func (r *GroupScheduler) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil {
		return nil
	}
	switch event.Type {
	case fwkdl.EventDelete:
		r.forgetEndpoint(event.Endpoint.GetMetadata())
		return nil
	case fwkdl.EventAddOrUpdate:
		// No capacity learning here: the creation event carries no metrics yet (it fires before the
		// first scrape); capacity is learned from the live candidates.
		r.rememberEndpoint(event.Endpoint.GetMetadata())
		return nil
	}
	return nil
}

// RegisterDependencies runs after every plugin is constructed: the STARTUP fail-closed gate lives
// here, because a plugin cannot see its siblings at Factory time. No thread identity — or one
// whose block length disagrees with ours — and the process refuses to start rather than running a
// degraded queue key or charging every request the wrong cost.
func (r *GroupScheduler) RegisterDependencies(reg fwkdl.Registrar) error {
	if err := r.validateThreadProducer(); err != nil {
		return err
	}
	return reg.Register(fwkdl.PendingRegistration{
		Owner:      r.typedName,
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  r,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// rememberEndpoint adds an endpoint to the KNOWN set, which is what capacity_missing reports on.
func (r *GroupScheduler) rememberEndpoint(md *fwkdl.EndpointMetadata) {
	if md == nil {
		return
	}
	key := endpointKey(md)
	r.capMu.Lock()
	defer r.capMu.Unlock()
	if _, ok := r.endpoints[key]; !ok {
		// the HANDLE is not stored on purpose: a remembered endpoint's metrics are a construction-time
		// clone, so the live candidate set is the only input that sees the geometry arrive
		r.endpoints[key] = struct{}{}
	}
}

// forgetEndpoint drops an endpoint from the KNOWN set and tears its group down.
func (r *GroupScheduler) forgetEndpoint(md *fwkdl.EndpointMetadata) {
	if md == nil {
		return
	}
	key := endpointKey(md)
	r.capMu.Lock()
	delete(r.endpoints, key)
	// THE MARKS ARE PART OF THE DELETION. A pod is deleted on every not-ready state and the same
	// name comes back on the next ready one, so a surviving "learned" would describe a group that no
	// longer exists: the re-added endpoint would never be learned again and would run with no seats
	// (FCFS at N=1) for the rest of the run. Clearing them, and bumping the epoch, is what makes a
	// flap a re-learn rather than a permanent hole.
	delete(r.learned, key)
	delete(r.rejected, key)
	r.epoch[key]++
	r.capMu.Unlock()
	// tear the engine's group down: its threads are exited, their seats and bindings dropped, its
	// budget/index gone (leaving charges on a dead engine poisons every later budget decision on a
	// scale-down).
	r.sched.RemoveEngine(key, r.now())
}

// knownEndpoints is the sorted KNOWN endpoint set (the capacity_missing gauge reports it).
func (r *GroupScheduler) knownEndpoints() []string {
	r.capMu.Lock()
	keys := make([]string, 0, len(r.endpoints))
	for key := range r.endpoints {
		keys = append(keys, key)
	}
	r.capMu.Unlock()
	sort.Strings(keys)
	return keys
}

// endpointKey is the engine id every capacity path speaks: the endpoint's namespaced name.
func endpointKey(md *fwkdl.EndpointMetadata) string {
	if md == nil {
		return ""
	}
	if md.ID.Name != "" {
		return md.ID.String()
	}
	return fmt.Sprint(md)
}
