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

// ENDPOINT FLAPS — the delete and re-add of the SAME endpoint name.
//
// The reconciler deletes a pod on every not-ready state (a readiness probe flap, a restart in place,
// a drain that is cancelled) and the same name comes back on the next ready one. Capacity is learned
// once per engine and then remembered, so the deletion has to take the memory with it: otherwise the
// re-added endpoint is "already learned", is never learned again, has no group and no seats, and the
// engine serves FCFS for the rest of the run — at N=1 that is the whole pool.
//
// The three tests here are the reproducer, and the two orders of the race between a delete and a
// capacity learn that is already in flight. Both race orders must leave NEITHER a learned mark
// without a group NOR a group for a deleted endpoint (a phantom: nothing will ever delete it again).

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// newBarePlugin is a plugin whose engines have NO capacity yet: the flap tests need the learn to
// be the first thing that happens to the engine, which a pre-seeded fixture would hide.
func newBarePlugin(t *testing.T) *GroupScheduler {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	require.NoError(t, err)
	return &GroupScheduler{typedName: fwkplugin.TypedName{Type: PluginType, Name: "gs"},
		cfg: defaultConfig(), sched: s, metrics: newUnitMetrics(s), now: time.Now,
		endpoints: map[string]struct{}{}, learned: map[string]struct{}{},
		rejected: map[string]struct{}{}, epoch: map[string]uint64{}}
}

// flapPodName is the pod NAME these tests flap; its endpoint key is engineID(0) ("ns/<name>").
const flapPodName = "a"

// flapDelete fires the endpoint-delete event the reconciler sends for that pod.
func flapDelete(t *testing.T, p *GroupScheduler, blockSize int) {
	t.Helper()
	require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: liveEndpoint(flapPodName, 0, 0, blockSize, 1000, time.Now()),
	}))
}

// flapAdd fires the endpoint add/update event for that pod.
func flapAdd(t *testing.T, p *GroupScheduler) {
	t.Helper()
	require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: liveEndpoint(flapPodName, 0, 0, 64, 1000, time.Now()),
	}))
}

// flapLearn hands the live candidate set to the detector, which is the only capacity path.
func flapLearn(p *GroupScheduler) {
	p.Saturation(context.Background(), livePods(liveEndpoint(flapPodName, 0, 0, 64, 1000, time.Now())))
}

// learnedMarks and groupCount read the plugin's own memory and the scheduler's groups.
func learnedMarks(p *GroupScheduler) (learned, rejected bool) {
	p.capMu.Lock()
	defer p.capMu.Unlock()
	_, learned = p.learned[engineID(0)]
	_, rejected = p.rejected[engineID(0)]
	return learned, rejected
}

func groupCount(p *GroupScheduler, key string) int {
	p.sched.mu.Lock()
	defer p.sched.mu.Unlock()
	if _, ok := p.sched.groups[key]; ok {
		return 1
	}
	return 0
}

// TestEndpointFlapRelearnsCapacity is the reproducer: the same endpoint name, deleted and re-added,
// must learn capacity again and seat a thread there. Before the fix the second learn was skipped by
// the remembered mark and the engine stayed at zero capacity forever.
func TestEndpointFlapRelearnsCapacity(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 1, 1000)
	key := engineID(0)

	flapAdd(t, p)
	flapLearn(p) // first life: capacity learned, a group exists
	require.Equal(t, 1, groupCount(p, key), "the first learn must create the group")
	learned, _ := learnedMarks(p)
	require.True(t, learned, "the first learn must be remembered")

	// THE FLAP: the pod goes not-ready and comes back under the same name.
	flapDelete(t, p, 64)
	require.Equal(t, 0, groupCount(p, key), "the delete must tear the group down")
	learned, rejected := learnedMarks(p)
	require.False(t, learned, "the delete must forget that this engine was learned")
	require.False(t, rejected, "and that it was rejected")

	flapAdd(t, p)
	flapLearn(p)

	require.True(t, p.sched.HasCapacityFor(key), "the re-added endpoint must learn its capacity again")
	require.Equal(t, 1, groupCount(p, key), "and get its group back")
	learned, _ = learnedMarks(p)
	require.True(t, learned, "and be remembered again, once")

	// and a thread SEATS there: the whole point is that the engine is usable again.
	if i := p.sched.Pass([]Head{head("m", time.Second, 100)}, p.now()); i != 0 {
		t.Fatal("a thread must seat on the relearned engine")
	}
	p.sched.mu.Lock()
	seated := p.sched.bindings["m"] == key
	p.sched.mu.Unlock()
	require.True(t, seated, "the thread must be seated on the re-added engine, got %q", p.sched.bindings["m"])
}

// TestDeleteDuringLearnLeavesNoMarkAndNoPhantom is race ORDER 1: the delete lands between the
// epoch read and the learn, so the learn creates a group for an endpoint that is already gone.
// The learn must be discarded: no mark (the re-added endpoint still has to learn) and no group
// left behind (nothing would ever delete it again).
func TestDeleteDuringLearnLeavesNoMarkAndNoPhantom(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 1, 1000)
	key := engineID(0)

	flapAdd(t, p)
	// the seam runs inside the learn window, after the scheduler took the capacity
	p.learnHook = func(string, learnPhase) {
		p.learnHook = nil // once
		flapDelete(t, p, 64)
	}
	flapLearn(p)

	learned, rejected := learnedMarks(p)
	require.False(t, learned, "a learn that outlived its endpoint must not be remembered")
	require.False(t, rejected, "and must not be remembered as a rejection either")
	require.Equal(t, 0, groupCount(p, key), "the group the learn created must be torn down (a phantom group is never deleted again)")
	require.False(t, p.sched.HasCapacityFor(key), "no capacity may survive for a deleted endpoint")

	// and the re-add still learns, because nothing was marked
	flapAdd(t, p)
	flapLearn(p)
	require.True(t, p.sched.HasCapacityFor(key))
	require.Equal(t, 1, groupCount(p, key))
}

// TestDeleteAfterLearnBeforeMarkIsDiscarded is race ORDER 2: the learn completed (the scheduler
// holds the capacity) and the delete lands before the learner marks the key. Same requirements:
// no mark, no group.
func TestDeleteAfterLearnBeforeMarkIsDiscarded(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 1, 1000)
	key := engineID(0)

	flapAdd(t, p)
	p.learnHook = func(string, learnPhase) {
		p.learnHook = nil
		// the group exists at this instant; the delete tears it down and bumps the epoch
		require.Equal(t, 1, groupCount(p, key), "the learn has created the group by now")
		flapDelete(t, p, 64)
	}
	flapLearn(p)

	learned, _ := learnedMarks(p)
	require.False(t, learned, "the mark must not be written across the delete")
	require.Equal(t, 0, groupCount(p, key), "and no group may remain")
	require.False(t, p.sched.HasCapacityFor(key))
}

// TestRelearnAfterFlapIsNotSkippedWhenRejected: a REJECTED mark is memory too. A pod that came up
// with the wrong block size and was restarted correctly must be judged again on the same name.
func TestRelearnAfterFlapIsNotSkippedWhenRejected(t *testing.T) {
	p := newBarePlugin(t)
	key := engineID(0)

	flapAdd(t, p)
	// a pod whose measured block size disagrees with the configured one: rejected, not learned
	p.Saturation(context.Background(), livePods(liveEndpoint(flapPodName, 0, 0, 32, 1000, time.Now())))
	learned, rejected := learnedMarks(p)
	require.False(t, learned)
	require.True(t, rejected, "the mismatch must be remembered as a rejection")

	flapDelete(t, p, 64)
	_, rejected = learnedMarks(p)
	require.False(t, rejected, "the delete must forget the rejection too")

	flapAdd(t, p)
	flapLearn(p)
	require.True(t, p.sched.HasCapacityFor(key),
		"the re-added pod with the right block size must be learned, not skipped by the old rejection")
}

// know registers endpoints the way the reconciler does before any candidate can carry them. The
// learn gate checks the KNOWN set, so a test that hands Saturation a candidate for an endpoint the
// plugin was never told about is modelling a state the plugin cannot reach — and, since the gate
// exists to refuse exactly that, the test must say so explicitly.
func know(t *testing.T, p *GroupScheduler, eps ...fwkdl.Endpoint) {
	t.Helper()
	for _, e := range eps {
		require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{
			Type:     fwkdl.EventAddOrUpdate,
			Endpoint: e,
		}))
	}
}
