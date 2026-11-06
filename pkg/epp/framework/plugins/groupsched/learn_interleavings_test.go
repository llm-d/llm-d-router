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

// THE EXHAUSTIVE INTERLEAVING OF THE CAPACITY LEARN WITH THE ENDPOINT LIFE CYCLE.
//
// One learn attempt touches the plugin's state at exactly three points: it reads the known set and
// the epoch (under capMu), it takes the scheduler's group lock (outside capMu), and it writes the
// mark (under capMu again). A delete can therefore land in two windows: after the read and before
// the learn (the seam's learnStart), or after the learn and before the mark (learnLearned). Those
// two windows, crossed with the three endpoint outcomes that matter — deleted, deleted and
// re-added (a flap), and untouched — is the whole space, and this file drives every cell.
//
// The invariants are the ones the reviewer asked for, asserted on all of them:
//
//	(a) a LEARNED mark implies a group that holds capacity;
//	(b) no group ever exists for a key this plugin does not KNOW (a snapshot can predate a delete);
//	(c) a known endpoint whose geometry arrives after a flap relearns on the next Saturation.
//
// The order that no epoch check can catch is the one the known-set gate exists for: the delete is
// fully processed BEFORE the learn starts (so the epoch has already moved and cannot move again),
// while the candidate slice driving the learn was snapshotted before it.

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// staleSnapshot is the candidate set a delete does not reach: Saturation is about to be handed it.
func staleSnapshot() []fwkdl.Endpoint {
	return livePods(liveEndpoint(flapPodName, 0, 0, 64, 1000, time.Now()))
}

// driveLearn feeds the detector a candidate slice that was snapshotted BEFORE the action ran, which
// is the shape of the race: the plugin's own state says one thing, the slice says another.
func driveLearn(p *GroupScheduler, snapshot []fwkdl.Endpoint) {
	p.Saturation(context.Background(), snapshot)
}

// assertNoPhantom is invariant (b) plus its consequences: an engine this plugin does not know holds
// no group, no capacity and no mark.
func assertNoPhantom(t *testing.T, p *GroupScheduler, key, why string) {
	t.Helper()
	require.Equal(t, 0, groupCount(p, key), "no group may exist for an endpoint this plugin does not host (%s)", why)
	require.False(t, p.sched.HasCapacityFor(key), "and no capacity (%s)", why)
	learned, rejected := learnedMarks(p)
	require.False(t, learned, "and the key must not be MARKED learned: the re-added endpoint has to learn again (%s)", why)
	require.False(t, rejected, "and must not be remembered as a rejection either (%s)", why)
}

// knownEndpoint reads the KNOWN set under its own lock (the tests are single-goroutine, but a
// racy read in a test is still a racy read).
func knownEndpoint(p *GroupScheduler, key string) bool {
	p.capMu.Lock()
	defer p.capMu.Unlock()
	_, ok := p.endpoints[key]
	return ok
}

// assertLearnedHealthy is invariant (a) — a learned mark implies a group with capacity — and (c):
// the engine is usable, a thread seats on it, and its capacity is the measured one.
func assertLearnedHealthy(t *testing.T, p *GroupScheduler, key, why string) {
	t.Helper()
	learned, rejected := learnedMarks(p)
	require.True(t, learned, "a live endpoint must be marked learned (%s)", why)
	require.False(t, rejected, "and not rejected (%s)", why)
	require.Equal(t, 1, groupCount(p, key), "a learned mark implies a group (%s)", why)
	require.True(t, p.sched.HasCapacityFor(key), "and that group holds capacity (%s)", why)
	if i := p.sched.Pass([]Head{head("m", time.Second, 100)}, p.now()); i != 0 {
		t.Fatalf("a thread must seat on a learned engine (%s)", why)
	}
}

// TestLearnInterleavings runs every cell: {delete, flap, untouched} x {learnStart, learnLearned},
// plus the two orders the seams cannot express — a delete processed entirely before the learn
// (the stale-snapshot order) and a flap completed before the learn.
func TestLearnInterleavings(t *testing.T) {
	type action struct {
		name string
		run  func(t *testing.T, p *GroupScheduler)
		// disappears says the endpoint is not known when the learn finally marks, so the invariants
		// of the case are the "no phantom" ones (a flap leaves it known, so it is not one).
		disappears bool
	}
	actions := []action{
		{name: "untouched", run: func(*testing.T, *GroupScheduler) {}, disappears: false},
		{name: "deleted", run: func(t *testing.T, p *GroupScheduler) { flapDelete(t, p, 64) }, disappears: true},
		{name: "flapped", run: func(t *testing.T, p *GroupScheduler) {
			flapDelete(t, p, 64)
			flapAdd(t, p)
		}, disappears: false},
	}
	phases := []learnPhase{learnStart, learnLearned}

	for _, a := range actions {
		for _, ph := range phases {
			t.Run(a.name+"/"+phaseName(ph), func(t *testing.T) {
				p := newTestPlugin(t, defaultConfig(), 1, 1000)
				key := engineID(0)
				flapAdd(t, p)

				snapshot := staleSnapshot()
				fired := false
				p.learnHook = func(_ string, phase learnPhase) {
					if fired || phase != ph {
						return
					}
					fired = true
					a.run(t, p)
				}
				driveLearn(p, snapshot)
				require.True(t, fired, "the seam must fire at %s", phaseName(ph))

				if a.disappears {
					assertNoPhantom(t, p, key, a.name+"/"+phaseName(ph))
					// and the re-add relearns, because nothing was marked (invariant c)
					flapAdd(t, p)
					driveLearn(p, staleSnapshot())
					assertLearnedHealthy(t, p, key, a.name+"/"+phaseName(ph)+" then re-add")
					return
				}
				if a.name == "flapped" {
					// A flap moves the epoch, so this learn is necessarily discarded: the group it
					// made was torn down by the delete and the mark would describe a group that no
					// longer exists. What must hold is that the endpoint is left CLEAN — known, no
					// mark, no group, no capacity — so the next Saturation relearns (invariant c).
					assertNoPhantom(t, p, key, a.name+"/"+phaseName(ph))
					require.True(t, knownEndpoint(p, key), "the re-added endpoint must still be known")
					driveLearn(p, staleSnapshot())
					assertLearnedHealthy(t, p, key, a.name+"/"+phaseName(ph)+" then next Saturation")
					return
				}
				assertLearnedHealthy(t, p, key, "untouched/"+phaseName(ph))
			})
		}
	}

	// THE STALE-SNAPSHOT ORDER: the delete is processed to completion BEFORE the learn is called,
	// so the epoch it reads is already the post-delete one and the epoch check cannot see anything.
	// Only the known-set gate stops this from creating a group for an engine that no longer exists.
	t.Run("stale-snapshot", func(t *testing.T) {
		p := newTestPlugin(t, defaultConfig(), 1, 1000)
		key := engineID(0)
		flapAdd(t, p)

		snapshot := staleSnapshot() // taken BEFORE the delete
		flapDelete(t, p, 64)        // processed to completion, teardown and epoch bump included
		driveLearn(p, snapshot)     // the detector is handed the pre-delete slice

		assertNoPhantom(t, p, key, "stale snapshot")

		// the re-added endpoint learns again and seats (invariant c)
		flapAdd(t, p)
		driveLearn(p, staleSnapshot())
		assertLearnedHealthy(t, p, key, "stale snapshot then re-add")
	})

	// THE FLAP-BEFORE-LEARN ORDER: same shape, but the endpoint is known again by the time the learn
	// runs. The learn is legitimate — the group it creates belongs to the re-added endpoint — so the
	// case must end learned and seating, not merely phantom-free.
	t.Run("flap-before-learn", func(t *testing.T) {
		p := newTestPlugin(t, defaultConfig(), 1, 1000)
		key := engineID(0)
		flapAdd(t, p)

		snapshot := staleSnapshot()
		flapDelete(t, p, 64)
		flapAdd(t, p)
		driveLearn(p, snapshot)
		assertLearnedHealthy(t, p, key, "flap before learn")
	})

	// THE DELETE-AFTER-MARK ORDER is serial, not a race: the mark and the teardown both take capMu,
	// so the delete simply clears a mark that existed. Asserted because "we covered every order"
	// has to be true, not assumed.
	t.Run("delete-after-mark", func(t *testing.T) {
		p := newTestPlugin(t, defaultConfig(), 1, 1000)
		key := engineID(0)
		flapAdd(t, p)
		driveLearn(p, staleSnapshot())
		assertLearnedHealthy(t, p, key, "before the delete")

		flapDelete(t, p, 64)
		assertNoPhantom(t, p, key, "delete after mark")
	})
}

// phaseName is for subtest names only.
func phaseName(ph learnPhase) string {
	if ph == learnStart {
		return "delete-at-start"
	}
	return "delete-at-mark"
}
