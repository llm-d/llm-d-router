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

// WHO LEAVES, AND WHEN: the tenure rule that releases a resident, the demand-driven exit, RULE O's
// no-thrash clause, the budget valve's order, and the enforcement loop that walks it.

import (
	"fmt"
	"sort"
	"time"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// ---- rotation: WHO leaves, and when ----

// drainExiting releases every thread that is on its way out and has nothing left in
// flight.
func (s *Scheduler) drainExiting(now time.Time) {
	for _, g := range s.activeGroups() {
		var ids []string
		for id, m := range g.threads {
			// a LIVE window counts as in flight: a marked thread must not be drained out from under a
			// request that is already on its way to the pod
			if m.busy() || !g.index.Exiting(id) {
				continue
			}
			ids = append(ids, id)
		}
		if len(ids) == 0 {
			continue
		}
		// deterministic order: this writes EXIT events, and the event stream's order has to be
		// reproducible in the artifacts
		sort.Strings(ids)
		for _, id := range ids {
			if m, ok := g.threads[id]; ok {
				s.exit(g, m, "exiting-drain", now)
			}
		}
	}
}

// eligible asks whether a resident may be released, and why: Q — the one rotation
// value — in COMPLETED BUSY PERIODS, or S of silence.
func (s *Scheduler) eligible(m *thread, now time.Time) (string, bool) {
	if m.busy() {
		// a request at a pod, OR a live released window: a turn is in progress either way, and
		// rotating it out would evict a thread that is about to run
		return "", false
	}
	if m.busyPeriods >= s.cfg.Quantum {
		return "quantum-complete", true
	}
	if s.silent(m, now) {
		return "idle", true
	}
	return "", false
}

// silent is the idle test: no queued head and no activity for S.
func (s *Scheduler) silent(m *thread, now time.Time) bool {
	return m.noQueuedHead() &&
		now.Sub(m.lastActivity) > time.Duration(s.cfg.IdleSeconds*float64(time.Second))
}

// rotationVictim is the demand-driven exit over ONE engine's threads, in the order:
// among the ELIGIBLE, prefer a thread WITH a queued turn (it will use the seat again the
// moment it gets it), then earliest-seated, then id.
func (s *Scheduler) rotationVictim(g *engineGroup, now time.Time) (string, *thread) {
	var best *thread
	var bestReason string
	bestHasWaiting := false
	for _, m := range g.threads {
		if g.index.Exiting(m.id) {
			continue
		}
		reason, ok := s.eligible(m, now)
		if !ok {
			continue
		}
		hasWaiting := !m.waitingSince.IsZero()
		better := best == nil ||
			(hasWaiting && !bestHasWaiting) ||
			(hasWaiting == bestHasWaiting && m.seatTime.Before(best.seatTime)) ||
			(hasWaiting == bestHasWaiting && m.seatTime.Equal(best.seatTime) && m.id < best.id)
		if better {
			best, bestReason, bestHasWaiting = m, reason, hasWaiting
		}
	}
	if best != nil {
		return bestReason, best
	}
	// engine-idle rule: no thread in flight AND no thread dispatchable, while a waiter is blocked
	// (a livelock guard: the seats are held by threads that cannot use them)
	idle := true
	for _, m := range g.threads {
		if m.busy() || !m.waitingSince.IsZero() {
			idle = false
			break
		}
	}
	if idle {
		for _, m := range g.threads {
			if g.index.Exiting(m.id) {
				continue
			}
			if best == nil || m.lastActivity.Before(best.lastActivity) ||
				(m.lastActivity.Equal(best.lastActivity) && m.id < best.id) {
				best, bestReason = m, "engine-idle"
			}
		}
	}
	return bestReason, best
}

// olderCompletion breaks a tie between two equally-costly victims: a REAL last completion
// beats "never completed", then oldest completion first, then the id.
func olderCompletion(a, b *thread) bool {
	az, bz := a.lastCompletion.IsZero(), b.lastCompletion.IsZero()
	if az != bz {
		return bz // a real completion is the cheaper victim
	}
	if az {
		return a.id < b.id // TOTAL tie-break among threads that never completed
	}
	return a.lastCompletion.Before(b.lastCompletion) ||
		(a.lastCompletion.Equal(b.lastCompletion) && a.id < b.id)
}

// budgetVictim is MEMORY PRESSURE: who leaves when this engine is over its budget, and
// why — RULE O's no-thrash clause first, then the rotation demand, then the pressure classes.
func (s *Scheduler) budgetVictim(g *engineGroup, now time.Time) (string, *thread) {
	// RULE O's NO-THRASH clause, FIRST — before the rotation victim and every pressure class:
	// a thread that is the engine's ONLY staying resident, whose own footprint alone exceeds the
	// budget, is the seat Rule O granted on an empty engine. Evicting it frees nothing it could
	// return to.
	var sole *thread
	stay := 0
	for _, m := range g.threads {
		if g.index.Exiting(m.id) {
			continue
		}
		sole = m
		if stay++; stay > 1 {
			break
		}
	}
	if stay == 1 && s.soleOversize(g, sole) {
		return "", nil
	}
	// The rotation demand comes before pressure: the tenure rule has already decided this
	// thread should be released.
	if reason, m := s.rotationVictim(g, now); m != nil {
		return reason, m
	}
	// ONE pass, three buckets, each the oldest last completion. Bucket membership IS the class, and
	// everything above them is already spoken for: a silent or quantum-complete resident is a
	// ROTATION victim and the rotation demand above returned it, so these three are what is left —
	//  2 (budget-noqueue) — no queued head, below Q and within S: it has a turn to run, it just has
	//                       none waiting yet.
	//  3 (budget-idle)    — a queued head, mid-quantum: it wants the seat and has not had it long
	//                       enough for reclaiming it to be worth the loss. (The ≥Q ones are already
	//                       rotation victims, which is why mid-quantum is the whole class.)
	//  4 (budget-exiting) — a request RUNNING.
	var noQueue, midQuantum, running *thread
	for _, m := range g.threads {
		if g.index.Exiting(m.id) {
			continue
		}
		switch {
		case m.busy():
			if running == nil || olderCompletion(m, running) {
				running = m
			}
		case m.noQueuedHead():
			if noQueue == nil || olderCompletion(m, noQueue) {
				noQueue = m
			}
		default:
			if midQuantum == nil || olderCompletion(m, midQuantum) {
				midQuantum = m
			}
		}
	}
	if noQueue != nil {
		return "budget-noqueue", noQueue
	}
	if midQuantum != nil {
		return "budget-idle", midQuantum
	}
	if running != nil {
		return "budget-exiting", running
	}
	return "", nil
}

// exit unseats one thread.
func (s *Scheduler) exit(g *engineGroup, m *thread, reason string, now time.Time) {
	s.releaseCold(g, m.id) // unseat releases the charge
	g.index.Unseat(m.id)   // and the exact chains leave the budget here
	delete(g.threads, m.id)
	s.countExit(g, reason)
	s.recordEvent("EXIT", m.id, fmt.Sprintf("reason=%s turns=%d fairness_id=%s wire_id=%s",
		reason, m.seatTurns, fairnessOrNone(m.fairnessID), fairnessOrNone(m.wireID)), now)
	s.metrics.ObserveExit(engineLabel(g.id), reason)
	delete(s.bindings, m.id)
	// THE RECORDS GO WITH THE THREAD: they name a seat that no longer exists, so no later hook
	// could act on one (and its settle, if it ever comes, closes nothing rather than a charge against
	// an engine that never ran it).
	s.sweepThread(m.id, now)
	// a thread with a queued head re-enters at its ORIGINAL waiting-since (the queue position
	// survives the exit) — back in the SHARED queue, with no engine attached, carrying the cost
	// its HEAD will pay and the class its head has.
	if !m.waitingSince.IsZero() {
		s.waiters[m.id] = &waiter{id: m.id, since: m.waitingSince,
			returner: m.headReturner, chains: m.headChains,
			fairnessID: m.fairnessID, wireID: m.wireID}
	}
}

// fairnessOrNone renders an absent fairness id as "-" so a parser never sees an empty field.
func fairnessOrNone(id string) string {
	if id == "" {
		return "-"
	}
	return id
}

// enforceBudget is MEMORY PRESSURE on the growth path: while the STAY side — the blocks
// of the threads that are not already leaving — is over the engine's budget, take victims in
// the valve's order until it fits or nothing is evictable.
func (s *Scheduler) enforceBudget(g *engineGroup, now time.Time) {
	if g.budgetBlocks <= 0 {
		return // no measured capacity yet for THIS engine: nothing to enforce
	}
	for s.overBudget(g) > 0 {
		reason, victim := s.budgetVictim(g, now)
		if victim == nil {
			return
		}
		if victim.busy() {
			// The valve never returns an exiting thread (every bucket and the rotation demand skip
			// them), so a busy victim is one that still has to be MARKED.
			g.index.MarkExiting(victim.id)
			victim.exitingSince = now
			// THE MARK IS A STATE, NOT AN EVENT: one departure is booked ONCE, when the thread actually
			// leaves (reason "exiting-drain").
			continue
		}
		s.exit(g, victim, reason, now)
	}
}

// soleOversize reports RULE O's no-thrash case for ONE candidate: it is the engine's
// ONLY staying resident, and its own charged footprint alone exceeds the engine's budget.
func (s *Scheduler) soleOversize(g *engineGroup, m *thread) bool {
	if g == nil || m == nil || g.budgetBlocks <= 0 {
		return false
	}
	stay := 0
	for id := range g.threads {
		if g.index.Exiting(id) {
			continue
		}
		if stay++; stay > 1 {
			return false
		}
	}
	return stay == 1 && g.index.ChargedBlocks(m.id) > g.budgetBlocks
}

// overBudget is enforcement's view of the overage, in blocks: the STAY side (the seated
// threads that are not exiting) over the budget.
func (s *Scheduler) overBudget(g *engineGroup) int64 {
	over := int64(g.index.StayBlocks()) - int64(g.budgetBlocks)
	if over < 0 {
		return 0
	}
	return over
}

// ---- engine lifecycle (the DISCOVERED engine set) ----

// RemoveEngine tears one engine's group down when the EPP reports the endpoint gone.
func (s *Scheduler) RemoveEngine(engine string, now time.Time) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, ok := s.groups[engine]
	if !ok {
		return 0
	}
	// deterministic order: the teardown writes one EXIT event per thread and a torn-down group's
	// event order must be reproducible in the artifacts
	ids := make([]string, 0, len(g.threads))
	for id := range g.threads {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	n := 0
	for _, id := range ids {
		if m, ok := g.threads[id]; ok {
			s.exit(g, m, "engine-gone", now)
			n++
		}
	}
	// The waiter QUEUE is shared and survives a teardown: those heads can seat on a survivor,
	// which is the whole point of pooling the waits. Every binding of this engine went with its
	// thread in exit.
	delete(s.groups, engine)
	s.rebuildGroupOrder()
	s.counters.engineGone++
	s.metrics.ObserveEngineGone()
	s.logger.V(logutil.DEFAULT).Info("group-scheduler: ENGINE GONE — threads exited and the group dropped",
		"engine", engineLabel(engine), "threads", n)
	return n
}
