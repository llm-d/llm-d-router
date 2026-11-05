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

// THE PASS. One decision per call, six steps in a fixed order — and the order IS the policy: the
// counts are derived before anything reads them, the queue is reconciled before the cold charges
// read it, the budget pressure is applied before anybody is seated, and the seat is taken before
// dispatch scores the seats.

import (
	"sort"
	"time"
)

// Pass is THE PASS, in six steps, and the order is the policy. It answers ONE question — which head
// runs next — so it returns the index into heads, or -1 when there is nothing to serve. The engine a
// head was served on is a fact the pass LEAVES BEHIND (the winner's released record, resolved to its
// thread's binding), not a second return value: no production caller wants it, and a return value
// only the tests read is a return value that will drift.
//
//	tidy              the derived counts, the reaps, the leavers (before anything reads a count)
//	syncQueue         reconciles the SHARED waiter queue with this pass's heads
//	deriveColdCharges the cold charges, AFTER syncQueue: they read waitingSince, which it writes
//	enforceBudgets    the per-engine budget pressure, before anybody is seated
//	seatWaiters       the shared queue's head to the engine admission chooses
//	dispatch          the winning thread head on its seat engine, or the fall-through
func (s *Scheduler) Pass(heads []Head, now time.Time) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	s.tidy(now)
	s.syncQueue(heads, now)
	s.deriveColdCharges()
	s.enforceBudgets(now)
	s.seatWaiters(now)
	return s.dispatch(heads, now)
}

// tidy is the pass's housekeeping, and the ORDER is the point: derive the cold charges, reap the
// records nothing can settle any more, then drain the threads already leaving — each step leaves the
// state the next one reads.
func (s *Scheduler) tidy(now time.Time) {
	s.derive()
	s.reapRequests(now)
	s.drainExiting(now)
}

// enforceBudgets is MEMORY PRESSURE over every active engine, before this pass seats anybody.
func (s *Scheduler) enforceBudgets(now time.Time) {
	for _, g := range s.activeGroups() {
		s.enforceBudget(g, now)
	}
}

// syncQueue reconciles the SHARED waiter queue with the heads this pass was handed: a seated
// thread's presence and class are refreshed, an unseated thread's head enters the queue, and
// whatever the framework no longer offers leaves it.
func (s *Scheduler) syncQueue(heads []Head, now time.Time) {
	present := map[string]time.Time{}
	for _, h := range heads {
		if h.Thread == "" {
			continue
		}
		if len(h.Chains) == 0 {
			// no tokens, no cost: the request cannot be seated or charged, so it is served like
			// anonymous traffic (pass-through, last) and never enters the shared queue.
			s.noteAnomaly(anomalyNoTokens)
			continue
		}
		present[h.Thread] = h.Arrival
		if g := s.groupOf(h.Thread); g != nil {
			if m, seated := g.threads[h.Thread]; seated {
				if h.FairnessID != "" && m.fairnessID != h.FairnessID {
					m.fairnessID = h.FairnessID
				}
				if h.WireID != "" {
					m.wireID = h.WireID
				}
				if m.waitingSince.IsZero() || h.Arrival.Before(m.waitingSince) {
					m.waitingSince = h.Arrival
				}
				// the head's own cost and class, for the re-queue an exit may have to make
				m.headChains, m.headReturner = h.Chains, h.Returner
				m.lastActivity = now
				continue
			}
		}
		// unseated: its demand is ADMISSION, so it waits in the one shared queue
		w, ok := s.waiters[h.Thread]
		if !ok {
			w = &waiter{id: h.Thread}
			s.waiters[h.Thread] = w
		}
		// the class rides with the HEAD: thread identity knows whether this is the thread's first
		// turn
		w.returner = h.Returner
		w.fairnessID = h.FairnessID
		w.wireID = h.WireID
		w.chains = h.Chains
		// the oldest arrival is what the wait is measured from: a fan-out re-arrival does not reset
		// it
		if w.since.IsZero() || h.Arrival.Before(w.since) {
			w.since = h.Arrival
		}
	}
	// vanished: threads clear their waiting marker; waiters leave
	for _, g := range s.activeGroups() {
		for id, m := range g.threads {
			if _, ok := present[id]; !ok {
				m.waitingSince = time.Time{}
			}
		}
	}
	for id := range s.waiters {
		if _, ok := present[id]; !ok {
			delete(s.waiters, id)
		}
	}
}

// dispatch hands the pass's decision back: among the heads whose thread is SEATED and within its
// in-flight cap, one head per engine (oldest arrival first), the engine whose queued demand per seat
// is highest wins, and its turn's chain is charged in this same critical section. With no thread head
// dispatchable, the OLDEST head that has neither a thread nor a costed chain passes through — an
// anonymous request, or a thread head whose render produced no tokens.
func (s *Scheduler) dispatch(heads []Head, now time.Time) int {
	// The same group set enforceBudgets saw: nothing between them (the seat and the exits
	// inside seatWaiters) creates or deletes a group, and s.mu is held throughout.
	groups := s.activeGroups()
	type cand struct {
		i      int
		h      Head
		engine string
	}
	perEngine := make(map[string][]cand, len(groups))
	for i, h := range heads {
		if h.Thread == "" {
			continue // anonymous traffic is not tracked (passes through last)
		}
		g := s.groupOf(h.Thread)
		if g == nil {
			continue // unseated: its head is ADMISSION, handled by seatWaiters
		}
		m, seated := g.threads[h.Thread]
		if !seated || g.index.Exiting(h.Thread) {
			continue
		}
		if m.inflight+m.pending >= s.cfg.InFlightCap {
			continue
		}
		perEngine[g.id] = append(perEngine[g.id], cand{i: i, h: h, engine: g.id})
	}
	type best struct {
		c     cand
		score float64
	}
	var winner *best
	for _, g := range groups {
		cands := perEngine[g.id]
		if len(cands) == 0 {
			continue
		}
		sort.SliceStable(cands, func(x, y int) bool { return cands[x].h.Arrival.Before(cands[y].h.Arrival) })
		// FIFO: the candidates are oldest-first, so the head is the first one
		chosen := &cands[0]
		score := s.queuedPerSeat(g)
		if winner == nil || score > winner.score ||
			(score == winner.score && chosen.h.Arrival.Before(winner.c.h.Arrival)) ||
			(score == winner.score && chosen.h.Arrival.Equal(winner.c.h.Arrival) && g.id < winner.c.engine) {
			winner = &best{c: *chosen, score: score}
		}
	}
	if winner != nil {
		g := s.groups[winner.c.engine]
		m := g.threads[winner.c.h.Thread]
		// THE OBSERVABLES ARE READ BEFORE THE CHARGE: the resident prefix as the decision found it.
		// Read after the Grow it would be the whole chain every time — a 100% predicted hit rate
		// that measures nothing.
		obs := s.observablesFor(g, winner.c.h, now)
		// THE DISPATCHED TURN'S CHAIN IS CHARGED HERE, as the pass's winner — in this pass's
		// critical section, so the charge is exact and atomic with the decision.
		if len(winner.c.h.Chains) > 0 {
			g.index.Grow(m.id, winner.c.h.Chains)
		}
		// The request is RELEASED: it is neither queued nor in flight until PreRequest records it,
		// and the record written here is what every later hook resolves it by.
		s.release(m, winner.c.h.ReqID, winner.c.h.Chains, now, obs)
		m.lastActivity = now
		g.counters.served++
		s.counters.served++
		s.metrics.ObserveServed(engineLabel(g.id))
		return winner.c.i
	}
	// No thread head is dispatchable — serve the OLDEST anonymous head (FCFS pass-through:
	// anonymous traffic is never tracked, never seated, never starved).
	fcfs := -1
	for i, h := range heads {
		if h.Thread != "" && len(h.Chains) > 0 {
			continue // a thread head that could be costed is not this pass's business
		}
		if fcfs < 0 || h.Arrival.Before(heads[fcfs].Arrival) {
			fcfs = i
		}
	}
	if fcfs >= 0 {
		// the pass-through's turn is a request like any other for the queue-wait histogram; it has
		// no engine, so it contributes nothing to the per-engine cache counters
		s.metrics.ObserveQueueWait(queueWaitUntracked, nonNegative(now.Sub(heads[fcfs].Arrival)))
	}
	return fcfs
}

// queuedPerSeat is dispatch's engine score: the fraction of this engine's SEATED threads that have
// a head waiting right now.
func (s *Scheduler) queuedPerSeat(g *engineGroup) float64 {
	seats := len(g.threads)
	if seats <= 0 {
		seats = 1
	}
	queued := 0
	for _, m := range g.threads {
		if !m.waitingSince.IsZero() {
			queued++
		}
	}
	return float64(queued) / float64(seats)
}

// seatWaiters is the admission loop: hand the shared queue's head to the engine admission chooses,
// until nothing more can be seated.
func (s *Scheduler) seatWaiters(now time.Time) {
	if len(s.waiters) == 0 {
		return
	}
	budget := len(s.waiters)
	for _, g := range s.activeGroups() {
		budget += len(g.threads)
	}
	for ; budget > 0; budget-- {
		head := s.headWaiter()
		if head == nil {
			return
		}
		cand := s.seatEngine(head, now)
		if cand == nil || cand.g == nil {
			// nothing to do this pass.
			if engine, held := s.coldValveHeldOn(head); held {
				s.metrics.ObserveColdValveHeld(engineLabel(engine))
			}
			return
		}
		if cand.fits {
			// ONE number, one moment: the fit test the chooser just ran and the seat's own cost are the
			// same computation inside the same critical section, so TrySeat can only reject a head the
			// chooser admitted if the two disagree — and then the loop must STOP, not re-pick the same
			// head until the budget runs out under the scheduler's mutex.
			if !s.seat(cand.g, head, cand.costTokens, now) {
				return
			}
			continue
		}
		// Nothing fits BY BUDGET on the chosen engine: eviction is the only way a budget-blocked
		// head ever runs, and it is demand-driven (a head is waiting right now).
		reason, victim := s.rotationVictim(cand.g, now)
		if victim == nil {
			return // this engine can free nothing; another pass will re-evaluate
		}
		// NOTE: a rotation victim is never in flight — the tenure rule refuses in-flight threads — so
		// there is no mid-flight exit to handle here; that class lives in budgetVictim (eviction.go),
		// which does reach running threads.
		s.exit(cand.g, victim, reason, now)
	}
}

// FCFSPick returns the index of the oldest head, or -1: the pre-capacity pass-through's order.
func FCFSPick(heads []Head) int {
	best := -1
	var bestArrival time.Time
	for i, h := range heads {
		if best < 0 || h.Arrival.Before(bestArrival) {
			best, bestArrival = i, h.Arrival
		}
	}
	return best
}

// queueWaitNew, queueWaitReturner and queueWaitUntracked are the queue-wait kinds: a thread's
// first turn, a later turn, and a head the pass served without seating (anonymous traffic, or a
// thread head with no rendered tokens).
const (
	queueWaitNew       = "new"
	queueWaitReturner  = "returner"
	queueWaitUntracked = "untracked"
)

// nonNegative clamps a duration to zero: a clock that went backwards must not observe a negative
// wait into an exponential histogram.
func nonNegative(d time.Duration) float64 {
	if d < 0 {
		return 0
	}
	return d.Seconds()
}

// observablesFor is one dispatched turn's observability: the queue wait it just served, and how
// many of its prompt tokens the engine is PREDICTED to hold as a resident prefix. It READS the
// index and writes nothing to it, and the caller runs it BEFORE the charge — after the charge the
// prediction would always be the whole prompt.
func (s *Scheduler) observablesFor(g *engineGroup, h Head, now time.Time) dispatchObservables {
	obs := dispatchObservables{queueWait: now.Sub(h.Arrival)}
	if m := g.threads[h.Thread]; m != nil && m.seatObs != nil {
		// the turn that SEATED this thread: its prediction was measured at the seat, before the
		// reservation. Consume it — every later turn measures the index directly.
		obs.predictedCached, obs.promptTokens = m.seatObs.predictedCached, m.seatObs.promptTokens
		m.seatObs = nil
		s.metrics.ObserveQueueWait(queueWaitKind(h), nonNegative(obs.queueWait))
		s.metrics.ObservePredictedCache(engineLabel(g.id), obs.predictedCached, obs.promptTokens)
		return obs
	}
	for _, c := range h.Chains {
		obs.promptTokens += int64(c.Tokens)
		prefix := g.index.residentPrefix(c.Keys)
		cached := int64(prefix * s.cfg.BlockSizeTokens)
		if cached > int64(c.Tokens) {
			cached = int64(c.Tokens) // a partial trailing block is not resident
		}
		obs.predictedCached += cached
	}
	s.metrics.ObserveQueueWait(queueWaitKind(h), nonNegative(obs.queueWait))
	s.metrics.ObservePredictedCache(engineLabel(g.id), obs.predictedCached, obs.promptTokens)
	return obs
}

// queueWaitKind classifies a dispatched head for the queue-wait histogram.
func queueWaitKind(h Head) string {
	if h.Returner {
		return queueWaitReturner
	}
	return queueWaitNew
}
