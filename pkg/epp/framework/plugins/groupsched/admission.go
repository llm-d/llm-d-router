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

// ADMISSION — one shared queue, one engine choice, ONE code path for N=1..N.

import (
	"fmt"
	"time"
)

// queuedPerSeat is the engine's OWN queued demand per seat it holds: seated threads

// freeBudget is what the engine can still admit: its budget minus its USED side (every
// seated thread's blocks, exiting ones included — they still hold them).
func (s *Scheduler) freeBudget(g *engineGroup) int64 {
	return int64(g.budgetBlocks) - int64(g.index.UniqueBlocks())
}

// coldAllowsTokens asks whether this head's cold prefill fits the engine's stagger, for the
// exact cost it brings.
func (s *Scheduler) coldAllowsTokens(g *engineGroup, tokens int64) bool {
	cap := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize)
	if cap <= 0 {
		return true // no measured capacity: the admission pass already refuses these engines
	}
	return g.coldInFlight+min(tokens, cap) <= cap
}

// seatChoice is placement's answer (chooseEngine) bound back to the engine it named.
type seatChoice struct {
	g *engineGroup
	// costTokens is the head's cold tokens ON THIS ENGINE: the fit test's input and the cold-
	// entry charge the seat takes (capped by the engine's own valve).
	costTokens int64
	// fits: the head's cost was inside the free budget AND the cold-entry valve allowed it.
	fits bool
}

// seatEngine is THE seat-time engine choice: the scheduler gathers the FACTS (per-engine
// cost, free budget, fit, whether a victim exists) and chooseEngine decides.
func (s *Scheduler) seatEngine(head *waiter, now time.Time) *seatChoice {
	groups := s.activeGroups()
	cands := make([]EngineCandidate, 0, len(groups))
	owners := make([]*engineGroup, 0, len(groups))
	for _, g := range groups {
		if g.numBlocks <= 0 {
			// NO measured capacity: the engine cannot be costed, and seating into an unknown budget is
			// how an engine takes unbounded seats during the metrics race.
			continue
		}
		// the head's exact cost ON THIS ENGINE: what this engine would have to compute (its own
		// resident prefix discounted) and the cold tokens behind it
		tokens := g.index.ColdTokens(head.chains)
		blocks := int64(g.index.Cost(head.chains))
		free := s.freeBudget(g)
		budget := int64(g.budgetBlocks)
		budgetFits := blocks <= free
		// RULE O — the EMPTY ENGINE ADMITS ONE.
		ruleO := len(g.threads) == 0 && blocks > budget
		gg := g
		cands = append(cands, EngineCandidate{
			Engine: gg.id, CostTokens: tokens, CostBlocks: blocks,
			FreeBudget: free, Budget: budget, BudgetFits: budgetFits,
			// FITS = (budget OR Rule O's empty engine) AND the cold valve: a head the valve holds must
			// WAIT for the valve, never evict for it — evicting frees budget, and the valve is not
			// budget.
			Fits:      (budgetFits || ruleO) && s.coldAllowsTokens(g, tokens),
			HasVictim: func() bool { return s.hasRotationVictim(gg, now) },
		})
		owners = append(owners, gg)
	}
	idx := chooseEngine(cands)
	if idx < 0 || idx >= len(cands) {
		return nil
	}
	return &seatChoice{g: owners[idx], costTokens: cands[idx].CostTokens, fits: cands[idx].Fits}
}

// coldValveHeldOn reports WHICH engine's cold-entry valve is the only thing keeping this head
// out — the engine whose BUDGET would take it while the valve refuses it — and whether any
// engine is in that state at all.
func (s *Scheduler) coldValveHeldOn(head *waiter) (string, bool) {
	best := EngineCandidate{Engine: ""}
	for _, g := range s.activeGroups() {
		if g.numBlocks <= 0 {
			continue // no measured capacity: the engine is not taking seats at all
		}
		tokens := g.index.ColdTokens(head.chains)
		if int64(g.index.Cost(head.chains)) > s.freeBudget(g) {
			continue // the BUDGET is the blocker on this engine: the valve is not the reason
		}
		if s.coldAllowsTokens(g, tokens) {
			continue // the valve allows it: whatever held the head, it was not this
		}
		c := EngineCandidate{Engine: g.id, FreeBudget: s.freeBudget(g)}
		if best.Engine == "" || betterEngine(c, best) {
			best = c
		}
	}
	return best.Engine, best.Engine != ""
}

// hasRotationVictim reports whether an engine can evict somebody right now (one
// definition of the victim criteria, in rotationVictim).
func (s *Scheduler) hasRotationVictim(g *engineGroup, now time.Time) bool {
	_, v := s.rotationVictim(g, now)
	return v != nil
}

// headWaiter picks the head of the ONE shared queue: newcomers before returning
// threads, then the oldest arrival, then the id.
func (s *Scheduler) headWaiter() *waiter {
	var head *waiter
	for _, w := range s.waiters {
		if head == nil || newcomerLess(w, head) {
			head = w
		}
	}
	return head
}

// newcomerLess is the entry order's total comparison (lower wins): a total order, so map
// iteration cannot leak into a decision.
func newcomerLess(a, b *waiter) bool {
	if a.returner != b.returner {
		return !a.returner // newcomers first
	}
	if !a.since.Equal(b.since) {
		return a.since.Before(b.since) // oldest first
	}
	return a.id < b.id
}

// seat seats one waiting head on the engine the admission pass chose.
func (s *Scheduler) seat(g *engineGroup, w *waiter, coldTokens int64, now time.Time) bool {
	// THE TURN'S PREDICTION IS MEASURED HERE, BEFORE THE SEAT RESERVES ANYTHING: from this moment the
	// thread's blocks are resident by construction, so a measurement taken after the insert would
	// report the reservation as cache and every seat-entry dispatch would predict its whole prompt.
	seatObs := dispatchObservables{
		predictedCached: g.index.CachedTokens(w.chains),
		promptTokens:    PromptTokens(w.chains),
	}
	if ok, cost := g.index.TrySeat(w.id, w.chains, g.budgetBlocks); !ok {
		// RULE O — AN EMPTY ENGINE ADMITS ONE OVERSIZE HEAD, charging its FULL exact chain.
		if len(g.threads) != 0 || cost <= g.budgetBlocks {
			// Not Rule O: the chooser and this seat read the same numbers in the same critical section,
			// so this means they disagreed.
			return false
		}
		g.index.SeatForce(w.id, w.chains)
	}
	s.bindings[w.id] = g.id
	delete(s.waiters, w.id)
	m := &thread{
		id:           w.id,
		seatTime:     now,
		lastActivity: now,
		// the head that triggered this seating is still queued (the serve happens after the seat):
		// inherit its presence so the engine-idle rule cannot evict a thread that is about to
		// dispatch — and its own chains and class, so a later exit re-queues with this head's cost
		// rather than the chain of whatever ran.
		waitingSince: w.since,
		fairnessID:   w.fairnessID,
		wireID:       w.wireID,
		headChains:   w.chains,
		headReturner: w.returner,
		seatObs:      &seatObs,
	}
	// EVERY entry carries a cold charge (in tokens), CAPPED at the engine's own stagger cap: the
	// valve's job is to serialise cold prefills inside the budget's headroom, not to account the
	// entry's whole footprint (the seat's blocks do that).
	m.coldEntry = min(coldTokens, g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
	g.coldInFlight += m.coldEntry
	// record EVERY seat wait, split by kind (a thread's first turn vs a returning thread): both
	// answer "how long did admission make this thread wait".
	eventKind, waitKind := "re-entry", "reentry"
	if !w.returner {
		eventKind, waitKind = "newcomer", "new"
	}
	s.metrics.ObserveEntry(engineLabel(g.id), waitKind, now.Sub(w.since).Seconds())
	// the fields a run's forensics need: thread id, wire id, engine (the Event's own field), the
	// seat kind, the wait, and the arrival the wait is measured from
	s.recordEvent("SEAT", w.id, fmt.Sprintf("kind=%s wait=%.3f fairness_id=%s wire_id=%s since=%s",
		eventKind, now.Sub(w.since).Seconds(), fairnessOrNone(w.fairnessID), fairnessOrNone(w.wireID),
		w.since.UTC().Format(time.RFC3339Nano)), now)
	g.threads[w.id] = m
	return true
}

// ---- PLACEMENT ---------------------------------------------------------------

// EngineCandidate is one engine's offer for one waiting head — the FACTS chooseEngine decides on.
type EngineCandidate struct {
	// Engine is the engine id (the deterministic tie-break).
	Engine string
	// CostTokens is this head's admission cost ON THIS ENGINE (tokens), and CostBlocks its block-
	// quantised form.
	CostTokens int64
	CostBlocks int64
	// FreeBudget is what the engine can still admit (budget − accounted footprint).
	FreeBudget int64
	// Budget is the engine's SEAT BUDGET in blocks (fraction × its own num_gpu_blocks): the
	// number an EMPTY engine could admit, which is what RULE O's trigger is measured against.
	Budget int64
	// BudgetFits reports CostBlocks <= FreeBudget.
	BudgetFits bool
	// Fits reports BudgetFits AND that the COLD-ENTRY VALVE allows this head on this engine.
	Fits bool
	// HasVictim is LAZY (it costs a scan): may the engine evict an eligible incumbent right now?
	HasVictim func() bool
}

// chooseEngine answers "which engine takes this head?": among the engines where the head FITS
// (budget AND the cold valve), the one with the MOST FREE BUDGET AFTER the seat — free minus
// this head's cost on it; ties → lowest engine id. It returns the index into cands, or -1.
func chooseEngine(cands []EngineCandidate) int {
	best := -1
	for i := range cands {
		if !cands[i].Fits {
			continue
		}
		if best < 0 || betterFit(cands[i], cands[best]) {
			best = i
		}
	}
	if best >= 0 {
		return best
	}
	if oversizeEverywhere(cands) {
		// RULE O, the no-empty-engine half: evict on the engine placement would pick anyway, and let
		// the next iteration re-ask.
		for i := range cands {
			if cands[i].HasVictim == nil || !cands[i].HasVictim() {
				continue
			}
			if best < 0 || betterEngine(cands[i], cands[best]) {
				best = i
			}
		}
		return best
	}
	// The cold valve is holding this head somewhere: WAIT — never evict — because evicting frees
	// budget and never the valve.
	for i := range cands {
		if cands[i].BudgetFits {
			return -1
		}
	}
	// Nothing fits by BUDGET either: eviction is the way forward.
	for i := range cands {
		if cands[i].HasVictim == nil || !cands[i].HasVictim() {
			continue
		}
		if best < 0 || betterEngine(cands[i], cands[best]) {
			best = i
		}
	}
	return best
}

// betterFit is the placement total order among engines the head FITS: the most free budget
// left AFTER the seat, then the engine id.
func betterFit(c, best EngineCandidate) bool {
	if c.FreeBudget-c.CostBlocks != best.FreeBudget-best.CostBlocks {
		return c.FreeBudget-c.CostBlocks > best.FreeBudget-best.CostBlocks
	}
	return c.Engine < best.Engine
}

// oversizeEverywhere reports RULE O's trigger: the head's exact cost exceeds the SEAT BUDGET
// of every candidate engine, so it could not fit one even if that engine were empty.
func oversizeEverywhere(cands []EngineCandidate) bool {
	if len(cands) == 0 {
		return false
	}
	for i := range cands {
		if cands[i].CostBlocks <= cands[i].Budget {
			return false
		}
	}
	return true
}

// betterEngine is the placement total order for the engines that cannot take the head (and for
// naming the engine a valve-held head waits for): most free budget, then engine id.
func betterEngine(c, best EngineCandidate) bool {
	if c.FreeBudget != best.FreeBudget {
		return c.FreeBudget > best.FreeBudget
	}
	return c.Engine < best.Engine
}
