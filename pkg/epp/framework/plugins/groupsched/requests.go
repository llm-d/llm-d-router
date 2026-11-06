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

// THE REQUEST'S LIFE: it leaves the queue at Pick, the framework hands it back through the scoped
// hooks, and it ends at a settle, an abandon or its thread's exit. This file owns the ONE record
// per tracked request, and the derive-don't-accumulate bookkeeping that re-derives the counts from
// that map at every decision point.

import (
	"fmt"
	"time"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// A request leaves the queue at Pick and comes back to the scheduler only through hooks the
// framework SCOPES: Filter and PreRequest get a request whose attribute access is confined to
// what this plugin declares, and an undeclared read resolves as absent.

// dispatchObservables is what one dispatched turn contributes to the span and the counters. The
// caller computes it BEFORE the charge, so predictedCached is the engine's resident prefix as it
// stood at the decision, not as the decision left it.
type dispatchObservables struct {
	queueWait       time.Duration
	predictedCached int64
	promptTokens    int64
}

// release opens one request's record: the thread that dispatched it, the chains it was charged
// with, and the observables PreRequest may put on the span.
func (s *Scheduler) release(m *thread, reqID string, chains []Chain, now time.Time, obs dispatchObservables) {
	if reqID == "" {
		// An id-less request cannot be tracked: nothing names it, so no window opens and the thread
		// is not held busy by it.
		s.noteAnomaly(anomalyNoRequestID)
		return
	}
	if _, tracked := s.reqs[reqID]; tracked {
		// a RE-release of a request already tracked (a retry that re-entered the queue): replacing
		// the record must correct the counts it was holding, or the thread is charged twice for one
		// turn.
		s.dropRequest(reqID, now)
	}
	s.reqs[reqID] = &reqRec{thread: m.id, chains: chains, released: now,
		queueWait: obs.queueWait, predictedCached: obs.predictedCached, promptTokens: obs.promptTokens,
		counted: true}
	m.pending++
	m.windows = append(m.windows, s.reqs[reqID])
}

// expireWindows drops the thread's windows that have outlived S from its count, and forgets the
// records that are no longer windows. It is the busy() read's own liveness pass: the settle asks
// "has this thread's work drained" at ITS time, so a window that died since the last pass must stop
// counting HERE — otherwise a dead window keeps the thread busy, the turn's busy period is lost, and
// the thread is never rotated at its quantum.
func (s *Scheduler) expireWindows(m *thread, now time.Time) {
	if m == nil || len(m.windows) == 0 {
		return
	}
	kept := m.windows[:0]
	for _, rec := range m.windows {
		if !rec.pod.IsZero() {
			continue // it reached a pod: not a window any more
		}
		if rec.counted && now.Sub(rec.released) > time.Duration(s.cfg.IdleSeconds*float64(time.Second)) {
			rec.counted = false
			m.pending = max(m.pending-1, 0)
		}
		kept = append(kept, rec)
	}
	m.windows = kept
}

// windowLive reports whether a released record's window is still inside S: the thread is
// presumed to be using its seat for that long and evictable after it (see thread.busy).
func (s *Scheduler) windowLive(rec *reqRec, now time.Time) bool {
	if rec == nil || !rec.pod.IsZero() {
		return false
	}
	return now.Sub(rec.released) <= time.Duration(s.cfg.IdleSeconds*float64(time.Second))
}

// atPod closes the released window: the request is running on an engine.
func (s *Scheduler) atPod(rec *reqRec, m *thread, engine string, now time.Time) dispatchObservables {
	if !rec.pod.IsZero() {
		rec.engine = engine
		return dispatchObservables{} // a repeated PreRequest: the record is already in flight
	}
	// LIVENESS IS READ BEFORE THE RECORD STOPS BEING PENDING: a window that already outlived S
	// was never counted (reapRequests adds only live windows to m.pending), so a LATE
	// PreRequest must not decrement the thread — it would take a count away from another live
	// window of the same thread and read the thread as not busy until the next pass re-derives.
	// THE COUNT IS THE RECORD'S OWN, not a guess about whether a pass ran: a window that has expired
	// no longer counts, so decrementing for it would take a count from another live window.
	live := rec.counted
	rec.counted = false
	rec.engine = engine
	rec.pod = now
	if m != nil && rec.thread == m.id {
		// THE ACTIVITY STAMP (baseline, scheduler.go:1023): a request arriving at a pod is activity on
		// its thread, so the idle rule measures silence from the arrival, not from the Pick.
		m.lastActivity = now
		if live {
			m.pending = max(m.pending-1, 0)
		}
		m.inflight++
	}
	return dispatchObservables{queueWait: rec.queueWait, predictedCached: rec.predictedCached, promptTokens: rec.promptTokens}
}

// unlinkWindow forgets a record's entry in its thread's window slice. The slice is the thread's own
// liveness list and records are dropped by IDENTITY: a pointer left there after the record has
// stopped existing is a count waiting to be taken off a live window.
func (s *Scheduler) unlinkWindow(m *thread, rec *reqRec) {
	if m == nil || rec == nil || len(m.windows) == 0 {
		return
	}
	kept := m.windows[:0]
	for _, w := range m.windows {
		if w != rec {
			kept = append(kept, w)
		}
	}
	for i := len(kept); i < len(m.windows); i++ {
		m.windows[i] = nil // let the dropped record go; the slice is reused, not leaked
	}
	m.windows = kept
}

// dropRequest deletes one record and re-derives the counts of the thread that owned it.
func (s *Scheduler) dropRequest(reqID string, now time.Time) {
	rec, ok := s.reqs[reqID]
	if !ok {
		return
	}
	delete(s.reqs, reqID)
	// The nil guard STAYS, and it is not defensiveness for a state that cannot happen: exit
	// deletes the thread's binding BEFORE sweepThread drops its records, so during an exit a
	// record briefly names a thread that is no longer seated. The counts are being discarded with
	// the thread, so there is nothing to correct — but the guard is what makes that non-event.
	if g := s.groupOf(rec.thread); g != nil {
		if m := g.threads[rec.thread]; m != nil {
			s.unlinkWindow(m, rec)
			if rec.pod.IsZero() {
				if rec.counted {
					// a window already expired was cleared by whatever noticed
					rec.counted = false
					m.pending = max(m.pending-1, 0)
				}
			} else {
				m.inflight = max(m.inflight-1, 0)
			}
		}
	}
}

// sweepThread drops every record a thread owns: they name a seat that no longer exists,
// so no later hook could act on them (and a settle for one closes nothing, not a mis-charged
// engine).
func (s *Scheduler) sweepThread(id string, now time.Time) {
	for reqID, rec := range s.reqs {
		if rec.thread == id {
			s.dropRequest(reqID, now)
		}
	}
}

// SeatEngineOfRequest is the Filter's one question: the engine the request's thread is seated on.
// It resolves the request to its thread and the thread to its engine, under s.mu.
func (s *Scheduler) SeatEngineOfRequest(reqID string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	thread := s.threadOfRequest(reqID)
	if thread == "" {
		return "", false
	}
	engine, bound := s.bindings[thread]
	if !bound || engine == "" {
		return "", false
	}
	return engine, true
}

// threadOfRequest is the ONE resolution the scoped hooks go through: the thread a
// request id names, or "" when nothing tracks it (anonymous traffic, a pre-capacity FCFS
// dispatch, a request whose thread has left, or one that no Pick released).
func (s *Scheduler) threadOfRequest(reqID string) string {
	if reqID == "" {
		return ""
	}
	if rec, ok := s.reqs[reqID]; ok {
		return rec.thread
	}
	return ""
}

// ---- PreRequest (post-dispatch): the in-flight record, and the observed engine ----

// AtPod records a released request at PreRequest on the endpoint the framework
// ACTUALLY picked. The thread and the chains come from the request's own record: a request with no
// record is untracked (anonymous, pre-capacity FCFS, or a thread that has already left).
func (s *Scheduler) AtPod(reqID, engine string, now time.Time) dispatchObservables {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec := s.reqs[reqID]
	if rec == nil {
		return dispatchObservables{} // untracked: anonymous traffic, a pre-capacity FCFS dispatch, or a thread that left
	}
	id := rec.thread
	if engine != "" && engine != s.bindings[id] {
		// the observed endpoint is authoritative
		s.rebind(id, engine, now)
	}
	return s.atPod(rec, s.groups[s.bindings[id]].threads[id], engine, now)
}

// Abandon ends a request's record from OUTSIDE the decision path: the request is gone before a
// pod served it (the pinned engine vanished, a screener refused it, scheduling failed), so the
// thread is not waiting on anything any more.
func (s *Scheduler) Abandon(reqID string, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.reqs[reqID]; !ok {
		return // nothing tracks this request: nothing to close
	}
	s.dropRequest(reqID, now)
}

// FirstChunk is the first response chunk for the scoped hook, which is handed a request and
// not a thread: the thread comes from the request's record.
func (s *Scheduler) FirstChunk(reqID string, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	id := s.threadOfRequest(reqID)
	if id == "" {
		return
	}
	if g := s.groupOf(id); g != nil {
		s.releaseCold(g, id)
	}
}

// releaseCold returns one thread's cold-entry charge to its engine's valve, once: at the first chunk,
// at the settle, at the exit, or when the pass finds its owner can no longer be prefilling.
func (s *Scheduler) releaseCold(g *engineGroup, id string) {
	m, ok := g.threads[id]
	if !ok || m.coldDone || m.coldEntry <= 0 {
		return
	}
	g.coldInFlight = max(g.coldInFlight-m.coldEntry, 0)
	m.coldEntry = 0
	m.coldDone = true
}

// ---- settle (ResponseBody at EndOfStream: the finalizer) ----

// Settle finalises a request at the end of its stream: the thread comes from the request's
// record (the scoped hook cannot read the thread id out of the request), so a settle that names no
// record ends a turn this policy never took on, and there is nothing to close.
func (s *Scheduler) Settle(reqID, engine string, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := s.threadOfRequest(reqID)
	if id == "" {
		// NO RECORD, NOTHING TO CLOSE: the policy served this request without taking it on — a head
		// with no rendered tokens (the pass-through, pick.go), a pre-capacity FCFS dispatch (outside
		// the pass, hooks_request.go), anonymous traffic, or a thread that has already left. All of
		// them settle exactly once and none of them is a fault, so this is a non-event, not a kind.
		return
	}
	g := s.groupOf(id)
	// WHERE IT LANDED is the record's engine (the endpoint the observation named at PreRequest),
	// falling back to the seat engine when the record names none.
	landed := g.id
	rec := s.reqs[reqID]
	if rec.engine != "" {
		landed = rec.engine
	}
	if engine != "" && landed != engine {
		s.counters.affinityViolations++
		s.recordEvent("AFFINITY-VIOLATION", id, fmt.Sprintf("bound=%s settled_on=%s", engineLabel(landed), engineLabel(engine)), now)
		s.metrics.ObserveAffinityViolation(engineLabel(landed), engineLabel(engine))
		s.logger.V(logutil.DEFAULT).Info("group-scheduler: AFFINITY VIOLATION at the settle — the request ran on an engine other than the one it was charged to",
			"thread", id, "charged", engineLabel(landed), "settledOn", engineLabel(engine))
	}
	// THE RECORD ENDS HERE, and the counts follow the map (derive-don't-accumulate).
	m := g.threads[id]
	s.dropRequest(reqID, now)
	m.lastActivity = now
	s.releaseCold(g, id) // any termination releases the charge
	// THE WINDOWS THAT DIED SINCE THE LAST PASS STOP COUNTING HERE, at this settle's own time.
	s.expireWindows(m, now)
	// THE BUSY PERIOD: the thread's work reached zero, and one thread is one chain, so that is the
	// whole of it draining — the rotation quantum's unit. busy(), not the at-a-pod count: a turn a
	// Pick has released and PreRequest has not brought to a pod is work too, and it closes the
	// period when IT settles.
	if !m.busy() {
		m.busyPeriods++
		m.lastCompletion = now
	}
	m.seatTurns++ // every settled turn counts in the residency
	// the settled request was the tracked queued head: clear the marker (the next Pick re-
	// establishes it if more heads remain)
	m.waitingSince = time.Time{}
	// an exiting thread leaves when its work drains — the same question as the period above, so the
	// same answer: a live released window keeps it seated
	if g.index.Exiting(id) && !m.busy() {
		s.exit(g, m, "exiting-drain", now)
	}
}

// rebind is the observation winning: a seated thread whose request lands on
// another engine is MOVED to it (one thread's correction instead of unbounded accounting drift
// — the index is an optimistic model of where blocks ARE, and a stale binding drifts
// permanently and silently).
func (s *Scheduler) rebind(id, observed string, now time.Time) *engineGroup {
	from := s.groupOf(id)
	if from == nil || from.id == observed {
		return from
	}
	to, err := s.groupFor(observed)
	if err != nil || to == nil {
		return from
	}
	m := from.threads[id]
	s.counters.affinityViolations++
	from.coldInFlight = max(from.coldInFlight-m.coldEntry, 0)
	chains := from.index.Chains(id)
	from.index.Unseat(id) // the blocks are NOT on this engine (observation wins)
	delete(from.threads, id)
	to.threads[id] = m
	s.bindings[id] = to.id
	to.coldInFlight += m.coldEntry
	// the engine is serving this thread, so its blocks ARE there: seat them without a fit test
	// and let enforcement reconcile the budget (an over-budget counter is recoverable; a thread
	// absent from the engine that is serving it is not).
	to.index.SeatForce(id, chains)
	s.recordEvent("REBIND", id, fmt.Sprintf("from=%s to=%s", engineLabel(from.id), engineLabel(to.id)), now)
	s.metrics.ObserveAffinityViolation(engineLabel(from.id), engineLabel(to.id))
	s.logger.V(logutil.DEFAULT).Info("group-scheduler: AFFINITY VIOLATION — a request landed off its seat; rebound to the observed endpoint",
		"thread", id, "charged", engineLabel(from.id), "observed", engineLabel(to.id))
	return to
}

// derive recomputes the derived aggregates from thread records (derive-
// don't-accumulate): a leaked or missed increment anywhere cannot persist past the next
// decision point.
func (s *Scheduler) derive() {
	for _, g := range s.groups {
		var cold int64
		for _, m := range g.threads {
			cold += m.coldEntry
		}
		g.coldInFlight = cold
	}
}

// reapRequests re-derives every thread's request counts from the map (derive-
// don't-accumulate: a leaked increment cannot survive a decision pass) and drops the records
// that outlived InflightReapSeconds.
func (s *Scheduler) reapRequests(now time.Time) {
	for _, g := range s.groups {
		for _, m := range g.threads {
			m.pending, m.inflight = 0, 0
		}
	}
	limit := time.Duration(s.cfg.InflightReapSeconds * float64(time.Second))
	reaped := map[string]*thread{} // thread id → its record, for the cold-charge release
	for rid, rec := range s.reqs {
		g := s.groupOf(rec.thread)
		var m *thread
		if g != nil {
			m = g.threads[rec.thread]
		}
		if rec.pod.IsZero() {
			if now.Sub(rec.released) > limit {
				// I1 ON THE REAP PATH: the record is about to stop existing, so its count and its
				// place in the thread's window slice go FIRST. A record older than the reap bound is
				// never counted in this pass (the bound is far longer than S), so there is nothing to
				// decrement here — but `counted` left TRUE is a count with no owner: the next settle
				// walks the slice, still finds this record, and takes one off a LIVE window, after
				// which the thread reads not-busy and is rotated (or exits) early.
				rec.counted = false
				s.unlinkWindow(m, rec)
				delete(s.reqs, rid)
				s.noteAnomaly(anomalyReapedWindow)
				continue
			}
			rec.counted = s.windowLive(rec, now)
			if rec.counted && m != nil {
				m.pending++
			}
			continue
		}
		rec.counted = false
		if now.Sub(rec.pod) > limit {
			delete(s.reqs, rid)
			reaped[rec.thread] = m
			s.noteAnomaly(anomalyReaped)
			continue
		}
		m.inflight++
	}
	for id, m := range reaped {
		// the reap dropped an at-a-pod record that outlived the bound, so THIS thread's work is not
		// necessarily over: a live released window is work as well, and the thread's completion is
		// the moment nothing is on its way either
		if m.busy() {
			continue
		}
		m.lastCompletion = now
		// the thread's requests are all dead — release its cold charge, or the valve freezes the
		// queue until the reap bound.
		if g := s.groupOf(id); g != nil {
			s.releaseCold(g, id)
		}
	}
}

// deriveColdCharges releases a cold charge whose owner can no longer be prefilling: no
// queued head, no LIVE released window, and nothing in flight.
func (s *Scheduler) deriveColdCharges() {
	for _, g := range s.activeGroups() {
		for _, m := range g.threads {
			if m.coldEntry == 0 || m.busy() || !m.waitingSince.IsZero() {
				continue
			}
			s.releaseCold(g, m.id)
		}
	}
}
