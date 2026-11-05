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

// THE TEST-ONLY API — the core's own accessors, moved out of the production build.
//
// Each of these has no caller outside the tests: they exist so a test can read the state machine
// and drive it by hand. They are the same code as before, byte for byte; only the file changed.

import (
	"fmt"
	"time"
)

// releaseThread is a release WITHOUT Pick's growth charge — test scaffolding, which names the
// THREAD rather than holding it. It calls the scheduler's own release, so the record and the pending
// count follow production exactly; the old fixtures never charged at dispatch either, and the GOLDEN
// drives the real Pick.
func (s *Scheduler) releaseThread(thread, reqID string, chains []Chain, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groupOf(thread)
	if g == nil {
		return
	}
	m, ok := g.threads[thread]
	if !ok {
		return
	}
	s.release(m, reqID, chains, now, dispatchObservables{})
}

// reqOf is the request record a thread holds ("" when none): test scaffolding for a helper that
// knows the thread but not the id the fixture gave the request.
func (s *Scheduler) reqOf(thread string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	for reqID, rec := range s.reqs {
		if rec.thread == thread {
			return reqID
		}
	}
	return ""
}

// Dispatched records a released request at PreRequest without an observed endpoint (the
// single-engine form): the thread's seat decides the engine.
func (s *Scheduler) Dispatched(reqID string, now time.Time) {
	s.DispatchedOn(reqID, "", now)
}

// DispatchedOn is the endpoint-aware Dispatched.
func (s *Scheduler) DispatchedOn(reqID, engine string, now time.Time) {
	s.AtPod(reqID, engine, now)
}

// Chunk is the first response chunk: prefill complete — the cold-entry charge is released
// HERE, at prefill completion rather than at end of decode, exactly once.
func (s *Scheduler) Chunk(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if g := s.groupOf(id); g != nil {
		s.releaseCold(g, id)
	}
}

// ChunkAt is Chunk for a caller that has the clock (kept so the live path and the virtual-
// clock tests take the same code).
func (s *Scheduler) ChunkAt(id string, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	if g := s.groupOf(id); g != nil {
		s.releaseCold(g, id)
	}
}

// Seated reports whether the thread holds a seat on ANY engine.
func (s *Scheduler) Seated(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seatedAnywhere(id)
}
func (s *Scheduler) seatedAnywhere(id string) bool {
	for _, g := range s.groups {
		if _, ok := g.threads[id]; ok {
			return true
		}
	}
	return false
}

// seatedIDs is the union of every engine's seated thread ids (the replay's state observation).
func (s *Scheduler) seatedIDs() map[string]bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]bool{}
	for _, g := range s.groups {
		for id := range g.threads {
			out[id] = true
		}
	}
	return out
}

// SeatRecords is the number of per-engine seat records: one per THREAD-SEAT, so it equals the
// seat count by construction and a number that drifts is a leak (the record-leak test's
// input).
func (s *Scheduler) SeatRecords() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, g := range s.groups {
		n += g.index.TrackedCount()
	}
	return n
}

// TrackedRequests is how many request records exist.
func (s *Scheduler) TrackedRequests() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.reqs)
}

// RequestThread is the scheduler's own request→thread resolution, for a caller that only has
// the request id (the scoped-hooks test asserts every hook agrees with it; nothing in the
// hooks reads a request attribute to get the same answer).
func (s *Scheduler) RequestThread(reqID string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	b := s.threadOfRequest(reqID)
	return b, b != ""
}

// Seated reports whether the thread holds a seat here.
func (b *BlockIndex) Seated(id string) bool {
	_, ok := b.seats[id]
	return ok
}

// TrackedCount is the number of seat records (one per seat, by construction).
func (b *BlockIndex) TrackedCount() int {
	return len(b.seats)
}

// BlockSizeTokens is the engine block size in tokens.
func (b *BlockIndex) BlockSizeTokens() int { return b.blockSizeTokens }

// checkInvariant recomputes both refcounts from the records and panics on divergence.
func (b *BlockIndex) checkInvariant() {
	want := make(map[Hash]int)
	wantStay := make(map[Hash]int)
	for _, s := range b.seats {
		for h := range s.charged {
			want[h]++
			if !s.exiting {
				wantStay[h]++
			}
		}
	}
	if !sameCounts(want, b.blocks) {
		panic(fmt.Sprintf("invariant: blocks %v != recomputed %v", b.blocks, want))
	}
	if !sameCounts(wantStay, b.blocksStay) {
		panic(fmt.Sprintf("invariant: blocksStay %v != recomputed %v", b.blocksStay, wantStay))
	}
}
func sameCounts(a, c map[Hash]int) bool {
	if len(a) != len(c) {
		return false
	}
	for h, n := range a {
		if c[h] != n {
			return false
		}
	}
	return true
}

// HasNE reports whether a key is on the USED side.
func (b *BlockIndex) HasNE(h Hash) bool {
	_, ok := b.blocks[h]
	return ok
}

// newUnitMetrics is the metric surface for a scheduler that has no plugin around it: a fixture. The
// plugin a live instance always has is built here with the maps and the clock Factory would give it,
// so the collector's scrape is the same code path in a unit test as in a run.
func newUnitMetrics(s *Scheduler) *groupMetrics {
	return newPluginMetrics(&GroupScheduler{
		sched: s, now: time.Now,
		endpoints: map[string]struct{}{}, learned: map[string]struct{}{},
		rejected: map[string]struct{}{},
	})
}

// clockNow is the scheduler's own clock (the last pass's `now`), for a test that has no virtual
// clock in scope at the call site.
func (s *Scheduler) clockNow() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.now
}

// engineOfDispatch is the engine a pass served a head ON, read from what the pass LEFT BEHIND: the
// winner's released record, its thread, and the thread's binding. "" when nothing tracks the id —
// which is exactly the answer for the FCFS fall-through (it releases no record) and for a head with
// no request id (there is nothing to record). Production ids are server-minted and never empty, so
// the empty-id case exists only for a fixture that invents heads.
func (s *Scheduler) engineOfDispatch(reqID string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.reqs[reqID]; !ok {
		return ""
	}
	id := s.threadOfRequest(reqID)
	if g := s.groupOf(id); g != nil {
		return g.id
	}
	return ""
}
