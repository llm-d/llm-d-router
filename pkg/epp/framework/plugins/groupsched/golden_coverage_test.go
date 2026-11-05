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

// THE COVERAGE SCENARIOS — the shapes that make each rule load-bearing in the trace.
//
// The first golden missed eleven one-rule mutants: no re-entry seat, no idle exit, no reap, no
// victim competition, no in-flight cap, no cold re-derive, no time at all. These scenarios exist
// to make every one of those rules change a decision, so a refactor that breaks it fails here.

import (
	"fmt"
	"time"

	"github.com/stretchr/testify/require"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

var goldenCoverageScenarios = []goldenScenario{
	{
		// M1: a newcomer and a returner both wait for the SAME room. The newcomer must seat first.
		name: "returnerOrder", blocks: 20, sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			first := h.request(1, 500)
			h.identify(first, "T1", 1)
			h.cycle(first, 500, 0, 20)
			h.advance(time.Second)
			// the returner (turn 2) and a newcomer arrive together, both too large to co-reside
			returner := h.request(2, 1400)
			h.identify(returner, "T1", 2)
			newcomer := h.request(1, 1400)
			h.identify(newcomer, "T2", 1)
			h.enqueue(returner)
			h.enqueue(newcomer)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 1400, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M11/M13: a resident goes SILENT with a waiter blocked, so the silence rule (not the
		// quantum) makes it the victim — and it is the one holding a queued turn, so it goes first.
		name: "silentVictim", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			quiet := h.request(1, 1000)
			h.identify(quiet, "T1", 1)
			h.cycle(quiet, 1000, 0, 20)
			busy := h.request(1, 500)
			h.identify(busy, "T2", 1)
			h.cycle(busy, 500, 0, 20)
			// T1 has been silent for longer than S, T2 has not
			h.advance(25 * time.Second)
			waiter := h.request(1, 500)
			h.identify(waiter, "T3", 1)
			h.enqueue(waiter)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M14/M9: two candidates in the pressure fallback classes with different completion
		// times — the OLDER completion must go first.
		name: "twoVictimsCompletion", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			early := h.request(1, 500)
			h.identify(early, "T1", 1)
			h.cycle(early, 500, 0, 20)
			h.advance(5 * time.Second)
			late := h.request(1, 500)
			h.identify(late, "T2", 1)
			h.cycle(late, 500, 0, 20)
			// both now hold a queued turn larger than what is left, and a third waits behind them
			g1 := h.request(2, 1400)
			h.identify(g1, "T1", 2)
			h.enqueue(g1)
			g2 := h.request(2, 1400)
			h.identify(g2, "T2", 2)
			h.enqueue(g2)
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},
	{
		// M21: the CLASS ORDER. One engine over budget with three residents, none of them
		// rotation-eligible (each below Q and within S) and a turn in flight so the engine is not
		// idle: T1 has no queued head (class 2), T2 has one queued at this very instant (class 3),
		// T3 is the running turn that put the engine over budget (class 4). The resident with
		// nothing waiting must go first.
		name: "classOrderNoQueueBeforeMidQuantum", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// three residents, 5 blocks each, all inside the 19-block budget
			noQueue := h.request(1, 300)
			h.identify(noQueue, "T1", 1)
			h.cycle(noQueue, 300, 0, 20) // one busy period, then nothing queued
			h.advance(5 * time.Second)
			queued := h.request(1, 300)
			h.identify(queued, "T2", 1)
			h.cycle(queued, 300, 0, 20)
			h.advance(time.Second)
			third := h.request(1, 300)
			h.identify(third, "T3", 1)
			h.cycle(third, 300, 0, 20)
			// T3's SECOND turn is the grower: it is seated, so Pick dispatches it with no rotate, and
			// its growth pushes `stay` past the budget while it is still in flight
			grow := h.request(2, 1400)
			h.identify(grow, "T3", 2)
			for _, d := range h.arrive(grow) {
				h.toPod(d, "")
			}
			// T2's second turn is queued AT THIS INSTANT: below Q, within S, with a queued head
			turn := h.request(2, 300)
			h.identify(turn, "T2", 2)
			h.enqueue(turn)
			h.finalPass()
		},
	},
	{
		// M13: two rotation-eligible residents, only one of them holding a queued turn. The one
		// that is WAITING for room goes first.
		name: "twoRotationVictims", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			waiting := h.request(1, 500)
			h.identify(waiting, "T1", 1)
			h.cycle(waiting, 500, 0, 20)
			for turn := 2; turn <= 5; turn++ {
				req := h.request(turn, 500+turn*10)
				h.identify(req, "T1", turn)
				h.cycle(req, 500+turn*10, 500, 20)
			}
			quiet := h.request(1, 500)
			h.identify(quiet, "T2", 1)
			h.cycle(quiet, 500, 0, 20)
			for turn := 2; turn <= 5; turn++ {
				req := h.request(turn, 500+turn*10)
				h.identify(req, "T2", turn)
				h.cycle(req, 500+turn*10, 500, 20)
			}
			// T1 now has a queued turn it cannot fit; T2 does not
			g := h.request(6, 1400)
			h.identify(g, "T1", 6)
			h.enqueue(g)
			h.saturate(true)
			h.pickLoop()
			newcomer := h.request(1, 500)
			h.identify(newcomer, "T3", 1)
			h.enqueue(newcomer)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M6: exactly Q completed busy periods and a blocked waiter — eligibility turns on the
		// boundary, so an off-by-one changes the exit.
		name: "quantumExactlyQ", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			resident := h.request(1, 1000)
			h.identify(resident, "T1", 1)
			h.cycle(resident, 1000, 0, 20)
			for turn := 2; turn <= 4; turn++ {
				req := h.request(turn, 1000)
				h.identify(req, "T1", turn)
				h.cycle(req, 1000, 1000, 20)
			}
			// T1 has completed exactly Q=4 busy periods; it is idle (no queued turn), so the
			// ONLY thing that can free this engine is the tenure rule firing on the boundary
			require.Equal(h.t, 4, h.busyPeriodsFor("T1"), "the scenario must sit exactly on Q")
			waiter := h.request(1, 500)
			h.identify(waiter, "T2", 1)
			h.enqueue(waiter)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M7: two engines both have a dispatchable turn, and the earlier arrival is NOT the one
		// with more waiting behind it — the dispatch-order score decides.
		name: "dispatchOrderAcrossEngines", blocks: 20, sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			busyThread := h.request(1, 500)
			h.identify(busyThread, "T1", 1)
			for _, d := range h.arrive(busyThread) {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			// two turns behind the first thread, so its engine has more waiting per seat
			for turn := 2; turn <= 3; turn++ {
				req := h.request(turn, 500+turn*10)
				h.identify(req, "T1", turn)
				h.enqueue(req)
			}
			h.saturate(true)
			h.pickLoop()
			h.advance(time.Second)
			// now a second thread on the other engine, whose single turn arrives later
			other := h.request(1, 500)
			h.identify(other, "T2", 1)
			h.enqueue(other)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M17: the in-flight cap binds — a thread's third turn waits although the budget would fit it.
		name: "inFlightCapBinds", blocks: 20, sizes: []int{1}, cfgExtra: `,"inFlightCap":1`,
		run: func(h *goldenHarness, n int) {
			h.learn()
			var out []dispatched
			for turn := 1; turn <= 3; turn++ {
				req := h.request(turn, 400+turn*10)
				h.identify(req, "T1", turn)
				out = append(out, h.arrive(req)...)
			}
			for _, d := range out {
				h.toPod(d, "")
			}
			for _, d := range out {
				h.finish(d, 500, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M12: the reap bounds the record map — both the released-window reap and the in-flight
		// reap, each with its own anomaly counter.
		name: "reapBothKinds", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// a request released by Pick that never reaches a pod
			windowed := h.request(1, 500)
			h.identify(windowed, "T1", 1)
			h.arrive(windowed)
			// a request that reaches a pod and never settles
			inflight := h.request(1, 500)
			h.identify(inflight, "T2", 1)
			for _, d := range h.arrive(inflight) {
				h.toPod(d, "")
			}
			// past the reap bound
			h.advance(9400 * time.Second)
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},
	{
		// M20: the cold charge is RE-DERIVED from the records, so a charge whose request ended
		// out of band is released rather than leaking until the valve closes the engine.
		name: "coldChargeReDerived", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			first := h.request(1, 240)
			h.identify(first, "T1", 1)
			h.arrive(first) // released, cold charged, then abandoned: no chunk ever releases it
			h.abandon(first)
			h.saturate(true)
			h.pickLoop()
			// a fresh newcomer must not be held by the abandoned charge
			for i := 0; i < 3; i++ {
				req := h.request(1, 240)
				h.identify(req, fmt.Sprintf("T%d", i+2), 1)
				h.enqueue(req)
			}
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 240, 0, 10)
			}
			h.finalPass()
		},
	},
	{
		// M3: an EXITING thread holds a live released window (inflight 0, pending 1). The drain
		// must wait for the window; an inflight-only test would release it early.
		name: "drainRespectsLiveWindow", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// T1 small and in flight; T2 grows past the budget so T1 is a legal victim
			a := h.request(1, 500)
			h.identify(a, "T1", 1)
			outA := h.arrive(a)
			require.Len(h.t, outA, 1)
			h.toPod(outA[0], "")
			t2 := h.request(1, 500)
			h.identify(t2, "T2", 1)
			for _, d := range h.arrive(t2) {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			grow := h.request(2, 1400)
			h.identify(grow, "T2", 2)
			for _, d := range h.arrive(grow) {
				h.toPod(d, "")
				h.finish(d, 1400, 500, 20)
			}
			// a waiter makes the overage enforceable: T1 is marked exiting
			waiter := h.request(1, 500)
			h.identify(waiter, "T3", 1)
			h.enqueue(waiter)
			h.saturate(true)
			h.pickLoop()
			// T1's next turn is released by Pick and never reaches a pod: a LIVE window on an
			// exiting thread, which is what the drain must respect
			held := h.request(2, 600)
			h.identify(held, "T1", 2)
			for _, d := range h.arrive(held) {
				_ = d // the turn is released, and deliberately never taken to a pod
			}
			// T1's first request settles now, so inflight is 0 and the window is all that is left
			h.finish(outA[0], 500, 0, 20)
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},

	{
		// M1: a thread is rotated out while its turn is still queued, so that turn returns to the
		// shared queue as a RE-ENTRY. It then competes with a newcomer for the freed room, and the
		// newcomer must seat first.
		name: "reentryVsNewcomer", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			resident := h.request(1, 1000)
			h.identify(resident, "T1", 1)
			h.cycle(resident, 1000, 0, 20)
			for turn := 2; turn <= 4; turn++ {
				req := h.request(turn, 1000)
				h.identify(req, "T1", turn)
				h.cycle(req, 1000, 1000, 20)
			}
			require.Equal(h.t, 4, h.busyPeriodsFor("T1"), "the resident must be exactly at Q")
			// the resident's next turn is queued but too large to co-exist, so the tenure rule
			// evicts the resident and the turn becomes an UNSEATED re-entry
			held := h.request(5, 1400)
			h.identify(held, "T1", 5)
			h.enqueue(held)
			newcomer := h.request(1, 500)
			h.identify(newcomer, "T2", 1)
			h.enqueue(newcomer)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M11: the SILENCE rule is what makes the victim eligible — one busy period, then longer
		// than S of quiet, with a newcomer blocked behind it.
		name: "silenceMakesEligible", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			quiet := h.request(1, 1000)
			h.identify(quiet, "T1", 1)
			h.cycle(quiet, 1000, 0, 20)
			next := h.request(2, 1000)
			h.identify(next, "T1", 2)
			h.cycle(next, 1000, 1000, 20)
			// past S of silence, and its next turn is queued but does not fit
			h.advance(25 * time.Second)
			waiter := h.request(1, 500)
			h.identify(waiter, "T2", 1)
			h.enqueue(waiter)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M13: two eligible victims, only one of them holding a queued turn. The one waiting for
		// room goes first.
		name: "victimWithQueuedTurnFirst", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			holder := h.request(1, 500)
			h.identify(holder, "T1", 1)
			h.cycle(holder, 500, 0, 20)
			for turn := 2; turn <= 5; turn++ {
				req := h.request(turn, 500)
				h.identify(req, "T1", turn)
				h.cycle(req, 500, 500, 20)
			}
			h.advance(time.Second)
			idle := h.request(1, 500)
			h.identify(idle, "T2", 1)
			h.cycle(idle, 500, 0, 20)
			for turn := 2; turn <= 5; turn++ {
				req := h.request(turn, 500)
				h.identify(req, "T2", turn)
				h.cycle(req, 500, 500, 20)
			}
			// T1's next turn and a newcomer that cannot fit arrive at the SAME instant: rotate runs
			// before dispatch, so T1 is not busy and holds a queued turn (the two eligible victims
			// differ only in hasWaiting).
			held := h.request(5, 1400)
			h.identify(held, "T1", 5)
			h.enqueue(held)
			newcomer := h.request(1, 500)
			h.identify(newcomer, "T3", 1)
			h.enqueue(newcomer)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		// M9/M14: two residents in the pressure fallback classes with different completion times.
		// The older completion must be the victim, whether it is chosen by class 2 (headless) or
		// class 3 (a queued turn).
		name: "olderCompletionFirst", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			early := h.request(1, 500)
			h.identify(early, "T1", 1)
			h.cycle(early, 500, 0, 20)
			h.advance(5 * time.Second)
			late := h.request(1, 500)
			h.identify(late, "T2", 1)
			h.cycle(late, 500, 0, 20)
			// T1 grows past what the engine has, and the growth is honoured (a seat is a right)
			big := h.request(2, 1400)
			h.identify(big, "T1", 2)
			h.cycle(big, 1400, 500, 20)
			// the engine is now over budget with both residents settled. Queue a next turn for each
			// and a third thread behind them: enforcement must choose between two live candidates.
			g1 := h.request(3, 600)
			h.identify(g1, "T1", 3)
			h.enqueue(g1)
			g2 := h.request(3, 600)
			h.identify(g2, "T2", 3)
			h.enqueue(g2)
			waiter := h.request(1, 500)
			h.identify(waiter, "T3", 1)
			h.enqueue(waiter)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 600, 0, 20)
			}
			h.finalPass()
		},
	},
	{
		name: "dispatchScoreDecides", blocks: 20, sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// two threads on e0, one on e1
			for _, b := range []string{"A", "B", "C"} {
				req := h.request(1, 500)
				h.identify(req, b, 1)
				for _, d := range h.arrive(req) {
					h.toPod(d, "")
					h.finish(d, 500, 0, 20)
				}
			}
			h.advance(time.Second)
			// the single thread on e1 queues its next turn FIRST, the two on e0 afterwards, so
			// arrival order and queue depth disagree
			c2 := h.request(2, 600)
			h.identify(c2, "C", 2)
			h.enqueue(c2)
			a2 := h.request(2, 600)
			h.identify(a2, "A", 2)
			h.enqueue(a2)
			b2 := h.request(2, 600)
			h.identify(b2, "B", 2)
			h.enqueue(b2)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 600, 0, 20)
			}
			h.finalPass()
		},
	},

	{
		// M14: the engine is over budget with two settled, head-less residents. Class 2 must take
		// the OLDER completion.
		name: "headlessTwoVictims", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// three residents, 5 blocks each, all inside the 19-block budget
			early := h.request(1, 300)
			h.identify(early, "T1", 1)
			h.cycle(early, 300, 0, 20) // one busy period at t=0
			h.advance(5 * time.Second)
			late := h.request(1, 300)
			h.identify(late, "T2", 1)
			h.cycle(late, 300, 0, 20) // one busy period at t=5s
			h.advance(time.Second)
			third := h.request(1, 300)
			h.identify(third, "T3", 1)
			h.cycle(third, 300, 0, 20)
			// T3's SECOND turn is the grower: it is seated, so Pick dispatches it with no rotate,
			// and the growth pushes `stay` past the budget. It stays in flight, so the next pass
			// has no waiter and no engine-idle victim.
			grow := h.request(2, 1400)
			h.identify(grow, "T3", 2)
			for _, d := range h.arrive(grow) {
				h.toPod(d, "")
			}
			// one waiter: a pass needs a head to run at all, and enforcement runs BEFORE rotate, so
			// the valve sees the two settled residents and the idle rule never gets to them
			waiter := h.request(1, 300)
			h.identify(waiter, "T4", 1)
			h.enqueue(waiter)
			h.finalPass()
		},
	},
	{
		// a repeated PreRequest and a late one (the window is already presumed dead after S)
		name: "preRequestRepeatAndLate", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			first := h.request(1, 300)
			h.identify(first, "T1", 1)
			out := h.arrive(first)
			require.Len(h.t, out, 1)
			h.filter(out[0].req)
			h.preRequest(out[0].req, podNameOf(out[0].engine))
			h.preRequest(out[0].req, podNameOf(out[0].engine)) // the repeat: no window left to close
			// a second request is released and reaches its pod AFTER S, so its window is dead
			late := h.request(2, 300)
			h.identify(late, "T1", 2)
			for _, d := range h.arrive(late) {
				_ = d
			}
			h.advance(25 * time.Second)
			h.preRequest(late, podNameOf(out[0].engine))
			h.finish(out[0], 300, 0, 10)
			h.settleCause(late, podNameOf(out[0].engine), fwkrc.TerminationCauseClientDisconnect)
			h.finalPass()
		},
	},
	{
		// the drain at the top of a pass (not the settle arm): an exiting thread with a dead window
		name: "drainAtPassStart", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			a := h.request(1, 500)
			h.identify(a, "T1", 1)
			outA := h.arrive(a)
			require.Len(h.t, outA, 1)
			h.toPod(outA[0], "")
			t2 := h.request(1, 500)
			h.identify(t2, "T2", 1)
			for _, d := range h.arrive(t2) {
				h.toPod(d, "")
				h.finish(d, 500, 0, 20)
			}
			grow := h.request(2, 1400)
			h.identify(grow, "T2", 2)
			for _, d := range h.arrive(grow) {
				h.toPod(d, "")
				h.finish(d, 1400, 500, 20)
			}
			waiter := h.request(1, 500)
			h.identify(waiter, "T3", 1)
			h.enqueue(waiter)
			h.saturate(true)
			h.pickLoop() // T1 is marked exiting here
			// T1's turn is released, then the window dies (past S): the next pass's drain releases it
			held := h.request(2, 400)
			h.identify(held, "T1", 2)
			for _, d := range h.arrive(held) {
				_ = d
			}
			h.advance(25 * time.Second)
			h.finish(outA[0], 500, 0, 20)
			h.finalPass()
		},
	},
	{
		// an in-flight record that never settles is reaped, and its thread is released
		name: "reapedInflight", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			req := h.request(1, 300)
			h.identify(req, "T1", 1)
			for _, d := range h.arrive(req) {
				h.toPod(d, "") // at a pod, never settled
			}
			h.advance(9400 * time.Second)
			other := h.request(1, 300)
			h.identify(other, "T2", 1)
			h.enqueue(other)
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},
	{
		// a settle that lands on the OTHER engine (the affinity path) and an abnormal cause
		name: "settleElsewhereAbnormal", blocks: 20, sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			req := h.request(1, 300)
			h.identify(req, "T1", 1)
			out := h.arrive(req)
			require.Len(h.t, out, 1)
			h.toPod(out[0], "")
			other := "e1"
			if out[0].engine == "golden/e1" {
				other = "e0"
			}
			h.firstChunk(out[0].req, other)
			h.settleCause(out[0].req, other, fwkrc.TerminationCauseEvicted)
			h.finalPass()
		},
	},
	{
		// two newcomers wait together: the entry order's tie-break decides, and a resident that has
		// never completed a turn is the cheaper victim (olderCompletion's zero arm)
		name: "newcomerTieNeverCompleted", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			fresh := h.request(1, 300)
			h.identify(fresh, "T1", 1)
			for _, d := range h.arrive(fresh) {
				h.toPod(d, "") // seated and never completed: lastCompletion stays zero
			}
			done := h.request(1, 300)
			h.identify(done, "T2", 1)
			h.cycle(done, 300, 0, 20)
			w1 := h.request(1, 400)
			h.identify(w1, "T3", 1)
			w2 := h.request(1, 400)
			h.identify(w2, "T4", 1)
			h.enqueue(w1)
			h.enqueue(w2)
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},
	{
		// two newcomers wait together: the entry order falls through to the arrival tie-break
		name: "newcomerTie", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			big := h.request(1, 1400)
			h.identify(big, "T1", 1)
			h.cycle(big, 1400, 0, 20) // takes the budget alone (Rule O)
			w1 := h.request(1, 300)
			h.identify(w1, "T2", 1)
			w2 := h.request(1, 300)
			h.identify(w2, "T3", 1)
			h.enqueue(w1)
			h.enqueue(w2) // same instant: the tie-break decides
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},
	{
		// two engines with the same free budget: the engine-id tie-break places the head
		name: "equalFreeBudgetTie", blocks: 20, sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			for i := 0; i < 4; i++ {
				req := h.request(1, 300)
				h.identify(req, fmt.Sprintf("T%d", i+1), 1)
				h.enqueue(req)
			}
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 300, 0, 10)
			}
			h.finalPass()
		},
	},
	{
		// the engine sits over budget with ONE staying resident whose own footprint IS the
		// overage: Rule O's no-thrash guard refuses it as a victim, and a waiter stays waiting.
		name: "soleOversizeNoThrash", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			huge := h.request(1, 1400)
			h.identify(huge, "T1", 1)
			h.cycle(huge, 1400, 0, 20) // alone on the engine, over the budget by construction
			waiter := h.request(1, 300)
			h.identify(waiter, "T2", 1)
			h.enqueue(waiter)
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},

	{
		// two engines, one reporting a DIFFERENT block size: its group exists with no budget, so the
		// valve skip and the enforcement return are both reachable, and the mismatch is loud.
		name: "wrongBlockSizeBudgetless", blocks: 20, sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			// e1 reports a different block size: its group exists with no budget at all
			h.setBlockSize("e1", 32)
			h.learn()
			// fill the healthy engine with ONE oversized resident and keep it in flight, so the
			// valve has no rotation victim there either
			huge := h.request(1, 1500)
			h.identify(huge, "T1", 1)
			var busy []dispatched
			busy = append(busy, h.arrive(huge)...)
			for _, d := range busy {
				require.Equal(h.t, "golden/e0", d.engine, "the oversize head must take the healthy engine")
				h.toPod(d, "")
			}
			// a normal head that fits nowhere: the healthy engine is full, the other has no budget
			head := h.request(1, 300)
			h.identify(head, "T2", 1)
			h.enqueue(head)
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},
	{
		// two BUSY victims in one pass: the second valve call sees the first, already marked
		name: "twoBusyVictims", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			a := h.request(1, 300)
			h.identify(a, "T1", 1)
			outA := h.arrive(a)
			require.Len(h.t, outA, 1)
			h.toPod(outA[0], "")
			b := h.request(1, 300)
			h.identify(b, "T2", 1)
			for _, d := range h.arrive(b) {
				h.toPod(d, "")
			}
			// a third thread grows past the budget while both residents are in flight
			c := h.request(1, 300)
			h.identify(c, "T3", 1)
			for _, d := range h.arrive(c) {
				h.toPod(d, "")
			}
			grow := h.request(2, 1400)
			h.identify(grow, "T3", 2)
			for _, d := range h.arrive(grow) {
				h.toPod(d, "")
			}
			waiter := h.request(1, 300)
			h.identify(waiter, "T4", 1)
			h.enqueue(waiter)
			h.finalPass()
		},
	},
	{
		// every resident is EXITING and the engine is still over budget: the valve has no victim
		name: "allExitingOverBudget", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			a := h.request(1, 300)
			h.identify(a, "T1", 1)
			outA := h.arrive(a)
			require.Len(h.t, outA, 1)
			h.toPod(outA[0], "")
			grow := h.request(2, 1400)
			h.identify(grow, "T1", 2)
			for _, d := range h.arrive(grow) {
				h.toPod(d, "")
			}
			waiter := h.request(1, 300)
			h.identify(waiter, "T2", 1)
			h.enqueue(waiter)
			h.saturate(true)
			h.pickLoop() // T1 is marked exiting here and stays (it is in flight)
			h.finalPass()
		},
	},
	{
		// an exiting thread whose request is PENDING (released, never at a pod) is drained at the
		// TOP of a later pass, after the window is abandoned
		name: "drainPassAfterAbandon", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// T1 completes first; T2 completes later, so T1 has the OLDER completion
			a := h.request(1, 300)
			h.identify(a, "T1", 1)
			h.cycle(a, 300, 0, 20)
			h.advance(2 * time.Second)
			b := h.request(1, 300)
			h.identify(b, "T2", 1)
			h.cycle(b, 300, 0, 20)
			// T1's next turn is RELEASED and never reaches a pod: a pending window keeps it busy
			held := h.request(2, 400)
			h.identify(held, "T1", 2)
			for _, d := range h.arrive(held) {
				_ = d
			}
			// T2's second turn is the grower: dispatched, so stay goes over budget and T2 is busy
			grow := h.request(2, 1400)
			h.identify(grow, "T2", 2)
			for _, d := range h.arrive(grow) {
				h.toPod(d, "")
			}
			// a pass with a head: enforce marks T1 (busy, older completion, class 4)
			waiter := h.request(1, 300)
			h.identify(waiter, "T3", 1)
			h.enqueue(waiter)
			h.finalPass()
			// the abandoned window leaves T1 not busy and NOT exited; the next pass's DRAIN exits it
			h.abandon(held)
			h.finalPass()
		},
	},
	{
		// DECLARED (S7c): with NO traffic there is no pass, and nothing is reaped until one runs.
		// The detector alone (which runs at ~1 kHz) must not reap — a reap is the pass's tidy, and the
		// tidy runs BEFORE the decision of the pass that follows it. This scenario has no old-code
		// baseline BY DESIGN: it pins the behaviour the ticker's removal introduced.
		name: "idleNoPassNoReap", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			first := h.request(1, 300)
			h.identify(first, "T1", 1)
			for _, d := range h.arrive(first) {
				h.toPod(d, "")
			}
			second := h.request(2, 400)
			h.identify(second, "T1", 2)
			for _, d := range h.arrive(second) {
				h.toPod(d, "")
			}
			// well past inflightReapSeconds, with NOTHING queued: the detector runs, no pass does
			h.advance(9400 * time.Second)
			h.line("IDLE detector only, no pass")
			h.saturate(false)
			h.digest() // the two records are still there: nothing reaped them
			// traffic arrives: the pass tidies FIRST, so the reap is visible in the state the
			// decision is taken on
			head := h.request(1, 300)
			h.identify(head, "T2", 1)
			h.enqueue(head)
			h.finalPass()
		},
	},
	{
		// the reap meets a thread that still has work in flight, under TRAFFIC: one head is queued at
		// the instant of the reap, so a real pass runs and its tidy (derive, reap, drain) runs BEFORE
		// the decision — which is what the deleted ticker used to do from a timer.
		name: "reapWithInflightLeft", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			first := h.request(1, 300)
			h.identify(first, "T1", 1)
			for _, d := range h.arrive(first) {
				h.toPod(d, "")
			}
			second := h.request(2, 400)
			h.identify(second, "T1", 2)
			for _, d := range h.arrive(second) {
				h.toPod(d, "")
			}
			h.advance(9400 * time.Second)
			// a head at the reap instant: with one queued, the pass runs, tidies first, and the reap
			// finds the two records it is meant to find
			head := h.request(1, 300)
			h.identify(head, "T2", 1)
			h.enqueue(head)
			h.finalPass()
			h.settleCause(second, "e0", fwkrc.TerminationCauseNatural)
			h.finalPass()
		},
	},
	{
		// two newcomers arrive a second apart: the entry order's arrival tie-break decides
		name: "newcomerStagger", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			big := h.request(1, 1400)
			h.identify(big, "T1", 1)
			h.cycle(big, 1400, 0, 20)
			w1 := h.request(1, 300)
			h.identify(w1, "T2", 1)
			h.enqueue(w1)
			h.advance(time.Second)
			w2 := h.request(1, 300)
			h.identify(w2, "T3", 1)
			h.enqueue(w2)
			h.saturate(true)
			h.pickLoop()
			h.finalPass()
		},
	},
	{
		// two residents that have NEVER completed, both in flight: the zero-completion id tie
		name: "twoNeverCompleted", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// two residents whose only request was abandoned before reaching a pod: seated,
			// headless, never completed, inside S
			for _, id := range []string{"T1", "T2"} {
				req := h.request(1, 300)
				h.identify(req, id, 1)
				for _, d := range h.arrive(req) {
					_ = d // released, never at a pod
				}
				h.abandon(req)
			}
			h.finalPass()
			// a third resident grows the engine past its budget and stays in flight
			t3 := h.request(1, 300)
			h.identify(t3, "T3", 1)
			for _, d := range h.arrive(t3) {
				h.toPod(d, "")
			}
			grow := h.request(2, 1400)
			h.identify(grow, "T3", 2)
			for _, d := range h.arrive(grow) {
				h.toPod(d, "")
			}
			// a head must be queued for a pass to run; the valve then sees {T1,T2} in class 2
			waiter := h.request(1, 300)
			h.identify(waiter, "T4", 1)
			h.enqueue(waiter)
			h.finalPass()
		},
	},
}
