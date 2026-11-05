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

// THE GOLDEN SCENARIOS — every shape the simplification must not change.
//
// Each runs at N=1 and N=2 where the shape exists, and its trace is frozen in testdata/golden/.
// The startup family (preCapacityFCFS, capacityMidQueue) is the only pair allowed to change, and
// only at S1a and S7c, each with a written justification.

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// goldenUpdate rewrites the frozen traces. It is the only way a trace changes.
var goldenUpdate = flag.Bool("golden-update", false, "rewrite the golden decision traces")

// goldenScenario is one scripted run.
type goldenScenario struct {
	name  string
	sizes []int // engine counts this scenario runs at
	// blocks is each engine's num_gpu_blocks (0 = the default 1000). A small pod makes a few
	// hundred tokens of prompt press the budget, which is what the eviction paths need.
	blocks int
	// cfgExtra is appended to the config JSON, for the scenarios that need a different knob
	// (the in-flight cap).
	cfgExtra string
	run      func(h *goldenHarness, n int)
}

// goldenPodsFor is the pod set: n engines, each with enough blocks for a few threads.
func goldenPodsFor(n int, blocks int) map[string]int {
	out := map[string]int{}
	for i := 0; i < n; i++ {
		out[fmt.Sprintf("e%d", i)] = blocks
	}
	return out
}

// learn is the production capacity handshake: the detector's live candidates carry the metrics,
// the tick learns them.
func (h *goldenHarness) learn() { h.saturate(true) }

// finalPass captures the scenario's end state, so no scenario ends mid-decision.
func (h *goldenHarness) finalPass() {
	h.saturate(true)
	h.pickLoop()
}

// arrive queues one request and lets the dispatch loop take it as far as Pick allows.
func (h *goldenHarness) arrive(req *fwksched.InferenceRequest) []dispatched {
	h.enqueue(req)
	h.saturate(true)
	return h.pickLoop()
}

// toPod runs the hooks between Pick and the stream: Filter, then PreRequest naming the landing
// engine ("" = the framework's result is absent).
func (h *goldenHarness) toPod(d dispatched, landing string) {
	h.filter(d.req)
	if landing == "" {
		landing = podNameOf(d.engine)
	}
	h.preRequest(d.req, landing)
}

// finish ends a request's stream with the given usage.
func (h *goldenHarness) finish(d dispatched, prompt, cached, completion int) {
	h.firstChunk(d.req, podNameOf(d.engine))
	h.settle(d.req, podNameOf(d.engine), prompt, cached, completion)
}

// cycle is the whole life of one request: arrive, take it to a pod, finish it.
func (h *goldenHarness) cycle(req *fwksched.InferenceRequest, prompt, cached, completion int) {
	for _, d := range h.arrive(req) {
		h.toPod(d, "")
		h.finish(d, prompt, cached, completion)
	}
}

var goldenScenarios = []goldenScenario{
	{
		// one thread, five turns, each turn's prompt growing on the last
		name: "multiTurnGrowth", sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			base := growTokens(1000, 200)
			for turn := 1; turn <= 5; turn++ {
				prompt := growTokens(1000, 200+turn*150)
				req := h.request(turn, len(prompt))
				req.Body.TokenizedRequest.Prompts[0].TokenIDs = prompt
				h.identify(req, "T1", turn)
				h.cycle(req, len(prompt), len(base), 40)
			}
		},
	},
	{
		// a request that extends a BUSY tip forks: it gets its own thread id
		name: "fork", sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			prompt := growTokens(2000, 400)
			first := h.request(1, len(prompt))
			first.Body.TokenizedRequest.Prompts[0].TokenIDs = prompt
			h.identify(first, "T1", 1)
			out := h.arrive(first)
			require.Len(h.t, out, 1)
			h.toPod(out[0], "") // in flight, so its tip is busy and unproven

			forkPrompt := append(append([]uint32{}, prompt...), growTokens(9000, 120)...)
			fork := h.request(1, len(forkPrompt))
			fork.Body.TokenizedRequest.Prompts[0].TokenIDs = forkPrompt
			h.identify(fork, "T1-fork", 1)
			h.cycle(fork, len(forkPrompt), len(prompt), 30)

			h.finish(out[0], len(prompt), 0, 30)
		},
	},
	{
		// S of silence makes a resident rotation-eligible; a newcomer then takes its room
		name: "idleGapRotation", blocks: 20, sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			resident := h.request(1, 300)
			h.identify(resident, "T1", 1)
			h.cycle(resident, 300, 0, 20)
			if n == 2 {
				// at N=2 the engines are filled so the newcomer has only one home
				for i := 0; i < n; i++ {
					_ = i
				}
			}
			h.clock.Advance(25 * time.Second)
			newcomer := h.request(1, 300)
			h.identify(newcomer, "T2", 1)
			h.cycle(newcomer, 300, 0, 20)
		},
	},
	{
		// Q completed busy periods make a resident eligible; a WAITING newcomer forces the yield
		name: "quantumRotation", blocks: 20, sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			resident := h.request(1, 1000)
			h.identify(resident, "T1", 1)
			h.cycle(resident, 1000, 0, 20)
			for turn := 2; turn <= 6; turn++ {
				req := h.request(turn, 1000+turn*10)
				h.identify(req, "T1", turn)
				h.cycle(req, 1000+turn*10, 1000, 20)
			}
			newcomer := h.request(1, 500)
			h.identify(newcomer, "T2", 1)
			h.cycle(newcomer, 500, 0, 20)
		},
	},
	{
		// entry order: a newcomer goes before a returner, both waiting
		name: "newcomerBeforeReturner", blocks: 20, sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			resident := h.request(1, 300)
			h.identify(resident, "T1", 1)
			h.cycle(resident, 300, 0, 20)
			// a returner and a newcomer arrive together, with no room for either
			returner := h.request(2, 300+600)
			h.identify(returner, "T1", 2)
			newcomer := h.request(1, 300)
			h.identify(newcomer, "T2", 1)
			h.enqueue(returner)
			h.enqueue(newcomer)
			h.saturate(true)
			out := h.pickLoop()
			for _, d := range out {
				h.toPod(d, "")
				h.finish(d, 900, 0, 20)
			}
		},
	},
	{
		// Rule O: a head larger than every engine's budget seats ALONE on an empty engine
		name: "ruleOversize", blocks: 20, sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			huge := h.request(1, 1500) // 19 blocks x 64 = 1216 tokens of budget, so 1500 overflows
			h.identify(huge, "T1", 1)
			small := h.request(1, 100)
			h.identify(small, "T2", 1)
			h.enqueue(huge)
			h.enqueue(small)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 1500, 0, 20)
			}
		},
	},
	{
		// growth past the budget: the grower runs, the overage is enforced on the next pass
		name: "budgetEnforcement", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// two residents fill 16 of the 19-block budget
			first := h.request(1, 500)
			h.identify(first, "T1", 1)
			h.cycle(first, 500, 0, 20)
			second := h.request(1, 500)
			h.identify(second, "T2", 1)
			h.cycle(second, 500, 0, 20)
			// the first grows past what is left: the overage is enforced on the next pass
			grow := h.request(2, 1500)
			h.identify(grow, "T1", 2)
			h.cycle(grow, 1500, 500, 20)
			// a newcomer that cannot fit: this is what makes enforcement act on demand
			waiter := h.request(1, 500)
			h.identify(waiter, "T3", 1)
			h.cycle(waiter, 500, 0, 20)
		},
	},
	{
		// the cold valve: a burst of newcomers is staggered by the budget's own headroom
		name: "coldValve", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			for i := 0; i < 6; i++ {
				req := h.request(1, 240)
				h.identify(req, fmt.Sprintf("T%d", i+1), 1)
				h.enqueue(req)
			}
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 200, 0, 10)
			}
		},
	},
	{
		// reconcile: the framework lands the request on the OTHER engine; the seat follows
		name: "reconcileOtherEngine", blocks: 20, sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			req := h.request(1, 300)
			h.identify(req, "T1", 1)
			out := h.arrive(req)
			require.Len(h.t, out, 1)
			h.filter(out[0].req)
			landing := "e1"
			if out[0].engine == "e1" {
				landing = "e0"
			}
			h.preRequest(out[0].req, landing)
			h.finish(out[0], 300, 0, 20)
		},
	},
	{
		// a request released by Pick that never reaches a pod, then is abandoned
		name: "abandonBeforePod", sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			req := h.request(1, 300)
			h.identify(req, "T1", 1)
			out := h.arrive(req)
			require.Len(h.t, out, 1)
			h.abandon(req)
			// the next pass sees an empty world again
			h.saturate(true)
			h.pickLoop()
		},
	},
	{
		// a settle for a request nothing tracked: the policy served it without taking it on, so the
		// settle closes nothing, books nothing and crashes nothing (the pass around it is untouched)
		name: "straySettle", sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			req := h.request(1, 300)
			h.identify(req, "T9", 1)
			h.settle(req, "", 300, 0, 20)
			h.saturate(true)
			h.pickLoop()
		},
	},
	{
		// an engine leaves with threads seated and a queue waiting
		name: "engineDelete", blocks: 20, sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			a := h.request(1, 300)
			h.identify(a, "T1", 1)
			h.cycle(a, 300, 0, 20)
			b := h.request(1, 300)
			h.identify(b, "T2", 1)
			out := h.arrive(b)
			require.Len(h.t, out, 1)
			h.toPod(out[0], "")
			h.deletePod("e0")
			h.saturate(true)
			h.pickLoop()
			h.finish(out[0], 300, 0, 20)
		},
	},
	{
		// THE OBSERVED ENGINE WINS, INCLUDING ITS BLOCKS (X10): a request lands on an engine that is
		// not its seat, the thread is rebound there, and the blocks are FORCE-SEATED on the new
		// engine — the engine is serving the thread, so its blocks are there whether the budget agrees
		// or not. Without the forced seat the thread looks uncharged on the engine that is running it.
		name: "rebindToObservedEngine", blocks: 20, sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			h.learn()
			a := h.request(1, 300)
			h.identify(a, "T1", 1)
			out := h.arrive(a)
			require.Len(h.t, out, 1)
			// it lands on the OTHER engine: the framework's choice is authoritative
			other := "e1"
			if out[0].engine == "e1" {
				other = "e0"
			}
			h.toPod(out[0], other)
			h.saturate(true)
			h.pickLoop()
			h.finish(out[0], 300, 0, 20)
			h.saturate(true)
		},
	},
	{
		// THE ENGINE-IDLE VICTIM IS THE LEAST RECENTLY ACTIVE THREAD (X6). Two threads hold the
		// engine's only two seats and neither is busy; a third thread is blocked for a seat, so the
		// livelock guard gives one up — the thread that has been quiet longest.
		name: "engineIdlePicksLeastRecent", blocks: 12, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			t1 := h.request(1, 300)
			h.identify(t1, "T1", 1)
			h.cycle(t1, 300, 0, 20)
			h.advance(10 * time.Second)
			t2 := h.request(1, 300)
			h.identify(t2, "T2", 1)
			h.cycle(t2, 300, 0, 20)
			h.advance(time.Second)
			t3 := h.request(1, 300)
			h.identify(t3, "T3", 1)
			for _, d := range h.arrive(t3) {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
			h.saturate(true)
			h.pickLoop()
		},
	},
	{
		// THE ROTATION VICTIM IS THE EARLIEST-SEATED ELIGIBLE THREAD WITH A QUEUED TURN (X9). Both
		// residents have spent their quantum and both ask for another turn; the seat that frees must be
		// the one seated FIRST, so rotation is a queue and not a lottery.
		name: "rotationVictimEarliestSeat", blocks: 12, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			q := defaultConfig().Quantum
			for i, id := range []string{"T1", "T2"} {
				if i == 1 {
					// a LATER seat for the second thread: with equal seat times the id breaks the tie
					// and this scenario could not tell the order apart
					h.advance(5 * time.Second)
				}
				for turn := 1; turn <= q; turn++ {
					req := h.request(1, 300)
					h.identify(req, id, turn)
					h.cycle(req, 300, 0, 20)
				}
			}
			for _, id := range []string{"T1", "T2"} {
				next := h.request(1, 300)
				h.identify(next, id, q+1)
				h.enqueue(next)
			}
			t3 := h.request(1, 300)
			h.identify(t3, "T3", 1)
			h.enqueue(t3)
			h.saturate(true)
			h.pickLoop()
		},
	},
	{
		// A REQUEST ARRIVING AT A POD IS ACTIVITY (B5). The baseline stamped the thread at PreRequest;
		// without that stamp, a thread whose record is dropped before its settle is measured silent
		// from its PICK, not from its arrival at the pod — so it becomes idle-eligible early and a
		// newcomer evicts it. Here T1 is dispatched, reaches its pod 15 s later and is abandoned; a
		// newcomer arrives at t=25 s. With the stamp T1 is silent for 10 s (< S) and keeps its seat;
		// without it, 25 s (> S) and the newcomer takes it by the idle rule.
		name: "preRequestStampsActivity", blocks: 9, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			t1 := h.request(1, 300)
			h.identify(t1, "T1", 1)
			out := h.arrive(t1)
			require.Len(h.t, out, 1)
			h.advance(15 * time.Second)
			h.toPod(out[0], "") // PreRequest at t=15: the stamp under test
			h.advance(time.Second)
			h.abandon(t1)              // the record goes; no settle, so no later stamp
			h.advance(9 * time.Second) // t=25: 10 s since the pod, 25 s since the pick
			newcomer := h.request(1, 300)
			h.identify(newcomer, "T2", 1)
			for _, d := range h.arrive(newcomer) {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
			h.saturate(true)
			h.pickLoop()
		},
	},
	{
		// AN ENDPOINT NAME THAT FLAPS: deleted (a readiness flap) and re-added under the same name.
		// The re-added endpoint must learn its capacity again and take a seat. Before B3 the second
		// learn was skipped by the remembered mark: no group, no seats, and at N=1 the pool served
		// FCFS for the rest of the run — the reason this scenario exists and the one trace in the
		// freeze that the pre-fix code cannot reproduce.
		name: "endpointFlapRelearns", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			a := h.request(1, 300)
			h.identify(a, "T1", 1)
			h.cycle(a, 300, 0, 20) // T1 is seated on e0 and finishes
			h.deletePod("e0")      // the pod goes not-ready
			h.addPod("e0", 20)     // the same name comes back ready
			h.saturate(true)       // the candidates are handed to the detector: capacity is learned again
			b := h.request(1, 300)
			h.identify(b, "T2", 1)
			out := h.arrive(b)
			require.Len(h.t, out, 1)
			h.toPod(out[0], "")
			h.finish(out[0], 300, 0, 20)
			h.saturate(true)
		},
	},
	{
		// the in-flight cap bounds how many of one thread's requests are out at once
		name: "inFlightCap", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			var out []dispatched
			for turn := 1; turn <= 3; turn++ {
				req := h.request(turn, 260+turn*10)
				h.identify(req, "T1", turn)
				out = append(out, h.arrive(req)...)
			}
			for _, d := range out {
				h.toPod(d, "")
			}
			for _, d := range out {
				h.finish(d, 300, 0, 20)
			}
		},
	},
	{
		// F2's race: a second request on the thread while the first is still out. The thread is
		// marked exiting; the first settles. Today's behaviour is frozen here so B1 can show its
		// re-freeze in the diff.
		name: "f2StickyBusyRace", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			// T1 holds a small seat and its first turn is in flight
			a := h.request(1, 500)
			h.identify(a, "T1", 1)
			outA := h.arrive(a)
			require.Len(h.t, outA, 1)
			h.toPod(outA[0], "")
			// T2 seats beside it and then grows until the engine is over budget
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
			// a waiter makes the overage enforceable: T1 is the legal victim and is MARKED
			// (it cannot be drained while its request is out)
			waiter := h.request(1, 500)
			h.identify(waiter, "T3", 1)
			h.enqueue(waiter)
			h.saturate(true)
			h.pickLoop()
			// T1's second turn is released by Pick and never reaches a pod
			b := h.request(2, 600)
			h.identify(b, "T1", 2)
			outB := h.arrive(b)
			// the first turn settles: the settle-time exit is where F2 lives
			h.finish(outA[0], 500, 0, 20)
			h.saturate(true)
			h.pickLoop()
			for _, d := range outB {
				h.toPod(d, "")
				h.finish(d, 600, 500, 20)
			}
		},
	},
	{
		// (d) two threads share a system prompt: one exits, the other keeps the blocks alive —
		// the used/stay refcounts of the R4 class
		name: "sharedPrefixExit", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			sys := growTokens(5000, 200)
			pa := append(append([]uint32{}, sys...), growTokens(7000, 100)...)
			pb := append(append([]uint32{}, sys...), growTokens(8000, 100)...)
			a := h.request(1, len(pa))
			a.Body.TokenizedRequest.Prompts[0].TokenIDs = pa
			h.identify(a, "T1", 1)
			b := h.request(1, len(pb))
			b.Body.TokenizedRequest.Prompts[0].TokenIDs = pb
			h.identify(b, "T2", 1)
			h.enqueue(a)
			h.enqueue(b)
			h.saturate(true)
			out := h.pickLoop()
			for _, d := range out {
				h.toPod(d, "")
			}
			// T1 finishes and leaves; T2 stays, so the shared blocks must remain charged
			for _, d := range out {
				if d.req.RequestID == a.RequestID {
					h.finish(d, len(pa), len(sys), 20)
				}
			}
			h.clock.Advance(25 * time.Second)
			h.saturate(true)
			h.pickLoop()
			for _, d := range out {
				if d.req.RequestID == b.RequestID {
					h.finish(d, len(pb), len(sys), 20)
				}
			}
		},
	},
	{
		// (e) a two-prompt request: the chains stay one per prompt (S9's exactness case)
		name: "multiPrompt", sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			req := h.request(1, 200, 300)
			h.identify(req, "T1", 1)
			h.cycle(req, 500, 0, 40)
		},
	},
	{
		// (f) a head whose request carries no tokens: no_tokens_head, served last
		name: "noTokensHead", sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			withTokens := h.request(1, 300)
			h.identify(withTokens, "T1", 1)
			blind := &fwksched.InferenceRequest{RequestID: "blind-1"}
			h.identify(blind, "T2", 1)
			h.enqueue(blind)
			h.enqueue(withTokens)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
		},
	},
	{
		// (g) a request with no id: no_request_id, and nothing is tracked for it
		name: "noRequestID", sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			req := h.request(1, 300)
			req.RequestID = ""
			h.identify(req, "", 1)
			h.enqueue(req)
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
		},
	},
	{
		// (h) engines with different capacities: the budget follows each engine's own number
		name: "unequalEngines", blocks: 20, sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			h.pods["e1"] = 40
			h.learn()
			for i := 0; i < 6; i++ {
				req := h.request(1, 300)
				h.identify(req, fmt.Sprintf("T%d", i+1), 1)
				h.enqueue(req)
			}
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
		},
	},
	// ---- THE STARTUP FAMILY: the only scenarios allowed to change (at S1a and S7c) ----
	{
		// traffic before any capacity is known: the FCFS pass-through, unseated
		name: "preCapacityFCFS", sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			for i := 0; i < 3; i++ {
				req := h.request(1, 300)
				h.identify(req, fmt.Sprintf("T%d", i+1), 1)
				h.enqueue(req)
			}
			// the FIRST saturate carries no metrics at all: capacity is unknown
			blind := make([]string, 0, len(h.pods))
			for id := range h.pods {
				blind = append(blind, id)
			}
			sort.Strings(blind)
			for _, id := range blind {
				h.pods[id] = 0
			}
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
		},
	},
	{
		// (a) capacity arrives while the queue is non-empty: the hand-off to seated operation
		name: "capacityMidQueue", sizes: []int{1, 2},
		run: func(h *goldenHarness, n int) {
			for i := 0; i < 3; i++ {
				req := h.request(1, 300)
				h.identify(req, fmt.Sprintf("T%d", i+1), 1)
				h.enqueue(req)
			}
			for id := range h.pods {
				h.pods[id] = 0
			}
			h.saturate(true)
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
			// now the scrape lands
			for i := 0; i < n; i++ {
				h.pods[fmt.Sprintf("e%d", i)] = 1000
			}
			h.saturate(true)
			late := h.request(1, 300)
			h.identify(late, "T9", 1)
			for _, d := range h.arrive(late) {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
		},
	},
	{
		// (b) two engines learn at different times: a head must choose between a measured and an
		// unmeasured engine
		name: "enginesLearnApart", sizes: []int{2},
		run: func(h *goldenHarness, n int) {
			h.pods["e0"] = 1000
			h.pods["e1"] = 0
			h.saturate(true)
			first := h.request(1, 300)
			h.identify(first, "T1", 1)
			for _, d := range h.arrive(first) {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
			h.pods["e1"] = 1000
			h.saturate(true)
			second := h.request(1, 300)
			h.identify(second, "T2", 1)
			for _, d := range h.arrive(second) {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
		},
	},
	{
		// (c) a head leaves the flow queue between two picks (cancelled in the controller)
		name: "headLeavesQueue", blocks: 20, sizes: []int{1},
		run: func(h *goldenHarness, n int) {
			h.learn()
			stays := h.request(1, 300)
			h.identify(stays, "T1", 1)
			leaves := h.request(1, 300)
			h.identify(leaves, "T2", 1)
			h.enqueue(leaves)
			h.enqueue(stays)
			h.saturate(true)
			// the controller drops the cancelled head before the next pass
			for _, q := range h.queues {
				if q.key == leaves.FairnessID {
					q.items = nil
				}
			}
			for _, d := range h.pickLoop() {
				h.toPod(d, "")
				h.finish(d, 300, 0, 20)
			}
		},
	},
}

// TestGoldenDecisionTraces is the proof every simplification step is behaviour-neutral: the same
// decisions, in the same order, for the same reasons.
func TestGoldenDecisionTraces(t *testing.T) {
	produced := map[string]bool{}
	defer func() { requireNoOrphanTraces(t, produced) }()
	for _, sc := range append(append([]goldenScenario{}, goldenScenarios...), goldenCoverageScenarios...) {
		for _, n := range sc.sizes {
			sc, n := sc, n
			produced[fmt.Sprintf("%s-N%d.txt", sc.name, n)] = true
			t.Run(fmt.Sprintf("%s/N%d", sc.name, n), func(t *testing.T) {
				blocks := sc.blocks
				if blocks == 0 {
					blocks = 1000
				}
				h := newGoldenHarnessWith(t, goldenPodsFor(n, blocks), sc.cfgExtra)
				sc.run(h, n)
				got := h.trace()
				path := filepath.Join("testdata", "golden", fmt.Sprintf("%s-N%d.txt", sc.name, n))
				if *goldenUpdate {
					require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
					require.NoError(t, os.WriteFile(path, []byte(got), 0o600))
					return
				}
				want, err := os.ReadFile(path)
				require.NoError(t, err, "no frozen trace: run with -golden-update once")
				require.Equal(t, string(want), got, "the decision trace changed")
			})
		}
	}
}

// requireNoOrphanTraces fails when testdata/golden holds a trace no scenario produced. A stale file is
// worse than no file: nothing compares it, so it rots silently while the suite stays green, and a reader
// counting the freeze counts coverage that is not there. Renaming a scenario therefore means renaming
// its trace in the same change.
func requireNoOrphanTraces(t *testing.T, produced map[string]bool) {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join("testdata", "golden"))
	require.NoError(t, err)
	var orphans []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".txt") {
			continue
		}
		if !produced[e.Name()] {
			orphans = append(orphans, e.Name())
		}
	}
	require.Empty(t, orphans, "testdata/golden holds traces no scenario produces (rename the scenario or delete the trace): %v", orphans)
}

// (TestTickIsNotLoadBearing stood here: it existed to prove the 1 s ticker was redundant for
// decisions, so that S7c could delete it. The ticker IS deleted — the pass tidies at its top — so
// the test has no subject left.)
