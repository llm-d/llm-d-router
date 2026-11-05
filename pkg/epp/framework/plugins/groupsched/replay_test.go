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

// THE N-ENGINE SIMULATION — the determinism guard that stands where the goldens did.

import (
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"
)

// ---- the virtual clock ----

var scenarioT0 = time.Unix(1793000000, 0).UTC()

// ---- the scenario model ----

// gsSession is one scripted chain: its turns arrive on a pattern, grow their context, and
// sometimes go silent or re-enter late.
type gsSession struct {
	id string
	// blocks is the context the thread has RUN so far (its chain's length).
	blocks   int
	turn     int
	kind     string // "main" | "single" | "reentry" | "canceller"
	nextTurn int    // sim second of the next arrival
}

// turnChain is the chain the thread's NEXT turn will run: everything the thread has run plus
// this turn's growth, with the keys ordered so that a later turn extends an earlier one (a
// thread's turns are a continuation).
func (g *gsSession) turnChain() []Chain {
	n := g.blocks + g.turnBlocks()
	return []Chain{{Keys: keys(n, seedFor(g.id)), Tokens: n * 64}}
}

// scenarioSessions builds the scripted session set: 46 sessions over 2,600 sim seconds against
// a 1000-block capacity (budget 950), enough pressure for seating, rotation, budget-idle
// exits, TTL and re-entry.
func scenarioSessions() []*gsSession {
	var ss []*gsSession
	for i := 0; i < 46; i++ {
		kind := "main"
		switch {
		case i%11 == 7:
			kind = "single" // one request then silence (TTL path)
		case i%13 == 5:
			kind = "reentry" // a few turns, long silence, returns
		case i%17 == 3:
			kind = "canceller" // one non-natural settle mid-run
		}
		ss = append(ss, &gsSession{
			id:       fmt.Sprintf("s%02d", i),
			blocks:   8, // the context it starts with
			kind:     kind,
			nextTurn: 20 + 9*i, // staggered arrivals, 20s..425s
		})
	}
	return ss
}

// turnBlocks is the per-turn context growth (blocks): mains grow steadily; singles stay small.
func (g *gsSession) turnBlocks() int {
	switch g.kind {
	case "single":
		return 6
	default:
		return 10 + g.turn/2 // 10..~40 blocks over a long chain
	}
}

// thinkGap is the inter-turn gap pattern (seconds).
func (g *gsSession) thinkGap() int {
	switch g.kind {
	case "single":
		return 1 << 30 // never returns
	case "reentry":
		if g.turn >= 3 {
			return 900 // long silence, then re-enter via nextTurn below
		}
		return 6 + g.turn*7
	default:
		return 3 + (g.turn*5)%23 // 3..25s
	}
}

// ---- the driver ----

type scenarioTrace struct {
	sb    strings.Builder
	seats map[string]bool
}

func (g *scenarioTrace) line(format string, args ...any) {
	fmt.Fprintf(&g.sb, format+"\n", args...)
}

// observeState appends the seated set ONLY when it changes (keeps the golden small without
// losing decisions).
func (g *scenarioTrace) observeState(s *Scheduler) {
	cur := s.seatedIDs()
	if len(cur) != len(g.seats) {
		goto diff
	}
	for id := range cur {
		if !g.seats[id] {
			goto diff
		}
	}
	return
diff:
	g.seats = cur
	ids := make([]string, 0, len(cur))
	for id := range cur {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	g.line("STATE seats=%d [%s]", len(ids), strings.Join(ids, ","))
}

// runScenario drives the scheduler and returns the trace.
func runScenario(t *testing.T) string {
	return runScenarioEnd(t, 2600)
}

// runScenarioCfg is the PARAMETRIZED driver used by the WAVE-1 session-ordering equality test:
// cfg is the scheduler config and sessions maps a scripted chain's id to its SESSION attribute
// (nil = every head carries none, i.e. session := thread — base mode).
func runScenarioCfg(t *testing.T, simEnd int, cfg Config) string {
	return runScenarioFull(t, simEnd, cfg)
}

// runScenarioEnd drives the scenario truncated at simEnd (the legacy-c57e92ed comparison uses
// 489 — the pre-C4 code stalls on this scenario shortly after ~500, AND at t=490 the always-on
// C8 fix deliberately diverges from legacy (a thread with a queued head is no longer idle-
// evicted — the reviewer's predicted A0 caveat).
func runScenarioEnd(t *testing.T, simEnd int) string {
	t.Helper()
	return runScenarioFull(t, simEnd, defaultConfig())
}

func runScenarioFull(t *testing.T, simEnd int, cfg Config) string {
	t.Helper()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// The scenario names its ONE engine: a real discovered engine whose budget is 950 of 1000
	// blocks.
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	return runScenarioOnSched(t, simEnd, s)
}

// runScenarioOn replays the golden scenario on an ALREADY-BUILT scheduler (the multi-engine
// properties are asserted this way).
func runScenarioOn(t *testing.T, s *Scheduler) string {
	t.Helper()
	return runScenarioOnSched(t, 2600, s)
}

func runScenarioOnSched(t *testing.T, simEnd int, s *Scheduler) string {
	tr := &scenarioTrace{seats: map[string]bool{}}

	// exit/entry observation: the ORDER relative to the op is part of the trace (the stream fires
	// inside the op). It reads the event stream, which is the one observer the core keeps — the
	// detail is trimmed back to the two fields this trace has always printed, so a per-engine
	// replay still compares against the same golden line for line (PREREG-multipod-offload
	// directive 1: the unit-equivalence gate).
	var exitLog, entryLog []string
	s.onEvent = func(ev Event) {
		head, _, _ := strings.Cut(ev.Detail, " fairness_id=")
		switch ev.Kind {
		case "EXIT":
			exitLog = append(exitLog, "EXIT "+head)
		case "SEAT":
			entryLog = append(entryLog, "SEAT "+head)
		}
	}

	ss := scenarioSessions()
	byID := func(id string) *gsSession {
		for _, g := range ss {
			if g.id == id {
				return g
			}
		}
		return nil
	}

	// reqSeq numbers a session's requests (reqIDs must be unique).
	reqSeq := map[string]int{}
	nextReqID := func(id string) string {
		reqSeq[id]++
		return fmt.Sprintf("%s-r%02d", id, reqSeq[id])
	}

	// inFlight: the requests dispatched and not yet settled; settle after a decode period.
	type flight struct {
		sess     *gsSession
		reqID    string
		arr      int
		blocks   int
		settleAt int
		natural  bool
	}
	var flying []*flight

	headsAt := func(sim int) []Head {
		var hs []Head
		for _, g := range ss {
			if g.nextTurn <= sim {
				// the head is queued since nextTurn
				hs = append(hs, Head{
					Thread:  g.id,
					ReqID:   fmt.Sprintf("%s-r%02d", g.id, reqSeq[g.id]+1),
					Arrival: scenarioT0.Add(time.Duration(g.nextTurn) * time.Second),
					Chains:  g.turnChain(),
					// the arrival record thread identity publishes: a thread's FIRST turn is a newcomer,
					// every later one a returner
					Returner: g.turn > 0,
				})
			}
		}
		return hs
	}

	for sim := 0; sim <= simEnd; sim++ {
		now := scenarioT0.Add(time.Duration(sim) * time.Second)

		// settle due flights (decode periods of 5..40s)
		var still []*flight
		for _, f := range flying {
			if f.settleAt <= sim {
				g := f.sess
				usage := struct{ Prompt, Cached int }{Prompt: f.blocks}
				if f.natural {
					usage.Cached = f.blocks - 2 // mostly warm
					if usage.Cached < 0 {
						usage.Cached = 0
					}
				}
				s.Chunk(g.id)
				s.Settle(f.reqID, "", scenarioT0.Add(time.Duration(f.settleAt)*time.Second))
				tr.line("t=%d SETTLE %s req=%s natural=%v", f.settleAt, g.id, f.reqID, f.natural)
				for _, l := range exitLog {
					tr.line("t=%d %s", f.settleAt, l)
				}
				exitLog = nil
				tr.observeState(s)
			} else {
				still = append(still, f)
			}
		}
		flying = still

		// the 1s ticker (as the plugin runs)
		s.Pass(nil, now)

		// dispatch attempts: up to 2 picks per sim second
		for pick := 0; pick < 2; pick++ {
			hs := headsAt(sim)
			if len(hs) == 0 {
				break
			}
			i := s.Pass(hs, now)
			if i < 0 {
				tr.line("t=%d PICK -1", sim)
				break
			}
			h := hs[i]
			g := byID(h.Thread)
			reqID := nextReqID(g.id)
			// the dispatched turn's context is the one the head carried
			g.turn++
			g.blocks += g.turnBlocks()
			s.Pass(nil, now) // the released window: a pass before PreRequest
			s.releaseThread(g.id, reqID, nil, now)
			s.Dispatched(reqID, now)
			// schedule the settle; cancellers fail their 3rd turn
			natural := true
			if g.kind == "canceller" && g.turn == 3 {
				natural = false
			}
			decode := 5 + (g.turn*7)%35
			flying = append(flying, &flight{sess: g, reqID: reqID, arr: sim,
				blocks: g.blocks, settleAt: sim + decode, natural: natural})
			// the next arrival (or silence)
			gap := g.thinkGap()
			if g.kind == "reentry" && g.turn == 3 {
				g.nextTurn = sim + 900 // the long silence, then re-entry
			} else if gap < 1<<29 {
				g.nextTurn = sim + gap
			} else {
				g.nextTurn = 1 << 29 // silent forever (single-shot)
			}
			tr.line("t=%d PICK %d -> %s req=%s blocks=%d", sim, i, g.id, reqID, g.blocks)
			for _, l := range entryLog {
				tr.line("t=%d %s", sim, l)
			}
			entryLog = nil
			tr.observeState(s)
		}
		// hook emissions from Tick (ttl/budget exits with no op line)
		for _, l := range exitLog {
			tr.line("t=%d %s", sim, l)
		}
		exitLog = nil
		for _, l := range entryLog {
			tr.line("t=%d %s", sim, l)
		}
		entryLog = nil
		tr.observeState(s)
	}

	// final gauges (derived, not per-event): exit counters and snapshot
	snap := s.Snapshot(s.clockNow())
	tr.line("SNAPSHOT served=%d fcfs=%d", snap.Served, snap.FCFSServed)
	var reasons []string
	for r := range snap.Exits {
		reasons = append(reasons, r)
	}
	sort.Strings(reasons)
	for _, r := range reasons {
		tr.line("EXITS %s=%d", r, snap.Exits[r])
	}
	return tr.sb.String()
}

// TestDeterminism_OneAndTwoEngines replays the SAME synthetic N=1 and N=2 event streams twice
// each and requires identical traces.
func TestDeterminism_OneAndTwoEngines(t *testing.T) {
	for _, n := range []int{1, 2} {
		first := runScenarioEngines(t, n)
		for i := 2; i <= 3; i++ {
			if again := runScenarioEngines(t, n); again != first {
				gotL, wantL := strings.Split(again, "\n"), strings.Split(first, "\n")
				for k := 0; k < len(gotL) || k < len(wantL); k++ {
					ga, wa := "", ""
					if k < len(gotL) {
						ga = gotL[k]
					}
					if k < len(wantL) {
						wa = wantL[k]
					}
					if ga != wa {
						t.Fatalf("N=%d: replay %d diverges at line %d:\n  run %d: %s\n  run 1: %s", n, i, k+1, i, ga, wa)
					}
				}
				t.Fatalf("N=%d: replay %d diverges (length)", n, i)
			}
		}
		if first == "" {
			t.Fatalf("N=%d: the scenario produced an empty trace", n)
		}
	}
}

// runScenarioEngines replays the scenario against n engines of equal capacity: the same event
// stream, one more placement choice per pass.
func runScenarioEngines(t *testing.T, n int) string {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := s.SetNumBlocksFor(fmt.Sprintf("engine-%c", 'a'+i), 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	return runScenarioOnSched(t, 2600, s)
}
