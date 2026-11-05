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

// THE OBSERVABILITY SUITE — the two series that made a real defect invisible (2026-11-01,
// reviewer's ask; both observability-only, golden-neutral): 1. cold_valve_held_total was
// attributed `engine="single"` on a multi-engine stack, because the admission loop passed an
// empty engine id.

import (
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// gatherValue pulls one sample out of a fresh gather, matching every label pair given.
func gatherValue(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) (float64, bool) {
	t.Helper()
	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			match := true
			seen := map[string]string{}
			for _, lp := range m.GetLabel() {
				seen[lp.GetName()] = lp.GetValue()
			}
			for k, v := range labels {
				if seen[k] != v {
					match = false
					break
				}
			}
			if !match {
				continue
			}
			return sampleValue(m), true
		}
	}
	return 0, false
}

// sampleValue reads whichever of the metric's value fields this series uses.
func sampleValue(m *dto.Metric) float64 {
	switch {
	case m.GetGauge() != nil:
		return m.GetGauge().GetValue()
	case m.GetCounter() != nil:
		return m.GetCounter().GetValue()
	case m.GetUntyped() != nil:
		return m.GetUntyped().GetValue()
	}
	return 0
}

// TestColdValveHeldIsAttributedToTheEngine: the valve-held series must name the engine whose
// prefill slot the head is waiting for, never "" (which the exporter renders `engine="single"`
// on a two-engine stack).
func TestColdValveHeldIsAttributedToTheEngine(t *testing.T) {
	// capacity 1,000 blocks: the DERIVED cap is 50 blocks = 3,200 tokens, so a 300-block
	// resident's cold charge fills it completely
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// the counter IS the observation: this is what the arm's series shows
		reg := prometheus.NewRegistry()
		m := newUnitMetrics(fx.s)
		if err := m.register(reg); err != nil {
			t.Fatalf("register: %v", err)
		}
		fx.s.metrics = m

		// one cold charge per engine, and the FIRST engine deliberately fuller so the two engines'
		// free budgets differ (300 blocks used on e0, 200 on e1).
		fx.seatOn(t, fx.ids[0], "cold-0", 300, 3*time.Second)
		if len(fx.ids) > 1 {
			fx.seatOn(t, fx.ids[1], "cold-1", 200, 3*time.Second)
		}

		// a head that fits every engine's BUDGET but is refused by every engine's VALVE
		if i := fx.pick(head("waiting", time.Second, 200)); i >= 0 {
			t.Fatalf("the head must NOT be seated (the valve holds it), got pick %d", i)
		}
		// EVERY held engine must be one of this stack's, booked under its own id: an empty label
		// would render engine="single" on a multi-engine stack (the defect this pins).
		held := map[string]float64{}
		for _, id := range fx.ids {
			if got, ok := gatherValue(t, reg, subsystem+"_cold_valve_held_total", map[string]string{"engine": id}); ok && got > 0 {
				held[id] = got
			}
		}
		if len(held) == 0 {
			t.Fatal("the valve held a head and fired no signal: cold_valve_held_total would stay 0 on a cap-binding arm")
		}
		if want := fx.ids[len(fx.ids)-1]; len(fx.ids) > 1 && held[want] == 0 {
			// e1 has the most free budget, so it is the engine the head is actually waiting for: the
			// attribution follows placement's own order, not the loop's.
			t.Fatalf("cold_valve_held_total{engine=%q} = 0, want > 0 (the engine with the MOST FREE BUDGET among those that fit by budget); held on %v", want, held)
		}
	})
}

// THE DERIVED VALVE CAP IS SCRAPEABLE PER ENGINE: a scorer that divides the cold charge by its
// cap must read the cap from the arm, not from a constant copied out of a config.
func TestColdCapIsExportedPerEngine(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 4000, func(t *testing.T, fx *engineSet) {
		reg := prometheus.NewRegistry()
		m := newUnitMetrics(fx.s)
		if err := m.register(reg); err != nil {
			t.Fatalf("register: %v", err)
		}
		want := float64(200 * 64) // 5 % of 4,000 blocks x 64 tokens
		for _, id := range fx.ids {
			if got, ok := gatherValue(t, reg, subsystem+"_cold_cap_tokens", map[string]string{"engine": id}); !ok || got != want {
				t.Fatalf("cold_cap_tokens{engine=%s} = %v (present=%v), want %v", id, got, ok, want)
			}
		}
	})
}

// THE EXITING LEDGER ON THE WIRE (2026-11-01 late, the reviewer's ask: "this leak class must
// be visible at T+5").
func TestExitingLedgerIsVisiblePerEngine(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.tighten(t)
		fx.seatOne(t, "keep", 400, 5*time.Second)
		fx.s.Chunk("keep")
		fx.seatOne(t, "gone", 300, 5*time.Second)
		m, g := fx.thread(t, "gone")

		reg := prometheus.NewRegistry()
		mtr := newUnitMetrics(fx.s)
		if err := mtr.register(reg); err != nil {
			t.Fatalf("register: %v", err)
		}
		if got, ok := gatherValue(t, reg, subsystem+"_exiting_threads", map[string]string{"engine": engineLabel(g.id)}); !ok || got != 0 {
			t.Fatalf("a settled engine reports %v exiting threads (present=%v), want 0", got, ok)
		}

		// the mark (the pressure valve's mid-flight class)
		fx.s.mu.Lock()
		inFlight(fx.s, m, fx.clock, "r-gone")
		g.index.MarkExiting("gone")
		m.exitingSince = fx.clock
		fx.s.mu.Unlock()

		fx.tick(7 * time.Second) // the decision clock advances; the drain has not (one request in flight)
		if got, ok := gatherValue(t, reg, subsystem+"_exiting_threads", map[string]string{"engine": engineLabel(g.id)}); !ok || got != 1 {
			t.Fatalf("exiting_threads = %v (present=%v), want 1", got, ok)
		}
		if got, ok := gatherValue(t, reg, subsystem+"_exiting_blocks", map[string]string{"engine": engineLabel(g.id)}); !ok || got != 300 {
			t.Fatalf("exiting_blocks = %v (present=%v), want 300 (the thread's own footprint)", got, ok)
		}
		if got, ok := gatherValue(t, reg, subsystem+"_exiting_oldest_age_seconds", map[string]string{"engine": engineLabel(g.id)}); !ok || got < 7 {
			t.Fatalf("exiting_oldest_age_seconds = %v (present=%v), want >= 7", got, ok)
		}
		// and the used side still counts them: the two views are visible side by side
		if got, ok := gatherValue(t, reg, subsystem+"_ne_blocks", map[string]string{"engine": engineLabel(g.id)}); !ok || got != 700 {
			t.Fatalf("ne_blocks = %v (present=%v), want 700 (admission still sees the exiting blocks)", got, ok)
		}

		// the drain: the ledger empties
		fx.s.Settle("r-gone", g.id, fx.clock)
		for _, name := range []string{"_exiting_threads", "_exiting_blocks", "_exiting_oldest_age_seconds"} {
			if got, ok := gatherValue(t, reg, subsystem+name, map[string]string{"engine": engineLabel(g.id)}); !ok || got != 0 {
				t.Fatalf("%s = %v (present=%v) after the drain, want 0", name, got, ok)
			}
		}
	})
}

// THE SEAT/EXIT LINE CARRIES THE CLIENT'S OWN ID.
func TestSeatAndExitCarryTheClientsRequestID(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		h := head("m", time.Second, 100)
		h.WireID = "thread-9-r17"
		events := newEventCollector(fx.s)
		if i := fx.pick(h); i < 0 {
			t.Fatal("m must be seated")
		}
		fx.s.Settle(h.ReqID, "", fx.clock)
		// the thread leaves (a teardown is the deterministic exit; the id must still be on the EXIT
		// line, which fires long after the head that carried it is gone)
		fx.s.RemoveEngine(fx.s.bindings["m"], fx.clock)

		var seat, exit string
		for _, ev := range events.events() {
			if ev.ID != "m" {
				continue
			}
			switch ev.Kind {
			case "SEAT":
				seat = ev.Detail
			case "EXIT":
				exit = ev.Detail
			}
		}
		if seat == "" || exit == "" {
			t.Fatalf("want a SEAT and an EXIT event for m, got seat=%q exit=%q", seat, exit)
		}
		for _, d := range []string{seat, exit} {
			if !strings.Contains(d, "wire_id=thread-9-r17") {
				t.Fatalf("the event detail must carry the client's request id, got %q", d)
			}
		}
	})
}
