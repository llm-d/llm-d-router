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

// EVERY KEPT COUNTER MOVES ON ITS OWN EVENT, and on nothing else.

import (
	"testing"
	"time"

	fcTypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"

	"github.com/prometheus/client_golang/prometheus"
)

// metricFixture wires the plugin's metric hooks to a fresh registry, so a test can read one
// series' delta without a scrape.
type metricFixture struct {
	s   *Scheduler
	reg *prometheus.Registry
	m   *groupMetrics
}

func newMetricFixture(t *testing.T, engines ...string) *metricFixture {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	m := newUnitMetrics(s)
	reg := prometheus.NewRegistry()
	if err := m.register(reg); err != nil {
		t.Fatal(err)
	}
	s.metrics = m
	for _, e := range engines {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	return &metricFixture{s: s, reg: reg, m: m}
}

// delta is one series' value in the registry as it stands.
func (f *metricFixture) value(t *testing.T, name string, labels map[string]string) float64 {
	t.Helper()
	families, err := f.reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, fam := range families {
		if fam.GetName() != subsystem+"_"+name {
			continue
		}
		for _, met := range fam.GetMetric() {
			match := true
			for k, want := range labels {
				found := false
				for _, lp := range met.GetLabel() {
					if lp.GetName() == k && lp.GetValue() == want {
						found = true
					}
				}
				if !found {
					match = false
				}
			}
			if !match {
				continue
			}
			switch {
			case met.GetCounter() != nil:
				return met.GetCounter().GetValue()
			case met.GetGauge() != nil:
				return met.GetGauge().GetValue()
			case met.GetHistogram() != nil:
				return float64(met.GetHistogram().GetSampleCount())
			}
		}
	}
	return 0
}

func (f *metricFixture) seat(t *testing.T, id string, blocks int, now time.Time) {
	t.Helper()
	if i := f.s.Pass([]Head{head(id, time.Second, blocks)}, now); i != 0 {
		t.Fatalf("%s must seat", id)
	}
	releasedWindow(f.s, now) // the release → PreRequest gap every dispatch passes through
	f.s.releaseThread(id, "r-"+id, nil, now)
	f.s.Dispatched("r-"+id, now)
}

// dispatches_total moves on a DISPATCH, and not on a seat alone.
func TestCounterDispatchesMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	if got := f.value(t, "dispatches_total", map[string]string{"engine": "e0"}); got != 0 {
		t.Fatalf("before any dispatch: %v, want 0", got)
	}
	f.seat(t, "m", 100, now)
	if got := f.value(t, "dispatches_total", map[string]string{"engine": "e0"}); got != 1 {
		t.Fatalf("one dispatch: %v, want 1", got)
	}
}

// exits_total moves on an EXIT with the reason it happened for.
func TestCounterExitsMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "m", 100, now)
	settle(f.s, "m", 100, 100) // the turn ends: nothing in flight, one busy period
	f.s.mu.Lock()
	f.s.groups["e0"].threads["m"].busyPeriods = 4 // past Q
	f.s.mu.Unlock()
	// a waiting head that cannot fit evicts the rotation-eligible thread
	f.s.mu.Lock()
	g := f.s.groups["e0"]
	g.budgetBlocks = 100
	reason, victim := f.s.rotationVictim(g, now)
	if victim == nil {
		t.Fatal("the thread must be rotation-eligible")
	}
	f.s.exit(g, victim, reason, now)
	f.s.mu.Unlock()
	if got := f.value(t, "exits_total", map[string]string{"engine": "e0", "reason": reason}); got != 1 {
		t.Fatalf("exits_total{reason=%q} = %v, want 1", reason, got)
	}
	// the other reasons are pre-instantiated at 0 (a zero is a fact, an absent series is a
	// question) — but only the reasons an exit can actually happen for: a stale one would sit in
	// every scrape as a line that can never move.
	if _, ok := gatherValue(t, f.reg, subsystem+"_exits_total", map[string]string{"engine": "e0", "reason": "quantum-complete"}); !ok {
		t.Fatal("every live exit reason must be pre-instantiated for a known engine")
	}
	if _, ok := gatherValue(t, f.reg, subsystem+"_exits_total", map[string]string{"engine": "e0", "reason": "ttl"}); ok {
		t.Fatal("a reason no exit can produce must have no series at all")
	}
}

// cold_valve_held_total moves when the VALVE holds a head — once per pass that holds it, not
// once per candidate engine and not once per loop iteration.
func TestCounterColdValveHeldMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0")
	// a resident whose cold charge fills the cap (5 % of 1000 blocks = 3200 tokens)
	f.s.mu.Lock()
	g := f.s.groups["e0"]
	g.budgetBlocks = 950
	f.s.mu.Unlock()
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "resident", 600, now)
	before := f.value(t, "cold_valve_held_total", map[string]string{"engine": "e0"})
	// a head the budget would take and only the valve refuses
	if i := f.s.Pass([]Head{head("cold", time.Second, 300)}, now); i != -1 {
		t.Fatalf("the valve must hold the head, got %d", i)
	}
	if got := f.value(t, "cold_valve_held_total", map[string]string{"engine": "e0"}); got != before+1 {
		t.Fatalf("cold_valve_held_total: %v -> %v, want exactly one increment for the held head", before, got)
	}
}

// gift: entry_wait_seconds moves once per SEAT with the kind that seat had.
func TestHistogramEntryWaitMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	before := f.value(t, "entry_wait_seconds", map[string]string{"kind": "new"})
	f.seat(t, "m", 100, now)
	if got := f.value(t, "entry_wait_seconds", map[string]string{"kind": "new"}); got != before+1 {
		t.Fatalf("entry_wait_seconds{kind=new}: %v -> %v, want +1", before, got)
	}
	if got := f.value(t, "entry_wait_seconds", map[string]string{"kind": "reentry"}); got != 0 {
		t.Fatalf("a first turn must not be counted as a re-entry, got %v", got)
	}
}

// engine_gone_total moves on a TEARDOWN.
func TestCounterEngineGoneMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0", "e1")
	if got := f.value(t, "engine_gone_total", nil); got != 0 {
		t.Fatalf("before any teardown: %v", got)
	}
	f.s.RemoveEngine("e1", time.Unix(1793000000, 0).UTC())
	if got := f.value(t, "engine_gone_total", nil); got != 1 {
		t.Fatalf("engine_gone_total: %v, want 1", got)
	}
}

// affinity_violations_total moves on an OBSERVED engine that disagrees with
// the seat.
func TestCountersAffinityAndRebindMoveOnTheirEvent(t *testing.T) {
	f := newMetricFixture(t, "e0", "e1")
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "m", 100, now)
	if got := f.value(t, "rebinds_total", nil); got != 0 {
		t.Fatalf("before any rebound: %v", got)
	}
	// the framework lands the request on e1 although the seat is on e0
	f.s.releaseThread("m", "r-m2", nil, now)
	f.s.DispatchedOn("r-m2", "e1", now)
	if got := f.value(t, "affinity_violations_total", map[string]string{"from": "e0", "to": "e1"}); got != 1 {
		t.Fatalf("affinity_violations_total{e0->e1}: %v, want 1", got)
	}
}

// and the SERIES covers BOTH sites: a violation whose symptom is the settle (the request ran
// on an engine other than the one it was charged to, so there is nothing left to rebind) moves
// the same counter — and a settle-side violation must not count a repair, because nothing was.
func TestCounterAffinityViolationMovesAtTheSettleToo(t *testing.T) {
	f := newMetricFixture(t, "e0", "e1")
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "m", 100, now) // seated on e0 and charged there
	f.s.releaseThread("m", "r-m", nil, now)
	f.s.DispatchedOn("r-m", "e0", now)
	// the response's own endpoint metadata says the turn ran on e1
	f.s.Settle("r-m", "e1", now)
	if got := f.value(t, "affinity_violations_total", map[string]string{"from": "e0", "to": "e1"}); got != 1 {
		t.Fatalf("affinity_violations_total{e0->e1} after the settle-side violation: %v, want 1", got)
	}
}

// anomalies_total is ONE counter for the events the policy cannot act on, and every kind is
// SHOULD-BE-ZERO.
func TestCounterAnomaliesMoveOnTheirEvent(t *testing.T) {
	now := time.Unix(1793000000, 0).UTC()

	t.Run("no_tokens_head", func(t *testing.T) {
		f := newMetricFixture(t, "e0")
		// a thread head the render produced no tokens for: it cannot be seated or charged
		f.s.Pass([]Head{{Thread: "m", ReqID: "r-m", Arrival: now}}, now)
		if got := f.value(t, "anomalies_total", map[string]string{"kind": anomalyNoTokens}); got != 1 {
			t.Fatalf("anomalies_total{no_tokens_head}: %v, want 1", got)
		}
		if got := f.value(t, "anomalies_total", map[string]string{"kind": anomalyReaped}); got != 0 {
			t.Fatalf("an unrelated kind moved: %v", got)
		}
	})

	t.Run("no_request_id", func(t *testing.T) {
		f := newMetricFixture(t, "e0")
		h := head("m", time.Second, 100)
		h.ReqID = "" // nothing can name this turn
		if i := f.s.Pass([]Head{h}, now); i < 0 {
			t.Fatal("the head must still be served (an unnameable turn is not refused)")
		}
		if got := f.value(t, "anomalies_total", map[string]string{"kind": anomalyNoRequestID}); got != 1 {
			t.Fatalf("anomalies_total{no_request_id}: %v, want 1", got)
		}
	})

	t.Run("settle_with_no_record", func(t *testing.T) {
		f := newMetricFixture(t, "e0")
		// a settle for a request nothing tracks ends a turn the policy never took on: no counter
		// moves and the dump stays empty (the two legitimate sources are tested in
		// stray_settle_test.go)
		f.s.Settle("never-released", "", now)
		if got := len(f.s.Snapshot(now).Anomalies); got != 0 {
			t.Fatalf("a settle with no record booked %d kind(s)", got)
		}
	})
}

// THE PRE-CAPACITY WINDOW IS VISIBLE IN dispatches_total.
func TestPreCapacityDispatchIsCountedOnTheNocapacitySeries(t *testing.T) {
	ctx := t.Context()
	plugin, err := Factory("gs", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	gs := plugin.(*GroupScheduler)
	reg := prometheus.NewRegistry()
	if err := gs.metrics.register(reg); err != nil {
		t.Fatal(err)
	}
	// NO SetNumBlocksFor: the policy is pre-capacity, so Pick passes every head through
	fc, _ := newIntegrationHarness(t, gs)
	req := bareRequest("s-1", 100, map[string]string{})
	req.key = fwkfc.FlowKey{ID: "s-1", Priority: 0}
	outcome, err := fc.EnqueueAndWait(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.String() != fcTypes.QueueOutcomeDispatched.String() {
		t.Fatalf("a pre-capacity stack must still serve: outcome %v", outcome)
	}
	families, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	got := -1.0
	for _, fam := range families {
		if fam.GetName() != subsystem+"_dispatches_total" {
			continue
		}
		for _, m := range fam.GetMetric() {
			for _, lp := range m.GetLabel() {
				if lp.GetName() == "engine" && lp.GetValue() == nocapacityEngine {
					got = m.GetCounter().GetValue()
				}
			}
		}
	}
	if got != 1 {
		t.Fatalf("dispatches_total{engine=%q} = %v, want 1", nocapacityEngine, got)
	}
}
