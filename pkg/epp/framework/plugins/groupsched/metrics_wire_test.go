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

// THE METRIC SURFACE, AS SCRAPED.

import (
	"sort"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// keptMetrics is the design's list, name for name.
var keptMetrics = []string{
	"seats", "ne_blocks", "budget_blocks", "waiters_shared",
	"exiting_threads", "exiting_blocks", "exiting_oldest_age_seconds",
	"cold_inflight_tokens", "cold_cap_tokens", "cold_valve_held_total",
	"capacity_missing",
	"dispatches_total", "exits_total",
	"entry_wait_seconds",
	"engine_gone_total", "affinity_violations_total",
	"anomalies_total",
}

// droppedMetrics must NOT be registered any more.
var droppedMetrics = []string{
	"unique_blocks", "num_gpu_blocks", "waiters", "residency_turns", "refill_share",
	"fcfs_dispatches_total", "filter_decisions_total", "queue_view_age_seconds",
	"event_ring_dropped_total", "capacity_heterogeneous", "queue_view_untrusted_passes_total",
	"engine_kv_usage", "engine_waiting_queue",
	"safety_fired", "safety_fires_total", "safety_fired_seconds_total", "safety_skips_total",
	"detector_pool_score", "detector_stale_samples_total",
	"running_blocks", "execution_reserved_blocks", "execution_budget_blocks",
	"dispatch_holds_total", "dispatch_held_turns_total", "dispatch_held_seconds",
	"cold_valve_held_gated_total", "gate_charge_hold_seconds",
	"tier_shadow_blocks", "tier_shadow_capacity_blocks", "tier_shadow_pairs_total",
}

// TestMetricSurfaceIsExactlyTheKeptSet drives one seat, one wait, one dispatch, one exit and
// one cold-valve hold through the real hooks, then scrapes: every kept name is present, every
// dropped name is absent, and the engine label carries the endpoint id (not "single", not "").
func TestMetricSurfaceIsExactlyTheKeptSet(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		reg := prometheus.NewRegistry()
		// the LIVE surface: capacity_missing is rendered from the plugin's known endpoints
		p := &GroupScheduler{sched: fx.s, now: time.Now, endpoints: map[string]struct{}{},
			learned: map[string]struct{}{}, rejected: map[string]struct{}{}}
		m := newPluginMetrics(p)
		if err := m.register(reg); err != nil {
			t.Fatalf("register: %v", err)
		}
		fx.s.metrics = m
		// the fixture built its engines before the metrics existed (production wires them first);
		// this is what groupFor does for every engine at the moment it is created.
		for _, id := range fx.ids {
			m.preInstantiateEngine(id)
		}

		// a KNOWN endpoint with no measured geometry: the capacity_missing series
		p.rememberEndpoint(metaFor(t, fx.ids[1%len(fx.ids)]))

		// one event per per-label-pair counter, so its series exists to be scraped
		fx.s.metrics.ObserveColdValveHeld(engineLabel(fx.ids[0]))
		fx.s.metrics.ObserveAffinityViolation(engineLabel(fx.ids[0]), engineLabel(fx.ids[len(fx.ids)-1]))
		fx.s.metrics.ObserveEngineGone()

		fx.seatOne(t, "m", 100, 5*time.Second)
		fx.s.Chunk("m")
		fx.s.Pass(nil, fx.clock)

		mfs, err := reg.Gather()
		if err != nil {
			t.Fatalf("gather: %v", err)
		}
		got := make(map[string]bool, len(mfs))
		for _, mf := range mfs {
			got[mf.GetName()] = true
		}
		for _, name := range keptMetrics {
			want := subsystem + "_" + name
			if !got[want] {
				missing := make([]string, 0, len(got))
				for n := range got {
					missing = append(missing, n)
				}
				sort.Strings(missing)
				t.Fatalf("kept metric %s is not exported; got %v", want, missing)
			}
		}
		for _, name := range droppedMetrics {
			if got[subsystem+"_"+name] {
				t.Fatalf("dropped metric %s is still exported", name)
			}
		}
		// the PRE-CAPACITY bucket exists at zero (a zero is a fact), and it is not an engine id
		if v, ok := gatherValue(t, reg, subsystem+"_dispatches_total",
			map[string]string{"engine": nocapacityEngine}); !ok || v != 0 {
			t.Fatalf("dispatches_total{engine=%q} must exist at 0 on a capacity-measured stack, got %v (present=%v)",
				nocapacityEngine, v, ok)
		}
		// the engine label is the endpoint id the framework gave us
		if _, ok := gatherValue(t, reg, subsystem+"_seats", map[string]string{"engine": engineLabel(fx.ids[0])}); !ok {
			t.Fatalf("seats must carry engine=%s", engineLabel(fx.ids[0]))
		}
		// the merged wait histogram carries the kind label with BOTH values in use
		if _, ok := gatherValue(t, reg, subsystem+"_entry_wait_seconds", map[string]string{"kind": "new"}); !ok {
			t.Fatal("entry_wait_seconds must carry kind=new for a newcomer's wait")
		}
		// its top finite bucket is above the request timeout: nothing may land only in +Inf
		for _, mf := range mfs {
			if mf.GetName() != subsystem+"_entry_wait_seconds" {
				continue
			}
			bounds := mf.GetMetric()[0].GetHistogram().GetBucket()
			top := bounds[len(bounds)-1].GetUpperBound()
			if top < 9000 {
				t.Fatalf("entry_wait_seconds top finite bucket is %v s, want >= 9000 (the request timeout)", top)
			}
		}
	})
}
