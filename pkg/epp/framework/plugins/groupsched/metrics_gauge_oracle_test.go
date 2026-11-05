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

// THE GAUGE EQUIVALENCE ORACLE.
//
// The gauges used to be pushed by Refresh from a Snapshot; they are now rendered at scrape by
// stateGauges. That is a MECHANISM change, so it has to be proven to be a VALUE change of
// nothing: refreshOracle below is the deleted code, kept here as the reference, and every
// scenario's end state must produce the same families — names, label sets and values — through
// both paths.

import (
	"fmt"
	"sort"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/stretchr/testify/require"
)

// refreshOracle is the Refresh body as it stood before the collector existed: it writes the same
// gauges into fresh GaugeVecs on a private registry.
type refreshOracle struct {
	seats, neBlocks, budgetBlocks, exitingThreads, exitingBlocks, exitingOldest *prometheus.GaugeVec
	coldInFlight, coldCap                                                       *prometheus.GaugeVec
	waitersShared                                                               prometheus.Gauge
	known                                                                       map[string]struct{}
}

func newRefreshOracle(reg prometheus.Registerer) *refreshOracle {
	o := &refreshOracle{known: map[string]struct{}{}}
	mk := func(name, help string) *prometheus.GaugeVec {
		v := prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: name, Help: help}, engineLabels)
		reg.MustRegister(v)
		return v
	}
	o.seats = mk("seats", "seated threads (residencies) on this engine")
	o.neBlocks = mk("ne_blocks", "distinct blocks charged to this engine's budget: every seated thread's set, exiting threads included until they drain")
	o.budgetBlocks = mk("budget_blocks", "seat budget in blocks (groupBudgetFraction x this engine's num_gpu_blocks)")
	o.exitingThreads = mk("exiting_threads", "threads on this engine that have been marked exiting and still hold their seat until their in-flight requests drain")
	o.exitingBlocks = mk("exiting_blocks", "distinct blocks of this engine's ne_blocks that belong to exiting threads: the part of the used side that is already leaving (admission still refuses it, enforcement no longer evicts for it)")
	o.exitingOldest = mk("exiting_oldest_age_seconds", "how long this engine's OLDEST exiting thread has been on its way out (a value that only grows is a drain that is not happening)")
	o.coldInFlight = mk("cold_inflight_tokens", "outstanding cold-entry charge in tokens against this engine's stagger cap")
	o.coldCap = mk("cold_cap_tokens", "DERIVED cold-entry stagger cap in tokens: floor((1-groupBudgetFraction) x this engine's num_gpu_blocks) x blockSizeTokens")
	o.waitersShared = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "waiters_shared",
		Help: "heads waiting in the ONE shared admission queue"})
	reg.MustRegister(o.waitersShared)
	return o
}

// refresh is the DELETED production body, verbatim.
func (o *refreshOracle) refresh(snap Snapshot) {
	seen := make(map[string]struct{}, len(snap.Engines))
	for label, es := range snap.Engines {
		seen[label] = struct{}{}
		o.known[label] = struct{}{}
		o.seats.WithLabelValues(label).Set(float64(es.Seats))
		o.neBlocks.WithLabelValues(label).Set(float64(es.NEBlocks))
		o.budgetBlocks.WithLabelValues(label).Set(float64(es.BudgetBlocks))
		o.coldInFlight.WithLabelValues(label).Set(float64(es.ColdInFlight))
		o.coldCap.WithLabelValues(label).Set(float64(es.ColdCapTokens))
		o.exitingThreads.WithLabelValues(label).Set(float64(es.ExitingThreads))
		o.exitingBlocks.WithLabelValues(label).Set(float64(es.ExitingBlocks))
		o.exitingOldest.WithLabelValues(label).Set(es.ExitingOldestAgeS)
	}
	for label := range o.known {
		if _, ok := seen[label]; ok {
			continue
		}
		delete(o.known, label)
		for _, vec := range []*prometheus.GaugeVec{o.seats, o.neBlocks,
			o.exitingThreads, o.exitingBlocks, o.exitingOldest,
			o.budgetBlocks, o.coldInFlight, o.coldCap} {
			vec.DeleteLabelValues(label)
		}
	}
	o.waitersShared.Set(float64(snap.WaitersShared))
}

// gaugeFamilies reduces a scrape to its GAUGE families, rendered as name{labels}=value lines so
// two registries can be compared exactly.
func gaugeFamilies(t *testing.T, reg *prometheus.Registry) []string {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	var out []string
	for _, mf := range mfs {
		if mf.GetType() != dto.MetricType_GAUGE {
			continue
		}
		for _, m := range mf.GetMetric() {
			out = append(out, fmt.Sprintf("%s%s=%v", mf.GetName(), labelString(m.GetLabel()), m.GetGauge().GetValue()))
		}
	}
	sort.Strings(out)
	return out
}

func labelString(pairs []*dto.LabelPair) string {
	if len(pairs) == 0 {
		return ""
	}
	s := "{"
	for i, p := range pairs {
		if i > 0 {
			s += ","
		}
		s += p.GetName() + "=" + `"` + p.GetValue() + `"`
	}
	return s + "}"
}

// TestGaugesMatchTheDeletedRefreshPath proves the collector change is value-neutral: for each
// scenario's END STATE, the deleted Refresh path and the new scrape-time collector produce the
// same gauge families.
func TestGaugesMatchTheDeletedRefreshPath(t *testing.T) {
	scenarios := []struct {
		name string
		n    int
	}{
		{"multiTurnGrowth", 1}, {"multiTurnGrowth", 2},
		{"engineDelete", 1}, {"engineDelete", 2},
		{"wrongBlockSizeBudgetless", 1}, {"wrongBlockSizeBudgetless", 2},
		{"silentVictim", 1}, {"classOrderNoQueueBeforeMidQuantum", 1},
	}
	for _, sc := range scenarios {
		sc := sc
		t.Run(fmt.Sprintf("%s/N%d", sc.name, sc.n), func(t *testing.T) {
			h := newGoldenHarness(t, goldenPodsFor(sc.n, 1000))
			scenarioByName(t, sc.name).run(h, sc.n)

			// the ORACLE: the deleted Refresh code, fed the same snapshot
			oracleReg := prometheus.NewRegistry()
			oracle := newRefreshOracle(oracleReg)
			oracle.refresh(h.r.sched.Snapshot(h.clock.Now()))

			// the NEW path: what a scrape of the plugin's own registration sees
			newReg := prometheus.NewRegistry()
			m := newUnitMetrics(h.r.sched)
			require.NoError(t, m.register(newReg))
			for id := range h.r.sched.groups {
				m.preInstantiateEngine(id)
			}

			require.Equal(t, gaugeFamilies(t, oracleReg), gaugeFamilies(t, newReg),
				"the scrape-time gauges differ from the deleted Refresh path")
		})
	}
}

// scenarioByName finds a scenario in either list.
func scenarioByName(t *testing.T, name string) goldenScenario {
	t.Helper()
	for _, sc := range append(append([]goldenScenario{}, goldenScenarios...), goldenCoverageScenarios...) {
		if sc.name == name {
			return sc
		}
	}
	t.Fatalf("no scenario named %q", name)
	return goldenScenario{}
}

// TestEngineRemovalDropsGaugesButKeepsCounters pins the asymmetry the collector exists for: a
// removed engine's GAUGE series must disappear (a stale gauge is a lie), and its COUNTER series
// must survive (the arms sum them).
func TestEngineRemovalDropsGaugesButKeepsCounters(t *testing.T) {
	h := newGoldenHarness(t, goldenPodsFor(2, 1000))
	h.learn()
	reg := prometheus.NewRegistry()
	m := newUnitMetrics(h.r.sched)
	require.NoError(t, m.register(reg))
	h.r.sched.metrics = m

	// seat something on e1 so it has non-zero counters and a live gauge
	req := h.request(1, 300)
	h.identify(req, "gone", 1)
	h.enqueue(req)
	h.saturate(true)
	for _, d := range h.pickLoop() {
		if d.engine == h.r.sched.groupOrder[len(h.r.sched.groupOrder)-1] {
			h.toPod(d, "")
		}
	}
	victimID := h.r.sched.groupOrder[len(h.r.sched.groupOrder)-1]
	m.ObserveExit(engineLabel(victimID), "exiting-drain")

	h.r.sched.RemoveEngine(victimID, h.clock.Now())
	after := gatherNames(t, reg)

	require.NotContains(t, after, fmt.Sprintf("%s_seats{engine=%q}", subsystem, victimID),
		"a removed engine's gauge must not be reported")
	require.Contains(t, after, fmt.Sprintf("%s_exits_total{engine=%q,reason=\"exiting-drain\"}", subsystem, victimID),
		"a removed engine's counter series must survive: the arms sum it")
}

// gatherNames renders one line per series (name + labels) so a presence check reads plainly.
func gatherNames(t *testing.T, reg *prometheus.Registry) []string {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	var out []string
	for _, mf := range mfs {
		for _, m := range mf.GetMetric() {
			out = append(out, mf.GetName()+labelString(m.GetLabel()))
		}
	}
	return out
}
