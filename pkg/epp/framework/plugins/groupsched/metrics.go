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

import (
	"context"
	"sort"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// groupMetrics exports the MINIMAL set an operator needs to see liveness and fairness or to
// diagnose a known failure class — twenty names, and nothing else.
type groupMetrics struct {
	sched *Scheduler

	// gauges renders every GAUGE at scrape time, from ONE Snapshot: the counters and
	// histograms below stay event-driven, because a per-engine counter series must survive its
	// engine being removed while a gauge must not.
	gauges *stateGauges
	exits  *prometheus.CounterVec
	// entryWait is ONE histogram for BOTH seat waits (newcomer and re-entry), split by `kind`:
	// they answer the same question (how long did admission make this thread wait), and two names
	// meant two bucket sets and two queries.
	entryWait *prometheus.HistogramVec
	// queueWait and the two token counters are OBSERVABILITY: nothing in the decision reads them.
	queueWait        *prometheus.HistogramVec
	predictedCached  *prometheus.CounterVec
	dispatchedPrompt *prometheus.CounterVec
	served           *prometheus.CounterVec

	// affinityViolations counts requests that landed off their thread's seat engine.
	affinityViolations *prometheus.CounterVec
	engineGone         prometheus.Counter
	// anomalies is the ONE kept counter for the events the policy cannot act on, by kind
	// (anomalyKinds).
	anomalies *prometheus.CounterVec
	// coldValveHeld counts heads the cold-entry VALVE held back: the budget would have admitted
	// the head, the valve refused it, so the head WAITED for the engine's prefill slot.
	coldValveHeld *prometheus.CounterVec
}

// subsystem namespaces every series (llm_d_epp_group_*).
const subsystem = "llm_d_epp_group"

// engineLabels is the ONE label every per-engine series carries: the endpoint id.
var engineLabels = []string{"engine"}

// newPluginMetrics is the metric surface, built on the plugin: the gauges are rendered from a
// Snapshot at scrape time and capacity_missing from the plugin's known endpoints.
func newPluginMetrics(r *GroupScheduler) *groupMetrics {
	return &groupMetrics{sched: r.sched, gauges: newStateGauges(r)}
}

// register creates every series and installs the global zero series (the pre-capacity dispatch label
// and one anomaly kind per kind): a zero is a fact, an absent series is a question.
func (g *groupMetrics) register(reg prometheus.Registerer) error {
	g.exits = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "exits_total",
		Help: "thread exits by reason"}, []string{"engine", "reason"})
	g.entryWait = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "entry_wait_seconds",
		Help:    "seat waits by kind (new = a thread's first turn, reentry = a returning thread): how long admission made it wait, TOP BUCKET 13,107 s so nothing lands only in +Inf",
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 18)}, []string{"engine", "kind"})
	// OBSERVABILITY ONLY (no decision reads either series): the queue wait a request spent between
	// arriving and being dispatched, and the cache the engine is PREDICTED to hold for it.
	g.queueWait = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "queue_wait_seconds",
		Help:    "queue waits by kind (new = a thread's first turn, returner = a later turn, untracked = served by the pass-through): dispatch time minus the head's arrival, PER REQUEST. Not the seat wait: seat_wait_seconds is a waiter's wait to be SEATED (arrival to first seat); this is every dispatched turn, including a seated thread's turn waiting on the in-flight cap or the dispatch order. Same buckets, so nothing lands only in +Inf",
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 18)}, []string{"kind"})
	g.predictedCached = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "predicted_cached_tokens_total",
		Help: "prompt tokens this engine was PREDICTED to hold as a resident prefix for the turn, summed over a request's prompts, computed read-only at dispatch BEFORE the charge. Its ratio to dispatched_prompt_tokens_total is the predicted prefix hit rate, comparable with the engine's own prefix-cache hit/query counters"}, engineLabels)
	g.dispatchedPrompt = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "dispatched_prompt_tokens_total",
		Help: "prompt tokens of every dispatched turn, summed over a request's prompts — the denominator of predicted_cached_tokens_total"}, engineLabels)
	g.served = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "dispatches_total",
		Help: "dispatches served: per engine, by that engine's seats; under engine=\"nocapacity\", the pre-capacity pass-through (no engine had a measured num_gpu_blocks yet, so nothing was seated and the request was served unseated)"}, engineLabels)
	g.affinityViolations = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "affinity_violations_total",
		Help: "requests that landed on an engine other than the thread's seat (an invariant: must be 0)"}, []string{"from", "to"})
	g.engineGone = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "engine_gone_total",
		Help: "engines torn down by an endpoint delete"})
	g.anomalies = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "anomalies_total",
		Help: "should-be-zero events the policy could not act on, by kind. no_tokens_head = a RENDER FAILURE: one Pick pass saw a thread head with no rendered tokens, so it could not be seated or charged (served last, unseated) — a PASS rate, not a request count, because Pick runs at ~1 kHz. no_request_id = a boundary guard: a turn was released for a request with no id, and the router gives every request one, so this reads 0 in practice. reaped_inflight / reaped_window = THE LEAK SIGNALS: a request (or a released window) outlived the reap and had to be closed by it — the thread's accounting is only right because the reap closed it"}, []string{"kind"})
	g.coldValveHeld = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "cold_valve_held_total",
		Help: "Pick passes in which the queue head was held by this engine's cold-entry valve (the budget would have admitted it; it WAITS, nothing is evicted). A PASS count, not a head count: Pick runs at ~1 kHz, so a head held for a second counts about a thousand times"}, engineLabels)

	// The GLOBAL zero series: a zero is a fact, an absent series is a question (and a
	// should-be-zero count with no series is the shape of a bug nobody notices).
	g.served.WithLabelValues(nocapacityEngine).Add(0)
	for _, kind := range anomalyKinds {
		g.anomalies.WithLabelValues(kind).Add(0)
	}
	for _, c := range []prometheus.Collector{g.gauges,
		g.exits, g.entryWait, g.served, g.queueWait, g.predictedCached, g.dispatchedPrompt,
		g.affinityViolations, g.engineGone, g.coldValveHeld, g.anomalies,
	} {
		if err := reg.Register(c); err != nil {
			// A SECOND instantiation in one process (a config reload, or a second plugin config in the
			// same test binary) must not fail startup: the series exist and belong to the first
			// instance.
			if _, dup := err.(prometheus.AlreadyRegisteredError); dup {
				// One group scheduler per process is the supported configuration; a second instance would
				// double-register every series.
				log.FromContext(context.Background()).V(logutil.DEFAULT).Info("group-scheduler: WARNING metrics already registered in this process — this instance will not emit its own series")
				continue
			}
			return err
		}
	}
	return nil
}

// exitReasons is one series per DEPARTURE reason, pre-instantiated per engine. "budget-exiting" is
// absent on purpose: the valve RETURNS it for a class-4 victim but the exit is booked later, as
// "exiting-drain" when the thread actually leaves.
var exitReasons = []string{"quantum-complete", "idle", "engine-idle",
	"exiting-drain", "budget-idle", "budget-noqueue", "engine-gone"}

// preInstantiateEngine creates this engine's zero-valued COUNTER series at the moment its group
// is created, so "no dispatch", "no exit of this reason" are readable as zeros from the first
// scrape of the engine's life (and survive the engine's removal, as the gauges must not).
func (g *groupMetrics) preInstantiateEngine(engine string) {
	if g == nil || g.served == nil || g.exits == nil {
		return
	}
	label := engineLabel(engine)
	g.served.WithLabelValues(label).Add(0)
	for _, reason := range exitReasons {
		g.exits.WithLabelValues(label, reason).Add(0)
	}
}

// ObserveExit records a thread exit (the reason + engine).
func (g *groupMetrics) ObserveExit(engine, reason string) {
	if g == nil || g.exits == nil {
		return
	}
	g.exits.WithLabelValues(engine, reason).Inc()
}

// ObserveEntry records ONE seat wait, split by kind: "new" for a thread's first turn,
// "reentry" for a returning thread.
func (g *groupMetrics) ObserveEntry(engine, kind string, waitSeconds float64) {
	if g == nil || g.entryWait == nil {
		return
	}
	g.entryWait.WithLabelValues(engine, kind).Observe(waitSeconds)
}

// ObserveQueueWait records how long one dispatched request waited in the queue, by kind.
func (g *groupMetrics) ObserveQueueWait(kind string, seconds float64) {
	if g == nil || g.queueWait == nil {
		return
	}
	if seconds < 0 {
		seconds = 0
	}
	g.queueWait.WithLabelValues(kind).Observe(seconds)
}

// ObservePredictedCache records, for one dispatched turn, the prompt tokens and how many of them
// this engine was predicted to hold already.
func (g *groupMetrics) ObservePredictedCache(engine string, predicted, prompt int64) {
	if g == nil || g.predictedCached == nil {
		return
	}
	g.predictedCached.WithLabelValues(engine).Add(float64(predicted))
	g.dispatchedPrompt.WithLabelValues(engine).Add(float64(prompt))
}

// ObserveServed records a dispatch.
func (g *groupMetrics) ObserveServed(engine string) {
	if g == nil || g.served == nil {
		return
	}
	g.served.WithLabelValues(engine).Inc()
}

// ObserveAffinityViolation records a request that landed off the engine it was charged to.
func (g *groupMetrics) ObserveAffinityViolation(from, to string) {
	if g == nil || g.affinityViolations == nil {
		return
	}
	g.affinityViolations.WithLabelValues(from, to).Inc()
}

// ObserveEngineGone records an engine torn down by an endpoint delete.
func (g *groupMetrics) ObserveEngineGone() {
	if g == nil || g.engineGone == nil {
		return
	}
	g.engineGone.Inc()
}

// ObserveAnomaly counts ONE should-be-zero event by kind (anomalyKinds).
func (g *groupMetrics) ObserveAnomaly(kind string) {
	if g == nil || g.anomalies == nil {
		return
	}
	g.anomalies.WithLabelValues(kind).Inc()
}

// ObserveColdValveHeld counts a head the cold-entry valve held back.
func (g *groupMetrics) ObserveColdValveHeld(engine string) {
	if g == nil || g.coldValveHeld == nil {
		return
	}
	g.coldValveHeld.WithLabelValues(engineLabel(engine)).Inc()
}

// ---- the gauges: rendered from ONE Snapshot at scrape time ----

// stateGauges renders every per-engine GAUGE from ONE Snapshot at scrape time, so a scrape sees the
// state as it is, and an engine that has been removed simply stops being reported (which is what a
// gauge must do — unlike a counter, whose series has to survive its engine, because totals sum it).
type stateGauges struct {
	// r is the plugin: capacity_missing comes from its KNOWN-endpoint set, and the clock that ages
	// an exiting thread is the one the framework hooks use.
	r *GroupScheduler
	// sched is r.sched, kept to avoid a hop in the scrape.
	sched *Scheduler
	// descs are built once: name, help and labels are what a scraper stores, so they are the part of
	// this file that must not drift.
	seats           *prometheus.Desc
	neBlocks        *prometheus.Desc
	budgetBlocks    *prometheus.Desc
	exiting         *prometheus.Desc
	exitingBlks     *prometheus.Desc
	exitingAge      *prometheus.Desc
	coldInFlight    *prometheus.Desc
	coldCap         *prometheus.Desc
	capacityMissing *prometheus.Desc
	waiters         *prometheus.Desc
}

// gaugeDesc builds one per-engine gauge descriptor under the plugin's subsystem.
func gaugeDesc(name, help string) *prometheus.Desc {
	return prometheus.NewDesc(prometheus.BuildFQName("", subsystem, name), help, engineLabels, nil)
}

// newStateGauges builds the collector for one plugin instance.
func newStateGauges(r *GroupScheduler) *stateGauges {
	return &stateGauges{
		r:     r,
		sched: r.sched,
		seats: gaugeDesc("seats",
			"seated threads (residencies) on this engine"),
		neBlocks: gaugeDesc("ne_blocks",
			"distinct blocks charged to this engine's budget: every seated thread's set, exiting threads included until they drain"),
		budgetBlocks: gaugeDesc("budget_blocks",
			"seat budget in blocks (groupBudgetFraction x this engine's num_gpu_blocks)"),
		exiting: gaugeDesc("exiting_threads",
			"threads on this engine that have been marked exiting and still hold their seat until their in-flight requests drain"),
		exitingBlks: gaugeDesc("exiting_blocks",
			"distinct blocks of this engine's ne_blocks that belong to exiting threads: the part of the used side that is already leaving (admission still refuses it, enforcement no longer evicts for it)"),
		exitingAge: gaugeDesc("exiting_oldest_age_seconds",
			"how long this engine's OLDEST exiting thread has been on its way out (a value that only grows is a drain that is not happening)"),
		coldInFlight: gaugeDesc("cold_inflight_tokens",
			"outstanding cold-entry charge in tokens against this engine's stagger cap"),
		coldCap: gaugeDesc("cold_cap_tokens",
			"DERIVED cold-entry stagger cap in tokens: floor((1-groupBudgetFraction) x this engine's num_gpu_blocks) x blockSizeTokens"),
		capacityMissing: gaugeDesc("capacity_missing",
			"1 while this engine's num_gpu_blocks is unknown"),
		waiters: prometheus.NewDesc(prometheus.BuildFQName("", subsystem, "waiters_shared"),
			"heads waiting in the ONE shared admission queue", nil, nil),
	}
}

// Describe sends every descriptor this collector can produce.
func (c *stateGauges) Describe(ch chan<- *prometheus.Desc) {
	for _, d := range []*prometheus.Desc{c.seats, c.neBlocks, c.budgetBlocks, c.exiting,
		c.exitingBlks, c.exitingAge, c.coldInFlight, c.coldCap, c.capacityMissing, c.waiters} {
		ch <- d
	}
}

// Collect reads ONE snapshot and emits const gauges from it.
func (c *stateGauges) Collect(ch chan<- prometheus.Metric) {
	// THE CALLER'S CLOCK: the oldest-exiting age is a duration, and the scheduler's own clock only
	// advances when the pass runs, so an idle stack would report a frozen age.
	snap := c.sched.Snapshot(c.r.now())
	gauge := func(d *prometheus.Desc, v float64, labels ...string) {
		ch <- prometheus.MustNewConstMetric(d, prometheus.GaugeValue, v, labels...)
	}
	labels := make([]string, 0, len(snap.Engines))
	for id := range snap.Engines {
		labels = append(labels, id)
	}
	sort.Strings(labels) // deterministic emit order: a scraper compares two scrapes row by row
	for _, id := range labels {
		es := snap.Engines[id]
		gauge(c.seats, float64(es.Seats), es.ID)
		gauge(c.neBlocks, float64(es.NEBlocks), es.ID)
		gauge(c.budgetBlocks, float64(es.BudgetBlocks), es.ID)
		gauge(c.exiting, float64(es.ExitingThreads), es.ID)
		gauge(c.exitingBlks, float64(es.ExitingBlocks), es.ID)
		gauge(c.exitingAge, es.ExitingOldestAgeS, es.ID)
		gauge(c.coldInFlight, float64(es.ColdInFlight), es.ID)
		gauge(c.coldCap, float64(es.ColdCapTokens), es.ID)
	}
	// capacity_missing: every KNOWN endpoint whose geometry the scheduler has not measured. An engine
	// counts as measured only once it has blocks; one with no group at all is the pre-capacity case.
	for _, key := range c.r.knownEndpoints() {
		label := c.sched.GroupLabel(key)
		es, has := snap.Engines[label]
		v := 0.0
		if !has || es.NumBlocks <= 0 {
			v = 1
		}
		gauge(c.capacityMissing, v, label)
	}
	gauge(c.waiters, float64(snap.WaitersShared))
}
