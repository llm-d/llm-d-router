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

package threadidentity

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// metricsSubsystem namespaces this plugin's series: llm_d_epp_thread_*.
const metricsSubsystem = "llm_d_epp_thread"

// threadMetrics is the legibility set for the derivation: the arrival-outcome
// mix, the table size, the three costs on the arrival path (render, chain,
// resolve), and the bypass/failure counters.
type threadMetrics struct {
	matched       prometheus.Counter
	forked        prometheus.Counter
	fresh         prometheus.Counter
	sticky        prometheus.Counter
	retried       prometheus.Counter
	evicted       prometheus.Counter
	renderFailure prometheus.Counter
	multiPrompt   prometheus.Counter
	overridden    prometheus.Counter
	records       prometheus.Gauge
	renderNs      prometheus.Histogram
	chainNs       prometheus.Histogram
	resolveNs     prometheus.Histogram
}

// newThreadMetrics builds the empty set; register fills it.
func newThreadMetrics() *threadMetrics { return &threadMetrics{} }

// register creates every series and installs it on the registry.
func (m *threadMetrics) register(reg prometheus.Registerer) error {
	m.matched = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "matched_total",
		Help: "requests that continued their thread (the arrival outcome 'continue'; same queue key as the thread's previous turn)"})
	m.forked = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "forks_total",
		Help: "requests that extended a BUSY tip and therefore forked a new thread (a parallel sibling)"})
	m.fresh = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "new_total",
		Help: "requests that matched no tip and started a new thread"})
	m.sticky = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "sticky_busy_total",
		Help: "requests whose continuation was proven by content (the H1 reply guard) despite a busy thread"})
	m.retried = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "retry_total",
		Help: "requests identical to a tip with nothing in flight, treated as a client retry of a still-queued turn (S2)"})
	m.evicted = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "evicted_total",
		Help: "threads removed by the table cap (the only eviction path; there is no idle GC)"})
	m.renderFailure = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "render_failure_total",
		Help: "requests with no identity because the arrival-time render failed or timed out; each starts its own thread"})
	m.multiPrompt = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "multi_prompt_requests_total",
		Help: "requests whose render produced more than one prompt; their per-prompt chains are concatenated"})
	m.overridden = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "explicit_override_total",
		Help: "requests that carried an explicit fairness id, so the derived thread did not become the queue key (a run whose whole point is the derivation must keep this at 0)"})
	m.records = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: metricsSubsystem, Name: "records",
		Help: "live threads in the table"})
	m.renderNs = prometheus.NewHistogram(prometheus.HistogramOpts{Subsystem: metricsSubsystem, Name: "render_ns",
		Help:    "arrival-time render latency per request (the token producer, relocated from dispatch)",
		Buckets: prometheus.ExponentialBuckets(50_000, 4, 10)})
	m.chainNs = prometheus.NewHistogram(prometheus.HistogramOpts{Subsystem: metricsSubsystem, Name: "chain_ns",
		Help:    "block-chain hashing time per request (the prefix scorer's chained hashes)",
		Buckets: prometheus.ExponentialBuckets(1_000, 4, 8)})
	m.resolveNs = prometheus.NewHistogram(prometheus.HistogramOpts{Subsystem: metricsSubsystem, Name: "resolve_ns",
		Help:    "thread table resolve time per request (under the table lock: match + mint)",
		Buckets: prometheus.ExponentialBuckets(500, 4, 8)})
	for _, c := range []prometheus.Collector{
		m.matched, m.forked, m.fresh, m.sticky, m.retried, m.evicted, m.renderFailure,
		m.multiPrompt, m.overridden, m.records, m.renderNs, m.chainNs, m.resolveNs,
	} {
		if err := reg.Register(c); err != nil {
			return err
		}
	}
	return nil
}

// observeRender records one render call's duration.
func (m *threadMetrics) observeRender(d time.Duration) {
	if m.renderNs != nil && d >= 0 {
		m.renderNs.Observe(float64(d.Nanoseconds()))
	}
}

// observeChain records one chain-construction duration.
func (m *threadMetrics) observeChain(d time.Duration) {
	if m.chainNs != nil && d >= 0 {
		m.chainNs.Observe(float64(d.Nanoseconds()))
	}
}

// observeResolve books one resolve: its outcome, its flags, and its duration.
func (m *threadMetrics) observeResolve(res ResolveResult) {
	switch res.Outcome {
	case OutcomeMatched:
		if m.matched != nil {
			m.matched.Inc()
		}
	case OutcomeForked:
		if m.forked != nil {
			m.forked.Inc()
		}
	case OutcomeNew:
		if m.fresh != nil {
			m.fresh.Inc()
		}
	}
	if res.StickyBusy && m.sticky != nil {
		m.sticky.Inc()
	}
	if res.Retry && m.retried != nil {
		m.retried.Inc()
	}
	if m.resolveNs != nil && res.Elapsed >= 0 {
		m.resolveNs.Observe(float64(res.Elapsed.Nanoseconds()))
	}
}

// observeRenderFailure books a render that produced no tokens.
func (m *threadMetrics) observeRenderFailure() {
	if m.renderFailure != nil {
		m.renderFailure.Inc()
	}
}

// addMultiPrompt books a request that carried more than one prompt.
func (m *threadMetrics) addMultiPrompt(n int) {
	if m.multiPrompt != nil && n > 0 {
		m.multiPrompt.Inc()
	}
}

// observeExplicitOverride books a request whose explicit fairness id won over the derived one.
func (m *threadMetrics) observeExplicitOverride() {
	if m.overridden != nil {
		m.overridden.Inc()
	}
}

// addEvicted books tracked threads discarded to keep the index bounded.
func (m *threadMetrics) addEvicted(n int) {
	if m.evicted != nil && n > 0 {
		m.evicted.Add(float64(n))
	}
}

// setRecords publishes how many threads the index holds right now.
func (m *threadMetrics) setRecords(n int) {
	if m.records != nil {
		m.records.Set(float64(n))
	}
}
