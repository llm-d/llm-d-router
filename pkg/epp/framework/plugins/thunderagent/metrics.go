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

package thunderagent

import (
	"errors"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// thunderMetrics holds the plugin's Prometheus collectors.
type thunderMetrics struct {
	ledger *ledgerCollector

	holds                *prometheus.CounterVec
	releases             *prometheus.CounterVec
	pauses               prometheus.Counter
	resumes              prometheus.Counter
	starvationPromotions prometheus.Counter
}

func newThunderMetrics(mgr *sessionManager) *thunderMetrics {
	return &thunderMetrics{
		ledger: newLedgerCollector(mgr),
		holds: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_holds_total",
			Help:      metricsutil.HelpMsgWithStability("Dispatches whose head waited past the hold floor in the flow-control queue, by session class.", compbasemetrics.ALPHA),
		}, []string{"class"}),
		releases: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_releases_total",
			Help:      metricsutil.HelpMsgWithStability("Dispatches picked by the thunder fairness policy, by session class.", compbasemetrics.ALPHA),
		}, []string{"class"}),
		pauses: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_pauses_total",
			Help:      metricsutil.HelpMsgWithStability("Idle sessions paused to make room for another session's turn; their next turn must fit their endpoint again.", compbasemetrics.ALPHA),
		}),
		resumes: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_resumes_total",
			Help:      metricsutil.HelpMsgWithStability("Paused sessions whose next turn was admitted and dispatched.", compbasemetrics.ALPHA),
		}),
		starvationPromotions: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_starvation_promotions_total",
			Help:      metricsutil.HelpMsgWithStability("Queue heads force-admitted ahead of class, size, and fit by the starvation backstop.", compbasemetrics.ALPHA),
		}),
	}
}

func (m *thunderMetrics) register(reg prometheus.Registerer) error {
	return errors.Join(
		reg.Register(m.ledger),
		registerOrReuse(reg, &m.holds),
		registerOrReuse(reg, &m.releases),
		registerOrReuse(reg, &m.pauses),
		registerOrReuse(reg, &m.resumes),
		registerOrReuse(reg, &m.starvationPromotions),
	)
}

func registerOrReuse[C prometheus.Collector](reg prometheus.Registerer, target *C) error {
	err := reg.Register(*target)
	if err == nil {
		return nil
	}
	var already prometheus.AlreadyRegisteredError
	if errors.As(err, &already) {
		if existing, ok := already.ExistingCollector.(C); ok {
			*target = existing
			return nil
		}
	}
	return err
}

// ledgerCollector reports the session ledger at scrape time, so the gauges
// stay current when no request arrives.
type ledgerCollector struct {
	mgr        *sessionManager
	sessions   *prometheus.Desc
	workingSet *prometheus.Desc
	capacity   *prometheus.Desc
}

func newLedgerCollector(mgr *sessionManager) *ledgerCollector {
	fqName := func(name string) string {
		return prometheus.BuildFQName("", eppmetrics.LLMDRouterEndpointPickerSubsystem, name)
	}
	return &ledgerCollector{
		mgr: mgr,
		sessions: prometheus.NewDesc(fqName("thunder_agent_sessions"),
			metricsutil.HelpMsgWithStability("Tracked sessions by state.", compbasemetrics.ALPHA),
			[]string{"state"}, nil),
		workingSet: prometheus.NewDesc(fqName("thunder_agent_endpoint_working_set_tokens"),
			metricsutil.HelpMsgWithStability("Session KV working set per endpoint in tokens. Compare with the engine's KV utilization: a large working set over a low engine utilization is the KV-thrashing signature.", compbasemetrics.ALPHA),
			[]string{"endpoint"}, nil),
		capacity: prometheus.NewDesc(fqName("thunder_agent_endpoint_capacity_tokens"),
			metricsutil.HelpMsgWithStability("KV token capacity per endpoint.", compbasemetrics.ALPHA),
			[]string{"endpoint"}, nil),
	}
}

func (c *ledgerCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.sessions
	ch <- c.workingSet
	ch <- c.capacity
}

func (c *ledgerCollector) Collect(ch chan<- prometheus.Metric) {
	snap := c.mgr.snapshot(time.Now())
	ch <- prometheus.MustNewConstMetric(c.sessions, prometheus.GaugeValue, float64(snap.running), "running")
	ch <- prometheus.MustNewConstMetric(c.sessions, prometheus.GaugeValue, float64(snap.idle), "idle")
	ch <- prometheus.MustNewConstMetric(c.sessions, prometheus.GaugeValue, float64(snap.paused), "paused")
	for endpoint, g := range snap.endpoints {
		ch <- prometheus.MustNewConstMetric(c.workingSet, prometheus.GaugeValue, g.workingSet, endpoint)
		ch <- prometheus.MustNewConstMetric(c.capacity, prometheus.GaugeValue, g.capacity, endpoint)
	}
}
