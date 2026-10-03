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
}

func newThunderMetrics(mgr *sessionManager) *thunderMetrics {
	return &thunderMetrics{
		ledger: newLedgerCollector(mgr),
	}
}

func (m *thunderMetrics) register(reg prometheus.Registerer) error {
	return registerReplacing(reg, m.ledger)
}

// registerReplacing registers c, replacing a collector with the same
// descriptors left by an earlier instance of the plugin (config reload), so
// scrapes read the live ledger instead of a discarded one.
func registerReplacing(reg prometheus.Registerer, c prometheus.Collector) error {
	err := reg.Register(c)
	var already prometheus.AlreadyRegisteredError
	if errors.As(err, &already) {
		reg.Unregister(already.ExistingCollector)
		return reg.Register(c)
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
			[]string{"endpoint", "view"}, nil),
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
	for endpoint, g := range snap.endpoints {
		ch <- prometheus.MustNewConstMetric(c.workingSet, prometheus.GaugeValue, g.undecayed, endpoint, "undecayed")
		ch <- prometheus.MustNewConstMetric(c.capacity, prometheus.GaugeValue, g.capacity, endpoint)
	}
}
