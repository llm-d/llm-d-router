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

package heavyhitter

import (
	"time"
	"unicode/utf8"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// collector reports the window at scrape time, so the gauges stay current when no request
// arrives, and series for IDs that leave the top-k disappear on the next scrape without
// DeleteLabelValues bookkeeping.
type collector struct {
	plugin       *Plugin
	weight       *prometheus.Desc
	weightError  *prometheus.Desc
	windowWeight *prometheus.Desc
	evictions    *prometheus.Desc
}

func newCollector(name string, p *Plugin) *collector {
	fqName := func(n string) string {
		return prometheus.BuildFQName("", eppmetrics.LLMDRouterEndpointPickerSubsystem, n)
	}
	labels := prometheus.Labels{"plugin_name": name}
	return &collector{
		plugin: p,
		weight: prometheus.NewDesc(fqName("heavy_hitter_weight"),
			metricsutil.HelpMsgWithStability("Estimated windowed load of each of the heaviest fairness IDs.", compbasemetrics.ALPHA),
			[]string{"fairness_id"}, labels),
		weightError: prometheus.NewDesc(fqName("heavy_hitter_weight_error"),
			metricsutil.HelpMsgWithStability("Lower-bound correction of the fairness ID's windowed load: the true load is at least weight minus this value.", compbasemetrics.ALPHA),
			[]string{"fairness_id"}, labels),
		windowWeight: prometheus.NewDesc(fqName("heavy_hitter_window_weight"),
			metricsutil.HelpMsgWithStability("Exact windowed load of all fairness IDs.", compbasemetrics.ALPHA),
			nil, labels),
		evictions: prometheus.NewDesc(fqName("heavy_hitter_evictions_total"),
			metricsutil.HelpMsgWithStability("Cumulative slot takeovers. A nonzero rate means the reported loads are approximate and slotsPerSlice is below the active fairness-ID count per slice.", compbasemetrics.ALPHA),
			nil, labels),
	}
}

// Describe implements prometheus.Collector.
func (c *collector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.weight
	ch <- c.weightError
	ch <- c.windowWeight
	ch <- c.evictions
}

// Collect implements prometheus.Collector.
func (c *collector) Collect(ch chan<- prometheus.Metric) {
	snap := c.plugin.window.Snapshot(time.Now(), c.plugin.params.MetricTopK)
	ch <- prometheus.MustNewConstMetric(c.windowWeight, prometheus.GaugeValue, float64(snap.Total))
	ch <- prometheus.MustNewConstMetric(c.evictions, prometheus.CounterValue, float64(snap.Evictions))
	// A configured threshold narrows the series to the heavy IDs on the same lower bound
	// IsHeavy uses; the snapshot already holds the heaviest metricTopK of them.
	threshold := c.plugin.params.HeavyThreshold
	for _, e := range snap.Entries {
		if threshold > 0 && float64(e.Weight-e.Error) < threshold {
			continue
		}
		// These series are rebuilt on every scrape and bounded by metricTopK, so they do not
		// pass through the shared fairness-ID label limiter: admitting top-k keys there would
		// consume the process-wide cap that flow-control and request metrics share. Keys must
		// be valid UTF-8 or MustNewConstMetric panics; truncation already bounds their length.
		if !utf8.ValidString(e.ID) {
			continue
		}
		ch <- prometheus.MustNewConstMetric(c.weight, prometheus.GaugeValue, float64(e.Weight), e.ID)
		ch <- prometheus.MustNewConstMetric(c.weightError, prometheus.GaugeValue, float64(e.Error), e.ID)
	}
}
