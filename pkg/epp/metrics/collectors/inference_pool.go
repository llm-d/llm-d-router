/*
Copyright 2025 The Kubernetes Authors.
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

package collectors

import (
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// nixlFailureCounters pairs each NIXL failure counter attribute with the
// descriptor it is exposed under.
var nixlFailureCounters = []struct {
	key  fwkplugin.DataKey
	desc *prometheus.Desc
}{
	{attrmetrics.NixlFailedTransfersDataKey, eppmetrics.DescInferencePoolPerEndpointNixlFailedTransfers},
	{attrmetrics.NixlFailedNotificationsDataKey, eppmetrics.DescInferencePoolPerEndpointNixlFailedNotifications},
	{attrmetrics.NixlKVExpiredRequestsDataKey, eppmetrics.DescInferencePoolPerEndpointNixlKVExpiredRequests},
}

type inferencePoolMetricsCollector struct {
	ds datastore.Datastore
}

// Check if inferencePoolMetricsCollector implements necessary interface
var _ prometheus.Collector = &inferencePoolMetricsCollector{}

// NewInferencePoolMetricsCollector implements the prometheus.Collector interface and
// exposes metrics about inference pool.
func NewInferencePoolMetricsCollector(ds datastore.Datastore) prometheus.Collector {
	return &inferencePoolMetricsCollector{
		ds: ds,
	}
}

// DescribeWithStability implements the prometheus.Collector interface.
func (c *inferencePoolMetricsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- eppmetrics.DescInferencePoolPerEndpointQueueSize
	for _, counter := range nixlFailureCounters {
		ch <- counter.desc
	}
}

// CollectWithStability implements the prometheus.Collector interface.
func (c *inferencePoolMetricsCollector) Collect(ch chan<- prometheus.Metric) {
	pool, err := c.ds.PoolGet()
	if err != nil {
		return
	}

	podMetrics := c.ds.PodList(datastore.AllPodsPredicate)
	if len(podMetrics) == 0 {
		return
	}

	for _, pod := range podMetrics {
		ch <- prometheus.MustNewConstMetric(
			eppmetrics.DescInferencePoolPerEndpointQueueSize,
			prometheus.GaugeValue,
			float64(pod.GetMetrics().WaitingQueueSize),
			pool.Name,
			pod.GetMetadata().ID.Name,
		)
		// An endpoint carries these attributes only if its model server reports the counters.
		for _, counter := range nixlFailureCounters {
			value, ok := attrmetrics.ReadScalarMetricValue(pod.GetAttributes(), counter.key)
			if !ok {
				continue
			}
			ch <- prometheus.MustNewConstMetric(
				counter.desc,
				prometheus.CounterValue,
				float64(value),
				pool.Name,
				pod.GetMetadata().ID.Name,
			)
		}
	}
}
