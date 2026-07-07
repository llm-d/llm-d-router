/*
Copyright 2026 The Kubernetes Authors.

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
package preciseprefixcache

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var (
	llmdPrefixCacheMaxHitRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "precise_prefix_indexer_max_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Ratio of matched KV blocks to total prompt KV blocks at the best-matched endpoint.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"plugin_name", "plugin_type"},
	)

	llmdPrefixCacheAvgHitRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "precise_prefix_indexer_avg_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Average Ratio of matched KV blocks to total prompt KV blocks across all candidate endpoints.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"plugin_name", "plugin_type"},
	)

	llmdPrefixCacheStdDevHitRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "precise_prefix_indexer_std_dev_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Standard Deviation in ratio of matched KV blocks to total prompt KV blocks across all candidate endpoints.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"plugin_name", "plugin_type"},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("precise prefix cache metrics registerer is required")
	}
	for _, collector := range []prometheus.Collector{
		llmdPrefixCacheMaxHitRatio,
		llmdPrefixCacheAvgHitRatio,
		llmdPrefixCacheStdDevHitRatio,
	} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register precise prefix cache metric: %w", err)
		}
	}
	return nil
}

func recordPrefixCacheMaxMatch(pluginName, pluginType string, maxBlocks int, totalBlocks int) {
	llmdPrefixCacheMaxHitRatio.WithLabelValues(pluginName, pluginType).Observe(float64(maxBlocks) / float64(totalBlocks))
}

func recordPrefixCacheAvgMatch(pluginName, pluginType string, avgBlocks float64, totalBlocks int) {
	llmdPrefixCacheAvgHitRatio.WithLabelValues(pluginName, pluginType).Observe(avgBlocks / float64(totalBlocks))
}

func recordPrefixCacheStdDevMatch(pluginName, pluginType string, stdDevBlocks float64, totalBlocks int) {
	llmdPrefixCacheStdDevHitRatio.WithLabelValues(pluginName, pluginType).Observe(stdDevBlocks / float64(totalBlocks))
}
