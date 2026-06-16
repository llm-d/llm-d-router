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

package approximateprefix

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var (
	prefixCacheSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.InferenceExtensionSubsystem,
			Name:      "prefix_indexer_size",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_prefix_indexer_size] Size of the prefix indexer.", compbasemetrics.ALPHA),
		},
		[]string{},
	)

	llmdPrefixCacheSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "prefix_indexer_size",
			Help:      metricsutil.HelpMsgWithStability("Size of the prefix indexer.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	prefixCacheHitRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.InferenceExtensionSubsystem,
			Name:      "prefix_indexer_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_prefix_indexer_hit_ratio] Ratio of prefix length matched to total prefix length in the cache lookup.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{},
	)

	llmdPrefixCacheHitRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "prefix_indexer_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Ratio of prefix length matched to total prefix length in the cache lookup.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"plugin_name", "plugin_type"},
	)

	prefixCacheHitLength = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.InferenceExtensionSubsystem,
			Name:      "prefix_indexer_hit_bytes",
			Help:      metricsutil.HelpMsgWithStability("[Deprecated: Use llm_d_epp_prefix_indexer_hit_bytes] Length of the prefix match in number of bytes in the cache lookup.", compbasemetrics.ALPHA),
			Buckets:   []float64{0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536},
		},
		[]string{},
	)

	llmdPrefixCacheHitLength = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "prefix_indexer_hit_bytes",
			Help:      metricsutil.HelpMsgWithStability("Length of the prefix match in number of bytes in the cache lookup.", compbasemetrics.ALPHA),
			Buckets:   []float64{0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536},
		},
		[]string{"plugin_name", "plugin_type"},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("approximate prefix metrics registerer is required")
	}
	for _, collector := range []prometheus.Collector{
		prefixCacheSize,
		llmdPrefixCacheSize,
		prefixCacheHitRatio,
		llmdPrefixCacheHitRatio,
		prefixCacheHitLength,
		llmdPrefixCacheHitLength,
	} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register approximate prefix metric: %w", err)
		}
	}
	return nil
}

// recordPrefixCacheSize records the size of the prefix indexer in megabytes.
func recordPrefixCacheSize(pluginName, pluginType string, size int64) {
	prefixCacheSize.WithLabelValues().Set(float64(size))
	llmdPrefixCacheSize.WithLabelValues(pluginName, pluginType).Set(float64(size))
}

// recordPrefixCacheMatch records both the hit ratio and hit length for a prefix indexer match.
// matchedLength is the number of characters that matched, and totalLength is the total prefix length.
func recordPrefixCacheMatch(pluginName, pluginType string, matchedLength, totalLength int) {
	prefixCacheHitLength.WithLabelValues().Observe(float64(matchedLength))
	llmdPrefixCacheHitLength.WithLabelValues(pluginName, pluginType).Observe(float64(matchedLength))

	if totalLength > 0 {
		ratio := float64(matchedLength) / float64(totalLength)
		prefixCacheHitRatio.WithLabelValues().Observe(ratio)
		llmdPrefixCacheHitRatio.WithLabelValues(pluginName, pluginType).Observe(ratio)
	}
}
