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

package multimodal

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var (
	// encoderCacheQueriesTotal counts every multimodal item hash lookup against the LRU.
	encoderCacheQueriesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "encoder_cache_queries_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of multimodal item hash lookups made against the encoder-cache affinity LRU.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_type", "plugin_name", "modality"},
	)

	// encoderCacheHitsTotal counts the subset of encoder_cache_queries_total where
	// the item hash was already present in the endpoint's LRU, labelled by pod.
	// Divide by queries_total for hit rate.
	encoderCacheHitsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "encoder_cache_hits_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of multimodal item hash lookups that found a match in the encoder-cache affinity LRU, by endpoint.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_type", "plugin_name", "pod", "modality"},
	)

	// encoderCacheHitRatio records, per endpoint and per request, the fraction of the
	// request's multimodal items already present in that endpoint's LRU. The counters
	// give an aggregate hit rate; this histogram exposes the distribution of
	// per-endpoint hit ratios, which surfaces uneven cache locality across pods.
	encoderCacheHitRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "encoder_cache_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Ratio of matched multimodal items to total items per endpoint in an encoder-cache lookup.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"plugin_type", "plugin_name"},
	)

	registerOnce sync.Once
)

func registerEncoderCacheMetrics() {
	registerOnce.Do(func() {
		metrics.Registry.MustRegister(encoderCacheQueriesTotal)
		metrics.Registry.MustRegister(encoderCacheHitsTotal)
		metrics.Registry.MustRegister(encoderCacheHitRatio)
	})
}
