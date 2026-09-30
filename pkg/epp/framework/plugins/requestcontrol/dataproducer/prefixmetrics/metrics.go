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

// Package prefixmetrics holds the metrics the approximate and precise
// prefix-cache producers share, so either deployment reports prefix-cache
// prediction under one set of metric names.
package prefixmetrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var predictedCachedTokens = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_predicted_cached_tokens",
		Help: metricsutil.HelpMsgWithStability(
			"Prompt tokens the producer predicted the scheduler's chosen endpoint holds in its prefix cache, per request.",
			compbasemetrics.ALPHA),
		Buckets: metricsutil.TokenCountBuckets,
	},
	[]string{"plugin_name", "plugin_type"},
)

var bestPredictedCachedTokens = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_best_predicted_cached_tokens",
		Help: metricsutil.HelpMsgWithStability(
			"Highest prefix-cache prediction among the endpoints the scheduler selected from, per request.",
			compbasemetrics.ALPHA),
		Buckets: metricsutil.TokenCountBuckets,
	},
	[]string{"plugin_name", "plugin_type"},
)

var bestAvailableCachedTokens = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_best_available_cached_tokens",
		Help: metricsutil.HelpMsgWithStability(
			"Highest prefix-cache prediction among the request's candidate endpoints before filtering, per request.",
			compbasemetrics.ALPHA),
		Buckets: metricsutil.TokenCountBuckets,
	},
	[]string{"plugin_name", "plugin_type"},
)

var promptTokens = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_prompt_tokens",
		Help: metricsutil.HelpMsgWithStability(
			"Prompt tokens the producer measured its prediction against, per request.",
			compbasemetrics.ALPHA),
		Buckets: metricsutil.TokenCountBuckets,
	},
	[]string{"plugin_name", "plugin_type"},
)

var registerOnce sync.Once

// Register makes the shared prefix metrics collectable. Every prefix producer
// instance calls it; the first call registers.
func Register() {
	registerOnce.Do(func() {
		metrics.Registry.MustRegister(predictedCachedTokens, bestPredictedCachedTokens,
			bestAvailableCachedTokens, promptTokens)
	})
}

// Prediction is one request's prefix-cache prediction in prompt tokens, taken
// over three endpoint sets that narrow into each other: every candidate the
// request could have reached, those that survived filtering and reached the
// picker, and the one the picker chose. Selected <= BestPredicted <=
// BestAvailable holds by construction.
type Prediction struct {
	// Selected is what the producer expects the chosen endpoint to serve from
	// its prefix cache.
	Selected int
	// BestPredicted is the highest prediction the picker could have chosen.
	// The distance from Selected is what the routing decision left behind.
	BestPredicted int
	// BestAvailable is the highest prediction among the candidate endpoints
	// before filters ran. Scheduling profiles filter by endpoint role, so under
	// disaggregated prefill/decode this spans both roles while the other two
	// fields follow the decode profile.
	BestAvailable int
	// PromptTokens is the prompt the predictions are measured against.
	PromptTokens int
}

// RecordPrediction records a request's prefix-cache prediction. Every field is
// observed in one call so each histogram covers the same requests, which is
// what lets their sums be divided by one another.
// llm_d_epp_request_input_tokens is not a usable denominator here: it is
// recorded from the model server's response, so it omits requests that fail or
// return no usage, which these metrics still count.
func RecordPrediction(pluginName, pluginType string, p Prediction) {
	predictedCachedTokens.WithLabelValues(pluginName, pluginType).Observe(float64(p.Selected))
	bestPredictedCachedTokens.WithLabelValues(pluginName, pluginType).Observe(float64(p.BestPredicted))
	bestAvailableCachedTokens.WithLabelValues(pluginName, pluginType).Observe(float64(p.BestAvailable))
	promptTokens.WithLabelValues(pluginName, pluginType).Observe(float64(p.PromptTokens))
}
