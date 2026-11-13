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

package prefixcacheaffinity

import (
	"errors"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Decision outcome labels. Each Filter call records exactly one of these,
// capturing whether kv-cache affinity or load decided the candidate set.
const (
	// outcomeSticky: the affinity gate narrowed the set to endpoints whose
	// prefix cache score met the threshold, and the load gate kept them.
	outcomeSticky = "sticky"
	// outcomeNoMatch: no endpoint met the affinity threshold, so all were kept.
	outcomeNoMatch = "no_match"
	// outcomeLoadOverride: the TTFT load gate discarded a non-empty sticky set
	// because those endpoints were too slow, reopening all endpoints.
	outcomeLoadOverride = "load_override"
	// outcomeExploration: the gate was skipped for exploration.
	outcomeExploration = "exploration"
	// outcomeNotApplicable: the filter had nothing to decide (a single
	// candidate, or the affinity threshold disabled).
	outcomeNotApplicable = "not_applicable"
	// outcomeMissingSignal: Either side of the TTFT comparison lacked the configured attribute
	// gate skipped and sticky set kept
	outcomeMissingSignal = "missing_signal"
)

// Labels recorded only on the P2P-paired counter.
const (
	// outcomeMultiple is the affinity_outcome of a request the instance decided
	// more than once because it runs in more than one scheduling profile, in
	// place of the individual outcomes.
	outcomeMultiple = "multiple"
	// p2pOutcomeUnreported is the p2p_outcome of a request whose
	// p2p-source-producer published no decision.
	p2pOutcomeUnreported = "unreported"
)

var filterDecisions = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_cache_affinity_filter_decisions_total",
		Help:      metricsutil.HelpMsgWithStability("Prefix cache affinity filter decisions, by outcome.", compbasemetrics.ALPHA),
	},
	[]string{"plugin_name", "outcome"},
)

var p2pDecisions = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_cache_affinity_filter_p2p_decisions_total",
		Help: metricsutil.HelpMsgWithStability(
			"Prefix cache affinity filter outcomes paired with the p2p-source-producer decision for the same request.",
			compbasemetrics.ALPHA),
	},
	[]string{"plugin_name", "endpoint_role", "affinity_outcome", "p2p_outcome"},
)

// ttftBuckets are the histogram buckets, in seconds, for the estimated and the
// actual TTFT of a request.
var ttftBuckets = []float64{
	0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45,
	60, 90, 120, 180, 300, 600,
}

// ttftLabels are the labels of the estimated and actual TTFT histograms, as on
// the predicted-latency producer's TTFT histograms.
var ttftLabels = []string{"plugin_name", "plugin_type", "target_model_name"}

var estimatedTTFT = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_cache_affinity_filter_estimated_ttft_seconds",
		Help:      metricsutil.HelpMsgWithStability("TTFT the filter estimated for the endpoint that served the request, in seconds.", compbasemetrics.ALPHA),
		Buckets:   ttftBuckets,
	},
	ttftLabels,
)

var actualTTFT = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "prefix_cache_affinity_filter_actual_ttft_seconds",
		Help:      metricsutil.HelpMsgWithStability("TTFT measured from the filter's decision to the first response chunk, for requests with an estimate, in seconds.", compbasemetrics.ALPHA),
		Buckets:   ttftBuckets,
	},
	ttftLabels,
)

// registerMetrics registers the filter collectors on the given registerer. A nil
// registerer is a no-op, so the filter runs without metrics when the handle
// supplies no recorder. Repeated registration of the same collector is tolerated.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{filterDecisions, p2pDecisions, estimatedTTFT, actualTTFT} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register prefix cache affinity filter metric: %w", err)
		}
	}
	return nil
}

func recordDecision(pluginName, outcome string) {
	filterDecisions.WithLabelValues(pluginName, outcome).Inc()
}

func recordP2PDecision(pluginName, role, affinityOutcome, p2pOutcome string) {
	p2pDecisions.WithLabelValues(pluginName, role, affinityOutcome, p2pOutcome).Inc()
}

// recordTTFT records a request's estimated and actual TTFT.
func recordTTFT(plugin fwkplugin.TypedName, targetModel string, estimated, actual time.Duration) {
	estimatedTTFT.WithLabelValues(plugin.Name, plugin.Type, targetModel).Observe(estimated.Seconds())
	actualTTFT.WithLabelValues(plugin.Name, plugin.Type, targetModel).Observe(actual.Seconds())
}

// resetMetrics clears the collectors between tests.
func resetMetrics() {
	filterDecisions.Reset()
	p2pDecisions.Reset()
	estimatedTTFT.Reset()
	actualTTFT.Reset()
}
