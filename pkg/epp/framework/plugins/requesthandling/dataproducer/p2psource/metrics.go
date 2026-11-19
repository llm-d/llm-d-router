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

package p2psource

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var sourceDecisions = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "p2p_source_decisions_total",
		Help: metricsutil.HelpMsgWithStability(
			"p2p-source-producer decisions, by outcome. outcome=pulled is a request sent to pull its prefix from a peer.",
			compbasemetrics.ALPHA),
	},
	[]string{"plugin_name", "endpoint_role", "outcome"},
)

var predictedPulledTokens = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "p2p_source_predicted_pulled_tokens",
		Help: metricsutil.HelpMsgWithStability(
			"Prompt tokens the selected source holds beyond the endpoint computing the prefix, per request sent to pull.",
			compbasemetrics.ALPHA),
		Buckets: metricsutil.TokenCountBuckets,
	},
	[]string{"plugin_name", "endpoint_role"},
)

var unpulledTokens = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "p2p_source_unpulled_tokens",
		Help: metricsutil.HelpMsgWithStability(
			"Prompt tokens the selected source holds beyond the endpoint computing the prefix, per request the producer did not send to pull, by outcome (declined, below-floor).",
			compbasemetrics.ALPHA),
		Buckets: metricsutil.TokenCountBuckets,
	},
	[]string{"plugin_name", "endpoint_role", "outcome"},
)

var requestTTFT = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "p2p_source_request_ttft_seconds",
		Help: metricsutil.HelpMsgWithStability(
			"Time from request receipt to the first response byte of a streaming response that ends naturally, by the producer's decision and, for a cost-model decision, its gain-minus-cost margin bucket. Recorded with ttftByDecision.",
			compbasemetrics.ALPHA),
		Buckets: metricsutil.GeneralLatencyBuckets,
	},
	[]string{"plugin_name", "endpoint_role", "outcome", "cost_margin"},
)

// costMarginNone labels a decision the cost model did not price.
const costMarginNone = "none"

// costMarginBucket buckets a cost-model decision's gain minus cost, in
// milliseconds. A positive margin is the cost model approving a pull.
func costMarginBucket(marginMs float64) string {
	switch {
	case marginMs < -1000:
		return "-inf..-1s"
	case marginMs < -250:
		return "-1s..-250ms"
	case marginMs <= 0:
		return "-250ms..0"
	case marginMs < 250:
		return "0..250ms"
	case marginMs < 1000:
		return "250ms..1s"
	default:
		return "1s..+inf"
	}
}

// registerMetrics registers the producer collectors on the given registerer. A
// nil registerer is a no-op, so the producer runs without metrics when the
// handle supplies no recorder. Repeated registration of the same collector is
// tolerated.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{sourceDecisions, predictedPulledTokens, unpulledTokens, requestTTFT} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register p2p-source-producer metric: %w", err)
		}
	}
	return nil
}

func recordDecision(pluginName, role, outcome string) {
	sourceDecisions.WithLabelValues(pluginName, role, outcome).Inc()
}

func recordPredictedPulledTokens(pluginName, role string, tokens int) {
	predictedPulledTokens.WithLabelValues(pluginName, role).Observe(float64(tokens))
}

func recordUnpulledTokens(pluginName, role, outcome string, tokens int) {
	unpulledTokens.WithLabelValues(pluginName, role, outcome).Observe(float64(tokens))
}

func recordRequestTTFT(pluginName string, decision *decisionState, seconds float64) {
	requestTTFT.WithLabelValues(pluginName, decision.role, decision.outcome, decision.costMargin).Observe(seconds)
}

// resetMetrics clears the collectors between tests.
func resetMetrics() {
	sourceDecisions.Reset()
	predictedPulledTokens.Reset()
	unpulledTokens.Reset()
	requestTTFT.Reset()
}
