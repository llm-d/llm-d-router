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

package cacheclass

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// These are deliberately recorded at classification time rather than at dispatch. Pairing
// admission_class_total with the flow-control outcome counters is what makes cache-blind
// shedding directly visible: equal rejection rates across the two class labels is the symptom
// this plugin exists to remove.
var (
	classifications = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "cache_class_requests_total",
			Help: metricsutil.HelpMsgWithStability(
				"Requests classified before admission, by cache class.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "cache_class"},
	)

	matchRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "cache_class_match_ratio",
			Help: metricsutil.HelpMsgWithStability(
				"Prefix match ratio observed at admission time, before any shedding decision.",
				compbasemetrics.ALPHA),
			Buckets: []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"plugin_name", "plugin_type"},
	)

	indexSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "cache_class_index_size",
			Help: metricsutil.HelpMsgWithStability(
				"Prefix blocks currently tracked by the admission-time index.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	// Cached tokens the pool declined at the door. Paired with the hit label this is the
	// stranded-investment figure: prefill already paid, turned away, with no later turn to
	// amortize it. This is the metric that makes cache-blind shedding cost visible in
	// production rather than something to infer from hit-rate dips.
	turnedAwayTokens = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "cache_class_turned_away_tokens_total",
			Help: metricsutil.HelpMsgWithStability(
				"Cached prefix tokens carried by requests that were classified but never dispatched.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "cache_class"},
	)

	turnedAwayRequests = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "cache_class_turned_away_requests_total",
			Help: metricsutil.HelpMsgWithStability(
				"Requests that were classified but never dispatched, by cache class.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "cache_class"},
	)
)

// registerMetrics publishes the collectors on the registerer EPP actually serves.
//
// This must be the handle's recorder, not prometheus.DefaultRegisterer: EPP exposes its own
// registry, so collectors registered globally are recorded correctly and then never appear on
// /metrics -- which reads as "the plugin did nothing" rather than "the metric is missing".
//
// Registration happens once at construction so every series exists at zero from startup. An
// absent family then unambiguously means the plugin is not loaded, instead of being
// indistinguishable from an idle router.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("cache class metrics registerer is required")
	}
	for _, c := range []prometheus.Collector{
		classifications, matchRatio, indexSize, turnedAwayTokens, turnedAwayRequests,
	} {
		if err := registerer.Register(c); err != nil {
			var already prometheus.AlreadyRegisteredError
			if errors.As(err, &already) && already.ExistingCollector == c {
				continue
			}
			return fmt.Errorf("register cache class metric: %w", err)
		}
	}
	return nil
}

// initSeries materializes the label combinations so each series reports zero before any
// traffic, rather than being absent.
func initSeries(name, typ string) {
	for _, class := range []string{"hit", "miss"} {
		classifications.WithLabelValues(name, typ, class)
		turnedAwayRequests.WithLabelValues(name, typ, class)
		turnedAwayTokens.WithLabelValues(name, typ, class)
	}
	matchRatio.WithLabelValues(name, typ)
	indexSize.WithLabelValues(name, typ).Set(0)
}

func recordClassification(tn fwkplugin.TypedName, hit bool, ratio float64) {
	class := "miss"
	if hit {
		class = "hit"
	}
	classifications.WithLabelValues(tn.Name, tn.Type, class).Inc()
	matchRatio.WithLabelValues(tn.Name, tn.Type).Observe(ratio)
}

func recordTurnedAway(tn fwkplugin.TypedName, hit bool, matchedTokens int) {
	class := "miss"
	if hit {
		class = "hit"
	}
	turnedAwayRequests.WithLabelValues(tn.Name, tn.Type, class).Inc()
	turnedAwayTokens.WithLabelValues(tn.Name, tn.Type, class).Add(float64(matchedTokens))
}

func recordIndexSize(tn fwkplugin.TypedName, size int) {
	indexSize.WithLabelValues(tn.Name, tn.Type).Set(float64(size))
}
