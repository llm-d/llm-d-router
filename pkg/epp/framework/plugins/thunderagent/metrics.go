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

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// thunderMetrics holds the plugin's Prometheus collectors. They are created
// per plugin instance and registered by the factory when a metrics recorder
// is available; unregistered collectors still accept writes, so the hooks
// never need to nil-check.
type thunderMetrics struct {
	podUtilization    *prometheus.GaugeVec
	podCapacityTokens *prometheus.GaugeVec
	programs          *prometheus.GaugeVec

	holds                *prometheus.CounterVec
	releases             *prometheus.CounterVec
	starvationPromotions prometheus.Counter
	rebinds              prometheus.Counter
	sheds                prometheus.Counter
	sessionFinalReleases prometheus.Counter
	ttlEvictions         prometheus.Counter
}

func newThunderMetrics() *thunderMetrics {
	return &thunderMetrics{
		podUtilization: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_pod_utilization",
			Help:      metricsutil.HelpMsgWithStability("Decayed program working set as a fraction of the pod's KV token capacity.", compbasemetrics.ALPHA),
		}, []string{"pod"}),
		podCapacityTokens: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_pod_capacity_tokens",
			Help:      metricsutil.HelpMsgWithStability("KV token capacity per pod; source is 'real' for scraped cache_config_info, 'fallback' for the configured value.", compbasemetrics.ALPHA),
		}, []string{"pod", "source"}),
		programs: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_programs",
			Help:      metricsutil.HelpMsgWithStability("Tracked programs by state.", compbasemetrics.ALPHA),
		}, []string{"state"}),
		holds: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_holds_total",
			Help:      metricsutil.HelpMsgWithStability("Dispatches whose head waited past the hold floor in the flow-control queue, by program class.", compbasemetrics.ALPHA),
		}, []string{"class"}),
		releases: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_releases_total",
			Help:      metricsutil.HelpMsgWithStability("Dispatches picked by the thunder fairness policy, by program class.", compbasemetrics.ALPHA),
		}, []string{"class"}),
		starvationPromotions: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_starvation_promotions_total",
			Help:      metricsutil.HelpMsgWithStability("Queue heads promoted ahead of class and size order by the starvation guard.", compbasemetrics.ALPHA),
		}),
		rebinds: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_rebinds_total",
			Help:      metricsutil.HelpMsgWithStability("Programs moved to a different pod by re-placement.", compbasemetrics.ALPHA),
		}),
		sheds: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_sheds_total",
			Help:      metricsutil.HelpMsgWithStability("Idle programs unbound from an over-ceiling pod; their next turn re-enters admission as new.", compbasemetrics.ALPHA),
		}),
		sessionFinalReleases: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_session_final_releases_total",
			Help:      metricsutil.HelpMsgWithStability("Programs released by the session-final header at end of stream.", compbasemetrics.ALPHA),
		}),
		ttlEvictions: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "thunder_agent_ttl_evictions_total",
			Help:      metricsutil.HelpMsgWithStability("Programs dropped by the idle eviction sweep.", compbasemetrics.ALPHA),
		}),
	}
}

// register registers the collectors, adopting an already-registered collector
// of the same name instead of failing. The supported topology is one plugin
// instance per process, but a registry outliving the instance (config load
// tests, restarts of the plugin without the process) must not panic.
func (m *thunderMetrics) register(reg prometheus.Registerer) error {
	return errors.Join(
		registerOrReuse(reg, &m.podUtilization),
		registerOrReuse(reg, &m.podCapacityTokens),
		registerOrReuse(reg, &m.programs),
		registerOrReuse(reg, &m.holds),
		registerOrReuse(reg, &m.releases),
		registerOrReuse(reg, &m.starvationPromotions),
		registerOrReuse(reg, &m.rebinds),
		registerOrReuse(reg, &m.sheds),
		registerOrReuse(reg, &m.sessionFinalReleases),
		registerOrReuse(reg, &m.ttlEvictions),
	)
}

func registerOrReuse[C prometheus.Collector](reg prometheus.Registerer, target *C) error {
	err := reg.Register(*target)
	if err == nil {
		return nil
	}
	var already prometheus.AlreadyRegisteredError
	if errors.As(err, &already) {
		if existing, ok := already.ExistingCollector.(C); ok {
			*target = existing
			return nil
		}
	}
	return err
}

// setPodCapacity reports a pod's capacity under the source it came from,
// dropping the series of the other source. A pod starts on the configured
// fallback and switches to the scraped value once the metrics extractor has
// run; leaving the stale series behind would report two capacities per pod
// and make "did we get real capacity" unanswerable.
func (m *thunderMetrics) setPodCapacity(pod string, source capacitySource, tokens float64) {
	other := capacityReal
	if source == capacityReal {
		other = capacityFallback
	}
	m.podCapacityTokens.DeleteLabelValues(pod, string(other))
	m.podCapacityTokens.WithLabelValues(pod, string(source)).Set(tokens)
}

// forgetPod drops the per-pod series of an endpoint no longer in the pool.
func (m *thunderMetrics) forgetPod(pod string) {
	m.podUtilization.DeleteLabelValues(pod)
	m.podCapacityTokens.DeletePartialMatch(prometheus.Labels{"pod": pod})
}
