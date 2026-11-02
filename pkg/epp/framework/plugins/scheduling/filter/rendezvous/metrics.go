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

package rendezvous

import (
	"errors"
	"fmt"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Decision labels.
const (
	decisionHome     = "home"     // the hashed cluster
	decisionPin      = "pin"      // the tenant's pinned cluster
	decisionKept     = "kept"     // a spilled conversation stays on its spill cluster
	decisionSpill    = "spill"    // home was full or down; the scorers picked another cluster
	decisionFailover = "failover" // home could not serve and no cluster had room
)

var (
	decisions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "multicluster_rendezvous_decisions_total",
			Help:      metricsutil.HelpMsgWithStability("Requests routed by the multicluster rendezvous filter, by cluster and decision.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "cluster", "decision"},
	)

	movedEntries = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "multicluster_rendezvous_moved_entries",
			Help:      metricsutil.HelpMsgWithStability("Conversations the multicluster rendezvous filter keeps on a cluster other than their home.", compbasemetrics.ALPHA),
		},
		[]string{"plugin_name"},
	)
)

// registerMetrics registers the filter collectors; a nil registerer is a no-op.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{decisions, movedEntries} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register multicluster rendezvous filter metric: %w", err)
		}
	}
	return nil
}

// record counts a decision for an endpoint ID ("namespace/name"); the cluster label is the name.
func (f *Filter) record(id, decision string) {
	decisions.WithLabelValues(f.typedName.Name, id[strings.LastIndex(id, "/")+1:], decision).Inc()
	if f.moved != nil {
		movedEntries.WithLabelValues(f.typedName.Name).Set(float64(f.moved.Len()))
	}
}
