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

package redis

import (
	"errors"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	directionPublish = "publish"
	directionRead    = "read"
)

var (
	streamBytes = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "redis_syncer_stream_bytes_total",
			Help:      metricsutil.HelpMsgWithStability("Encoded element bytes this replica appended to its streams (publish) or read from peer streams (read).", compbasemetrics.ALPHA),
		},
		[]string{"state_key", "direction"},
	)

	streamBudgetExhausted = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "redis_syncer_stream_budget_exhausted_total",
			Help:      metricsutil.HelpMsgWithStability("Endpoint publishes or peer reads deferred to a later tick because maxBytesPerSecond was spent.", compbasemetrics.ALPHA),
		},
		[]string{"state_key", "direction"},
	)

	streamResets = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "redis_syncer_stream_resets_total",
			Help:      metricsutil.HelpMsgWithStability("Streams this replica deleted and started republishing under a new epoch.", compbasemetrics.ALPHA),
		},
		[]string{"state_key"},
	)

	streamPeerLag = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "redis_syncer_stream_peer_lag_generations",
			Help:      metricsutil.HelpMsgWithStability("Elements a peer has published that this replica has not read yet, observed after each peer read.", compbasemetrics.ALPHA),
			Buckets:   append([]float64{0}, prometheus.ExponentialBuckets(1, 4, 10)...),
		},
		[]string{"state_key"},
	)

	setBatchDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "redis_syncer_set_batch_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration of one SetBatch call, which publishes every endpoint for one state key. Compare with syncInterval.", compbasemetrics.ALPHA),
			Buckets:   metricsutil.GeneralLatencyBuckets,
		},
		[]string{"state_key"},
	)
)

// registerMetrics registers the syncer metrics. The vectors are package-level,
// so a second syncer instance registering them again is a no-op.
func registerMetrics(registerer prometheus.Registerer) error {
	for _, collector := range []prometheus.Collector{streamBytes, streamBudgetExhausted, streamResets, streamPeerLag, setBatchDuration} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("redis-syncer: register metric: %w", err)
		}
	}
	return nil
}

func observeSetBatch(key fwkdl.StateKey, start time.Time) {
	setBatchDuration.WithLabelValues(string(key)).Observe(time.Since(start).Seconds())
}
