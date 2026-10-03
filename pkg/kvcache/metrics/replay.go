// Copyright 2026 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// KV-event replay metrics describe the replay-on-connect / replay-on-gap
// lifecycle of a single subscriber, keyed by the same pod_identifier label as the
// per-pod subscriber metrics above. They answer "is this endpoint's index being
// rebuilt, how far along, and when did a rebuild last succeed" — the questions a
// stale or partially rebuilt prefix index raises, which the aggregate
// kv_cache_events_active_subscribers gauge cannot answer per endpoint.
//
// Why each series is needed rather than derived from the others:
//   - ReplayActive is the only in-progress signal. MessagesReceived and the pool
//     gauges also move during live traffic, so progress alone cannot tell a replay
//     from normal ingestion.
//   - ReplayCompleted and ReplayFailures count whole attempts. A failure is not
//     simply "completed did not increment": an attempt that never received a
//     terminal frame, or that was cancelled with the subscriber's context, returns
//     false without completing.
//   - ReplayProcessed counts events handed to the ordered worker during replays,
//     which distinguishes a replay that failed after making progress from one that
//     never connected at all.
//   - ReplayLastCompletionTimestamp is the freshness signal: the age of the last
//     successful rebuild is what tells an operator how long an endpoint's index may
//     have been serving scheduling decisions from a partial state.
//
// Failure reasons are deliberately not a label here; they stay in
// kv_cache_events_zmq_errors_total{pod_identifier,operation}, whose operation
// values already name each replay exit path (replay-connect, replay-capacity,
// replay-send, replay-incomplete, replay-no-progress).
var (
	// ReplayActive is 1 while a replay attempt is in flight for the pod, and 0
	// once the attempt returns. Every exit path clears it — including cancellation
	// — so a subscriber that dies mid-replay cannot leave the gauge stuck at 1.
	ReplayActive = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_replay_active",
		Help: metricsutil.HelpMsgWithStability(
			"1 while a KV-event replay attempt is in flight for the subscriber",
			compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
	// ReplayCompleted counts replay attempts that received the full requested
	// history and were accepted.
	ReplayCompleted = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_replay_completed_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-event replay attempts that completed successfully",
			compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
	// ReplayFailures counts replay attempts that ended without completing.
	ReplayFailures = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_replay_failures_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-event replay attempts that ended without completing; the reason is in "+
				"kv_cache_events_zmq_errors_total",
			compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
	// ReplayProcessed counts events forwarded from a replay into the ordered
	// event-processing pool.
	ReplayProcessed = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_replay_processed_total",
		Help: metricsutil.HelpMsgWithStability(
			"Total number of KV-events forwarded to the processing pool by replay",
			compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
	// ReplayLastCompletionTimestamp is the Unix time of the most recent successful
	// replay for the pod. Absent until the first success; a rising
	// now()-value gap with ReplayActive staying 0 means the index is aging without
	// a rebuild.
	ReplayLastCompletionTimestamp = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem: routerSubsystem, Name: "kv_cache_events_replay_last_completion_timestamp_seconds",
		Help: metricsutil.HelpMsgWithStability(
			"Unix timestamp of the most recent successful KV-event replay for the subscriber",
			compbasemetrics.ALPHA),
	}, []string{podIdentifierLabel})
)

// cleanupReplaySubscriber drops every per-pod replay series for podIdentifier.
// The active gauge is zeroed before deletion so a scrape landing between the
// subscriber's exit and the deletion cannot report a removed pod as mid-replay.
// Callers must invoke it only once the subscriber goroutine has exited.
func cleanupReplaySubscriber(podIdentifier string) {
	ReplayActive.WithLabelValues(podIdentifier).Set(0)
	ReplayActive.DeleteLabelValues(podIdentifier)
	ReplayCompleted.DeleteLabelValues(podIdentifier)
	ReplayFailures.DeleteLabelValues(podIdentifier)
	ReplayProcessed.DeleteLabelValues(podIdentifier)
	ReplayLastCompletionTimestamp.DeleteLabelValues(podIdentifier)
}
