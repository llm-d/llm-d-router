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

package kvevents_test

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

// The replay harness shares one subscriber identifier across tests, so every
// assertion below compares against a snapshot taken before the attempt instead of
// an absolute value.

const replayTestPodIdentifier = "test-pod"

func counterValue(t *testing.T, counter prometheus.Counter) float64 {
	t.Helper()
	var m dto.Metric
	require.NoError(t, counter.Write(&m))
	return m.GetCounter().GetValue()
}

func gaugeValue(t *testing.T, gauge prometheus.Gauge) float64 {
	t.Helper()
	var m dto.Metric
	require.NoError(t, gauge.Write(&m))
	return m.GetGauge().GetValue()
}

type replayMetrics struct {
	active, completed, failures, processed, lastCompletion float64
}

func snapshotReplayMetrics(t *testing.T) replayMetrics {
	t.Helper()
	pod := replayTestPodIdentifier
	return replayMetrics{
		active:         gaugeValue(t, metrics.ReplayActive.WithLabelValues(pod)),
		completed:      counterValue(t, metrics.ReplayCompleted.WithLabelValues(pod)),
		failures:       counterValue(t, metrics.ReplayFailures.WithLabelValues(pod)),
		processed:      counterValue(t, metrics.ReplayProcessed.WithLabelValues(pod)),
		lastCompletion: gaugeValue(t, metrics.ReplayLastCompletionTimestamp.WithLabelValues(pod)),
	}
}

// TestZMQSubscriber_ReplayMetrics_TrackCompletedReplay covers the success path:
// a replay that receives the full history counts once as completed, counts each
// forwarded event as processed, refreshes the last-completion timestamp, and
// leaves the per-attempt active gauge back at 0.
func TestZMQSubscriber_ReplayMetrics_TrackCompletedReplay(t *testing.T) {
	before := snapshotReplayMetrics(t)

	h := newReplayHarness(t, []replayMessage{
		{seq: 0, payload: buildDistinctBlockStoredPayload(t, 900)},
		{seq: 1, payload: buildDistinctBlockStoredPayload(t, 901)},
	}, false)

	// Anchor the metric expectations on the index actually being rebuilt: without
	// this, a completion count could come from an attempt that ingested nothing.
	require.Eventually(t, func() bool {
		_, err := h.index.GetRequestKey(h.ctx, kvblock.BlockHash(901))
		return err == nil
	}, 5*time.Second, 50*time.Millisecond, "replayed events must reach the index")

	require.Eventually(t, func() bool {
		now := snapshotReplayMetrics(t)
		return now.completed == before.completed+1 && now.processed == before.processed+2
	}, 5*time.Second, 50*time.Millisecond, "one completed attempt, two forwarded events")

	after := snapshotReplayMetrics(t)
	assert.Equal(t, before.failures, after.failures, "a completed replay is not also a failure")
	assert.Zero(t, after.active, "the attempt must not leave replay_active set")
	assert.Greater(t, after.lastCompletion, 0.0, "a successful replay stamps its completion time")
	assert.InDelta(t, float64(time.Now().Unix()), after.lastCompletion, 10,
		"the completion stamp is the time of this attempt")
}

// TestZMQSubscriber_ReplayMetrics_TrackFailedReplay covers the failure path: a
// replay whose history does not start where it was asked to is counted as a
// failure, never as completed, forwards nothing, and still clears the active
// gauge — the case a stuck-at-1 gauge would otherwise hide.
func TestZMQSubscriber_ReplayMetrics_TrackFailedReplay(t *testing.T) {
	before := snapshotReplayMetrics(t)

	h := newReplayHarness(t, []replayMessage{
		{seq: 1, payload: buildDistinctBlockStoredPayload(t, 902)},
	}, false)

	require.Eventually(t, func() bool {
		return snapshotReplayMetrics(t).failures == before.failures+1
	}, 5*time.Second, 50*time.Millisecond, "a truncated replay must count as one failure")

	after := snapshotReplayMetrics(t)
	assert.Equal(t, before.completed, after.completed, "a failed replay must not be counted as completed")
	assert.Equal(t, before.processed, after.processed, "no event may be counted as processed")
	assert.Zero(t, after.active, "the failure path must clear replay_active")
	assert.Equal(t, before.lastCompletion, after.lastCompletion,
		"a failure must not refresh the last successful completion stamp")

	_, err := h.index.GetRequestKey(h.ctx, kvblock.BlockHash(902))
	require.Error(t, err, "the rejected replay must not populate the index")
}

// TestZMQSubscriber_ReplayMetrics_ActiveTracksInFlightAttempt covers an exit the
// two outcome tests cannot observe on their own: the attempt while it is still
// running, and a publisher that never answers. The gauge has to read 1 in flight
// and return to 0 when the attempt finally gives up, or an endpoint stuck in a
// dead replay would look permanently healthy-but-busy.
func TestZMQSubscriber_ReplayMetrics_ActiveTracksInFlightAttempt(t *testing.T) {
	before := snapshotReplayMetrics(t)

	// silent keeps the buffer from answering, so the attempt stays inside
	// requestReplay across the idle-timeout retries until the no-progress budget
	// is spent.
	h := newReplayHarnessWithBehavior(t, []replayMessage{
		{seq: 0, payload: buildDistinctBlockStoredPayload(t, 903)},
	}, false, 0, 0, true)

	require.Eventually(t, func() bool { return h.buffer.requests.Load() >= 1 },
		5*time.Second, 50*time.Millisecond, "replay request expected")
	assert.Equal(t, 1.0, gaugeValue(t, metrics.ReplayActive.WithLabelValues(replayTestPodIdentifier)),
		"an in-flight attempt must report itself active")

	require.Eventually(t, func() bool {
		now := snapshotReplayMetrics(t)
		return now.active == 0 && now.failures >= before.failures+1
	}, 12*time.Second, 50*time.Millisecond, "a silent replay must give up and clear replay_active")

	after := snapshotReplayMetrics(t)
	assert.Equal(t, before.completed, after.completed, "a replay that never completed must not count as completed")
	assert.Equal(t, before.processed, after.processed, "a silent replay forwarded nothing")
}
