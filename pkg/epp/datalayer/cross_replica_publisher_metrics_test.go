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

package datalayer

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Each test uses its own state key: the metric vectors are process-wide, so a
// shared key would let one test's publishes leak into another's assertions.

func publishes(key, outcome string) float64 {
	return testutil.ToFloat64(metrics.LlmdCrossReplicaPublishesTotal.WithLabelValues(key, outcome))
}

func lastPublish(key string) float64 {
	return testutil.ToFloat64(metrics.LlmdCrossReplicaLastPublishTimestamp.WithLabelValues(key))
}

func publishCycles(t *testing.T) uint64 {
	t.Helper()
	var m dto.Metric
	require.NoError(t, metrics.LlmdCrossReplicaPublishCycleDuration.WithLabelValues().(prometheus.Histogram).Write(&m))
	return m.GetHistogram().GetSampleCount()
}

// publishSeries counts the outcome series that exist for key without creating any.
func publishSeries(key string) int {
	return seriesFor(metrics.LlmdCrossReplicaPublishesTotal, key)
}

// lastPublishSeries counts the last-publish timestamp series that exist for key
// without creating one. lastPublish cannot be used to assert absence: its
// WithLabelValues call creates the series at 0, which would hide a series that
// time() - llm_d_epp_cross_replica_last_publish_timestamp_seconds turns into a
// ~56 year staleness instead of an empty vector.
func lastPublishSeries(key string) int {
	return seriesFor(metrics.LlmdCrossReplicaLastPublishTimestamp, key)
}

// seriesFor counts the series of c labeled state_key=key, collecting them
// without creating any.
func seriesFor(c prometheus.Collector, key string) int {
	ch := make(chan prometheus.Metric, 64)
	go func() {
		c.Collect(ch)
		close(ch)
	}()
	n := 0
	for metric := range ch {
		var m dto.Metric
		if metric.Write(&m) != nil {
			continue
		}
		for _, label := range m.GetLabel() {
			if label.GetName() == "state_key" && label.GetValue() == key {
				n++
			}
		}
	}
	return n
}

func registerEndpoints(t *testing.T, pub *crossReplicaPublisher, names ...string) {
	t.Helper()
	for _, name := range names {
		require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: name}))
	}
}

func TestCrossReplicaPublisherMetrics_CountsEachPublishedEndpoint(t *testing.T) {
	const key = "metrics:per-endpoint-success"
	pub := testCrossReplicaPublisher(&fakeSyncer{}, fakeContributor{key: key})
	registerEndpoints(t, pub, "ep-a", "ep-b")
	before := time.Now()
	cyclesBefore := publishCycles(t)

	pub.publishAll(context.Background())

	assert.Equal(t, 2.0, publishes(key, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Zero(t, publishes(key, metrics.CrossReplicaPublishOutcomeError))
	assert.GreaterOrEqual(t, lastPublish(key), float64(before.Unix()))
	assert.Equal(t, cyclesBefore+1, publishCycles(t))
}

func TestCrossReplicaPublisherMetrics_CountsFailuresAndKeepsTimestamp(t *testing.T) {
	const key = "metrics:per-endpoint-failure"
	syncer := &fakeSyncer{}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: key})
	registerEndpoints(t, pub, "ep-a", "ep-b")

	pub.publishAll(context.Background())
	stamped := lastPublish(key)
	require.NotZero(t, stamped)

	// The syncer goes away: every publish fails and the timestamp must stop
	// moving, which is what an operator alerts on.
	syncer.mu.Lock()
	syncer.setErr = errors.New("syncer unreachable")
	syncer.mu.Unlock()
	pub.publishAll(context.Background())
	pub.publishAll(context.Background())

	assert.Equal(t, 2.0, publishes(key, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, 4.0, publishes(key, metrics.CrossReplicaPublishOutcomeError))
	assert.Equal(t, stamped, lastPublish(key))
}

func TestCrossReplicaPublisherMetrics_BatchCountsPartialFailure(t *testing.T) {
	const key = "metrics:batch-partial"
	syncer := &batchSyncer{onBatch: func(_ context.Context, _ fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
		for _, id := range ids {
			if id == "ns/ep-b" {
				return map[string]error{id: errors.New("chunk failed")}
			}
		}
		return nil
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: key})
	registerEndpoints(t, pub, "ep-a", "ep-b", "ep-c")

	pub.publishAll(context.Background())

	assert.Equal(t, 2.0, publishes(key, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, 1.0, publishes(key, metrics.CrossReplicaPublishOutcomeError))
	// A cycle with any failure is not clean, so it stamps nothing.
	assert.Zero(t, lastPublishSeries(key))
}

func TestCrossReplicaPublisherMetrics_PartialFailureKeepsLastCleanTimestamp(t *testing.T) {
	const key = "metrics:partial-after-clean"
	var failB atomic.Bool
	syncer := &batchSyncer{onBatch: func(_ context.Context, _ fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
		if !failB.Load() {
			return nil
		}
		for _, id := range ids {
			if id == "ns/ep-b" {
				return map[string]error{id: errors.New("chunk failed")}
			}
		}
		return nil
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: key})
	registerEndpoints(t, pub, "ep-a", "ep-b")

	pub.publishAll(context.Background())
	stamped := lastPublish(key)
	require.NotZero(t, stamped)

	// One endpoint keeps failing while the other publishes: the clock stops,
	// so a staleness alert catches an endpoint its peers never see updated.
	failB.Store(true)
	pub.publishAll(context.Background())

	assert.Equal(t, stamped, lastPublish(key))
	assert.Equal(t, 3.0, publishes(key, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, 1.0, publishes(key, metrics.CrossReplicaPublishOutcomeError))
}

func TestCrossReplicaPublisherMetrics_BatchTotalFailureLeavesTimestampUnset(t *testing.T) {
	const key = "metrics:batch-total"
	syncer := &batchSyncer{onBatch: func(_ context.Context, _ fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
		failures := make(map[string]error, len(ids))
		for _, id := range ids {
			failures[id] = errors.New("syncer unreachable")
		}
		return failures
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: key})
	registerEndpoints(t, pub, "ep-a", "ep-b")

	pub.publishAll(context.Background())

	assert.Zero(t, publishes(key, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, 2.0, publishes(key, metrics.CrossReplicaPublishOutcomeError))
	assert.Zero(t, lastPublishSeries(key), "no last-publish series before the first successful publish")
}

func TestCrossReplicaPublisherMetrics_SeriesExistBeforeFirstPublish(t *testing.T) {
	const key = "metrics:init"
	require.Zero(t, publishSeries(key))

	pub := newCrossReplicaPublisher(&fakeSyncer{}, extractorMapWith(fakeContributor{key: key}), 0, 0)
	require.NotNil(t, pub)

	// Every outcome exists at zero, so rate() sees the first failure as an increase.
	assert.Equal(t, 3, publishSeries(key))
	// The timestamp stays absent until a publish succeeds.
	assert.Zero(t, lastPublishSeries(key))
}

func TestCrossReplicaPublisherMetrics_SkippedEndpointIsNotASuccess(t *testing.T) {
	const key = "metrics:skipped-endpoint"
	syncer := &fakeSyncer{setErr: errors.New("syncer unreachable")}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: key})

	// The endpoint was removed after publishAll took its snapshot, so set
	// skips it without calling the syncer.
	cycle := newPublishCycle()
	pub.publish(context.Background(), types.NamespacedName{Namespace: "ns", Name: "removed"}, cycle)
	cycle.record(time.Now())

	assert.Zero(t, publishes(key, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Zero(t, publishes(key, metrics.CrossReplicaPublishOutcomeError))
	assert.Zero(t, lastPublishSeries(key), "a skipped endpoint must not stamp the last-publish time")
}

func TestCrossReplicaPublisherMetrics_EmptyCycleIsNotTimed(t *testing.T) {
	const key = "metrics:empty-cycle"
	pub := testCrossReplicaPublisher(&fakeSyncer{}, fakeContributor{key: key})
	cyclesBefore := publishCycles(t)

	pub.publishAll(context.Background())

	assert.Equal(t, cyclesBefore, publishCycles(t))
}

func TestCrossReplicaPublisherMetrics_SyncDisabledContributorHasNoSeries(t *testing.T) {
	const key = "metrics:sync-disabled"
	pub := newCrossReplicaPublisher(&fakeSyncer{}, extractorMapWith(fakeContributor{key: key, syncDisabled: true}), 0, 0)
	require.Nil(t, pub)
	assert.Zero(t, publishSeries(key))
}

func TestPublishCycle_RecordsOncePerStateKey(t *testing.T) {
	const key = "metrics:cycle-aggregate"
	cycle := newPublishCycle()
	cycle.addSuccess(fwkdl.StateKey(key))
	cycle.addBatch(fwkdl.StateKey(key), 3, map[string]error{
		"ns/ep-b": errors.New("refused"),
		"ns/ep-c": context.DeadlineExceeded,
	})
	cycle.addBatch(fwkdl.StateKey(key), 0, nil)

	cycle.record(time.Now())

	assert.Equal(t, 2.0, publishes(key, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, 1.0, publishes(key, metrics.CrossReplicaPublishOutcomeError))
	assert.Equal(t, 1.0, publishes(key, metrics.CrossReplicaPublishOutcomeTimeout))
	assert.Zero(t, lastPublishSeries(key), "a failure in the cycle keeps it from being clean")
}

func TestPublishCycle_NilCycleIgnoresReports(t *testing.T) {
	var cycle *publishCycle
	assert.NotPanics(t, func() {
		cycle.addSuccess(fwkdl.StateKey("metrics:nil-cycle"))
		cycle.addFailure(fwkdl.StateKey("metrics:nil-cycle"), errors.New("x"), false)
		cycle.addBatch(fwkdl.StateKey("metrics:nil-cycle"), 1, nil)
	})
}

type netTimeoutError struct{}

func (netTimeoutError) Error() string { return "i/o timeout" }
func (netTimeoutError) Timeout() bool { return true }

func TestCrossReplicaPublisherMetrics_TimeoutsAreSeparateFromErrors(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{name: "context deadline", err: context.DeadlineExceeded},
		{name: "wrapped context deadline", err: fmt.Errorf("redis-syncer: set: %w", context.DeadlineExceeded)},
		{name: "network timeout", err: netTimeoutError{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			key := "metrics:timeout:" + tc.name
			pub := testCrossReplicaPublisher(&fakeSyncer{setErr: tc.err}, fakeContributor{key: fwkdl.StateKey(key)})
			registerEndpoints(t, pub, "ep-a")

			pub.publishAll(context.Background())

			assert.Equal(t, 1.0, publishes(key, metrics.CrossReplicaPublishOutcomeTimeout))
			assert.Zero(t, publishes(key, metrics.CrossReplicaPublishOutcomeError))
			assert.Zero(t, lastPublishSeries(key))
		})
	}
}

func TestCrossReplicaPublisherMetrics_ShutdownCycleIsNotRecorded(t *testing.T) {
	const key = "metrics:shutdown"
	pub := testCrossReplicaPublisher(&fakeSyncer{setErr: context.Canceled}, fakeContributor{key: key})
	registerEndpoints(t, pub, "ep-a", "ep-b")
	cyclesBefore := publishCycles(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	pub.publishAll(ctx)

	assert.Zero(t, publishes(key, metrics.CrossReplicaPublishOutcomeError))
	assert.Zero(t, publishes(key, metrics.CrossReplicaPublishOutcomeTimeout))
	assert.Equal(t, cyclesBefore, publishCycles(t))
}
