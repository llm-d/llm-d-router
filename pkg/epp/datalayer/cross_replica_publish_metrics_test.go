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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/prometheus/client_golang/prometheus/testutil"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const publishMetricTestKey = "inflight:metrics"

// resetPublishMetrics registers the EPP metrics once and clears them so a test
// starts from zero series rather than from an earlier test's counts.
func resetPublishMetrics(t *testing.T) {
	t.Helper()
	metrics.Register()
	metrics.Reset()
	t.Cleanup(metrics.Reset)
}

// publishCounter returns the value of one state key and outcome series, or 0
// when the series does not exist.
func publishCounter(t *testing.T, outcome string) float64 {
	t.Helper()
	return testutil.ToFloat64(metrics.LlmdCrossReplicaPublishesTotal.WithLabelValues(publishMetricTestKey, outcome))
}

// lastSuccess returns the timestamp gauge for one state key and whether the
// series exists at all.
func lastSuccess(t *testing.T) (float64, bool) {
	t.Helper()
	family := gatherFamily(t, "llm_d_epp_cross_replica_last_success_timestamp_seconds")
	if family == nil {
		return 0, false
	}
	for _, m := range family.GetMetric() {
		for _, l := range m.GetLabel() {
			if l.GetName() == "state_key" && l.GetValue() == publishMetricTestKey {
				return m.GetGauge().GetValue(), true
			}
		}
	}
	return 0, false
}

// cycleSamples returns the observation count of the publish-cycle histogram.
func cycleSamples(t *testing.T) uint64 {
	t.Helper()
	family := gatherFamily(t, "llm_d_epp_cross_replica_publish_duration_seconds")
	if family == nil || len(family.GetMetric()) == 0 {
		return 0
	}
	return family.GetMetric()[0].GetHistogram().GetSampleCount()
}

func TestCrossReplicaPublisher_CountsPerEndpointPublishes(t *testing.T) {
	resetPublishMetrics(t)
	syncer := &fakeSyncer{}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: publishMetricTestKey})
	endpointID := types.NamespacedName{Namespace: "ns", Name: "ep-a"}
	require.True(t, pub.registerEndpoint(endpointID))

	pub.publish(context.Background(), endpointID)

	assert.Equal(t, float64(1), publishCounter(t, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, float64(0), publishCounter(t, metrics.CrossReplicaPublishOutcomeError))
}

func TestCrossReplicaPublisher_RecordsLastSuccessPerSuccessfulCycle(t *testing.T) {
	resetPublishMetrics(t)
	syncer := &fakeSyncer{}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: publishMetricTestKey})
	require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: "ep-a"}))
	require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: "ep-b"}))

	pub.publishAll(context.Background())

	assert.Equal(t, float64(2), publishCounter(t, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, float64(0), publishCounter(t, metrics.CrossReplicaPublishOutcomeError))
	stamp, ok := lastSuccess(t)
	require.True(t, ok, "a cycle where every endpoint landed records its time")
	assert.InDelta(t, float64(time.Now().Unix()), stamp, 60)
	assert.Equal(t, uint64(1), cycleSamples(t), "one publishAll observes one cycle")
}

func TestCrossReplicaPublisher_DoesNotMoveLastSuccessOnPartialCycle(t *testing.T) {
	resetPublishMetrics(t)
	syncer := &fakeSyncer{setErrOn: map[string]error{"ns/ep-b": assert.AnError}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: publishMetricTestKey})
	require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: "ep-a"}))
	require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: "ep-b"}))

	pub.publishAll(context.Background())

	assert.Equal(t, float64(1), publishCounter(t, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, float64(1), publishCounter(t, metrics.CrossReplicaPublishOutcomeError))

	_, ok := lastSuccess(t)
	assert.False(t, ok, "one endpoint landing and another failing has not succeeded as a cycle")
}

func TestCrossReplicaPublisher_EmptySnapshotObservesNoCycle(t *testing.T) {
	resetPublishMetrics(t)
	syncer := &fakeSyncer{}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: publishMetricTestKey})

	pub.publishAll(context.Background())

	assert.Empty(t, syncer.sets)
	assert.Equal(t, uint64(0), cycleSamples(t), "an empty snapshot has no cycle duration to record")
	_, ok := lastSuccess(t)
	assert.False(t, ok)
}

func TestCrossReplicaPublisher_CountsFailedPublishes(t *testing.T) {
	resetPublishMetrics(t)
	syncer := &fakeSyncer{setErr: assert.AnError}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: publishMetricTestKey})
	endpointID := types.NamespacedName{Namespace: "ns", Name: "ep-a"}
	require.True(t, pub.registerEndpoint(endpointID))

	pub.publish(context.Background(), endpointID)

	assert.Equal(t, float64(0), publishCounter(t, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, float64(1), publishCounter(t, metrics.CrossReplicaPublishOutcomeError))

	_, ok := lastSuccess(t)
	assert.False(t, ok, "a failed publish must not move the last-success timestamp")
}

func TestCrossReplicaPublisher_SkipsCountingRemovedEndpoint(t *testing.T) {
	resetPublishMetrics(t)
	syncer := &fakeSyncer{}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: publishMetricTestKey})
	endpointID := types.NamespacedName{Namespace: "ns", Name: "ep-a"}
	require.True(t, pub.registerEndpoint(endpointID))
	pub.mu.Lock()
	pub.endpoints.Delete(endpointID)
	pub.mu.Unlock()

	pub.publish(context.Background(), endpointID)

	assert.Empty(t, syncer.sets)
	assert.Equal(t, float64(0), publishCounter(t, metrics.CrossReplicaPublishOutcomeSuccess),
		"a publish skipped because the endpoint left must not count as a success")
}

func TestCrossReplicaPublisher_CountsPartialBatchFailure(t *testing.T) {
	resetPublishMetrics(t)
	syncer := &batchSyncer{onBatch: func(_ context.Context, _ fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
		return map[string]error{ids[0]: assert.AnError}
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: publishMetricTestKey})
	require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: "ep-a"}))
	require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: "ep-b"}))

	pub.publishAll(context.Background())

	assert.Equal(t, float64(1), publishCounter(t, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, float64(1), publishCounter(t, metrics.CrossReplicaPublishOutcomeError))

	_, ok := lastSuccess(t)
	assert.False(t, ok, "one failed endpoint keeps the last-success timestamp from moving")
}

func TestCrossReplicaPublisher_RecordsBatchSuccessAndCycle(t *testing.T) {
	resetPublishMetrics(t)
	syncer := &batchSyncer{onBatch: func(context.Context, fwkdl.CrossReplicaSpec, []string, time.Duration) map[string]error {
		return nil
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: publishMetricTestKey})
	require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: "ep-a"}))
	require.True(t, pub.registerEndpoint(types.NamespacedName{Namespace: "ns", Name: "ep-b"}))

	pub.publishAll(context.Background())

	assert.Equal(t, float64(2), publishCounter(t, metrics.CrossReplicaPublishOutcomeSuccess))
	assert.Equal(t, float64(0), publishCounter(t, metrics.CrossReplicaPublishOutcomeError))
	_, ok := lastSuccess(t)
	assert.True(t, ok, "a fully successful batch records its time")
	assert.Equal(t, uint64(1), cycleSamples(t), "one publishAll observes one cycle")
}
