// Copyright 2025 The llm-d Authors.
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

//nolint:testpackage // testing unexported functions
package metrics

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func TestCollectorsIncludesAllMetrics(t *testing.T) {
	collectors := Collectors()

	// Build a set of collector pointers for lookup and ensure Collectors()
	// does not return duplicates, which would panic during MustRegister.
	collectorSet := make(map[prometheus.Collector]bool, len(collectors))
	for _, c := range collectors {
		if collectorSet[c] {
			t.Fatalf("Collectors() contains a duplicate collector: %T", c)
		}
		collectorSet[c] = true
	}

	expected := []struct {
		name      string
		collector prometheus.Collector
	}{
		{"Admissions", Admissions},
		{"Evictions", Evictions},
		{"LookupRequests", LookupRequests},
		{"LookupHits", LookupHits},
		{"LookupLatency", LookupLatency},
		{"MaxPodHitCount", MaxPodHitCount},
		{"DedupRemovedHashesSuppressed", DedupRemovedHashesSuppressed},
		{"DedupRemovedHashesForwarded", DedupRemovedHashesForwarded},
		{"KVEventStoresSkipped", KVEventStoresSkipped},
		{"KVEventRemovalsSkipped", KVEventRemovalsSkipped},
		{"SubscriberActive", SubscriberActive},
		{"SubscriberReconnections", SubscriberReconnections},
		{"MessagesReceived", MessagesReceived},
		{"ZMQErrors", ZMQErrors},
		{"PoolQueueDepth", PoolQueueDepth},
		{"PoolCapacity", PoolCapacity},
		{"ReplayActive", ReplayActive},
		{"ReplayCompleted", ReplayCompleted},
		{"ReplayFailures", ReplayFailures},
		{"ReplayProcessed", ReplayProcessed},
		{"ReplayLastCompletionTimestamp", ReplayLastCompletionTimestamp},
	}

	for _, e := range expected {
		if !collectorSet[e.collector] {
			t.Errorf("Collectors() is missing %s", e.name)
		}
	}
}

func TestKVEventsMetricNames(t *testing.T) {
	// The kvevents observability metrics must be emitted under the router EPP
	// subsystem with the kv_cache_events prefix.
	reg := prometheus.NewRegistry()
	kvevents := []prometheus.Collector{
		KVEventStoresSkipped, KVEventRemovalsSkipped,
		SubscriberActive, SubscriberReconnections, MessagesReceived,
		ZMQErrors, PoolQueueDepth, PoolCapacity,
		ReplayActive, ReplayCompleted, ReplayFailures, ReplayProcessed,
		ReplayLastCompletionTimestamp,
	}
	for _, c := range kvevents {
		reg.MustRegister(c)
	}

	// Emit a sample for each labeled metric so it appears in the gather output.
	SubscriberActive.Set(1)
	SubscriberReconnections.WithLabelValues("pod-a").Inc()
	MessagesReceived.WithLabelValues("pod-a").Inc()
	ZMQErrors.WithLabelValues("pod-a", "recv").Inc()
	KVEventStoresSkipped.WithLabelValues("sliding_window", "unsupported_cache_kind").Inc()
	KVEventRemovalsSkipped.WithLabelValues("sliding_window", "unsupported_cache_kind").Inc()
	PoolQueueDepth.Set(3)
	PoolCapacity.Set(4)
	ReplayActive.WithLabelValues("pod-a").Set(1)
	ReplayCompleted.WithLabelValues("pod-a").Inc()
	ReplayFailures.WithLabelValues("pod-a").Inc()
	ReplayProcessed.WithLabelValues("pod-a").Add(3)
	ReplayLastCompletionTimestamp.WithLabelValues("pod-a").Set(1700000000)

	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	got := make(map[string]bool, len(mfs))
	for _, mf := range mfs {
		got[mf.GetName()] = true
	}

	for _, name := range []string{
		"llm_d_epp_kv_cache_events_active_subscribers",
		"llm_d_epp_kv_cache_events_subscriber_reconnections_total",
		"llm_d_epp_kv_cache_events_messages_received_total",
		"llm_d_epp_kv_cache_events_zmq_errors_total",
		"llm_d_epp_kv_cache_events_stores_skipped_total",
		"llm_d_epp_kv_cache_events_removals_skipped_total",
		"llm_d_epp_kv_cache_events_pool_queue_depth",
		"llm_d_epp_kv_cache_events_pool_capacity",
		"llm_d_epp_kv_cache_events_replay_active",
		"llm_d_epp_kv_cache_events_replay_completed_total",
		"llm_d_epp_kv_cache_events_replay_failures_total",
		"llm_d_epp_kv_cache_events_replay_processed_total",
		"llm_d_epp_kv_cache_events_replay_last_completion_timestamp_seconds",
	} {
		if !got[name] {
			t.Errorf("expected metric %q to be registered, got names: %v", name, got)
		}
	}
}

func TestCleanupSubscriberDropsPerPodSeries(t *testing.T) {
	// Removing a subscriber must drop only its own series, leaving other pods'
	// series intact.
	SubscriberReconnections.WithLabelValues("pod-a").Inc()
	MessagesReceived.WithLabelValues("pod-a").Inc()
	ZMQErrors.WithLabelValues("pod-a", "recv").Inc()
	ZMQErrors.WithLabelValues("pod-a", "connect").Inc()
	SubscriberReconnections.WithLabelValues("pod-b").Inc()
	MessagesReceived.WithLabelValues("pod-b").Inc()
	ZMQErrors.WithLabelValues("pod-b", "recv").Inc()

	CleanupSubscriber("pod-a")

	reg := prometheus.NewRegistry()
	reg.MustRegister(SubscriberReconnections, MessagesReceived, ZMQErrors)
	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	for _, mf := range mfs {
		for _, m := range mf.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == podIdentifierLabel && l.GetValue() == "pod-a" {
					t.Errorf("metric %q still has a series for pod-a", mf.GetName())
				}
			}
		}
	}

	// pod-b must be untouched: three series across the three metrics.
	remaining := 0
	for _, mf := range mfs {
		remaining += len(mf.GetMetric())
	}
	if remaining != 3 {
		t.Errorf("expected 3 remaining pod-b series, got %d", remaining)
	}

	CleanupSubscriber("pod-b")
}

// TestCleanupSubscriberDropsReplaySeries pins the lifecycle guarantee the replay
// panels rely on: when a subscriber's goroutine has exited and its series are
// cleaned, no per-pod replay series survives — in particular none left stuck
// reading as an in-flight replay for an endpoint that no longer exists.
func TestCleanupSubscriberDropsReplaySeries(t *testing.T) {
	ReplayActive.WithLabelValues("pod-gone").Set(1)
	ReplayCompleted.WithLabelValues("pod-gone").Inc()
	ReplayFailures.WithLabelValues("pod-gone").Inc()
	ReplayProcessed.WithLabelValues("pod-gone").Add(4)
	ReplayLastCompletionTimestamp.WithLabelValues("pod-gone").Set(1700000000)

	// A second pod keeps its series; cleanup must be scoped to one identifier.
	ReplayCompleted.WithLabelValues("pod-alive").Inc()
	ReplayActive.WithLabelValues("pod-alive").Set(1)

	reg := prometheus.NewRegistry()
	reg.MustRegister(ReplayActive, ReplayCompleted, ReplayFailures, ReplayProcessed,
		ReplayLastCompletionTimestamp)

	CleanupSubscriber("pod-gone")

	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}
	for _, mf := range mfs {
		for _, m := range mf.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == podIdentifierLabel && l.GetValue() == "pod-gone" {
					t.Errorf("metric %q still has a replay series for the removed subscriber", mf.GetName())
				}
			}
		}
	}

	// The surviving pod's in-flight replay must still be reported as active.
	var activeSeen bool
	for _, mf := range mfs {
		if mf.GetName() != "llm_d_epp_kv_cache_events_replay_active" {
			continue
		}
		for _, m := range mf.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == podIdentifierLabel && l.GetValue() == "pod-alive" && m.GetGauge().GetValue() == 1 {
					activeSeen = true
				}
			}
		}
	}
	if !activeSeen {
		t.Error("expected the live subscriber's replay_active series to survive cleanup")
	}

	CleanupSubscriber("pod-alive")
}

func TestLogMetrics(t *testing.T) {
	// Set up a buffer to capture log output
	var buf bytes.Buffer
	handler := slog.NewTextHandler(&buf, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})
	logrLogger := logr.FromSlogHandler(handler)
	log.SetLogger(logrLogger)

	ctx := context.Background()

	t.Run("no_latency", func(t *testing.T) {
		// Reset buffer
		buf.Reset()

		// Set test values for metrics
		Admissions.Inc()       // 1 admission
		Evictions.Add(2)       // 2 evictions
		LookupRequests.Add(10) // 10 lookups
		LookupHits.Add(5)      // 5 hits

		// Call logMetrics
		logMetrics(ctx)

		// Get the logged output
		output := buf.String()

		// Verify that the log contains expected key-value pairs
		expectedParts := []string{
			"admissions=1",
			"evictions=2",
			"lookups=10",
			"hits=5",
			"latency_count=0",
			"latency_sum=0",
			"latency_avg=0",
		}

		for _, part := range expectedParts {
			if !strings.Contains(output, part) {
				t.Errorf("Expected '%s' in log output, but it was not found. Full output: %s", part, output)
			}
		}
	})

	t.Run("with_latency", func(t *testing.T) {
		// Reset buffer
		buf.Reset()

		LookupLatency.Observe(0.1) // Observe latency
		LookupLatency.Observe(0.2) // Another observation

		logMetrics(ctx)
		// Get the logged output
		output := buf.String()
		// Verify that the log contains expected key-value pairs
		expectedParts := []string{
			"latency_count=2",
			"latency_sum=0.3",
			"latency_avg=0.15",
		}

		for _, part := range expectedParts {
			if !strings.Contains(output, part) {
				t.Errorf("Expected '%s' in log output, but it was not found. Full output: %s", part, output)
			}
		}
	})

	t.Run("max_pod_hit_count_logged", func(t *testing.T) {
		buf.Reset()

		MaxPodHitCount.Add(42)

		logMetrics(ctx)

		output := buf.String()
		if !strings.Contains(output, "max_pod_hit_count=42") {
			t.Errorf("Expected 'max_pod_hit_count=42' in log output, but it was not found. Full output: %s", output)
		}
	})
}
