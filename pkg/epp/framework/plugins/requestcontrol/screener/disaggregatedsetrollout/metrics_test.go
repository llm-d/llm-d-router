/*
Copyright 2026 The Kubernetes Authors.

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

package disaggregatedsetrollout

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	corev1 "k8s.io/api/core/v1"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Metric tests share process-global Prometheus collectors registered once via
// sync.Once. Do not call t.Parallel() in this file.
func resetMetrics(t *testing.T) {
	t.Helper()
	registerMetrics()
	strictHeaderNoMatchTotal.Reset()
	revisionGatingShare.Reset()
}

func TestMetricStrictHeaderNoMatch(t *testing.T) {
	resetMetrics(t)
	config := validConfig()
	config.RevisionGating = &RevisionGating{Mode: GatingModeDisabled}
	screener := newTestScreener(config)
	screener.screenStrictSelectors(context.Background(),
		&fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v99"}},
		[]fwksched.Endpoint{endpoint("p1", revLabels("v1"))},
	)

	got := testutil.ToFloat64(strictHeaderNoMatchTotal.WithLabelValues(PluginType, "test-screener", "revision"))
	if got != 1 {
		t.Fatalf("strict no-match: want 1, got %v", got)
	}
}

func TestMetricRevisionGatingShare(t *testing.T) {
	resetMetrics(t)
	screener := newTestScreener(validConfig())
	seedPods(t, screener,
		readyPod("v1-p", "v1", "prefill"),
		readyPod("v1-d", "v1", "decode"),
		readyPod("v2-p", "v2", "prefill"),
	)

	want := map[string]float64{"v1": 1, "v2": 0}
	for revision, expected := range want {
		got := testutil.ToFloat64(revisionGatingShare.WithLabelValues(PluginType, "test-screener", string(GatingModeSum), revision))
		if got != expected {
			t.Fatalf("revision %s: want share %v, got %v", revision, expected, got)
		}
	}
}

func TestMetricRevisionGatingShareDeletesRemovedRevision(t *testing.T) {
	resetMetrics(t)
	screener := newTestScreener(validConfig())
	v1Prefill := readyPod("v1-p", "v1", "prefill")
	v1Decode := readyPod("v1-d", "v1", "decode")
	seedPods(t, screener,
		v1Prefill,
		v1Decode,
		readyPod("v2-p", "v2", "prefill"),
		readyPod("v2-d", "v2", "decode"),
	)
	if got := testutil.CollectAndCount(revisionGatingShare); got != 2 {
		t.Fatalf("want 2 revision series, got %d", got)
	}

	handler := &podNotificationHandler{screener: screener}
	for _, pod := range []*corev1.Pod{v1Prefill, v1Decode} {
		if err := handler.Extract(context.Background(), podEvent(t, pod, fwkdl.EventDelete)); err != nil {
			t.Fatal(err)
		}
	}
	if got := testutil.CollectAndCount(revisionGatingShare); got != 1 {
		t.Fatalf("want 1 revision series after removal, got %d", got)
	}
}
