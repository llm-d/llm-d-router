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

package disaggregatedsetrollout

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	corev1 "k8s.io/api/core/v1"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// Metric tests share process-global Prometheus collectors registered once via
// sync.Once. Do not call t.Parallel() in this file.
func resetMetrics(t *testing.T) {
	t.Helper()
	registerMetrics()
	strictRevisionNoMatchTotal.Reset()
	revisionOverflowTotal.Reset()
	revisionGatingShare.Reset()
	revisionSelectionTotal.Reset()
}

func selectionCount(revision string, source selectionSource) float64 {
	return testutil.ToFloat64(revisionSelectionTotal.WithLabelValues(PluginType, "test-screener", revision, string(source)))
}

func TestMetricRevisionSelectionSource(t *testing.T) {
	twoRevisions := map[string]map[string]int{
		"v1": {"prefill": 1, "decode": 1},
		"v2": {"prefill": 1, "decode": 1},
	}
	sessionConfig := prefixAffinityConfig(sessionaffinity.SessionIDSource{Header: testSessionHeader})
	tests := []struct {
		name    string
		config  Config
		counts  map[string]map[string]int
		request *fwksched.InferenceRequest
		want    selectionSource
	}{
		{"strict header", validConfig(), twoRevisions,
			&fwksched.InferenceRequest{Headers: map[string]string{DefaultRevisionHeader: "v2"}}, sourceHeader},
		{"session identifier", sessionConfig, twoRevisions,
			chatRequest(map[string]string{testSessionHeader: "s1"}, chatMessage("user", "hi")), sourceSession},
		{"conversation prefix", sessionConfig, twoRevisions, chatRequest(nil, chatMessage("user", "hi")), sourcePrefix},
		{"nothing to key on", sessionConfig, twoRevisions, &fwksched.InferenceRequest{Headers: map[string]string{}}, sourceRandom},
		{"affinity off", validConfig(), twoRevisions, chatRequest(nil, chatMessage("user", "hi")), sourceRandom},
		{"one covered revision", sessionConfig, map[string]map[string]int{"v1": {"prefill": 1, "decode": 1}},
			chatRequest(nil, chatMessage("user", "hi")), sourceSingle},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics(t)
			screener := newTestScreener(test.config)
			seedCounts(t, screener, test.counts)
			revision := screenedRevision(t, screener, test.request, candidateRevisions("v1", "v2"))
			if got := selectionCount(revision, test.want); got != 1 {
				t.Fatalf("want one %s selection of %s, got %v", test.want, revision, got)
			}
			if got := testutil.CollectAndCount(revisionSelectionTotal); got != 1 {
				t.Fatalf("want one selection series, got %d", got)
			}
		})
	}
}

func TestMetricRevisionSelectionCoordinated(t *testing.T) {
	resetMetrics(t)
	screener := newTestScreener(validConfig())
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 1, "decode": 1},
		"v2": {"prefill": 1, "decode": 1},
	})
	request := &fwksched.InferenceRequest{Headers: map[string]string{reqcommon.RevisionDecisionIDHeaderKey: "parallel-1"}}
	first := screenedRevision(t, screener, request, candidateRevisions("v1", "v2"))
	if second := screenedRevision(t, screener, request, candidateRevisions("v1", "v2")); second != first {
		t.Fatalf("coordinated sibling served by %s, want %s", second, first)
	}
	if got := selectionCount(first, sourceRandom); got != 1 {
		t.Fatalf("want one random selection, got %v", got)
	}
	if got := selectionCount(first, sourceCoordinated); got != 1 {
		t.Fatalf("want one coordinated selection, got %v", got)
	}
}

func TestMetricRevisionSelectionSkipsStrictNoMatch(t *testing.T) {
	for _, mode := range []GatingMode{GatingModeSum, GatingModeDisabled} {
		resetMetrics(t)
		config := validConfig()
		config.RevisionGating.Mode = mode
		screener := newTestScreener(config)
		seedCounts(t, screener, map[string]map[string]int{"v1": {"prefill": 1, "decode": 1}})
		request := &fwksched.InferenceRequest{Headers: map[string]string{DefaultRevisionHeader: "v99"}}
		if got := screenCandidates(t, screener, request, candidateRevisions("v1")); len(got) != 0 {
			t.Fatalf("%s: unknown revision kept %d endpoints", mode, len(got))
		}
		if got := testutil.CollectAndCount(revisionSelectionTotal); got != 0 {
			t.Fatalf("%s: a strict header that matched nothing added %d selection series", mode, got)
		}
		if got := testutil.ToFloat64(strictRevisionNoMatchTotal.WithLabelValues(PluginType, "test-screener")); got != 1 {
			t.Fatalf("%s: want one strict no-match, got %v", mode, got)
		}
	}
}

func TestMetricRevisionSelectionDeletesRemovedRevision(t *testing.T) {
	resetMetrics(t)
	screener := newTestScreener(validConfig())
	v1Prefill := readyPod("v1-p", "v1", "prefill")
	v1Decode := readyPod("v1-d", "v1", "decode")
	seedPods(t, screener, v1Prefill, v1Decode, readyPod("v2-p", "v2", "prefill"), readyPod("v2-d", "v2", "decode"))
	for _, revision := range []string{"v1", "v2"} {
		request := &fwksched.InferenceRequest{Headers: map[string]string{DefaultRevisionHeader: revision}}
		screenedRevision(t, screener, request, candidateRevisions("v1", "v2"))
	}

	handler := &podNotificationHandler{screener: screener}
	for _, pod := range []*corev1.Pod{v1Prefill, v1Decode} {
		if err := handler.Extract(context.Background(), podEvent(t, pod, fwkdl.EventDelete)); err != nil {
			t.Fatal(err)
		}
	}
	if got := testutil.CollectAndCount(revisionSelectionTotal); got != 1 {
		t.Fatalf("want 1 selection series after removal, got %d", got)
	}
	if got := selectionCount("v2", sourceHeader); got != 1 {
		t.Fatalf("v2 selection series must survive, got %v", got)
	}
}

func TestMetricStrictHeaderNoMatch(t *testing.T) {
	resetMetrics(t)
	config := validConfig()
	config.RevisionGating = &RevisionGating{Mode: GatingModeDisabled}
	screener := newTestScreener(config)
	screener.Screen(context.Background(),
		&fwksched.InferenceRequest{Headers: map[string]string{"x-llm-d-disagg-revision": "v99"}},
		[]fwksched.Endpoint{endpoint("p1", revLabels("v1"))},
	)

	got := testutil.ToFloat64(strictRevisionNoMatchTotal.WithLabelValues(PluginType, "test-screener"))
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
