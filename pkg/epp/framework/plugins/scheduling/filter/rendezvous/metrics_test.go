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
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// TestDecisionMetrics: each routing decision is counted per cluster, and the moved gauge follows the table.
func TestDecisionMetrics(t *testing.T) {
	decisions.Reset()
	movedEntries.Reset()
	guard := newGuard()
	f := newFilter(t, baseConfig(), guard)
	endpoints := []scheduling.Endpoint{shared("a", "1"), shared("b", "1"), shared("c", "1")}
	s := sessionOnA(t, f, endpoints)
	turn := func(id string) *scheduling.InferenceRequest {
		r := withSession(s)
		r.RequestID = id
		return r
	}
	count := func(cluster, decision string) float64 {
		return testutil.ToFloat64(decisions.WithLabelValues(f.typedName.Name, cluster, decision))
	}
	home := count("a", decisionHome)

	if got := pick(t, f, turn("r0"), endpoints); got != "a" || count("a", decisionHome) != home+1 {
		t.Fatalf("home pick: went to %s, home count %v", got, count("a", decisionHome)-home)
	}
	guard.full = map[string]bool{"a": true}
	f.Filter(t.Context(), turn("r1"), endpoints)
	if err := f.PreRequest(t.Context(), turn("r1"), picked("c")); err != nil {
		t.Fatal(err)
	}
	if count("c", decisionSpill) != 1 {
		t.Fatalf("spill to c counted %v, want 1", count("c", decisionSpill))
	}
	if got := pick(t, f, turn("r2"), endpoints); got != "c" || count("c", decisionKept) != 1 {
		t.Fatalf("kept: went to %s, counted %v", got, count("c", decisionKept))
	}
	if got := testutil.ToFloat64(movedEntries.WithLabelValues(f.typedName.Name)); got != 1 {
		t.Fatalf("moved entries %v, want 1", got)
	}
}

func TestRegisterMetrics(t *testing.T) {
	registry := prometheus.NewRegistry()
	for i := 0; i < 2; i++ {
		if err := registerMetrics(registry); err != nil {
			t.Fatalf("register %d: %v", i, err)
		}
	}
	if err := registerMetrics(nil); err != nil {
		t.Fatal(err)
	}
}
