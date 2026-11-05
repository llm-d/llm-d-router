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

package groupsched

// PLUGIN-LEVEL TESTS: the framework-facing surface (detector, Filter, capacity learning, the
// live-metrics rule) plus the config/emitter guard.

import (
	"context"
	"testing"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
	"k8s.io/apimachinery/pkg/types"
)

// ---- framework test doubles ----

// liveEndpoint is ONE datalayer endpoint (the type the DETECTOR receives).
func liveEndpoint(id string, kv float64, waiting, blockSize, numBlocks int, at time.Time) *fwkdl.ModelServer {
	meta := &fwkdl.EndpointMetadata{Name: id, Address: id + ":8080",
		ID: types.NamespacedName{Namespace: "ns", Name: id}}
	return fwkdl.NewEndpoint(meta, &fwkdl.Metrics{
		KVCacheUsagePercent: kv, WaitingQueueSize: waiting,
		CacheBlockSize: blockSize, CacheNumBlocks: numBlocks, UpdateTime: at,
	})
}

// schedulerEndpoint is the SCHEDULING-layer endpoint (the type the Filter takes).
func schedulerEndpoint(id string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Name: id, Address: id + ":8080",
		ID: types.NamespacedName{Namespace: "ns", Name: id}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, CacheBlockSize: 64, CacheNumBlocks: 1000}, nil)
}

// livePods is the detector's input: live datalayer endpoints.
func livePods(eps ...*fwkdl.ModelServer) []fwkdl.Endpoint {
	out := make([]fwkdl.Endpoint, 0, len(eps))
	for _, e := range eps {
		out = append(out, e)
	}
	return out
}

// podsOf is the Filter's input: scheduling-layer endpoints.
func podsOf(eps ...fwksched.Endpoint) []fwksched.Endpoint { return eps }

// requestFor builds the request the Filter and PreRequest are handed for a thread's released
// turn — "" means an untracked (anonymous) request.
func requestFor(thread string) *fwksched.InferenceRequest {
	ir := &fwksched.InferenceRequest{RequestID: "r-" + thread, RequestSizeBytes: 4096}
	if thread != "" {
		ir.PutAttribute(threadidentity.ThreadIdentityKey, thread)
	}
	return ir
}

// newTestPlugin builds the plugin around a scheduler with n engines of `blocks` capacity each
// (no metrics registrar: unit mode).
func newTestPlugin(t *testing.T, cfg Config, n, blocks int) *GroupScheduler {
	t.Helper()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := s.SetNumBlocksFor(engineID(i), cfg.BlockSizeTokens, blocks); err != nil {
			t.Fatal(err)
		}
	}
	p := &GroupScheduler{typedName: fwkplugin.TypedName{Type: PluginType, Name: "gs"},
		cfg: cfg, sched: s, metrics: newUnitMetrics(s), now: time.Now,
		endpoints: map[string]struct{}{}, learned: map[string]struct{}{}, rejected: map[string]struct{}{}}
	return p
}

// engineID is the engine id the core will see for the i-th test endpoint: the endpoint's own
// datalayer key (namespace/name) — the same value endpointKey yields.
func engineID(i int) string { return "ns/" + string(rune('a'+i)) }

// ---- the detector ----

// The DEFAULT detector answer is "not saturated": the seat budget is the admission control, so
// the flow controller's pool gate never closes on this policy.
func TestDetectorDefaultsToNotSaturated(t *testing.T) {
	for _, n := range []int{1, 2} {
		p := newTestPlugin(t, defaultConfig(), n, 1000)
		now := time.Now()
		eps := make([]*fwkdl.ModelServer, 0, n)
		for i := 0; i < n; i++ {
			eps = append(eps, liveEndpoint(engineID(i), 0.99, 500, 64, 1000, now))
		}
		if got := p.Saturation(context.Background(), livePods(eps...)); got != 0 {
			t.Fatalf("N=%d: the default detector must answer 0 (not saturated), got %v", n, got)
		}
		// the detector's own call is the capacity path's only input
		if !p.sched.HasCapacityFor(engineID(0)) {
			t.Fatalf("N=%d: the live candidate's geometry must be learned by Saturation alone", n)
		}
	}
}

// The detector CONFLICT guard: a second detector in the same config is announced (exactly one
// may answer the flow-control gate).
func TestDetectorIdentityGate(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 1, 1000)
	if got := p.otherDetectors(); len(got) != 0 {
		t.Fatalf("no handle → no other detectors, got %v", got)
	}
}

// Capacity rides the LIVE path (the event-time copy may carry an empty geometry), and the per-
// engine liveness alarm is per engine.
func TestCapacityLearnedFromLiveSamples(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 0, 0)
	if p.sched.HasCapacity() {
		t.Fatal("no capacity yet")
	}
	p.Saturation(context.Background(), livePods(liveEndpoint("a", 0.1, 0, 64, 1000, time.Now()),
		liveEndpoint("b", 0.1, 0, 64, 2000, time.Now())))
	if !p.sched.HasCapacityFor(engineID(0)) || !p.sched.HasCapacityFor(engineID(1)) {
		t.Fatal("both engines' capacity must be learned from the live samples")
	}
	snap := p.sched.Snapshot(p.now())
	a, b := snap.Engines[engineLabel(engineID(0))], snap.Engines[engineLabel(engineID(1))]
	if a.BudgetBlocks != 950 || b.BudgetBlocks != 1900 {
		t.Fatalf("per-engine budgets wrong: a=%d b=%d", a.BudgetBlocks, b.BudgetBlocks)
	}
}

// BENCHMARK: the flow controller polls the detector at ~1 kHz, so Saturation must be a couple
// of loads and NO allocation on the default (noop) path. go test -run xxx -bench
// BenchmarkSaturation -benchmem ./pkg/epp/framework/plugins/groupsched/
func BenchmarkSaturationNoop(b *testing.B) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		b.Fatal(err)
	}
	p := &GroupScheduler{cfg: cfg, sched: s, metrics: newUnitMetrics(s),
		endpoints: map[string]struct{}{}, learned: map[string]struct{}{}, rejected: map[string]struct{}{}}
	now := time.Now()
	eps := make([]fwkdl.Endpoint, 0, 2)
	for i := 0; i < 2; i++ {
		eps = append(eps, liveEndpoint(engineID(i), 0.5, 10, 64, 1000, now))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := p.Saturation(context.Background(), eps); got != 0 {
			b.Fatalf("noop detector must answer 0, got %v", got)
		}
	}
}
