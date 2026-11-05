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

// SATURATION IS THE HOT PATH: the flow controller polls it about once a millisecond, and it used to
// be a pure atomic read. It is now also where capacity is learned, so these tests exist to keep that
// from becoming a scheduler mutex on every poll.

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// TestSaturationTakesNoSchedulerLock is the LOCK PROBE: with s.mu held by someone else, a
// Saturation over candidates whose geometry is already learned must return promptly. If the learn
// path ever takes s.mu before checking what it already knows, this test blocks and fails.
func TestSaturationTakesNoSchedulerLock(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 0, 0)
	ep0 := liveEndpoint(engineID(0), 0.1, 500, 64, 1000, time.Now())
	ep1 := liveEndpoint(engineID(1), 0.1, 500, 64, 1000, time.Now())
	key0, key1 := endpointKey(ep0.GetMetadata()), endpointKey(ep1.GetMetadata())
	pods := livePods(ep0, ep1)
	// first call: learns both engines (this is the only time it may take the lock)
	if got := p.Saturation(context.Background(), pods); got != 0 {
		t.Fatalf("Saturation must stay 0, got %v", got)
	}
	for _, key := range []string{key0, key1} {
		if !p.sched.HasCapacityFor(key) {
			t.Fatalf("engine %q's geometry was not learned by the first Saturation", key)
		}
	}

	// hold the scheduler's mutex from another goroutine and poll Saturation under it.
	p.sched.mu.Lock()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 5; i++ {
			if got := p.Saturation(context.Background(), pods); got != 0 {
				t.Errorf("Saturation must stay 0, got %v", got)
			}
		}
	}()
	select {
	case <-done:
		p.sched.mu.Unlock()
	case <-time.After(2 * time.Second):
		p.sched.mu.Unlock()
		t.Fatal("Saturation blocked on the scheduler's mutex with every engine already learned: the hot path must take capMu only")
	}
}

// TestRejectedGeometryIsTriedOnce pins the other half of the fast path: a candidate whose measured
// block size disagrees with the configured one is rejected ONCE and not retried on every poll (a pod's
// geometry is fixed for its life, and the rejection is loud).
func TestRejectedGeometryIsTriedOnce(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 0, 0)
	// 32 measured against 64 configured: a mismatch
	bad := liveEndpoint(engineID(0), 0.1, 500, 32, 1000, time.Now())
	badKey := endpointKey(bad.GetMetadata())
	_ = p.Saturation(context.Background(), livePods(bad))
	p.capMu.Lock()
	_, rejected := p.rejected[badKey]
	p.capMu.Unlock()
	if !rejected {
		t.Fatal("a mismatched block size must be recorded as rejected")
	}
	// a second poll must not try again: the engine stays unlearned and the set is unchanged
	_ = p.Saturation(context.Background(), livePods(bad))
	if p.sched.HasCapacityFor(badKey) {
		t.Fatal("a mismatched engine must never be learned")
	}
	// and a NEW endpoint key (a rollout: a new pod name) gets its own attempt
	rolled := liveEndpoint("rolled", 0.1, 500, 64, 1000, time.Now())
	rolledKey := endpointKey(rolled.GetMetadata())
	_ = p.Saturation(context.Background(), livePods(rolled))
	if !p.sched.HasCapacityFor(rolledKey) {
		t.Fatal("a new endpoint's geometry must get its own attempt after a rollout")
	}
}

// TestCapacityMissingSurvivesOnlyKnownEndpoints pins the DECLARED change: capacity_missing is
// rendered at scrape from the KNOWN endpoint set, so it disappears when the engine does (it used to
// be set to 0 once).
func TestCapacityMissingIsRenderedForKnownEndpoints(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 0, 0)
	k0, k1 := "ns/a", "ns/b"
	p.rememberEndpoint(metaFor(t, k0))
	p.rememberEndpoint(metaFor(t, k1))

	missing := scrapedCapacityMissing(t, p)
	if v, ok := missing[engineLabel(k0)]; !ok || v != 1 {
		t.Fatalf("a known endpoint with no measured capacity must report 1, got %v (present=%v)", v, ok)
	}
	// learn one: its series drops to 0, the other stays 1
	_ = p.Saturation(context.Background(), []fwkdl.Endpoint{metaEndpoint(k0, 64, 1000)})
	missing = scrapedCapacityMissing(t, p)
	if v := missing[engineLabel(k0)]; v != 0 {
		t.Fatalf("a learned engine must report 0, got %v", v)
	}
	if v := missing[engineLabel(k1)]; v != 1 {
		t.Fatalf("the unlearned engine must still report 1, got %v", v)
	}
	// forget it: the series must be GONE, not zero (declared: a removed pod stops being reported)
	p.forgetEndpoint(metaFor(t, k1))
	missing = scrapedCapacityMissing(t, p)
	if _, ok := missing[engineLabel(k1)]; ok {
		t.Fatal("a removed endpoint must not report capacity_missing at all")
	}
}

// metaFor is an endpoint's metadata for the capacity bookkeeping paths.
func metaFor(t *testing.T, id string) *fwkdl.EndpointMetadata {
	t.Helper()
	ns, name, ok := strings.Cut(id, "/")
	if !ok {
		ns, name = "ns", id
	}
	return &fwkdl.EndpointMetadata{Name: name, Address: name + ":8080",
		ID: types.NamespacedName{Namespace: ns, Name: name}}
}

// scrapedCapacityMissing reads the capacity_missing family out of a scrape of this plugin's own
// metrics, which is where it now lives.
func scrapedCapacityMissing(t *testing.T, p *GroupScheduler) map[string]float64 {
	t.Helper()
	reg := prometheus.NewRegistry()
	m := newPluginMetrics(p)
	if err := m.register(reg); err != nil {
		t.Fatalf("register: %v", err)
	}
	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	out := map[string]float64{}
	for _, mf := range mfs {
		if mf.GetName() != prometheus.BuildFQName("", subsystem, "capacity_missing") {
			continue
		}
		for _, mm := range mf.GetMetric() {
			for _, lp := range mm.GetLabel() {
				out[lp.GetValue()] = mm.GetGauge().GetValue()
			}
		}
	}
	return out
}

// metaEndpoint is an endpoint whose id is exactly the given engine key.
func metaEndpoint(key string, blockSize, numBlocks int) fwkdl.Endpoint {
	ns, name, _ := strings.Cut(key, "/")
	meta := &fwkdl.EndpointMetadata{Name: name, Address: name + ":8080",
		ID: types.NamespacedName{Namespace: ns, Name: name}}
	return fwkdl.NewEndpoint(meta, &fwkdl.Metrics{
		KVCacheUsagePercent: 0.1, CacheBlockSize: blockSize, CacheNumBlocks: numBlocks,
		UpdateTime: time.Now()})
}
