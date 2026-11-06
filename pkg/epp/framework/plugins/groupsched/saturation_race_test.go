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

// SATURATION RUNS ON SEVERAL SHARD GOROUTINES AT ONCE (the flow controller's, one per shard), so the
// learn path it now carries has to be safe under real concurrency, not only under a single-goroutine
// test run.

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// TestSaturationIsConcurrencySafe hammers the hot path from several goroutines while a Pick loop and
// a scrape run: learned and unlearned candidates mixed, one engine joining part-way through, and a
// mismatched engine that must be rejected once and never retried. Under -race this is the proof the
// learn path has no unlocked shared write and no lock nesting.
func TestSaturationIsConcurrencySafe(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 0, 0)
	m := newPluginMetrics(p)
	reg := prometheus.NewRegistry()
	if err := m.register(reg); err != nil {
		t.Fatalf("register: %v", err)
	}
	p.metrics = m

	ready0, ready1 := metaEndpoint(refEngine(0), 64, 1000), metaEndpoint(refEngine(1), 64, 1000)
	late := metaEndpoint(refEngine(2), 64, 1000) // joins mid-flight
	bad := metaEndpoint(refEngine(3), 32, 1000)  // 32 measured against 64 configured

	know(t, p, ready0, ready1, late, bad)

	stop := make(chan struct{})
	var work, scrape sync.WaitGroup

	// the detector, from eight shard goroutines
	for g := 0; g < 8; g++ {
		work.Add(1)
		go func() {
			defer work.Done()
			for i := 0; i < 200; i++ {
				cands := []fwkdl.Endpoint{ready0, ready1, bad}
				if i > 5 {
					cands = append(cands, late)
				}
				if got := p.Saturation(context.Background(), cands); got != 0 {
					t.Errorf("Saturation must stay 0, got %v", got)
					return
				}
			}
		}()
	}
	// a pass loop over the same plugin
	work.Add(1)
	go func() {
		defer work.Done()
		for i := 0; i < 200; i++ {
			p.sched.Pass(nil, time.Now())
		}
	}()
	// a scrape loop, until the detector and the pass loop are done
	scrape.Add(1)
	go func() {
		defer scrape.Done()
		for {
			select {
			case <-stop:
				return
			default:
				if _, err := reg.Gather(); err != nil {
					t.Errorf("gather: %v", err)
					return
				}
			}
		}
	}()

	workDone := make(chan struct{})
	go func() { work.Wait(); close(workDone) }()
	select {
	case <-workDone:
	case <-time.After(60 * time.Second):
		close(stop)
		t.Fatal("the concurrent run did not finish in a minute")
	}
	close(stop)
	scrape.Wait()

	// the invariants the concurrency must not have broken
	if !p.sched.HasCapacityFor(refEngine(0)) || !p.sched.HasCapacityFor(refEngine(1)) {
		t.Fatal("the ready engines must be learned")
	}
	if p.sched.HasCapacityFor(refEngine(3)) {
		t.Fatal("a mismatched engine must never be learned")
	}
	p.capMu.Lock()
	_, badRejected := p.rejected[refEngine(3)]
	learned := len(p.learned)
	p.capMu.Unlock()
	if !badRejected {
		t.Fatal("the mismatched engine must be in the rejected set")
	}
	if learned < 2 {
		t.Fatalf("learned set has %d engines, want at least 2", learned)
	}
}

// refEngine is an engine key in the form the capacity paths use.
func refEngine(i int) string { return fmt.Sprintf("ns/e%d", i) }
