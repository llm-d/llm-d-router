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

package scheduling_test

// TestReserveAtPickRace is the in-process A/B for issue #1598 (thundering herd
// under bursty traffic): the REAL inflight-load-producer, token-load-scorer and
// max-score-picker, one shared endpoint set, and a concurrent burst of
// scheduling goroutines — mirroring the EPP's per-request goroutines.
//
// Setup: 4 endpoints with pre-existing in-flight load 2/2/2/1 (one endpoint
// uniquely least loaded, as when a burst arrives while previous requests are
// still being served). K goroutines then Score+Pick simultaneously. No
// PreRequest accounting runs during the burst — the pick->PreRequest gap was
// measured at ~0.4ms in a live v1.0.0 EPP, i.e. the whole burst is scored
// inside it.
//
// Without reservations: every goroutine scores the same snapshot, so the
// uniquely-least-loaded endpoint strictly wins every pick -> the whole burst
// herds onto it.
//
// With reserve-at-pick: each pick immediately increments a reservation the
// scorers read, so the herd self-corrects and the burst spreads.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/apimachinery/pkg/util/sets"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	inflightload "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
)

// fakeHandle satisfies fwkplugin.Handle for factory construction.
type fakeHandle struct{}

func (fakeHandle) Context() context.Context            { return context.Background() }
func (fakeHandle) Plugin(string) fwkplugin.Plugin      { return nil }
func (fakeHandle) AddPlugin(string, fwkplugin.Plugin)  {}
func (fakeHandle) GetAllPlugins() []fwkplugin.Plugin   { return nil }
func (fakeHandle) GetAllPluginsWithNames() map[string]fwkplugin.Plugin { return nil }
func (fakeHandle) PodList() []k8stypes.NamespacedName  { return nil }
func (fakeHandle) Metrics() fwkplugin.MetricsRecorder  { return prometheus.DefaultRegisterer }
func (fakeHandle) CrossReplicaSyncer() fwkplugin.Plugin { return nil }
func (fakeHandle) SetCrossReplicaSyncer(fwkplugin.Plugin) {}
func (fakeHandle) RefreshMetricsInterval() time.Duration { return 0 }
func (fakeHandle) SchedulingProfileNames() sets.Set[string]  { return nil }

const (
	producerName = "inflight-load-producer"
	numEndpoints = 4
	burstSize    = 64
)

func runBurst(t *testing.T, withReservations bool) map[string]int {
	t.Helper()
	ctx := context.Background()

	// Real producer (reservations included in the InFlightLoad attribute).
	pl, err := inflightload.InFlightLoadProducerFactory(producerName, nil, fakeHandle{})
	if err != nil {
		t.Fatalf("producer factory: %v", err)
	}
	prod := pl.(*inflightload.InFlightLoadProducer)

	// Real endpoints: datalayer ModelServers registered with the producer
	// (injects the live InFlightLoad dynamic attribute), then converted to
	// scheduler endpoints exactly like Director.toSchedulerEndpoints does
	// (same metadata/metrics/attribute map).
	servers := make([]fwkdl.Endpoint, numEndpoints)
	eps := make([]fwksched.Endpoint, numEndpoints)
	for i := range eps {
		srv := fwkdl.NewEndpoint(
			&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: fmt.Sprintf("pod%d", i)}},
			&fwkdl.Metrics{})
		if err := prod.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: srv}); err != nil {
			t.Fatalf("extract: %v", err)
		}
		servers[i] = srv
		eps[i] = fwksched.NewEndpoint(srv.GetMetadata(), srv.GetMetrics(), srv.GetAttributes())
	}

	// Pre-existing in-flight load 2/2/2/1 (in both requests and tokens:
	// 2 prior requests x 2000 uncached tokens each): pod3 uniquely least loaded.
	preload := []int{2, 2, 2, 1}
	for i, n := range preload {
		for j := 0; j < n; j++ {
			prod.Reserve(eps[i].GetMetadata().ID.String(), 2000)
		}
	}

	// Real token-load scorer, pointed at the producer's data key.
	cfg, _ := json.Marshal(map[string]any{"inFlightLoadProducerName": producerName})
	spl, err := tokenload.TokenLoadScorerFactory("token-load-scorer", json.NewDecoder(bytes.NewReader(cfg)), fakeHandle{})
	if err != nil {
		t.Fatalf("scorer factory: %v", err)
	}
	scorer := spl.(fwksched.Scorer)
	uncachedKey := concurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(producerName)

	// Real max-score picker; the ONLY difference between the arms.
	picker := maxscore.NewMaxScorePicker(1).WithName("max-score-picker")
	if withReservations {
		picker = picker.WithReserver(prod)
	}

	var mu sync.Mutex
	counts := make(map[string]int, numEndpoints)
	var wg sync.WaitGroup
	start := make(chan struct{})
	req := &fwksched.InferenceRequest{}
	for i := 0; i < burstSize; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			// Model the request's Produce phase: record this request's
			// uncached-token estimate on every endpoint (~1500 tokens, small
			// per-request jitter so estimates are not bit-identical).
			own := int64(1400 + i%200)
			for _, ep := range eps {
				ep.Put(uncachedKey, &concurrency.UncachedRequestTokens{Tokens: own})
			}
			// Score and pick against the SHARED endpoint set — exactly the
			// concurrent-goroutine reality of the EPP scheduler.
			scores := scorer.Score(ctx, req, eps)
			scored := make([]*fwksched.ScoredEndpoint, 0, len(eps))
			for _, ep := range eps {
				scored = append(scored, &fwksched.ScoredEndpoint{Endpoint: ep, Score: scores[ep]})
			}
			res := picker.Pick(ctx, scored)
			if len(res.TargetEndpoints) == 0 {
				return
			}
			mu.Lock()
			counts[res.TargetEndpoints[0].GetMetadata().ID.Name]++
			mu.Unlock()
			// NOTE: no PreRequest here — the whole burst is scored inside the
			// measured ~0.4ms pick->PreRequest window.
		}(i)
	}
	close(start)
	wg.Wait()
	return counts
}

func TestReserveAtPickRace(t *testing.T) {
	fair := burstSize / numEndpoints

	without := runBurst(t, false)
	topWithout := 0
	var totalWithout int
	for _, n := range without {
		totalWithout += n
		if n > topWithout {
			topWithout = n
		}
	}
	t.Logf("WITHOUT reservations: picks=%v (fair share=%d)", without, fair)
	if totalWithout != burstSize {
		t.Fatalf("lost requests: %d of %d", totalWithout, burstSize)
	}
	// The herd: the uniquely-least-loaded pod (pod3) takes (nearly) everything.
	if topWithout < 3*fair {
		t.Errorf("expected the herd without reservations (top >= %d), got %d", 3*fair, topWithout)
	}

	with := runBurst(t, true)
	topWith := 0
	var totalWith int
	for _, n := range with {
		totalWith += n
		if n > topWith {
			topWith = n
		}
	}
	t.Logf("WITH reservations:    picks=%v (fair share=%d)", with, fair)
	if totalWith != burstSize {
		t.Fatalf("lost requests: %d of %d", totalWith, burstSize)
	}
	// The fix: no pod takes more than twice its fair share.
	if topWith > 2*fair {
		t.Errorf("expected spread with reservations (top <= %d), got %d", 2*fair, topWith)
	}
}
