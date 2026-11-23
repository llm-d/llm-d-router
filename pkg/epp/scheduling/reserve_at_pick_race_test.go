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

// Tests for the pick-time reservation mechanism. The A/B test
// below models the director's per-request pipeline: score, pick, then
// ReserveResult. Without the reservation between pick and the next request's
// scoring, a concurrent burst is scored against one stale snapshot and the
// least-loaded endpoint wins every pick (the thundering herd). With it, each
// pick is visible to the requests that score after it.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
)

// fakeHandle satisfies fwkplugin.Handle for factory construction.
type fakeHandle struct{}

func (fakeHandle) Context() context.Context                            { return context.Background() }
func (fakeHandle) Plugin(string) fwkplugin.Plugin                      { return nil }
func (fakeHandle) AddPlugin(string, fwkplugin.Plugin)                  {}
func (fakeHandle) GetAllPlugins() []fwkplugin.Plugin                   { return nil }
func (fakeHandle) GetAllPluginsWithNames() map[string]fwkplugin.Plugin { return nil }
func (fakeHandle) PodList() []k8stypes.NamespacedName                  { return nil }
func (fakeHandle) Metrics() fwkplugin.MetricsRecorder                  { return prometheus.DefaultRegisterer }
func (fakeHandle) CrossReplicaSyncer() fwkplugin.Plugin                { return nil }
func (fakeHandle) SetCrossReplicaSyncer(fwkplugin.Plugin)              {}
func (fakeHandle) RefreshMetricsInterval() time.Duration               { return 0 }
func (fakeHandle) SchedulingProfileNames() sets.Set[string]            { return nil }

const (
	producerName = "inflight-load-producer"
	numEndpoints = 4
	burstSize    = 64
)

func newProducer(t testing.TB) *inflightload.InFlightLoadProducer {
	t.Helper()
	pl, err := inflightload.InFlightLoadProducerFactory(producerName, nil, fakeHandle{})
	if err != nil {
		t.Fatalf("producer factory: %v", err)
	}
	return pl.(*inflightload.InFlightLoadProducer)
}

// newRig builds the real producer, token-load scorer and endpoints wired the
// same way the director wires them (datalayer ModelServers registered with
// the producer, then converted with the same metadata/metrics/attributes).
func newRig(t testing.TB) (*inflightload.InFlightLoadProducer, fwksched.Scorer, []fwksched.Endpoint, fwkplugin.DataKey) {
	t.Helper()
	ctx := context.Background()
	prod := newProducer(t)

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

	cfg, _ := json.Marshal(map[string]any{"inFlightLoadProducerName": producerName})
	spl, err := tokenload.TokenLoadScorerFactory("token-load-scorer", json.NewDecoder(bytes.NewReader(cfg)), fakeHandle{})
	if err != nil {
		t.Fatalf("scorer factory: %v", err)
	}
	uncachedKey := concurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(producerName)
	return prod, spl.(fwksched.Scorer), eps, uncachedKey
}

// runBurst fires burstSize concurrent scheduling cycles, each following the
// director's order: record the per-request uncached estimate (the Produce
// phase), score, pick, then optionally ReserveResult. Goroutines are
// released staggerStart apart, modeling the per-request pacing of the real
// pipeline; without reservations nothing changes state mid-burst, so that arm
// stays deterministic regardless of interleaving.
func runBurst(t *testing.T, withReservations bool, staggerStart time.Duration) map[string]int {
	t.Helper()
	ctx := context.Background()
	prod, scorer, eps, uncachedKey := newRig(t)
	picker := maxscore.NewMaxScorePicker(1).WithName("max-score-picker")

	// Pre-existing in-flight load 2/2/2/1 (in both requests and tokens:
	// 2 prior requests x 2000 uncached tokens each): pod3 uniquely least loaded.
	// The estimate attribute is set first, as the Produce phase would for
	// those prior requests, so ReserveResult reserves token load too.
	for _, ep := range eps {
		ep.Put(uncachedKey, &concurrency.UncachedRequestTokens{Tokens: 2000})
	}
	preload := []int{2, 2, 2, 1}
	for i, n := range preload {
		for j := 0; j < n; j++ {
			prod.ReserveProfileResult(ctx, fmt.Sprintf("preload-%d-%d", i, j), "default",
				&fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{eps[i]}})
		}
	}

	var mu sync.Mutex
	counts := make(map[string]int, numEndpoints)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < burstSize; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			if staggerStart > 0 {
				time.Sleep(time.Duration(i) * staggerStart)
			}
			// Produce phase: this request's uncached-token estimate on every
			// endpoint (~1500 tokens with per-request jitter).
			own := int64(1400 + i%200)
			for _, ep := range eps {
				ep.Put(uncachedKey, &concurrency.UncachedRequestTokens{Tokens: own})
			}
			scores := scorer.Score(ctx, &fwksched.InferenceRequest{}, eps)
			scored := make([]*fwksched.ScoredEndpoint, 0, len(eps))
			for _, ep := range eps {
				scored = append(scored, &fwksched.ScoredEndpoint{Endpoint: ep, Score: scores[ep]})
			}
			result := picker.Pick(ctx, scored)
			if len(result.TargetEndpoints) == 0 {
				return
			}
			if withReservations {
				prod.ReserveProfileResult(ctx, fmt.Sprintf("req-%d", i), "default", result)
			}
			mu.Lock()
			counts[result.TargetEndpoints[0].GetMetadata().ID.Name]++
			mu.Unlock()
		}(i)
	}
	close(start)
	wg.Wait()
	return counts
}

func TestReservationSpreadsPacedBurst(t *testing.T) {
	fair := burstSize / numEndpoints

	// Both arms use the same pacing: goroutines released staggerStart apart,
	// modeling the per-request pacing of the real scheduling pipeline (a
	// truly simultaneous release makes the "with" arm depend on goroutine
	// interleaving and flake on many-core hosts). Under paced arrivals the
	// claim under test is exact: without reservations every cycle sees the
	// same snapshot; with them, each pick is visible to the cycles after it.
	without := runBurst(t, false, 250*time.Microsecond)
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
	// Nothing changes state during the burst in this arm, so every cycle sees
	// the same snapshot and the least-loaded pod wins them all.
	if topWithout < 3*fair {
		t.Errorf("expected the herd without reservations (top >= %d), got %d", 3*fair, topWithout)
	}

	with := runBurst(t, true, 250*time.Microsecond)
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
	if topWith > 2*fair {
		t.Errorf("expected spread with reservations (top <= %d), got %d", 2*fair, topWith)
	}
}

type dumpState struct {
	Endpoints []struct {
		Endpoint     string `json:"endpoint"`
		Requests     int64  `json:"requests"`
		Tokens       int64  `json:"tokens"`
		Reservations int64  `json:"reservations"`
	} `json:"endpoints"`
}

// dumpReservations returns per-endpoint request, token and reservation counts
// from the producer's debug dump.
func dumpReservations(t *testing.T, prod *inflightload.InFlightLoadProducer) dumpState {
	t.Helper()
	raw, err := prod.DumpState()
	if err != nil {
		t.Fatalf("DumpState: %v", err)
	}
	var state dumpState
	if err := json.Unmarshal(raw, &state); err != nil {
		t.Fatalf("unmarshal dump state: %v", err)
	}
	return state
}

// TestReserveConvertsAtPreRequest verifies the conversion: after PreRequest
// runs for a request that reserved, the reservation counters are back to zero
// and the real in-flight counters carry the load.
func TestReserveConvertsAtPreRequest(t *testing.T) {
	ctx := context.Background()
	prod, _, eps, _ := newRig(t)
	ep := eps[0]
	eid := ep.GetMetadata().ID.String()

	result := &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{ep}}
	prod.ReserveProfileResult(ctx, "req-convert", "default", result)

	if got := prod.GetRequests(eid); got != 0 {
		t.Errorf("real counter should be 0 before PreRequest, got %d", got)
	}

	req := &fwksched.InferenceRequest{RequestID: "req-convert", SchedulingResult: &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{"default": result}}}
	if err := prod.PreRequest(ctx, req, req.SchedulingResult); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}
	if got := prod.GetRequests(eid); got != 1 {
		t.Errorf("real counter should be 1 after PreRequest, got %d", got)
	}
}

// TestPreRequestDoesNotReleaseForeignReservation verifies that a request
// which never reserved cannot decrement another request's reservation
// (the release is keyed by request ID).
func TestPreRequestDoesNotReleaseForeignReservation(t *testing.T) {
	ctx := context.Background()
	prod, _, eps, _ := newRig(t)
	ep := eps[0]
	eid := ep.GetMetadata().ID.String()

	result := &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{ep}}
	// req-a reserves; req-b never does.
	prod.ReserveProfileResult(ctx, "req-a", "default", result)

	// req-b's PreRequest on the same endpoint must not touch req-a's
	// reservation.
	reqB := &fwksched.InferenceRequest{RequestID: "req-b", SchedulingResult: &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{"default": result}}}
	if err := prod.PreRequest(ctx, reqB, reqB.SchedulingResult); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}
	// req-b's real accounting landed (1 request) and req-a's reservation is
	// still outstanding: DumpState exposes both.
	state := dumpReservations(t, prod)
	var found bool
	for _, e := range state.Endpoints {
		if e.Endpoint == eid {
			found = true
			if e.Requests != 1 {
				t.Errorf("real requests: want 1 (req-b only), got %d", e.Requests)
			}
			if e.Reservations != 1 {
				t.Errorf("reservations: want 1 (req-a still outstanding), got %d", e.Reservations)
			}
		}
	}
	if !found {
		t.Fatalf("endpoint %s missing from dump state", eid)
	}
}

// TestAbandonedReservationReclaimed verifies the janitor path: a request that
// reserved but never dispatched has its reservation rolled back when its
// plugin state is reclaimed.
func TestAbandonedReservationReclaimed(t *testing.T) {
	ctx := context.Background()
	prod, _, eps, _ := newRig(t)
	ep := eps[0]
	eid := ep.GetMetadata().ID.String()

	result := &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{ep}}
	prod.ReserveProfileResult(ctx, "req-abandoned", "default", result)

	state := dumpReservations(t, prod)
	for _, e := range state.Endpoints {
		if e.Endpoint == eid && e.Reservations != 1 {
			t.Fatalf("reservations before reclaim: want 1, got %d", e.Reservations)
		}
	}

	// The director calls RequestAbandoned on every path that exits after
	// scheduling but before PreRequest.
	prod.RequestAbandoned(ctx, &fwksched.InferenceRequest{RequestID: "req-abandoned"})

	state = dumpReservations(t, prod)
	for _, e := range state.Endpoints {
		if e.Endpoint == eid && e.Reservations != 0 {
			t.Errorf("reservations after reclaim: want 0, got %d", e.Reservations)
		}
	}
}

// TestAbandonAfterPreRequestRollsBackBothCounters covers the case where
// prepareRequest fails partway: the producer's PreRequest has already
// converted the reservation into real counters, and RequestAbandoned must
// roll those back too, because the request is never forwarded.
func TestAbandonAfterPreRequestRollsBackBothCounters(t *testing.T) {
	ctx := context.Background()
	prod, _, eps, _ := newRig(t)
	ep := eps[0]
	eid := ep.GetMetadata().ID.String()

	result := &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{ep}}
	prod.ReserveProfileResult(ctx, "req-flaky", "default", result)

	req := &fwksched.InferenceRequest{RequestID: "req-flaky", SchedulingResult: &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{"default": result}}}
	if err := prod.PreRequest(ctx, req, req.SchedulingResult); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}
	// Conversion happened: real counter up, reservation gone.
	state := dumpReservations(t, prod)
	for _, e := range state.Endpoints {
		if e.Endpoint == eid {
			if e.Requests != 1 || e.Reservations != 0 {
				t.Fatalf("after PreRequest: want requests=1 reservations=0, got %d/%d", e.Requests, e.Reservations)
			}
		}
	}

	// The request is abandoned after PreRequest ran.
	prod.RequestAbandoned(ctx, req)

	// Both the real counter and any leftover state are rolled back.
	state = dumpReservations(t, prod)
	for _, e := range state.Endpoints {
		if e.Endpoint == eid {
			if e.Requests != 0 || e.Reservations != 0 {
				t.Errorf("after abandon: want requests=0 reservations=0, got %d/%d", e.Requests, e.Reservations)
			}
		}
	}
}
