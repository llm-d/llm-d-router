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

// Tests for pick-time scoring. A scorer that opts into
// fwksched.PickTimeScorer runs inside the scheduler profile's pick critical
// section, together with the picker and the reservation of the picked
// endpoint, so a burst of concurrently scheduled requests is placed exactly
// as if its requests had been scheduled one at a time.
//
// The tests drive the real scheduler (Scheduler.Schedule) with the real
// inflight-load-producer, token-load-scorer and max-score-picker. Each
// request gets its own scheduling endpoints, cloned from the shared ones the
// way the director builds them, carrying the request's own uncached-token
// estimate. A barrier scorer holds every request of a burst at the scoring
// step until the whole burst has arrived, which puts all of them between
// scoring and picking at once: the worst case for the race, and independent
// of goroutine timing.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

const (
	pickTimeProfile = "default"
	// preloadTokens is the uncached-token estimate of each preloaded request.
	preloadTokens = 2000
)

// barrierScorer holds every request at the scoring step until n requests
// have reached it, then lets all of them continue. It scores every endpoint
// 0, so it never changes the ranking. A nil barrier does not wait.
type barrierScorer struct {
	n       int32
	arrived atomic.Int32
	release chan struct{}
}

func newBarrierScorer(n int) *barrierScorer {
	return &barrierScorer{n: int32(n), release: make(chan struct{})}
}

func (b *barrierScorer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "barrier-scorer", Name: "barrier-scorer"}
}

func (b *barrierScorer) Category() fwksched.ScorerCategory { return fwksched.Distribution }

func (b *barrierScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	if b != nil {
		if b.arrived.Add(1) == b.n {
			close(b.release)
		}
		select {
		case <-b.release:
		case <-time.After(30 * time.Second): // fail rather than hang if a request never arrives
		}
	}
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = 0
	}
	return scores
}

// pickTimeRig is a scheduler with one profile: token-load-scorer, then a
// barrier scorer, then the max-score picker, reserving through the real
// inflight-load-producer.
type pickTimeRig struct {
	prod        *inflightload.InFlightLoadProducer
	shared      []fwksched.Endpoint
	uncachedKey fwkplugin.DataKey
	inFlightKey fwkplugin.DataKey
	sched       *scheduling.Scheduler
}

func newPickTimeRig(t testing.TB, scoreAtPick bool, barrier *barrierScorer, preload []int) *pickTimeRig {
	t.Helper()
	prod, _, shared, uncachedKey := newRig(t)

	cfg, _ := json.Marshal(map[string]any{
		"inFlightLoadProducerName": producerName,
		"scoreAtPick":              scoreAtPick,
	})
	plugin, err := tokenload.TokenLoadScorerFactory("token-load-scorer", json.NewDecoder(bytes.NewReader(cfg)), fakeHandle{})
	if err != nil {
		t.Fatalf("scorer factory: %v", err)
	}
	profile := scheduling.NewSchedulerProfile().
		WithScorers(
			scheduling.NewWeightedScorer(plugin.(fwksched.Scorer), 1),
			scheduling.NewWeightedScorer(barrier, 1),
		).
		WithPicker(maxscore.NewMaxScorePicker(1))
	sched := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{pickTimeProfile: profile}).
		WithReservers([]fwkrh.EndpointReserver{prod}))

	rig := &pickTimeRig{
		prod:        prod,
		shared:      shared,
		uncachedKey: uncachedKey,
		inFlightKey: concurrency.InFlightLoadDataKey.WithNonEmptyProducerName(producerName),
		sched:       sched,
	}
	ctx := context.Background()
	for i, n := range preload {
		endpoint := rig.requestEndpoints([]int64{preloadTokens, preloadTokens, preloadTokens, preloadTokens})[i]
		for j := range n {
			prod.ReserveProfileResult(ctx, fmt.Sprintf("preload-%d-%d", i, j), pickTimeProfile,
				&fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{endpoint}})
		}
	}
	return rig
}

// requestEndpoints returns per-request scheduling endpoints cloned from the
// shared ones, as the director builds them, with the request's uncached-token
// estimate per endpoint (what the Produce phase records).
func (r *pickTimeRig) requestEndpoints(uncached []int64) []fwksched.Endpoint {
	endpoints := make([]fwksched.Endpoint, len(r.shared))
	for i, shared := range r.shared {
		endpoints[i] = fwksched.NewEndpointFromSnapshot(shared.GetMetadata(), shared.GetMetrics(), shared)
		endpoints[i].Put(r.uncachedKey, &concurrency.UncachedRequestTokens{Tokens: uncached[i]})
	}
	return endpoints
}

// schedule schedules one request and returns the index of the picked
// endpoint, or -1 on failure.
func (r *pickTimeRig) schedule(t *testing.T, requestID string, uncached []int64) int {
	result, err := r.sched.Schedule(context.Background(), &fwksched.InferenceRequest{RequestID: requestID}, r.requestEndpoints(uncached))
	if err != nil {
		t.Errorf("schedule %s: %v", requestID, err)
		return -1
	}
	targets := result.ProfileResults[pickTimeProfile].TargetEndpoints
	if len(targets) != 1 {
		t.Errorf("schedule %s: got %d targets, want 1", requestID, len(targets))
		return -1
	}
	index, err := strconv.Atoi(strings.TrimPrefix(targets[0].GetMetadata().ID.Name, "pod"))
	if err != nil {
		t.Errorf("schedule %s: unexpected endpoint %s", requestID, targets[0].GetMetadata().ID.Name)
		return -1
	}
	return index
}

// burst schedules n identical requests concurrently and returns how many
// each endpoint received.
func (r *pickTimeRig) burst(t *testing.T, n int, uncached []int64) []int {
	counts := make([]int, len(r.shared))
	var mu sync.Mutex
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if index := r.schedule(t, fmt.Sprintf("burst-%d", i), uncached); index >= 0 {
				mu.Lock()
				counts[index]++
				mu.Unlock()
			}
		}()
	}
	close(start)
	wg.Wait()
	return counts
}

// sequential schedules n identical requests one at a time.
func (r *pickTimeRig) sequential(t *testing.T, n int, uncached []int64) []int {
	counts := make([]int, len(r.shared))
	for i := range n {
		if index := r.schedule(t, fmt.Sprintf("seq-%d", i), uncached); index >= 0 {
			counts[index]++
		}
	}
	return counts
}

// sortedTokenLoads returns every endpoint's in-flight token load (reserved
// plus dispatched), sorted. Endpoints that tie in a scheduling decision are
// interchangeable, so the sorted loads are the placement's invariant.
func (r *pickTimeRig) sortedTokenLoads(t *testing.T) []int64 {
	t.Helper()
	loads := make([]int64, len(r.shared))
	for i, shared := range r.shared {
		value, ok := shared.Get(r.inFlightKey)
		if !ok {
			t.Fatalf("endpoint %d has no in-flight load", i)
		}
		loads[i] = value.(*concurrency.InFlightLoad).Tokens
	}
	slices.Sort(loads)
	return loads
}

type pickTimeScenario struct {
	name     string
	preload  []int   // requests already in flight per endpoint
	n        int     // burst size
	uncached []int64 // each burst request's uncached tokens per endpoint
}

var pickTimeScenarios = []pickTimeScenario{
	{name: "uniform burst on 2/2/2/1 load", preload: []int{2, 2, 2, 1}, n: 64, uncached: []int64{1500, 1500, 1500, 1500}},
	{name: "one idle endpoint", preload: []int{6, 6, 6, 0}, n: 16, uncached: []int64{1500, 1500, 1500, 1500}},
	{name: "prompt cached on one endpoint", preload: []int{0, 0, 0, 0}, n: 16, uncached: []int64{200, 6000, 6000, 6000}},
}

// TestScoreAtPickPlacesBurstLikeSequential is the acceptance test: with the
// load scorer at pick time, a fully concurrent burst ends with the same
// per-endpoint load as the same requests scheduled one at a time.
func TestScoreAtPickPlacesBurstLikeSequential(t *testing.T) {
	for _, tc := range pickTimeScenarios {
		t.Run(tc.name, func(t *testing.T) {
			sequentialRig := newPickTimeRig(t, true, nil, tc.preload)
			sequentialRig.sequential(t, tc.n, tc.uncached)
			want := sequentialRig.sortedTokenLoads(t)

			burstRig := newPickTimeRig(t, true, newBarrierScorer(tc.n), tc.preload)
			counts := burstRig.burst(t, tc.n, tc.uncached)
			if got := burstRig.sortedTokenLoads(t); !slices.Equal(got, want) {
				t.Errorf("concurrent burst token loads = %v (picks %v), want the sequential placement's %v", got, counts, want)
			}
		})
	}
}

// TestBurstConvergesWithoutScoreAtPick shows the race that pick-time scoring
// closes: when every request reads the load before any of them is picked,
// reserving at pick time is too late and the whole burst lands on the
// endpoint that looked least loaded.
func TestBurstConvergesWithoutScoreAtPick(t *testing.T) {
	const n = 16
	rig := newPickTimeRig(t, false, newBarrierScorer(n), []int{2, 2, 2, 1})
	counts := rig.burst(t, n, []int64{1500, 1500, 1500, 1500})
	if want := []int{0, 0, 0, n}; !slices.Equal(counts, want) {
		t.Errorf("picks without pick-time scoring = %v, want the whole burst on the least-loaded endpoint %v", counts, want)
	}
}

// TestScoreAtPickSequentialMatchesDefault verifies that pick-time scoring
// changes nothing for requests that do not overlap: scheduled one at a time,
// both settings pick the same endpoint for every request.
func TestScoreAtPickSequentialMatchesDefault(t *testing.T) {
	preload := []int{2, 1, 0, 3}
	withDefault := newPickTimeRig(t, false, nil, preload)
	atPick := newPickTimeRig(t, true, nil, preload)
	for i := range 32 {
		// Vary the request size and its cache locality across endpoints.
		size := int64(500 + (i*389)%3000)
		uncached := []int64{size, size / 2, size, size / 4}
		id := fmt.Sprintf("req-%d", i)
		if got, want := atPick.schedule(t, id, uncached), withDefault.schedule(t, id, uncached); got != want {
			t.Fatalf("request %d: pick-time scoring picked pod%d, default picked pod%d", i, got, want)
		}
	}
}

// pickTimeFake is a pick-time scorer that scores every endpoint 1.
type pickTimeFake struct{}

func (pickTimeFake) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "pick-time-fake", Name: "pick-time-fake"}
}
func (pickTimeFake) Category() fwksched.ScorerCategory { return fwksched.Distribution }
func (pickTimeFake) ScoreAtPick() bool                 { return true }
func (pickTimeFake) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = 1
	}
	return scores
}

var _ fwksched.PickTimeScorer = pickTimeFake{}

// staticProfile is a SchedulerProfile implementation other than
// scheduling.SchedulerProfile; it always picks the same endpoint.
type staticProfile struct{ target fwksched.Endpoint }

func (s staticProfile) Run(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	return &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{s.target}}, nil
}

// TestReservesOncePerProfileRun verifies that every successful profile run is
// reserved exactly once, under the profile's name: inside Run for a
// SchedulerProfile, with or without pick-time scorers, and by the scheduler
// for any other SchedulerProfile implementation.
func TestReservesOncePerProfileRun(t *testing.T) {
	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: "pod1"}},
		&fwkdl.Metrics{}, fwkdl.NewAttributes())
	tests := []struct {
		name    string
		profile fwksched.SchedulerProfile
	}{
		{name: "no pick-time scorer", profile: scheduling.NewSchedulerProfile().
			WithPicker(maxscore.NewMaxScorePicker(1))},
		{name: "pick-time scorer", profile: scheduling.NewSchedulerProfile().
			WithScorers(scheduling.NewWeightedScorer(pickTimeFake{}, 1)).
			WithPicker(maxscore.NewMaxScorePicker(1))},
		{name: "other profile implementation", profile: staticProfile{target: endpoint}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reserver := &recordingReserver{}
			sched := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(single.NewSingleProfileHandler(),
				map[string]fwksched.SchedulerProfile{"p1": tc.profile}).
				WithReservers([]fwkrh.EndpointReserver{reserver}))
			if _, err := sched.Schedule(context.Background(), &fwksched.InferenceRequest{RequestID: "r1"},
				[]fwksched.Endpoint{endpoint}); err != nil {
				t.Fatalf("Schedule: %v", err)
			}
			if want := []string{"r1/p1"}; !slices.Equal(reserver.calls, want) {
				t.Errorf("reservations = %v, want %v", reserver.calls, want)
			}
		})
	}
}

// BenchmarkScoreAtPick measures the scheduling throughput of the token-load
// stack with every CPU scheduling concurrently, with and without pick-time
// scoring, to show the cost of serializing the pick-time critical section.
// Each request's reservation is released after it is scheduled, as an
// abandoned request's would be, so the producer's state stays bounded.
//
//	go test -run='^$' -bench=BenchmarkScoreAtPick -benchmem ./pkg/epp/scheduling/
func BenchmarkScoreAtPick(b *testing.B) {
	for _, scoreAtPick := range []bool{false, true} {
		b.Run(fmt.Sprintf("scoreAtPick=%t", scoreAtPick), func(b *testing.B) {

			rig := newPickTimeRig(b, scoreAtPick, nil, []int{2, 2, 2, 1})
			uncached := []int64{1500, 1500, 1500, 1500}
			var next atomic.Int64
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					request := &fwksched.InferenceRequest{RequestID: fmt.Sprintf("bench-%d", next.Add(1))}
					if _, err := rig.sched.Schedule(context.Background(), request, rig.requestEndpoints(uncached)); err != nil {
						b.Errorf("schedule: %v", err)
						return
					}
					rig.prod.RequestAbandoned(context.Background(), request)
				}
			})
		})
	}
}
