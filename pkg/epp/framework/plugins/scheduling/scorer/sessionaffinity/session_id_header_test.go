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

package sessionaffinity

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/test/utils"
)

var _ plugin.Handle = &fakeHandle{}

// fakeHandle is a minimal plugin.Handle for constructing a strategy in tests.
type fakeHandle struct {
	ctx context.Context
}

func newFakeHandle(ctx context.Context) *fakeHandle {
	return &fakeHandle{ctx: ctx}
}

func (h *fakeHandle) Context() context.Context                         { return h.ctx }
func (h *fakeHandle) Plugin(string) plugin.Plugin                      { return nil }
func (h *fakeHandle) AddPlugin(string, plugin.Plugin)                  {}
func (h *fakeHandle) GetAllPlugins() []plugin.Plugin                   { return nil }
func (h *fakeHandle) GetAllPluginsWithNames() map[string]plugin.Plugin { return nil }
func (h *fakeHandle) PodList() []k8stypes.NamespacedName               { return nil }
func (h *fakeHandle) Metrics() plugin.MetricsRecorder                  { return nil }

func newTestSessionIDHeaderStrategy(t *testing.T, overrides func(*parameters)) *sessionIDHeaderStrategy {
	t.Helper()
	params := parameters{
		Strategy:             StrategySessionIDHeader,
		EvictionTTLSeconds:   300,
		EvictionSweepSeconds: 10,
	}
	if overrides != nil {
		overrides(&params)
	}
	built := newStrategy(params, newFakeHandle(utils.NewTestContext(t)))
	s, ok := built.(*sessionIDHeaderStrategy)
	if !ok {
		t.Fatalf("expected *sessionIDHeaderStrategy, got %T", built)
	}
	return s
}

// newTestEndpoint returns an endpoint whose pod key (NamespacedName.String())
// is "/"+name, since these test pods have no namespace set.
func newTestEndpoint(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{},
		nil,
	)
}

func podKey(name string) string {
	return k8stypes.NamespacedName{Name: name}.String()
}

func requestWithSession(sessionID string) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID: "req-" + sessionID,
		Headers:   map[string]string{sessionutil.DefaultHeader: sessionID},
	}
}

func schedulingResultFor(endpoint scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

func TestSessionIDHeaderStrategy_Score_BoundSessionPresent(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	req := requestWithSession("s1")
	s.preRequest(context.Background(), req, schedulingResultFor(endpointB))

	got := s.score(context.Background(), requestWithSession("s1"), endpoints)
	want := map[scheduling.Endpoint]float64{endpointA: 0.0, endpointB: 1.0}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected scores (-want +got):\n%s", diff)
	}
}

func TestSessionIDHeaderStrategy_Score_UnboundPicksLeastLoaded(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	// Bind one session to pod-a so pod-b is strictly less loaded.
	s.preRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	got := s.score(context.Background(), requestWithSession("s2"), endpoints)
	want := map[scheduling.Endpoint]float64{endpointA: 0.0, endpointB: 1.0}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected scores (-want +got):\n%s", diff)
	}
}

func TestSessionIDHeaderStrategy_Score_NoEndpoints(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	got := s.score(context.Background(), requestWithSession("s1"), []scheduling.Endpoint{})
	if len(got) != 0 {
		t.Errorf("expected empty score map, got %v", got)
	}
}

func TestSessionIDHeaderStrategy_PreRequest_FirstBind(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	endpointA := newTestEndpoint("pod-a")

	s.preRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	if got, want := s.boundPod("s1"), podKey("pod-a"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q", got, want)
	}
	if got := s.podSessionCount(podKey("pod-a")); got != 1 {
		t.Errorf("podSessionCount(pod-a) = %d, want 1", got)
	}
}

func TestSessionIDHeaderStrategy_PreRequest_MigratesOnGenuineAbsence(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")

	// Bind s1 to pod-a.
	s.preRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	// Score with only pod-b as a candidate: pod-a is genuinely absent.
	req := requestWithSession("s1")
	s.score(context.Background(), req, []scheduling.Endpoint{endpointB})

	// PreRequest for the same request now sees the picker chose pod-b.
	s.preRequest(context.Background(), req, schedulingResultFor(endpointB))

	if got, want := s.boundPod("s1"), podKey("pod-b"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q after migration", got, want)
	}
	if got := s.podSessionCount(podKey("pod-a")); got != 0 {
		t.Errorf("podSessionCount(pod-a) = %d, want 0 after migration", got)
	}
	if got := s.podSessionCount(podKey("pod-b")); got != 1 {
		t.Errorf("podSessionCount(pod-b) = %d, want 1 after migration", got)
	}
}

func TestSessionIDHeaderStrategy_PreRequest_HoldsPinWhenPresentButOutvoted(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")

	// Bind s1 to pod-a.
	s.preRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	// Score with pod-a present as a candidate (not absent).
	req := requestWithSession("s1")
	s.score(context.Background(), req, []scheduling.Endpoint{endpointA, endpointB})

	// The picker chooses pod-b anyway (another scorer outvoted affinity).
	s.preRequest(context.Background(), req, schedulingResultFor(endpointB))

	if got, want := s.boundPod("s1"), podKey("pod-a"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q (pin should hold)", got, want)
	}
	if got := s.podSessionCount(podKey("pod-a")); got != 1 {
		t.Errorf("podSessionCount(pod-a) = %d, want 1 (unchanged)", got)
	}
}

func TestSessionIDHeaderStrategy_PreRequest_FirstBindRaceDoesNotDoubleCount(t *testing.T) {
	// Two concurrent requests for the same never-before-seen session ID, each
	// picking a DIFFERENT pod. Neither had Score() write a presence record
	// (score() only writes one when a binding already existed), so the request
	// that loses the LoadOrStore sees existing.podName != podName with no
	// record to consult. It must not read that absent record as "bound pod
	// gone" and migrate: the first bind stands, and podCount stays on the pod
	// that won with exactly one session. Repeated so either goroutine can win.
	for i := 0; i < 200; i++ {
		s := newTestSessionIDHeaderStrategy(t, nil)
		endpointA := newTestEndpoint("pod-a")
		endpointB := newTestEndpoint("pod-b")

		req1 := requestWithSession("new-session")
		req2 := &scheduling.InferenceRequest{RequestID: "req-new-session-2", Headers: req1.Headers}

		start := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			<-start
			s.preRequest(context.Background(), req1, schedulingResultFor(endpointA))
		}()
		go func() {
			defer wg.Done()
			<-start
			s.preRequest(context.Background(), req2, schedulingResultFor(endpointB))
		}()
		close(start)
		wg.Wait()

		countA := s.podSessionCount(podKey("pod-a"))
		countB := s.podSessionCount(podKey("pod-b"))
		if countA+countB != 1 {
			t.Fatalf("iteration %d: podCount summed to %d across pod-a/pod-b, want 1 (no double count)", i, countA+countB)
		}
	}

	// Same race with the ordering pinned, so the outcome is checkable: pod-a
	// binds first, then a second request for the same session picks pod-b with
	// no presence record of its own. The pin must hold on pod-a.
	s := newTestSessionIDHeaderStrategy(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")

	req1 := requestWithSession("pinned-session")
	req2 := &scheduling.InferenceRequest{RequestID: "req-pinned-session-2", Headers: req1.Headers}

	s.preRequest(context.Background(), req1, schedulingResultFor(endpointA))
	s.preRequest(context.Background(), req2, schedulingResultFor(endpointB))

	if got, want := s.boundPod("pinned-session"), podKey("pod-a"); got != want {
		t.Errorf("boundPod = %q, want %q: a missing presence record must not force a migration", got, want)
	}
	if got := s.podSessionCount(podKey("pod-a")); got != 1 {
		t.Errorf("podSessionCount(pod-a) = %d, want 1", got)
	}
	if got := s.podSessionCount(podKey("pod-b")); got != 0 {
		t.Errorf("podSessionCount(pod-b) = %d, want 0 (no spurious migration)", got)
	}
}

func TestSessionIDHeaderStrategy_ProfileName_BindsProfilePod(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, func(p *parameters) {
		p.ProfileName = "prefill"
	})
	prefillPod := newTestEndpoint("prefill-pod")

	schedResult := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"prefill": {TargetEndpoints: []scheduling.Endpoint{prefillPod}},
		},
	}

	s.preRequest(context.Background(), requestWithSession("s1"), schedResult)

	if got, want := s.boundPod("s1"), podKey("prefill-pod"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q", got, want)
	}
}

func TestSessionIDHeaderStrategy_ProfileName_DecodeOnlyDoesNotBind(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, func(p *parameters) {
		p.ProfileName = "prefill"
	})

	// Decode-only request: no "prefill" entry in ProfileResults.
	schedResult := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults:     map[string]*scheduling.ProfileRunResult{},
	}

	s.preRequest(context.Background(), requestWithSession("s1"), schedResult)

	if got := s.boundPod("s1"); got != "" {
		t.Errorf("boundPod(s1) = %q, want unbound", got)
	}
}

func TestSessionIDHeaderStrategy_Expired(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, func(p *parameters) {
		p.EvictionTTLSeconds = 60
	})
	now := time.Now()

	if s.expired(now, now) {
		t.Error("a binding just seen should not be expired")
	}
	if !s.expired(now.Add(-2*time.Minute), now) {
		t.Error("a binding unused past the TTL should be expired")
	}
}

func TestSessionIDHeaderStrategy_RunEvictionDropsExpiredBinding(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	s.ttl = 10 * time.Millisecond

	s.bindings.Store("s1", binding{podName: podKey("pod-a"), lastSeen: time.Now().Add(-time.Hour)})
	s.podCount.Store(podKey("pod-a"), 1)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	done := make(chan struct{})
	go func() {
		s.runEviction(ctx, 5*time.Millisecond)
		close(done)
	}()

	// Poll on both the binding removal and the count decrement: the sweep does
	// them as two statements, so observing only the former can race ahead of
	// the latter.
	deadline := time.After(150 * time.Millisecond)
	for s.boundPod("s1") != "" || s.podSessionCount(podKey("pod-a")) != 0 {
		select {
		case <-deadline:
			t.Fatalf("binding was not fully evicted within the deadline (boundPod=%q podCount=%d)",
				s.boundPod("s1"), s.podSessionCount(podKey("pod-a")))
		case <-time.After(5 * time.Millisecond):
		}
	}

	cancel()
	<-done
}

func TestSessionIDHeaderStrategy_RunEvictionSparesRefreshedBinding(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	// TTL comfortably longer than this test's sleeps, so the refreshed binding
	// stays genuinely unexpired for the rest of the run and only the stale
	// snapshot could ever be a candidate for eviction.
	s.ttl = 10 * time.Second

	// A binding old enough that the sweeper will select it for eviction.
	stale := binding{podName: podKey("pod-a"), lastSeen: time.Now().Add(-time.Hour)}
	s.bindings.Store("s1", stale)
	s.podCount.Store(podKey("pod-a"), 1)

	// Hold the lock so the sweeper blocks after it has already snapshotted the
	// stale value in Range but before it acts on it, then refresh the binding
	// underneath it. This is the interleaving the CompareAndDelete guard exists
	// for: on release the sweeper's compare must fail against the new value.
	s.mu.Lock()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() {
		s.runEviction(ctx, time.Millisecond)
		close(done)
	}()

	// Give the sweeper time to reach the blocked critical section.
	time.Sleep(30 * time.Millisecond)

	// The concurrent legitimate refresh, as preRequest's CompareAndSwap path
	// would perform it.
	refreshed := binding{podName: podKey("pod-a"), lastSeen: time.Now()}
	if !s.bindings.CompareAndSwap("s1", stale, refreshed) {
		s.mu.Unlock()
		cancel()
		<-done
		t.Fatal("refresh CompareAndSwap failed: binding was already evicted")
	}

	s.mu.Unlock()

	// The refresh won, so the binding must survive the sweep that was in flight.
	time.Sleep(30 * time.Millisecond)

	cancel()
	<-done

	if got, want := s.boundPod("s1"), podKey("pod-a"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q: a refreshed binding must not be evicted", got, want)
	}
	if got := s.podSessionCount(podKey("pod-a")); got != 1 {
		t.Errorf("podSessionCount(pod-a) = %d, want 1: count must track the surviving binding", got)
	}
}

func TestSessionIDHeaderStrategy_ResponseHeader_IsNoOp(t *testing.T) {
	s := newTestSessionIDHeaderStrategy(t, nil)
	s.responseHeader(context.Background(), nil, nil, nil)
}
