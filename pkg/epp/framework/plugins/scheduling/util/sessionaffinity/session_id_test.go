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

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

// testPluginKey is the owning-plugin identity a test SessionIDHeader is built
// with; any stable string works since it only keys per-instance state.
const testPluginKey = plugin.StateKey("session-affinity-test")

// testOptions overrides the defaults a test SessionIDHeader is built with.
type testOptions struct {
	profileName        string
	evictionTTLSeconds float64
}

func newTestSessionIDHeader(t *testing.T, overrides func(*testOptions)) *SessionIDHeader {
	t.Helper()
	opts := testOptions{evictionTTLSeconds: 300}
	if overrides != nil {
		overrides(&opts)
	}
	config := SessionIDConfig{
		Sources:              []SessionIDSource{{Header: DefaultHeader}},
		EvictionTTLSeconds:   opts.evictionTTLSeconds,
		EvictionSweepSeconds: 10,
	}
	return NewSessionIDHeader(
		config, opts.profileName, testPluginKey,
		utils.NewTestHandle(utils.NewTestContext(t)),
	)
}

// newTestEndpoint returns an endpoint whose pod key (NamespacedName.String())
// is "/"+name, since these test pods have no namespace set.
func newTestEndpoint(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}},
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
		Headers:   map[string]string{DefaultHeader: sessionID},
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

func TestSessionIDHeader_Choose_BoundSessionPresent(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	req := requestWithSession("s1")
	s.PreRequest(context.Background(), req, schedulingResultFor(endpointB))

	got, ok := s.Choose(context.Background(), requestWithSession("s1"), endpoints)
	if !ok || got != endpointB {
		t.Errorf("Choose = (%v, %v), want (pod-b, true)", got.GetMetadata().ID.Name, ok)
	}
}

func TestSessionIDHeader_Choose_UnboundAbstains(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	// A prior session is bound: an unbound session still expresses no
	// preference. The plugin does not place it; the consumer decides.
	s.PreRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	if got, ok := s.Choose(context.Background(), requestWithSession("s2"), endpoints); ok || got != nil {
		t.Errorf("Choose for unbound session = (%v, %v), want (nil, false)", got, ok)
	}
}

func TestSessionIDHeader_Choose_NoEndpoints(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	if got, ok := s.Choose(context.Background(), requestWithSession("s1"), []scheduling.Endpoint{}); ok || got != nil {
		t.Errorf("Choose with no endpoints = (%v, %v), want (nil, false)", got, ok)
	}
}

func TestSessionIDHeader_Choose_NoSessionAbstains(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	// A prior session is bound; the no-session request must still get no
	// preference. No session means abstain.
	s.PreRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	noSession := &scheduling.InferenceRequest{RequestID: "req-no-session"}
	if got, ok := s.Choose(context.Background(), noSession, endpoints); ok || got != nil {
		t.Errorf("Choose with no session = (%v, %v), want (nil, false)", got, ok)
	}
}

func TestSessionIDHeader_PreRequest_FirstBind(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	endpointA := newTestEndpoint("pod-a")

	s.PreRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	if got, want := s.boundPod("s1"), podKey("pod-a"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q", got, want)
	}
}

func TestSessionIDHeader_PreRequest_MigratesOnGenuineAbsence(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")

	// Bind s1 to pod-a.
	s.PreRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	// Choose with only pod-b as a candidate records pod-a as genuinely absent.
	req := requestWithSession("s1")
	s.Choose(context.Background(), req, []scheduling.Endpoint{endpointB})

	// PreRequest for the same request now sees the picker chose pod-b.
	s.PreRequest(context.Background(), req, schedulingResultFor(endpointB))

	if got, want := s.boundPod("s1"), podKey("pod-b"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q after migration", got, want)
	}
}

func TestSessionIDHeader_PreRequest_HoldsPinWhenPresentButOutvoted(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")

	// Bind s1 to pod-a.
	s.PreRequest(context.Background(), requestWithSession("s1"), schedulingResultFor(endpointA))

	// Choose with pod-a present as a candidate (not absent).
	req := requestWithSession("s1")
	s.Choose(context.Background(), req, []scheduling.Endpoint{endpointA, endpointB})

	// The picker chooses pod-b anyway (another scorer outvoted affinity).
	s.PreRequest(context.Background(), req, schedulingResultFor(endpointB))

	if got, want := s.boundPod("s1"), podKey("pod-a"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q (pin should hold)", got, want)
	}
}

func TestSessionIDHeader_PreRequest_FirstBindRaceKeepsSingleBinding(t *testing.T) {
	// Two concurrent requests for the same never-before-seen session ID, each
	// picking a DIFFERENT pod. Neither had Choose() write a presence record
	// (Choose only writes one when a binding already existed), so the request
	// that loses the LoadOrStore sees existing.podName != podName with no
	// record to consult. It must not read that absent record as "bound pod
	// gone" and migrate: the first bind stands. Repeated so either goroutine
	// can win.
	for i := 0; i < 200; i++ {
		s := newTestSessionIDHeader(t, nil)
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
			s.PreRequest(context.Background(), req1, schedulingResultFor(endpointA))
		}()
		go func() {
			defer wg.Done()
			<-start
			s.PreRequest(context.Background(), req2, schedulingResultFor(endpointB))
		}()
		close(start)
		wg.Wait()

		got := s.boundPod("new-session")
		if got != podKey("pod-a") && got != podKey("pod-b") {
			t.Fatalf("iteration %d: boundPod = %q, want the pod of whichever request won the first bind", i, got)
		}
	}

	// Same race with the ordering pinned, so the outcome is checkable: pod-a
	// binds first, then a second request for the same session picks pod-b with
	// no presence record of its own. The pin must hold on pod-a.
	s := newTestSessionIDHeader(t, nil)
	endpointA := newTestEndpoint("pod-a")
	endpointB := newTestEndpoint("pod-b")

	req1 := requestWithSession("pinned-session")
	req2 := &scheduling.InferenceRequest{RequestID: "req-pinned-session-2", Headers: req1.Headers}

	s.PreRequest(context.Background(), req1, schedulingResultFor(endpointA))
	s.PreRequest(context.Background(), req2, schedulingResultFor(endpointB))

	if got, want := s.boundPod("pinned-session"), podKey("pod-a"); got != want {
		t.Errorf("boundPod = %q, want %q: a missing presence record must not force a migration", got, want)
	}
}

func TestSessionIDHeader_ProfileName_BindsProfilePod(t *testing.T) {
	s := newTestSessionIDHeader(t, func(o *testOptions) {
		o.profileName = "prefill"
	})
	prefillPod := newTestEndpoint("prefill-pod")

	schedResult := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"prefill": {TargetEndpoints: []scheduling.Endpoint{prefillPod}},
		},
	}

	s.PreRequest(context.Background(), requestWithSession("s1"), schedResult)

	if got, want := s.boundPod("s1"), podKey("prefill-pod"); got != want {
		t.Errorf("boundPod(s1) = %q, want %q", got, want)
	}
}

func TestSessionIDHeader_ProfileName_DecodeOnlyDoesNotBind(t *testing.T) {
	s := newTestSessionIDHeader(t, func(o *testOptions) {
		o.profileName = "prefill"
	})

	// Decode-only request: no "prefill" entry in ProfileResults.
	schedResult := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults:     map[string]*scheduling.ProfileRunResult{},
	}

	s.PreRequest(context.Background(), requestWithSession("s1"), schedResult)

	if got := s.boundPod("s1"); got != "" {
		t.Errorf("boundPod(s1) = %q, want unbound", got)
	}
}

func TestSessionIDHeader_Expired(t *testing.T) {
	s := newTestSessionIDHeader(t, func(o *testOptions) {
		o.evictionTTLSeconds = 60
	})
	now := time.Now()

	if s.expired(now, now) {
		t.Error("a binding just seen should not be expired")
	}
	if !s.expired(now.Add(-2*time.Minute), now) {
		t.Error("a binding unused past the TTL should be expired")
	}
}

func TestSessionIDHeader_RunEvictionDropsExpiredBinding(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	s.ttl = 10 * time.Millisecond

	s.bindings.Store("s1", binding{podName: podKey("pod-a"), lastSeen: time.Now().Add(-time.Hour)})

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	done := make(chan struct{})
	go func() {
		s.runEviction(ctx, 5*time.Millisecond)
		close(done)
	}()

	deadline := time.After(150 * time.Millisecond)
	for s.boundPod("s1") != "" {
		select {
		case <-deadline:
			t.Fatalf("binding was not evicted within the deadline (boundPod=%q)", s.boundPod("s1"))
		case <-time.After(5 * time.Millisecond):
		}
	}

	cancel()
	<-done
}

func TestSessionIDHeader_RunEvictionSparesRefreshedBinding(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	// TTL comfortably longer than this test's sleeps, so the refreshed binding
	// stays genuinely unexpired for the rest of the run and only the stale
	// snapshot could ever be a candidate for eviction.
	s.ttl = 10 * time.Second

	// A binding old enough that the sweeper will select it for eviction.
	stale := binding{podName: podKey("pod-a"), lastSeen: time.Now().Add(-time.Hour)}
	s.bindings.Store("s1", stale)

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

	// The concurrent legitimate refresh, as PreRequest's CompareAndSwap path
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
}

func TestSessionIDHeader_ResponseHeader_IsNoOp(t *testing.T) {
	s := newTestSessionIDHeader(t, nil)
	s.ResponseHeader(context.Background(), nil, nil, nil)
}

// headerWithSources builds a SessionIDHeader configured with the given sources,
// for exercising resolveSessionID in isolation.
func headerWithSources(t *testing.T, sources ...SessionIDSource) *SessionIDHeader {
	t.Helper()
	return NewSessionIDHeader(
		SessionIDConfig{Sources: sources, EvictionTTLSeconds: 300, EvictionSweepSeconds: 10},
		"", testPluginKey, utils.NewTestHandle(utils.NewTestContext(t)),
	)
}

// namedSessionID mimics a producer publishing its identifier under a named
// string type (e.g. session.SessionID) rather than a plain string.
type namedSessionID string

func TestResolveSessionID(t *testing.T) {
	const attr = "agent-identity"
	attrKey := plugin.NewDataKey(attr, "")

	withHeader := func(name, value string) *scheduling.InferenceRequest {
		return &scheduling.InferenceRequest{Headers: map[string]string{name: value}}
	}
	withAttr := func(value string) *scheduling.InferenceRequest {
		req := &scheduling.InferenceRequest{Headers: map[string]string{}}
		req.PutAttribute(attrKey, value)
		return req
	}
	// withAttrValue stores an arbitrary value under the attribute key, to cover
	// producers that publish a named string type or a non-string value.
	withAttrValue := func(value any) *scheduling.InferenceRequest {
		req := &scheduling.InferenceRequest{Headers: map[string]string{}}
		req.PutAttribute(attrKey, value)
		return req
	}

	tests := []struct {
		name    string
		sources []SessionIDSource
		request *scheduling.InferenceRequest
		want    string
	}{
		{
			name:    "header source resolves",
			sources: []SessionIDSource{{Header: "x-session-id"}},
			request: withHeader("x-session-id", "sess-1"),
			want:    "sess-1",
		},
		{
			name:    "attribute source resolves",
			sources: []SessionIDSource{{Attribute: attr}},
			request: withAttr("agent-42"),
			want:    "agent-42",
		},
		{
			name:    "attribute stored as a named string type resolves",
			sources: []SessionIDSource{{Attribute: attr}},
			request: withAttrValue(namedSessionID("agent-99")),
			want:    "agent-99",
		},
		{
			name:    "attribute stored as a non-string value is skipped",
			sources: []SessionIDSource{{Attribute: attr}},
			request: withAttrValue(42),
			want:    "",
		},
		{
			name:    "header wins over attribute in priority order",
			sources: []SessionIDSource{{Header: "x-session-id"}, {Attribute: attr}},
			request: func() *scheduling.InferenceRequest {
				req := withHeader("x-session-id", "from-header")
				req.PutAttribute(attrKey, "from-attr")
				return req
			}(),
			want: "from-header",
		},
		{
			name:    "falls through to attribute when header empty",
			sources: []SessionIDSource{{Header: "x-session-id"}, {Attribute: attr}},
			request: withAttr("from-attr"),
			want:    "from-attr",
		},
		{
			name:    "none match yields empty",
			sources: []SessionIDSource{{Header: "x-session-id"}, {Attribute: attr}},
			request: &scheduling.InferenceRequest{Headers: map[string]string{}},
			want:    "",
		},
		{
			name:    "whitespace-only value is treated as empty",
			sources: []SessionIDSource{{Header: "x-session-id"}},
			request: withHeader("x-session-id", "   "),
			want:    "",
		},
		{
			name:    "nil request yields empty",
			sources: []SessionIDSource{{Header: "x-session-id"}},
			request: nil,
			want:    "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := headerWithSources(t, test.sources...)
			if got := s.resolveSessionID(context.Background(), test.request); got != test.want {
				t.Errorf("resolveSessionID = %q, want %q", got, test.want)
			}
		})
	}
}

func TestSessionIDConfig_ApplyDefaults(t *testing.T) {
	t.Run("empty config gets default source and eviction schedule", func(t *testing.T) {
		c := SessionIDConfig{}
		c.ApplyDefaults()
		if len(c.Sources) != 1 || c.Sources[0].Header != DefaultSessionIDHeader {
			t.Errorf("Sources = %+v, want single %q header source", c.Sources, DefaultSessionIDHeader)
		}
		if c.EvictionTTLSeconds != defaultEvictionTTLSeconds || c.EvictionSweepSeconds != defaultEvictionSweepSeconds {
			t.Errorf("eviction = (%v, %v), want (%v, %v)", c.EvictionTTLSeconds, c.EvictionSweepSeconds, defaultEvictionTTLSeconds, defaultEvictionSweepSeconds)
		}
	})

	t.Run("configured sources are left untouched", func(t *testing.T) {
		c := SessionIDConfig{Sources: []SessionIDSource{{Attribute: "agent-identity"}}}
		c.ApplyDefaults()
		if len(c.Sources) != 1 || c.Sources[0].Attribute != "agent-identity" {
			t.Errorf("Sources = %+v, want the configured attribute source unchanged", c.Sources)
		}
	})
}

func TestSessionIDConfig_Validate(t *testing.T) {
	tests := []struct {
		name      string
		config    SessionIDConfig
		expectErr bool
	}{
		{
			name: "single header source is valid",
			config: SessionIDConfig{
				Sources:              []SessionIDSource{{Header: DefaultSessionIDHeader}},
				EvictionTTLSeconds:   300,
				EvictionSweepSeconds: 10,
			},
		},
		{
			name: "source with both header and attribute rejected",
			config: SessionIDConfig{
				Sources:              []SessionIDSource{{Header: "h", Attribute: "a"}},
				EvictionTTLSeconds:   300,
				EvictionSweepSeconds: 10,
			},
			expectErr: true,
		},
		{
			name: "source with neither header nor attribute rejected",
			config: SessionIDConfig{
				Sources:              []SessionIDSource{{}},
				EvictionTTLSeconds:   300,
				EvictionSweepSeconds: 10,
			},
			expectErr: true,
		},
		{
			name: "zero ttl rejected",
			config: SessionIDConfig{
				Sources:              []SessionIDSource{{Header: "x-session-id"}},
				EvictionTTLSeconds:   0,
				EvictionSweepSeconds: 10,
			},
			expectErr: true,
		},
		{
			name: "negative sweep rejected",
			config: SessionIDConfig{
				Sources:              []SessionIDSource{{Header: "x-session-id"}},
				EvictionTTLSeconds:   300,
				EvictionSweepSeconds: -1,
			},
			expectErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.config.Validate()
			if test.expectErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
