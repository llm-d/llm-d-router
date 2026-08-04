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
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/test/utils"
)

func newTestFilterStrategy(t *testing.T) *sessionIDHeaderStrategy {
	t.Helper()
	built := newSessionIDHeaderStrategy(parameters{
		Strategy: StrategySessionID,
		SessionIDConfig: sessionutil.SessionIDConfig{
			Sources:              []sessionutil.SessionIDSource{{Header: sessionutil.DefaultHeader}},
			EvictionTTLSeconds:   300,
			EvictionSweepSeconds: 10,
		},
	}, utils.NewTestHandle(utils.NewTestContext(t)))
	return built.(*sessionIDHeaderStrategy)
}

func filterTestEndpoint(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{},
		nil,
	)
}

func filterRequest(sessionID string) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID: "req-" + sessionID,
		Headers:   map[string]string{sessionutil.DefaultHeader: sessionID},
	}
}

func filterResultFor(endpoint scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

func filterPodNames(endpoints []scheduling.Endpoint) []string {
	names := make([]string, len(endpoints))
	for i, e := range endpoints {
		names[i] = e.GetMetadata().ID.Name
	}
	return names
}

// TestSessionIDHeaderFilter_BoundPresentReturnsAlone checks the pinned pod is
// returned as the sole candidate when present.
func TestSessionIDHeaderFilter_BoundPresentReturnsAlone(t *testing.T) {
	s := newTestFilterStrategy(t)
	endpointA := filterTestEndpoint("pod-a")
	endpointB := filterTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	s.preRequest(context.Background(), filterRequest("s1"), filterResultFor(endpointB))

	got := s.filter(context.Background(), filterRequest("s1"), endpoints)
	assert.Equal(t, []string{"pod-b"}, filterPodNames(got))
}

// TestSessionIDHeaderFilter_NoSessionReturnsAll checks a request with no
// session identifier passes every candidate through.
func TestSessionIDHeaderFilter_NoSessionReturnsAll(t *testing.T) {
	s := newTestFilterStrategy(t)
	endpointA := filterTestEndpoint("pod-a")
	endpointB := filterTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	noSession := &scheduling.InferenceRequest{RequestID: "req-no-session"}
	got := s.filter(context.Background(), noSession, endpoints)
	assert.ElementsMatch(t, []string{"pod-a", "pod-b"}, filterPodNames(got))
}

// TestSessionIDHeaderFilter_UnboundReturnsAll checks an unbound session is a
// pass-through: the filter expresses no preference and returns all candidates.
func TestSessionIDHeaderFilter_UnboundReturnsAll(t *testing.T) {
	s := newTestFilterStrategy(t)
	endpointA := filterTestEndpoint("pod-a")
	endpointB := filterTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	// A prior session is bound; the unbound session must still see all pods.
	s.preRequest(context.Background(), filterRequest("s1"), filterResultFor(endpointA))

	got := s.filter(context.Background(), filterRequest("s2"), endpoints)
	assert.ElementsMatch(t, []string{"pod-a", "pod-b"}, filterPodNames(got))
}

// TestSessionIDHeaderFilter_MigratesOnAbsence checks a session whose bound pod
// dropped out of the candidate set rebinds to the present pod on PreRequest.
func TestSessionIDHeaderFilter_MigratesOnAbsence(t *testing.T) {
	s := newTestFilterStrategy(t)
	endpointA := filterTestEndpoint("pod-a")
	endpointB := filterTestEndpoint("pod-b")

	s.preRequest(context.Background(), filterRequest("s1"), filterResultFor(endpointA))

	// Only pod-b is a candidate: pod-a is genuinely absent.
	req := filterRequest("s1")
	s.filter(context.Background(), req, []scheduling.Endpoint{endpointB})
	s.preRequest(context.Background(), req, filterResultFor(endpointB))

	// After migration the session is pinned to pod-b: filtering against both
	// candidates now returns pod-b alone.
	got := s.filter(context.Background(), filterRequest("s1"), []scheduling.Endpoint{endpointA, endpointB})
	assert.Equal(t, []string{"pod-b"}, filterPodNames(got))
}

// TestSessionIDHeaderFilter_NeverNarrowsToEmpty guards the one invariant whose
// violation is a hard scheduling failure: the filter must never turn a
// non-empty candidate set into an empty one, in any session state. An empty
// input is passed through unchanged (that is the scheduler's condition, not the
// filter's to invent an endpoint for).
func TestSessionIDHeaderFilter_NeverNarrowsToEmpty(t *testing.T) {
	s := newTestFilterStrategy(t)
	endpointA := filterTestEndpoint("pod-a")
	endpointB := filterTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	// s1 is bound to a pod absent from the candidate set below; s3 is bound to a
	// present pod. Together with unbound and no-session, this covers every state.
	s.preRequest(context.Background(), filterRequest("s1"), filterResultFor(filterTestEndpoint("pod-gone")))
	s.preRequest(context.Background(), filterRequest("s3"), filterResultFor(endpointA))

	nonEmpty := map[string]*scheduling.InferenceRequest{
		"no session":    {RequestID: "req-none"},
		"unbound":       filterRequest("s2"),
		"bound absent":  filterRequest("s1"),
		"bound present": filterRequest("s3"),
	}
	for name, req := range nonEmpty {
		t.Run(name, func(t *testing.T) {
			if got := s.filter(context.Background(), req, endpoints); len(got) == 0 {
				t.Errorf("filter narrowed a non-empty set to empty for %q; must never happen", name)
			}
		})
	}

	t.Run("empty input passes through", func(t *testing.T) {
		if got := s.filter(context.Background(), filterRequest("s3"), []scheduling.Endpoint{}); len(got) != 0 {
			t.Errorf("filter with empty input = %d endpoints, want 0", len(got))
		}
	})
}
