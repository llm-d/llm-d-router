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

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/test/utils"
)

func newTestScorerStrategy(t *testing.T) *sessionIDHeaderStrategy {
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

func scorerTestEndpoint(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{},
		nil,
	)
}

func scorerRequest(sessionID string) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID: "req-" + sessionID,
		Headers:   map[string]string{sessionutil.DefaultHeader: sessionID},
	}
}

func scorerResultFor(endpoint scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

// TestSessionIDHeaderScorer_ScoresBoundPod checks the glue over the embed: the
// pinned pod scores 1.0, every other candidate 0.0.
func TestSessionIDHeaderScorer_ScoresBoundPod(t *testing.T) {
	s := newTestScorerStrategy(t)
	endpointA := scorerTestEndpoint("pod-a")
	endpointB := scorerTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	s.preRequest(context.Background(), scorerRequest("s1"), scorerResultFor(endpointB))

	got := s.score(context.Background(), scorerRequest("s1"), endpoints)
	want := map[scheduling.Endpoint]float64{endpointA: 0.0, endpointB: 1.0}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected scores (-want +got):\n%s", diff)
	}
}

// TestSessionIDHeaderScorer_NoSessionAbstains checks the scorer returns all
// zeros when the request carries no session identifier.
func TestSessionIDHeaderScorer_NoSessionAbstains(t *testing.T) {
	s := newTestScorerStrategy(t)
	endpointA := scorerTestEndpoint("pod-a")
	endpointB := scorerTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	// A prior session is bound; the no-session request must still not pick it.
	s.preRequest(context.Background(), scorerRequest("s1"), scorerResultFor(endpointA))

	noSession := &scheduling.InferenceRequest{RequestID: "req-no-session"}
	got := s.score(context.Background(), noSession, endpoints)
	want := map[scheduling.Endpoint]float64{endpointA: 0.0, endpointB: 0.0}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected scores (-want +got):\n%s", diff)
	}
}

// TestSessionIDHeaderScorer_UnboundAbstains checks the scorer returns all zeros
// for a session that carries an identifier but is not yet bound: the scorer
// does not place it, the picker decides.
func TestSessionIDHeaderScorer_UnboundAbstains(t *testing.T) {
	s := newTestScorerStrategy(t)
	endpointA := scorerTestEndpoint("pod-a")
	endpointB := scorerTestEndpoint("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	// A prior session is bound; the unbound session must still score all zeros.
	s.preRequest(context.Background(), scorerRequest("s1"), scorerResultFor(endpointA))

	got := s.score(context.Background(), scorerRequest("s2"), endpoints)
	want := map[scheduling.Endpoint]float64{endpointA: 0.0, endpointB: 0.0}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected scores (-want +got):\n%s", diff)
	}
}
