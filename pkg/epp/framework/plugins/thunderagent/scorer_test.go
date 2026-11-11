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

package thunderagent

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

// A bound session's pod gets 1.0, every other candidate 0.0.
func TestScorePinsBoundSession(t *testing.T) {
	l := newTestLedger(testConfig())
	l.runTurn("s1", "pod-a", 400, 300)
	podA, podB := schedEndpoint("pod-a"), schedEndpoint("pod-b")

	scores := l.agent.Score(context.Background(), l.request("s1", 400), []fwksched.Endpoint{podA, podB})
	require.Equal(t, 1.0, scores[podA])
	require.Equal(t, 0.0, scores[podB])
}

// When RequestHeader has not published SessionState on the request, Score
// falls back to EndpointSessions attached to Saturation's live endpoints.
func TestScoreFallsBackToBoundEndpointSessions(t *testing.T) {
	l := newTestLedger(testConfig())
	epA := l.pod("pod-a", 0, 0)
	l.runTurn("s1", "pod-a", 400, 300)
	l.prime(epA)

	req := &fwksched.InferenceRequest{RequestID: "req-1", RequestSizeBytes: 400}
	req.PutAttribute(agentidentity.AgentIdentityKey, "s1")

	podA, podB := schedEndpoint("pod-a"), schedEndpoint("pod-b")
	scores := l.agent.Score(context.Background(), req, []fwksched.Endpoint{podA, podB})
	require.Equal(t, 1.0, scores[podA])
	require.Equal(t, 0.0, scores[podB])
}

// A Pick reservation pins a new session before PreRequest binds it, and expires
// once PreRequest advances TurnsTaken or the reservation TTL elapses.
func TestScoreHonorsAndExpiresReservation(t *testing.T) {
	l := newTestLedger(testConfig())
	epA := l.pod("pod-a", 0, 0)
	epB := l.pod("pod-b", 0, 0)
	l.seed("existing", "pod-a", 800)
	l.prime(epA, epB)

	q := l.makeQueue("s-new", l.now(), 400)
	require.Equal(t, q, pick(t, l.agent, q))

	podA, podB := schedEndpoint("pod-a"), schedEndpoint("pod-b")
	scores := l.agent.Score(context.Background(), l.request("s-new", 400), []fwksched.Endpoint{podA, podB})
	require.Equal(t, 0.0, scores[podA])
	require.Equal(t, 1.0, scores[podB])

	// Once PreRequest dispatches onto pod-a (TurnsTaken=1 > reservedAtTurns=0),
	// Score follows the bound Endpoint instead of the stale reservation.
	l.runTurn("s-new", "pod-a", 400, 200)
	scores = l.agent.Score(context.Background(), l.request("s-new", 400), []fwksched.Endpoint{podA, podB})
	require.Equal(t, 1.0, scores[podA])
	require.Equal(t, 0.0, scores[podB])

	// An unconfirmed reservation expires after reservationTTL.
	l.removePod("pod-a")
	l.prime(epB)
	q2 := l.makeQueue("s-other", l.now(), 400)
	require.Equal(t, q2, pick(t, l.agent, q2))
	l.step(reservationTTL + time.Second)
	require.Nil(t, l.agent.Score(context.Background(), l.request("s-other", 400), []fwksched.Endpoint{podA, podB}))
}

// Unknown sessions and anonymous requests get no scores.
func TestScoreAbstains(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := schedEndpoint("pod-a")

	require.Nil(t, l.agent.Score(context.Background(), l.request("unknown", 400), []fwksched.Endpoint{podA}))
	require.Nil(t, l.agent.Score(context.Background(), l.request("", 400), []fwksched.Endpoint{podA}))
}

// When the bound pod is not among the candidates (it left the pool or a
// filter removed it), the scorer abstains rather than zeroing everything.
func TestScoreAbstainsWhenBoundPodMissing(t *testing.T) {
	l := newTestLedger(testConfig())
	podB := schedEndpoint("pod-b")
	l.runTurn("s1", "pod-a", 400, 300)

	require.Nil(t, l.agent.Score(context.Background(), l.request("s1", 400), []fwksched.Endpoint{podB}))
}

func TestScorerCategory(t *testing.T) {
	require.Equal(t, fwksched.Affinity, newTestLedger(testConfig()).agent.Category())
}
