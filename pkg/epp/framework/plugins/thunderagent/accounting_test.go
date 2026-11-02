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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// A turn charges its byte estimate while in flight and settles to the
// usage-reported total when it completes.
func TestTurnAccounting(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	req := startTurn(t, a, "s1", ep, 400) // 400 bytes / 4 bytes-per-token = 100 tokens
	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Equal(t, int64(100), s.inflightTokens())
	require.Equal(t, float64(100), endpointTokens(a, "default/pod-a"))

	a.ResponseBody(context.Background(), req, endOfStream(250, 100), nil)
	s, _ = sessionOf(a, "s1")
	require.Equal(t, int64(0), s.inflightTokens())
	require.Equal(t, int64(250), s.committedTokens())
	require.Equal(t, float64(250), endpointTokens(a, "default/pod-a"))
}

// Within one lane the footprint is the max of the committed total and the turn
// in flight, not their sum, because the new prefill covers the same history.
// Across lanes the agents hold separate histories, so the lanes add up.
func TestFootprintIsMaxPerLane(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	req := startTurn(t, a, "s1", ep, 400)
	a.ResponseBody(context.Background(), req, endOfStream(300, 100), nil)

	_ = startTurn(t, a, "s1", ep, 800) // in-flight estimate 200 < committed 300
	require.Equal(t, float64(300), endpointTokens(a, "default/pod-a"))

	// An overlapping turn is a second agent, so it opens a lane of its own:
	// max(200, 300) for the first lane plus 500 for the second.
	_ = startTurn(t, a, "s1", ep, 2000)
	require.Equal(t, float64(800), endpointTokens(a, "default/pod-a"))
}

// Each overlapping turn removes only its own estimate from its own lane: when a
// small turn ends first, the larger turn still in flight keeps counting and the
// small turn's usage total stays in its own lane.
func TestOverlappingTurnsSettleIndependently(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	runTurn(t, a, "s1", ep, 400, 300)

	big := startTurn(t, a, "s1", ep, 2000)  // 500 in flight
	small := startTurn(t, a, "s1", ep, 400) // 100 more
	require.Equal(t, float64(600), endpointTokens(a, "default/pod-a"))

	a.ResponseBody(context.Background(), small, endOfStream(150, 100), nil)
	s, _ := sessionOf(a, "s1")
	require.Equal(t, int64(500), s.inflightTokens())
	require.Equal(t, float64(650), endpointTokens(a, "default/pod-a"),
		"the big turn still counts, and the small agent's 150 settled beside it")

	a.ResponseBody(context.Background(), big, endOfStream(520, 500), nil)
	s, _ = sessionOf(a, "s1")
	require.Equal(t, int64(0), s.inflightTokens())
	require.Equal(t, float64(520), endpointTokens(a, "default/pod-a"),
		"nothing is in flight, so the lanes fold to the larger history")
}

// A small turn settling after a large one must not shrink the session to its
// own usage while the large agent's history is still resident. The lanes fold
// to the larger total once nothing is in flight, so the footprint holds the
// big agent rather than collapsing to the last turn to finish.
func TestReversedCompletionOrderKeepsFootprint(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	runTurn(t, a, "s1", ep, 400, 300)

	big := startTurn(t, a, "s1", ep, 2000)  // 500 in flight
	small := startTurn(t, a, "s1", ep, 400) // 100 more
	require.Equal(t, float64(600), endpointTokens(a, "default/pod-a"))

	a.ResponseBody(context.Background(), big, endOfStream(520, 500), nil)
	require.Equal(t, float64(620), endpointTokens(a, "default/pod-a"),
		"the big turn settled at 520 and the small one still has 100 in flight")

	a.ResponseBody(context.Background(), small, endOfStream(150, 100), nil)
	require.Equal(t, float64(520), endpointTokens(a, "default/pod-a"),
		"the big agent's history survives the small turn settling after it")
}

// One agent's turns reuse its lane, so a compacted history still shrinks the
// footprint. Keeping a high-water mark per session would hold 900 forever.
func TestSequentialTurnsStillShrink(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	runTurn(t, a, "s1", ep, 3600, 900)
	require.Equal(t, float64(900), endpointTokens(a, "default/pod-a"))

	runTurn(t, a, "s1", ep, 1200, 300)
	require.Equal(t, float64(300), endpointTokens(a, "default/pod-a"),
		"the agent compacted its history, so its lane follows the usage total down")

	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Len(t, s.lanes, 1, "sequential turns never open a second lane")
}

// Lane count follows the turns in flight right now, not the session's history:
// a session that once ran two agents does not carry two lanes forever.
func TestLaneCountFollowsPeakParallelism(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	first := startTurn(t, a, "s1", ep, 2000)
	second := startTurn(t, a, "s1", ep, 400)
	s, _ := sessionOf(a, "s1")
	require.Len(t, s.lanes, 2, "two turns at once are two agents")

	a.ResponseBody(context.Background(), first, endOfStream(520, 500), nil)
	a.ResponseBody(context.Background(), second, endOfStream(150, 100), nil)

	s, _ = sessionOf(a, "s1")
	require.Len(t, s.lanes, 1, "with nothing in flight the lanes fold together")

	for range 5 {
		runTurn(t, a, "s1", ep, 400, 150)
	}
	s, _ = sessionOf(a, "s1")
	require.Len(t, s.lanes, 1, "sequential turns never reopen a lane")
	require.Equal(t, float64(150), endpointTokens(a, "default/pod-a"),
		"the folded lane follows its latest turn down")
}

// A response for a request PreRequest never charged leaves the session alone.
func TestResponseWithoutEstimateIgnored(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	runTurn(t, a, "s1", ep, 400, 300)

	a.ResponseBody(context.Background(), newRequest("s1", 400), endOfStream(900, 900), nil)
	s, _ := sessionOf(a, "s1")
	require.Equal(t, int64(300), s.committedTokens())
}

// A turn charges at least one token, so a session with an empty request in
// flight is not taken for idle.
func TestEmptyRequestStillInFlight(t *testing.T) {
	a := newTestAgent(testConfig())
	_ = startTurn(t, a, "s1", schedEndpoint("pod-a", 0, 0), 0)
	s, _ := sessionOf(a, "s1")
	require.Equal(t, int64(1), s.inflightTokens())
}

// Idle sessions past the TTL are dropped by the sweep, and a pod that then
// holds nothing is dropped too. Fresh sessions and pods stay.
func TestTTLEviction(t *testing.T) {
	cfg := testConfig()
	a := newTestAgent(cfg)
	ep := schedEndpoint("pod-a", 0, 0)
	runTurn(t, a, "s1", ep, 400, 300)

	a.mgr.mu.Lock()
	a.mgr.sessions["s1"].lastActivity = time.Now().Add(-2 * a.mgr.ttl)
	a.mgr.endpoints["default/pod-a"].updatedAt = time.Now().Add(-2 * endpointStaleAfter)
	a.mgr.mu.Unlock()
	runTurn(t, a, "s2", schedEndpoint("pod-b", 0, 0), 4, 1)

	a.mgr.sweep(time.Now())
	_, ok := sessionOf(a, "s1")
	require.False(t, ok)
	require.Equal(t, float64(-1), endpointTokens(a, "default/pod-a"))
	_, ok = sessionOf(a, "s2")
	require.True(t, ok)
	require.Equal(t, float64(1), endpointTokens(a, "default/pod-b"))
}

// A session with a turn in flight is never TTL-evicted, so the completion
// bookkeeping always finds its state.
func TestTTLKeepsInflightSessions(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	_ = startTurn(t, a, "s1", ep, 400)

	a.mgr.mu.Lock()
	a.mgr.sessions["s1"].lastActivity = time.Now().Add(-2 * a.mgr.ttl)
	a.mgr.mu.Unlock()

	a.mgr.sweep(time.Now())
	_, ok := sessionOf(a, "s1")
	require.True(t, ok)
}

// Real scraped capacity wins over the configured fallback.
func TestCapacity(t *testing.T) {
	a := newTestAgent(testConfig())
	require.Equal(t, float64(16000), a.endpointCapacity(schedEndpoint("pod-a", 16, 1000).GetMetrics()))
	require.Equal(t, float64(1000), a.endpointCapacity(schedEndpoint("pod-a", 0, 0).GetMetrics()))
}

// Anonymous requests (no session identity) are not tracked.
func TestAnonymousIgnored(t *testing.T) {
	a := newTestAgent(testConfig())
	anon := newRequest("", 400)
	require.NoError(t, a.PreRequest(context.Background(), anon, schedulingResultFor(schedEndpoint("pod-a", 0, 0))))
	a.mgr.mu.Lock()
	n := len(a.mgr.sessions)
	a.mgr.mu.Unlock()
	require.Equal(t, 0, n)
}

// The session is the agent-identity attribute, also when an explicit
// fairness header sets FairnessID to another value. FairnessID alone does not
// identify a session.
func TestSessionFollowsAgentIdentity(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	withTenant := newRequest("s1", 400)
	withTenant.FairnessID = "tenant-a"
	require.NoError(t, a.PreRequest(context.Background(), withTenant, schedulingResultFor(ep)))
	a.ResponseBody(context.Background(), withTenant, endOfStream(300, 100), nil)
	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Equal(t, int64(300), s.committedTokens())

	fairnessOnly := newRequest("", 400)
	fairnessOnly.FairnessID = "tenant-a"
	require.NoError(t, a.PreRequest(context.Background(), fairnessOnly, schedulingResultFor(ep)))
	_, ok = sessionOf(a, "tenant-a")
	require.False(t, ok)
}

// Two agents settling at the same moment each update their own lane: the
// ledger takes one lock per settle, so neither total is lost.
func TestConcurrentSettleKeepsBothLanes(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	first := startTurn(t, a, "s1", ep, 2000)
	second := startTurn(t, a, "s1", ep, 400)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		a.ResponseBody(context.Background(), first, endOfStream(520, 500), nil)
	}()
	go func() {
		defer wg.Done()
		a.ResponseBody(context.Background(), second, endOfStream(150, 100), nil)
	}()
	wg.Wait()

	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Equal(t, int64(0), s.inflightTokens())
	require.Equal(t, float64(520), endpointTokens(a, "default/pod-a"))
}

// A turn whose response reports no usage falls back to its own estimate as a
// floor for its own lane, and never lowers the lane it settles in.
func TestNoUsageKeepsLaneFloor(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	runTurn(t, a, "s1", ep, 2000, 900)

	req := startTurn(t, a, "s1", ep, 400) // estimate 100, below the committed 900
	a.ResponseBody(context.Background(), req, endOfStream(0, 0), nil)
	require.Equal(t, float64(900), endpointTokens(a, "default/pod-a"),
		"a turn reporting no usage must not shrink its lane below what is already committed")

	req = startTurn(t, a, "s1", ep, 8000) // estimate 2000, above the committed 900
	a.ResponseBody(context.Background(), req, endOfStream(0, 0), nil)
	require.Equal(t, float64(2000), endpointTokens(a, "default/pod-a"),
		"without a usage total the estimate becomes the lane's floor")
}

// A pod leaving the pool unbinds the session and keeps its footprint, so the
// next turn carries it to the pod that serves it.
func TestUnbindKeepsLanes(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	first := startTurn(t, a, "s1", ep, 2000)
	second := startTurn(t, a, "s1", ep, 400)
	a.ResponseBody(context.Background(), first, endOfStream(520, 500), nil)
	a.ResponseBody(context.Background(), second, endOfStream(150, 100), nil)

	a.mgr.removeEndpoint("default/pod-a")

	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Nil(t, s.endpoint)
	require.Equal(t, float64(520), s.size(),
		"the footprint survives the unbind and moves with the session")
}

// An agent that compacts its history reports a smaller total, and no lane an
// earlier turn opened keeps the pre-compaction history alive.
func TestCompactionAfterParallelTurnsDropsTheOldTotal(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	ctx := context.Background()

	main := startTurn(t, a, "s1", ep, 200000) // estimate 50000
	sub := startTurn(t, a, "s1", ep, 4000)    // estimate 1000
	a.ResponseBody(ctx, main, endOfStream(50000, 50000), nil)
	a.ResponseBody(ctx, sub, endOfStream(1000, 1000), nil)
	require.Equal(t, float64(50000), endpointTokens(a, "default/pod-a"))

	runTurn(t, a, "s1", ep, 40000, 10500)
	require.Equal(t, float64(10500), endpointTokens(a, "default/pod-a"),
		"the compacted total replaces the history, it does not sit beside it")
	s, _ := sessionOf(a, "s1")
	require.Len(t, s.lanes, 1)
}

// Turns of one conversation that overlap, through a retry or a pipelining
// client, must not leave the session carrying a history per turn.
func TestOverlappingTurnsDoNotAccumulateLanes(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	ctx := context.Background()

	held := make([]*fwksched.InferenceRequest, 0, 10)
	for range 10 {
		held = append(held, startTurn(t, a, "s1", ep, 400)) // estimate 100 each
	}
	for _, req := range held {
		a.ResponseBody(ctx, req, endOfStream(100, 100), nil)
	}

	s, _ := sessionOf(a, "s1")
	require.Len(t, s.lanes, 1)
	require.Equal(t, float64(100), endpointTokens(a, "default/pod-a"),
		"ten turns of one conversation hold one history, not ten")
}

// A pod going stale unbinds its sessions while a turn is still in flight. The
// lane keeps its charge, so the settle still finds it.
func TestStalePodUnbindKeepsLaneCharge(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	ctx := context.Background()
	runTurn(t, a, "s1", ep, 400, 300)
	req := startTurn(t, a, "s1", ep, 2000) // estimate 500, in flight

	// Age pod-a past endpointStaleAfter, then report only pod-b, which is what
	// the dispatch cycle does when a pod leaves the pool.
	a.mgr.mu.Lock()
	a.mgr.endpoints["default/pod-a"].updatedAt = time.Now().Add(-2 * endpointStaleAfter)
	a.mgr.mu.Unlock()
	primeFitView(a, dlEndpoint("pod-b"))

	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Nil(t, s.endpoint)
	require.Equal(t, int64(500), s.inflightTokens(), "the charge survives the unbind")

	a.ResponseBody(ctx, req, endOfStream(620, 500), nil)
	s, _ = sessionOf(a, "s1")
	require.Equal(t, int64(0), s.inflightTokens(), "the settle found its lane")
	require.Equal(t, float64(620), s.size())
}

// Three agents settling in any order leave the session holding the largest
// history, never the one that happened to finish last.
func TestThreeAgentsSettleInAnyOrder(t *testing.T) {
	t.Parallel()
	orders := map[string][]int{
		"largest first":  {0, 1, 2},
		"largest last":   {2, 1, 0},
		"largest middle": {1, 0, 2},
	}
	for name, order := range orders {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			a := newTestAgent(testConfig())
			ep := schedEndpoint("pod-a", 0, 0)
			// Charged together, so each takes a lane of its own.
			turns := []*fwksched.InferenceRequest{
				startTurn(t, a, "s1", ep, 8000), // estimate 2000
				startTurn(t, a, "s1", ep, 2000), // estimate 500
				startTurn(t, a, "s1", ep, 400),  // estimate 100
			}
			totals := []int{4000, 900, 150}

			for _, i := range order {
				a.ResponseBody(context.Background(), turns[i], endOfStream(totals[i], totals[i]), nil)
			}
			require.Equal(t, float64(4000), endpointTokens(a, "default/pod-a"),
				"the largest agent's history survives whatever the settle order")
		})
	}
}

// A turn the client abandons still settles, so it releases its lane and leaves
// the other agents' accounting alone. The director forces an end-of-stream for
// any request that picked a pod and never completed.
func TestAbandonedTurnReleasesItsLaneOnly(t *testing.T) {
	t.Parallel()
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	ctx := context.Background()

	live := startTurn(t, a, "s1", ep, 8000)      // estimate 2000
	abandoned := startTurn(t, a, "s1", ep, 2000) // estimate 500
	require.Equal(t, float64(2500), endpointTokens(a, "default/pod-a"))

	// The abort path reports no usage, so the estimate becomes the lane's floor.
	a.ResponseBody(ctx, abandoned, endOfStream(0, 0), nil)
	s, _ := sessionOf(a, "s1")
	require.Equal(t, int64(2000), s.inflightTokens(), "only the abandoned turn's charge was released")
	require.Equal(t, float64(2500), endpointTokens(a, "default/pod-a"),
		"the live turn keeps its estimate and the abandoned lane holds its floor")

	a.ResponseBody(ctx, live, endOfStream(4200, 4000), nil)
	require.Equal(t, float64(4200), endpointTokens(a, "default/pod-a"),
		"the live agent settles and the abandoned lane folds away")
}
