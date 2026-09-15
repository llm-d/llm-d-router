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

package flowcontrol_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// This simulation demonstrates proactive shedding on the failure mode the
// fit check cannot prevent: admission sized to sessions' arrival footprint
// while their contexts grow far past it.
//
// Both arms run the identical workload through the real plugin and flow
// controller with bufferTokensPerProgram deliberately 0 (the mis-sized
// operator case) and acting decay off, so shedding is the only difference.
// At arrival every session fits (12 x 300 tokens over two 1800-token
// ceilings) and all are admitted; after growth each pod carries 6 x 550 =
// 3300 tokens against 2000 of physical KV.
//
// Without shed, REASONING bypass keeps all 12 running: cyclic LRU access
// over capacity evicts exactly the session that returns next, so nearly
// every turn re-prefills its whole history, and no request is ever held.
// With shed, over-ceiling pods unbind their smallest idle sessions; the
// survivors keep their prefix resident while shed sessions are held at
// admission (their holds are the discriminating signal) and readmit as
// capacity frees.

const shedSimParamsOff = `{
	"capacityTokens": 2000,
	"utilThreshold": 0.9,
	"actingHalfLifeSeconds": 0,
	"bufferTokensPerProgram": 0,
	"headWaitStarvationMs": 30000,
	"evictionTtlSeconds": 3600,
	"evictionSweepSeconds": 300,
	"sessionFinalHeader": "x-session-final",
	"shedIdleSeconds": 0
}`

const shedSimParamsOn = `{
	"capacityTokens": 2000,
	"utilThreshold": 0.9,
	"actingHalfLifeSeconds": 0,
	"bufferTokensPerProgram": 0,
	"headWaitStarvationMs": 30000,
	"evictionTtlSeconds": 3600,
	"evictionSweepSeconds": 300,
	"sessionFinalHeader": "x-session-final",
	"shedIdleSeconds": 0.005
}`

func TestThunderAgentShedRecoversFromOverAdmission(t *testing.T) {
	if testing.Short() {
		t.Skip("shed simulation uses wall-clock pacing")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	go func() {
		<-ctx.Done()
		if ctx.Err() == context.DeadlineExceeded {
			panic("shed simulation exceeded its deadline; an arm is likely stalled")
		}
	}()

	noShed := runThunderArm(t, shedSimParamsOff)
	shed := runThunderArm(t, shedSimParamsOn)

	turns := int64(simSessions * simTurns)
	t.Logf("workload: %d sessions x %d turns, context %d..%d tokens, 2 pods x %d-token KV (LRU), buffer mis-sized to 0",
		simSessions, simTurns, simTurnContext(0), simTurnContext(simTurns-1), simPodCapacityTokens)
	t.Logf("%-22s %14s %9s %9s %12s %7s %10s", "arm", "prefill_tokens", "hits", "misses", "makespan", "holds", "held_time")
	t.Logf("%-22s %14d %9d %9d %12s %7d %10s", "thunder shed=off",
		noShed.prefilledTokens, noShed.hits, noShed.misses, noShed.makespan.Round(time.Millisecond),
		noShed.holds, noShed.heldTime.Round(time.Millisecond))
	t.Logf("%-22s %14d %9d %9d %12s %7d %10s", "thunder shed=on",
		shed.prefilledTokens, shed.hits, shed.misses, shed.makespan.Round(time.Millisecond),
		shed.holds, shed.heldTime.Round(time.Millisecond))
	t.Logf("sheds: no-shed arm %d, shed arm %d", noShed.sheds, shed.sheds)
	t.Logf("prefill ratio (shed/no-shed): %.2f; hit rates: no-shed %.0f%%, shed %.0f%%",
		float64(shed.prefilledTokens)/float64(noShed.prefilledTokens),
		100*float64(noShed.hits)/float64(turns), 100*float64(shed.hits)/float64(turns))

	// The over-admission regime must be real: without shed nothing is held
	// (everything was admitted at arrival) and the pods thrash.
	require.Zero(t, noShed.holds,
		"shed-off arm holding requests means the workload did not over-admit; the comparison is void")
	require.Greater(t, noShed.misses, noShed.hits,
		"shed-off arm should thrash (mostly full re-prefills) for the comparison to mean anything")

	// Shed converts over-admission back into held sessions: holds appear, and
	// the surviving sessions' resident prefixes cut total re-prefill work.
	require.Greater(t, shed.holds, int64(0),
		"shed must have unbound sessions whose next turn was then held")
	require.Less(t, float64(shed.prefilledTokens), 0.65*float64(noShed.prefilledTokens),
		"shedding should eliminate most of the re-prefill caused by over-admission")
}

// The flood scenario: veterans over-admit and get shed while a wave of
// newcomers arrives mid-run. Every held request has a blocked client, so the
// resuming class is what keeps shed mid-trajectory sessions from being
// starved by a stream of smaller newcomers: when room frees, a shed veteran
// resumes ahead of any never-admitted session. Without the class, smallest
// footprint would win and veterans would drain toward the 30s starvation
// bound.
func TestThunderAgentResumingPriorityUnderNewcomerFlood(t *testing.T) {
	if testing.Short() {
		t.Skip("flood simulation uses wall-clock pacing")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	go func() {
		<-ctx.Done()
		if ctx.Err() == context.DeadlineExceeded {
			panic("flood simulation exceeded its deadline; a cohort is likely stalled")
		}
	}()

	result, stats := runThunderCohorts(t, shedSimParamsOn, []cohortSpec{
		{name: "veteran", sessions: simSessions, turns: simTurns, context: simTurnContext},
		{name: "newcomer", sessions: 6, turns: 3, startDelay: 150 * time.Millisecond, context: simTurnContext},
	})

	veterans, newcomers := stats["veteran"], stats["newcomer"]
	t.Logf("sheds=%d makespan=%s", result.sheds, result.makespan.Round(time.Millisecond))
	t.Logf("%-10s %7s %12s", "cohort", "holds", "avg_held")
	t.Logf("%-10s %7d %12s", "veteran", veterans.holds.Load(), veterans.avgHeld().Round(time.Millisecond))
	t.Logf("%-10s %7d %12s", "newcomer", newcomers.holds.Load(), newcomers.avgHeld().Round(time.Millisecond))

	require.Greater(t, result.sheds, int64(0), "the flood must run in the shed regime")
	require.Greater(t, veterans.holds.Load(), int64(0), "shed veterans must have been held at readmission")
	require.Greater(t, newcomers.holds.Load(), int64(0), "newcomers arriving under pressure must have been held")
	require.Less(t, veterans.avgHeld(), newcomers.avgHeld(),
		"resuming priority should readmit shed veterans faster than it admits newcomers")
}
