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
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	contractmocks "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	fcTypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/thunderagent"
)

// This test runs one agentic workload through two routing arms on identical
// simulated pods and compares the KV recompute each arm causes.
//
// Baseline arm: ideal session affinity. Every session is perfectly sticky to
// its pod and there is no admission control. Sessions per pod exceed the
// pod's KV capacity, and cyclic multi-turn access over an LRU cache evicts
// exactly the session that returns next, so most turns re-prefill their full
// history.
//
// Thunder arm: the real ThunderAgent plugin wired into the real
// FlowController (saturation detector + fairness policy) with plugin-driven
// placement, binding, accounting, and session-final release. Admission holds
// sessions that do not fit, so admitted sessions keep their prefix resident.
//
// Decision metric: total prefilled (cache-miss) tokens. Secondary: hit rate,
// makespan, and hold count (holds > 0 proves the thunder arm engaged).

// --- Simulated pod: LRU prefix cache + serialized prefill ---

const (
	simPodCapacityTokens = 2000
	simPrefillTokPerSec  = 50000.0
	simDecodeTime        = 2 * time.Millisecond
	simToolGap           = 30 * time.Millisecond
	simSessions          = 12
	simTurns             = 6
)

// simTurnContext is the full (resent) history length at a turn.
func simTurnContext(turn int) int { return 300 + 50*turn }

type simCacheEntry struct {
	tokens   int
	lastUsed int64
}

type simPod struct {
	name     string
	capacity int

	mu    sync.Mutex // serializes turn execution, approximating a busy engine
	cache map[string]*simCacheEntry
	seq   int64

	prefillRate float64
	decodeTime  time.Duration

	prefilledTokens atomic.Int64
	hits            atomic.Int64
	misses          atomic.Int64
}

func newSimPod(name string) *simPod {
	return newSimPodWith(name, simPodCapacityTokens, simPrefillTokPerSec, simDecodeTime)
}

func newSimPodWith(name string, capacity int, prefillRate float64, decodeTime time.Duration) *simPod {
	return &simPod{
		name: name, capacity: capacity,
		prefillRate: prefillRate, decodeTime: decodeTime,
		cache: make(map[string]*simCacheEntry),
	}
}

// runTurn executes one turn: a session found resident pays only the context
// delta, an evicted or new session re-prefills its full context. The cache
// then holds the session's context and evicts least-recently-used sessions
// down to capacity. Execution time is prefill tokens at the prefill rate plus
// a fixed decode time, held under the pod lock.
func (p *simPod) runTurn(session string, contextTokens int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	prefill := contextTokens
	if e, ok := p.cache[session]; ok && e.tokens > 0 {
		prefill = contextTokens - e.tokens
		p.hits.Add(1)
	} else {
		p.misses.Add(1)
	}
	p.prefilledTokens.Add(int64(prefill))

	p.seq++
	p.cache[session] = &simCacheEntry{tokens: contextTokens, lastUsed: p.seq}
	p.evictLRU(session)

	time.Sleep(time.Duration(float64(prefill)/p.prefillRate*float64(time.Second)) + p.decodeTime)
}

func (p *simPod) evictLRU(current string) {
	for {
		total := 0
		var victim string
		var victimUsed int64
		for id, e := range p.cache {
			total += e.tokens
			if id == current {
				continue
			}
			if victim == "" || e.lastUsed < victimUsed {
				victim, victimUsed = id, e.lastUsed
			}
		}
		if total <= p.capacity || victim == "" {
			return
		}
		delete(p.cache, victim)
	}
}

type armResult struct {
	prefilledTokens int64
	hits, misses    int64
	makespan        time.Duration
	holds           int64
	heldTime        time.Duration
	sheds           int64
}

func collect(pods []*simPod, makespan time.Duration, holds int64, heldTime time.Duration) armResult {
	r := armResult{makespan: makespan, holds: holds, heldTime: heldTime}
	for _, p := range pods {
		r.prefilledTokens += p.prefilledTokens.Load()
		r.hits += p.hits.Load()
		r.misses += p.misses.Load()
	}
	return r
}

// --- Baseline arm: ideal session affinity, no admission control ---

func runBaselineArm(t *testing.T) armResult {
	t.Helper()
	pods := []*simPod{newSimPod("pod-a"), newSimPod("pod-b")}

	start := time.Now()
	var wg sync.WaitGroup
	for i := 0; i < simSessions; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			pod := pods[i%len(pods)] // perfectly sticky, balanced binding
			session := fmt.Sprintf("session-%d", i)
			for turn := 0; turn < simTurns; turn++ {
				pod.runTurn(session, simTurnContext(turn))
				time.Sleep(simToolGap)
			}
		}(i)
	}
	wg.Wait()
	return collect(pods, time.Since(start), 0, 0)
}

// --- Thunder arm: real plugin + real flow controller ---

// thunderSimParams sizes admission so ~3 sessions fit a pod at their grown
// size: a session arrives at 300 tokens and grows to 550, so the 250-token
// buffer makes its projected footprint 550 and the 1800-token fit ceiling
// (0.9 x 2000) admits three per pod. The mild half-life decays 4% across a
// 30ms tool gap, so footprints are effectively kept between turns.
const thunderSimParams = `{
	"capacityTokens": 2000,
	"utilThreshold": 0.9,
	"actingHalfLifeSeconds": 0.5,
	"bufferTokensPerProgram": 250,
	"headWaitStarvationMs": 30000,
	"evictionTtlSeconds": 3600,
	"evictionSweepSeconds": 300,
	"sessionFinalHeader": "x-session-final"
}`

// cohortSpec describes one group of sessions in a thunder-arm run.
type cohortSpec struct {
	name       string
	sessions   int
	turns      int
	startDelay time.Duration
	context    func(turn int) int
}

// cohortStats aggregates hold behavior per cohort.
type cohortStats struct {
	holds     atomic.Int64
	heldNanos atomic.Int64
}

func (s *cohortStats) avgHeld() time.Duration {
	h := s.holds.Load()
	if h == 0 {
		return 0
	}
	return time.Duration(s.heldNanos.Load() / h)
}

func runThunderArm(t *testing.T, params string) armResult {
	t.Helper()
	result, _ := runThunderCohorts(t, params, []cohortSpec{
		{name: "session", sessions: simSessions, turns: simTurns, context: simTurnContext},
	})
	return result
}

func runThunderCohorts(t *testing.T, params string, cohorts []cohortSpec) (armResult, map[string]*cohortStats) {
	t.Helper()
	pods := []*simPod{newSimPod("pod-a"), newSimPod("pod-b")}

	plugin, err := thunderagent.Factory("thunder-sim", fwkplugin.StrictDecoder([]byte(params)), nil)
	require.NoError(t, err)
	ta := plugin.(*thunderagent.ThunderAgent)

	dlEndpoints := make([]datalayer.Endpoint, 0, len(pods))
	schedEndpoints := make([]fwksched.Endpoint, 0, len(pods))
	byName := make(map[string]*simPod, len(pods))
	schedByName := make(map[string]fwksched.Endpoint, len(pods))
	for _, p := range pods {
		meta := &datalayer.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: p.name}}
		dlEndpoints = append(dlEndpoints, datalayer.NewEndpoint(meta, datalayer.NewMetrics()))
		se := fwksched.NewEndpoint(meta, &datalayer.Metrics{}, nil)
		schedEndpoints = append(schedEndpoints, se)
		byName[meta.ID.String()] = p
		schedByName[meta.ID.String()] = se
	}

	h := newHarness(t, harnessOpts{
		detector:           ta,
		fairness:           ta,
		endpointCandidates: &contractmocks.MockEndpointCandidates{Candidates: dlEndpoints},
	})

	stats := make(map[string]*cohortStats, len(cohorts))
	for _, c := range cohorts {
		stats[c.name] = &cohortStats{}
	}

	start := time.Now()
	var wg sync.WaitGroup
	for _, cohort := range cohorts {
		for i := 0; i < cohort.sessions; i++ {
			wg.Add(1)
			go func(cohort cohortSpec, i int) {
				defer wg.Done()
				if cohort.startDelay > 0 {
					time.Sleep(cohort.startDelay)
				}
				session := fmt.Sprintf("%s-%d", cohort.name, i)
				cs := stats[cohort.name]
				for turn := 0; turn < cohort.turns; turn++ {
					contextTokens := cohort.context(turn)

					// Admission through the real controller; ThunderAgent's
					// Saturation and Pick decide hold and release.
					fcReq := &testRequest{
						id:       fmt.Sprintf("%s-turn-%d", session, turn),
						key:      flowcontrol.FlowKey{ID: session, Priority: 0},
						byteSize: uint64(contextTokens * 4),
						ttl:      time.Minute,
					}
					waitStart := time.Now()
					outcome, err := h.fc.EnqueueAndWait(h.ctx, fcReq)
					waited := time.Since(waitStart)
					require.NoError(t, err)
					require.Equal(t, fcTypes.QueueOutcomeDispatched, outcome,
						"turn %d of %s must dispatch, not time out", turn, session)
					if waited > 50*time.Millisecond {
						cs.holds.Add(1)
						cs.heldNanos.Add(int64(waited))
					}

					// Placement by the plugin's own scorer: sticky when bound,
					// least token load otherwise. Deterministic tie-break by name.
					req := &fwksched.InferenceRequest{
						RequestID:        fcReq.id,
						FairnessID:       session,
						RequestSizeBytes: contextTokens * 4,
						Headers:          map[string]string{},
					}
					if turn == cohort.turns-1 {
						req.Headers["x-session-final"] = "true"
					}
					scores := ta.Score(h.ctx, req, schedEndpoints)
					podName := pickBest(scores)
					result := &fwksched.SchedulingResult{
						PrimaryProfileName: "default",
						ProfileResults: map[string]*fwksched.ProfileRunResult{
							"default": {TargetEndpoints: []fwksched.Endpoint{schedByName[podName]}},
						},
					}
					require.NoError(t, ta.PreRequest(h.ctx, req, result))

					byName[podName].runTurn(session, contextTokens)

					ta.ResponseBody(h.ctx, req, &fwkrc.Response{
						EndOfStream: true,
						Usage:       requesthandling.Usage{TotalTokens: contextTokens, PromptTokens: contextTokens},
					}, nil)

					time.Sleep(simToolGap)
				}
			}(cohort, i)
		}
	}
	wg.Wait()

	var holds, heldNanos int64
	for _, cs := range stats {
		holds += cs.holds.Load()
		heldNanos += cs.heldNanos.Load()
	}
	result := collect(pods, time.Since(start), holds, time.Duration(heldNanos))
	if raw, err := ta.DumpState(); err == nil {
		var state struct {
			ShedsTotal int64 `json:"shedsTotal"`
		}
		if json.Unmarshal(raw, &state) == nil {
			result.sheds = state.ShedsTotal
		}
	}
	return result, stats
}

func pickBest(scores map[fwksched.Endpoint]float64) string {
	best := ""
	bestScore := -1.0
	for e, s := range scores {
		name := e.GetMetadata().ID.String()
		if s > bestScore || (s == bestScore && (best == "" || name < best)) {
			best, bestScore = name, s
		}
	}
	return best
}

func TestThunderAgentBeatsSessionAffinityOnThrashingWorkload(t *testing.T) {
	if testing.Short() {
		t.Skip("A/B simulation uses wall-clock pacing")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	go func() {
		<-ctx.Done()
		if ctx.Err() == context.DeadlineExceeded {
			panic("A/B simulation exceeded its deadline; the thunder arm is likely stalled")
		}
	}()

	baseline := runBaselineArm(t)
	thunder := runThunderArm(t, thunderSimParams)

	turns := int64(simSessions * simTurns)
	t.Logf("workload: %d sessions x %d turns, context %d..%d tokens, 2 pods x %d-token KV (LRU)",
		simSessions, simTurns, simTurnContext(0), simTurnContext(simTurns-1), simPodCapacityTokens)
	t.Logf("%-22s %14s %9s %9s %12s %7s %10s", "arm", "prefill_tokens", "hits", "misses", "makespan", "holds", "held_time")
	t.Logf("%-22s %14d %9d %9d %12s %7d %10s", "session-affinity",
		baseline.prefilledTokens, baseline.hits, baseline.misses, baseline.makespan.Round(time.Millisecond), baseline.holds, "-")
	t.Logf("%-22s %14d %9d %9d %12s %7d %10s", "thunder-agent",
		thunder.prefilledTokens, thunder.hits, thunder.misses, thunder.makespan.Round(time.Millisecond),
		thunder.holds, thunder.heldTime.Round(time.Millisecond))
	t.Logf("prefill ratio (thunder/baseline): %.2f; hit rates: baseline %.0f%%, thunder %.0f%%",
		float64(thunder.prefilledTokens)/float64(baseline.prefilledTokens),
		100*float64(baseline.hits)/float64(turns), 100*float64(thunder.hits)/float64(turns))

	require.Greater(t, thunder.holds, int64(0),
		"the thunder arm must have held at least one turn, otherwise the workload never saturated and the comparison is void")
	require.Less(t, float64(thunder.prefilledTokens), 0.7*float64(baseline.prefilledTokens),
		"thunder admission control should eliminate most re-prefill on a thrashing workload")
}
