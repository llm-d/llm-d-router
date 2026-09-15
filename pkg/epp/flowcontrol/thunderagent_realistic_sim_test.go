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
	"math/rand"
	"sort"
	"sync"
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

// Synthetic workload modeled on the empirical profile of real Weka traces
// (semianalysisai/cc-traces-weka-with-subagents-060826-256k, 391 Claude Code
// sessions): heterogeneous contexts around a ~76k-token median input with a
// heavy tail, ~170:1 input-to-output ratio, think times mostly ~0.3s with a
// ~10s p90 tail, context compaction on ~6% of turns, an H100-TP2-class KV
// pool that saturates at a handful of concurrent sessions, and benchmark
// concurrency of dozens of sessions.
//
// To run as a test, every token quantity is divided by 8 and every duration
// by 100. Both scalings preserve the ratios that drive the comparison:
// working set vs KV capacity, prefill cost vs think time, and per-class
// heterogeneity. Both arms replay the exact same pre-generated schedule
// (seeded), so the only difference is routing.

const (
	realTokenScale = 8   // trace tokens / 8
	realTimeScale  = 100 // trace seconds / 100

	// The trace's H100 TP2 pool is 2.34M tokens and saturates at 3-4
	// concurrent sessions, because a session's resident footprint (mean 726k
	// unique KV tokens: old-turn and compacted-away blocks stay cached) is
	// roughly 8x its latest context. This cache model keeps only the latest
	// context resident, so the pool is scaled to preserve the trace's
	// pool-to-session-footprint ratio rather than its absolute size.
	realPodCapacity = 100_000
	// 20k tok/s prefill per pod, time-compressed 100x.
	realPrefillRate = 20_000.0 / realTokenScale * realTimeScale
	// Roughly the median turn's decode span (~0.5s real), compressed.
	realDecodeTime = 5 * time.Millisecond

	realSessions = 48
)

type realTurn struct {
	contextTokens int
	gap           time.Duration // think time after the turn
}

type realSession struct {
	id     string
	class  string
	start  time.Duration // arrival offset
	turns  []realTurn
	podIdx int // sticky assignment for the baseline arm
}

// buildRealisticWorkload generates the seeded schedule shared by both arms.
// Classes approximate the trace percentiles (post-scaling): light sessions
// near the lower half, mediums near the median, heavies toward p90; heavier
// sessions run more turns, echoing the trace's turn-count skew.
func buildRealisticWorkload() []realSession {
	rng := rand.New(rand.NewSource(1))
	classes := []struct {
		name        string
		count       int
		startTokens int
		growth      int
		turns       int
	}{
		{"light", 24, 24_000 / realTokenScale, 4_000 / realTokenScale, 10},
		{"medium", 15, 48_000 / realTokenScale, 8_000 / realTokenScale, 14},
		{"heavy", 9, 96_000 / realTokenScale, 12_000 / realTokenScale, 18},
	}

	gap := func() time.Duration {
		r := rng.Float64()
		switch {
		case r < 0.70: // median think time ~0.31s
			return 310 * time.Millisecond / realTimeScale
		case r < 0.95: // mid-range pauses ~2s
			return 2 * time.Second / realTimeScale
		default: // p90 tail ~10s
			return 10 * time.Second / realTimeScale
		}
	}

	var sessions []realSession
	idx := 0
	for _, c := range classes {
		for i := 0; i < c.count; i++ {
			s := realSession{
				id:     fmt.Sprintf("%s-%d", c.name, i),
				class:  c.name,
				start:  time.Duration(rng.Int63n(int64(500 * time.Millisecond))),
				podIdx: idx % 2,
			}
			contextTokens := c.startTokens
			for turn := 0; turn < c.turns; turn++ {
				// Context compaction on ~5.9% of turns: input drops below
				// half of the previous turn, as in 63% of real traces.
				if turn > 0 && rng.Float64() < 0.059 {
					contextTokens = int(float64(contextTokens) * 0.4)
				}
				s.turns = append(s.turns, realTurn{contextTokens: contextTokens, gap: gap()})
				contextTokens += c.growth
			}
			sessions = append(sessions, s)
			idx++
		}
	}
	return sessions
}

type realArmResult struct {
	prefilledTokens int64
	hits, misses    int64
	makespan        time.Duration
	holds           int64
	sheds           int64
	latencies       []time.Duration // per turn: admission wait + pod service time
}

func percentile(latencies []time.Duration, p float64) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), latencies...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	i := int(p * float64(len(sorted)-1))
	return sorted[i]
}

func newRealPods() []*simPod {
	return []*simPod{
		newSimPodWith("pod-a", realPodCapacity, realPrefillRate, realDecodeTime),
		newSimPodWith("pod-b", realPodCapacity, realPrefillRate, realDecodeTime),
	}
}

// runRealisticBaseline: ideal session affinity, no admission control.
func runRealisticBaseline(t *testing.T, sessions []realSession) realArmResult {
	t.Helper()
	pods := newRealPods()
	var mu sync.Mutex
	var latencies []time.Duration

	start := time.Now()
	var wg sync.WaitGroup
	for _, s := range sessions {
		wg.Add(1)
		go func(s realSession) {
			defer wg.Done()
			time.Sleep(s.start)
			pod := pods[s.podIdx]
			for _, turn := range s.turns {
				t0 := time.Now()
				pod.runTurn(s.id, turn.contextTokens)
				lat := time.Since(t0)
				mu.Lock()
				latencies = append(latencies, lat)
				mu.Unlock()
				time.Sleep(turn.gap)
			}
		}(s)
	}
	wg.Wait()

	r := realArmResult{makespan: time.Since(start), latencies: latencies}
	for _, p := range pods {
		r.prefilledTokens += p.prefilledTokens.Load()
		r.hits += p.hits.Load()
		r.misses += p.misses.Load()
	}
	return r
}

// realisticThunderParams: capacity matches the simulated pods; the growth
// buffer is sized for the light class (heavier sessions overshoot it, which
// is what shedding is for); shedding and mild decay are on.
const realisticThunderParams = `{
	"capacityTokens": 100000,
	"utilThreshold": 0.9,
	"actingHalfLifeSeconds": 0.5,
	"bufferTokensPerProgram": 5000,
	"headWaitStarvationMs": 30000,
	"evictionTtlSeconds": 3600,
	"evictionSweepSeconds": 300,
	"sessionFinalHeader": "x-session-final",
	"shedIdleSeconds": 0.02
}`

// runRealisticThunder: the real plugin and flow controller, plugin-driven
// placement, same schedule.
func runRealisticThunder(t *testing.T, sessions []realSession) realArmResult {
	t.Helper()
	pods := newRealPods()

	plugin, err := thunderagent.Factory("thunder-realistic", fwkplugin.StrictDecoder([]byte(realisticThunderParams)), nil)
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

	var mu sync.Mutex
	var latencies []time.Duration
	var holds int64

	start := time.Now()
	var wg sync.WaitGroup
	for _, s := range sessions {
		wg.Add(1)
		go func(s realSession) {
			defer wg.Done()
			time.Sleep(s.start)
			for i, turn := range s.turns {
				t0 := time.Now()
				fcReq := &testRequest{
					id:       fmt.Sprintf("%s-turn-%d", s.id, i),
					key:      flowcontrol.FlowKey{ID: s.id, Priority: 0},
					byteSize: uint64(turn.contextTokens * 4),
					ttl:      time.Minute,
				}
				outcome, err := h.fc.EnqueueAndWait(h.ctx, fcReq)
				require.NoError(t, err)
				require.Equal(t, fcTypes.QueueOutcomeDispatched, outcome,
					"turn %d of %s must dispatch, not time out", i, s.id)
				if time.Since(t0) > 50*time.Millisecond {
					mu.Lock()
					holds++
					mu.Unlock()
				}

				req := &fwksched.InferenceRequest{
					RequestID:        fcReq.id,
					FairnessID:       s.id,
					RequestSizeBytes: turn.contextTokens * 4,
					Headers:          map[string]string{},
				}
				if i == len(s.turns)-1 {
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

				byName[podName].runTurn(s.id, turn.contextTokens)

				ta.ResponseBody(h.ctx, req, &fwkrc.Response{
					EndOfStream: true,
					Usage:       requesthandling.Usage{TotalTokens: turn.contextTokens, PromptTokens: turn.contextTokens},
				}, nil)

				lat := time.Since(t0)
				mu.Lock()
				latencies = append(latencies, lat)
				mu.Unlock()
				time.Sleep(turn.gap)
			}
		}(s)
	}
	wg.Wait()

	r := realArmResult{makespan: time.Since(start), latencies: latencies, holds: holds}
	for _, p := range pods {
		r.prefilledTokens += p.prefilledTokens.Load()
		r.hits += p.hits.Load()
		r.misses += p.misses.Load()
	}
	if raw, err := ta.DumpState(); err == nil {
		var state struct {
			ShedsTotal int64 `json:"shedsTotal"`
		}
		if json.Unmarshal(raw, &state) == nil {
			r.sheds = state.ShedsTotal
		}
	}
	return r
}

func TestThunderAgentOnRealisticWekaProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("realistic simulation uses wall-clock pacing")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	go func() {
		<-ctx.Done()
		if ctx.Err() == context.DeadlineExceeded {
			panic("realistic simulation exceeded its deadline; an arm is likely stalled")
		}
	}()

	sessions := buildRealisticWorkload()
	var turns int
	for _, s := range sessions {
		turns += len(s.turns)
	}

	baseline := runRealisticBaseline(t, sessions)
	thunder := runRealisticThunder(t, sessions)

	t.Logf("workload: %d sessions (24 light / 15 medium / 9 heavy), %d turns, contexts %d..%d tokens (trace/8), 2 pods x %d-token KV, compaction on ~6%% of turns",
		len(sessions), turns, 24_000/realTokenScale, (96_000+17*12_000)/realTokenScale, realPodCapacity)
	t.Logf("%-18s %14s %8s %8s %10s %10s %10s %7s %7s", "arm", "prefill_tokens", "hits", "misses", "turn_p50", "turn_p95", "makespan", "holds", "sheds")
	t.Logf("%-18s %14d %8d %8d %10s %10s %10s %7d %7s", "session-affinity",
		baseline.prefilledTokens, baseline.hits, baseline.misses,
		percentile(baseline.latencies, 0.50).Round(time.Millisecond), percentile(baseline.latencies, 0.95).Round(time.Millisecond),
		baseline.makespan.Round(time.Millisecond), baseline.holds, "-")
	t.Logf("%-18s %14d %8d %8d %10s %10s %10s %7d %7d", "thunder-agent",
		thunder.prefilledTokens, thunder.hits, thunder.misses,
		percentile(thunder.latencies, 0.50).Round(time.Millisecond), percentile(thunder.latencies, 0.95).Round(time.Millisecond),
		thunder.makespan.Round(time.Millisecond), thunder.holds, thunder.sheds)
	t.Logf("prefill ratio %.2f; hit rates: baseline %.0f%%, thunder %.0f%%",
		float64(thunder.prefilledTokens)/float64(baseline.prefilledTokens),
		100*float64(baseline.hits)/float64(turns), 100*float64(thunder.hits)/float64(turns))

	// Regime checks: the pool must be under enough pressure that the
	// baseline misses a meaningful share of turns, and the thunder arm must
	// actually have engaged.
	require.Greater(t, baseline.misses, int64(turns/4),
		"the baseline should miss on a meaningful share of turns for the comparison to mean anything")
	require.Greater(t, thunder.holds, int64(0), "the thunder arm must have held at least one turn")

	require.Less(t, float64(thunder.prefilledTokens), 0.85*float64(baseline.prefilledTokens),
		"program-aware admission should cut re-prefill on the trace-shaped workload")
}
