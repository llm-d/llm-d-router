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

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// markIdle backdates a program's last response so it passes any shed
// idle-eligibility bound.
func markIdle(a *ThunderAgent, id string, idle time.Duration) {
	a.table.mu.Lock()
	a.table.programs[id].lastResponseAt = time.Now().Add(-idle)
	a.table.mu.Unlock()
}

func boundPod(a *ThunderAgent, id string) string {
	a.table.mu.Lock()
	defer a.table.mu.Unlock()
	st, ok := a.table.programs[id]
	if !ok {
		return ""
	}
	return st.podName
}

func TestShedDisabledByDefault(t *testing.T) {
	a := newTestAgent(testConfig()) // shedIdleSeconds 0
	sched := newTestEndpoints("pod1")

	// 1200 tokens on a 900-token ceiling (capacity 1000, threshold 0.9).
	seedProgram(t, a, "program-a", sched[0], 1200)
	markIdle(a, "program-a", time.Hour)

	a.Saturation(context.Background(), dlEndpoints("pod1"))

	assert.NotEmpty(t, boundPod(a, "program-a"), "shed off: over-ceiling pods keep their programs")
	assert.Equal(t, 0.0, testutil.ToFloat64(a.metrics.sheds))
}

func TestShedUnbindsSmallestIdleFirst(t *testing.T) {
	cfg := testConfig()
	cfg.ShedIdleSeconds = 0.001
	a := newTestAgent(cfg)
	sched := newTestEndpoints("pod1")

	// 100 + 200 + 900 = 1200 tokens on a 900-token ceiling. Shedding smallest
	// first drops 100 (1100, still over) then 200 (900, at the ceiling); the
	// 900-token program survives.
	seedProgram(t, a, "small", sched[0], 100)
	seedProgram(t, a, "medium", sched[0], 200)
	seedProgram(t, a, "large", sched[0], 900)
	for _, id := range []string{"small", "medium", "large"} {
		markIdle(a, id, time.Second)
	}

	a.Saturation(context.Background(), dlEndpoints("pod1"))

	assert.Empty(t, boundPod(a, "small"))
	assert.Empty(t, boundPod(a, "medium"))
	assert.NotEmpty(t, boundPod(a, "large"))
	assert.Equal(t, 2.0, testutil.ToFloat64(a.metrics.sheds))

	a.table.mu.Lock()
	snap := a.table.snapshot["default/pod1"]
	a.table.mu.Unlock()
	require.NotNil(t, snap)
	assert.Equal(t, 900.0, snap.tokens, "the fit view reflects the post-shed working set")
}

func TestShedSkipsInflightAndRecentlyIdle(t *testing.T) {
	cfg := testConfig()
	cfg.ShedIdleSeconds = 10
	a := newTestAgent(cfg)
	sched := newTestEndpoints("pod1")

	// In flight: PreRequest ran, no response yet (400 bytes -> 100 tokens
	// inflight at the seeded 4.0 bytes-per-token, plus 1000 committed from a
	// prior turn).
	seedProgram(t, a, "inflight", sched[0], 1000)
	req := newRequest("inflight", 400)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(sched[0])))

	// Idle, but only just: under the 10s eligibility bound.
	seedProgram(t, a, "fresh", sched[0], 500)

	a.Saturation(context.Background(), dlEndpoints("pod1"))

	assert.NotEmpty(t, boundPod(a, "inflight"), "a program with a request in flight is never shed")
	assert.NotEmpty(t, boundPod(a, "fresh"), "a program idle for less than shedIdleSeconds is not shed")
	assert.Equal(t, 0.0, testutil.ToFloat64(a.metrics.sheds))
}

func TestShedProgramReentersAsResuming(t *testing.T) {
	cfg := testConfig()
	cfg.ShedIdleSeconds = 0.001
	a := newTestAgent(cfg)
	sched := newTestEndpoints("pod1")

	seedProgram(t, a, "squatter", sched[0], 600)
	seedProgram(t, a, "runner", sched[0], 700)
	markIdle(a, "squatter", time.Second)

	a.Saturation(context.Background(), dlEndpoints("pod1"))
	require.Empty(t, boundPod(a, "squatter"), "the smaller idle program is shed")

	// Class: RESUMING (no bypass, but ahead of never-admitted programs),
	// with its full committed footprint.
	a.table.mu.Lock()
	class, tokens := a.table.classAndTokens("squatter", time.Now())
	a.table.mu.Unlock()
	assert.Equal(t, classResuming, class)
	assert.Equal(t, 600.0, tokens)

	// Admission: held while the survivor leaves no room (700 + 600 > 900).
	q := makeQueueWithBytes("squatter", 1, time.Now(), 600*4)
	picked, err := a.Pick(context.Background(), bandOf(q))
	require.NoError(t, err)
	assert.Nil(t, picked, "a shed program's next turn is fit-checked, not bypassed")

	// Placement: no sticky branch for an unbound program.
	scores := a.Score(context.Background(), newRequest("squatter", 600*4), sched)
	assert.Less(t, scores[sched[0]], 1.0, "shed programs are re-placed by load, not pinned")
}

// A shed program outranks a never-admitted one when both fit, regardless of
// size and enqueue order: it is mid-trajectory, and its smaller rival's
// arrival size is transient anyway.
func TestPick_ResumingBeatsNewcomer(t *testing.T) {
	cfg := testConfig()
	cfg.ShedIdleSeconds = 0.001
	a := newTestAgent(cfg)
	sched := newTestEndpoints("pod1")

	seedProgram(t, a, "veteran", sched[0], 400)
	seedProgram(t, a, "runner", sched[0], 600) // 1000 > 900 ceiling
	markIdle(a, "veteran", time.Second)
	a.Saturation(context.Background(), dlEndpoints("pod1"))
	require.Empty(t, boundPod(a, "veteran"))

	// Free the pod so both candidates fit.
	seedProgram(t, a, "runner", sched[0], 100)
	primeFitView(a, "pod1")

	newcomer := makeQueueWithBytes("newcomer", 1, time.Now().Add(-time.Second), 300*4) // smaller and older
	veteran := makeQueueWithBytes("veteran", 1, time.Now(), 400*4)
	picked, err := a.Pick(context.Background(), bandOf(newcomer, veteran))
	require.NoError(t, err)
	assert.Same(t, veteran, picked, "resuming outranks new even when the newcomer is smaller and waited longer")
}

// Class priority applies among fitting candidates only: a resuming program
// that does not fit must not block a fitting newcomer (work conservation,
// same as upstream's cumulative resume selection).
func TestPick_NonFittingResumingDoesNotBlockFittingNewcomer(t *testing.T) {
	cfg := testConfig()
	cfg.ShedIdleSeconds = 0.001
	a := newTestAgent(cfg)
	sched := newTestEndpoints("pod1")

	seedProgram(t, a, "veteran", sched[0], 800)
	seedProgram(t, a, "runner", sched[0], 700) // 1500 > 900 ceiling, veteran... runner is bigger
	markIdle(a, "veteran", time.Second)
	a.Saturation(context.Background(), dlEndpoints("pod1"))
	require.Empty(t, boundPod(a, "veteran"), "the smaller idle program is shed first")

	// Room is 900 - 700 = 200: the 800-token veteran cannot fit, a 150-token
	// newcomer can.
	newcomer := makeQueueWithBytes("newcomer", 1, time.Now(), 150*4)
	veteran := makeQueueWithBytes("veteran", 1, time.Now(), 800*4)
	picked, err := a.Pick(context.Background(), bandOf(veteran, newcomer))
	require.NoError(t, err)
	assert.Same(t, newcomer, picked)
}
