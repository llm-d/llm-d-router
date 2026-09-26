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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPick_NilBand(t *testing.T) {
	a := newTestAgent(testConfig())
	got, err := a.Pick(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestPick_InTrajectoryBeatsNew(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")
	now := time.Now()

	seedProgram(t, a, "in-trajectory", endpoints[0], 100000)

	got, err := a.Pick(context.Background(), bandOf(
		makeQueue("in-trajectory", 1, now),
		makeQueue("new", 1, now),
	))
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "in-trajectory", got.FlowKey().ID)
}

// Every queued program has a request pending, so a long gap since its last
// response must not demote it. Upstream sets REASONING on arrival precisely to
// avoid that, and this guards against reintroducing an idle-based ACTING class.
func TestPick_IdleGapDoesNotDemote(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")
	now := time.Now()

	seedProgram(t, a, "long-idle", endpoints[0], 5000)
	a.table.mu.Lock()
	a.table.programs["long-idle"].lastResponseAt = now.Add(-10 * time.Minute)
	a.table.programs["long-idle"].lastActivity = now.Add(-10 * time.Minute)
	a.table.mu.Unlock()

	got, err := a.Pick(context.Background(), bandOf(
		makeQueue("long-idle", 1, now),
		makeQueue("new", 1, now),
	))
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "long-idle", got.FlowKey().ID,
		"a served program with a pending request outranks a new one regardless of idle gap")
}

func TestPick_SmallestFootprintFirstWithinClass(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")
	now := time.Now()

	seedProgram(t, a, "big", endpoints[0], 200000)
	seedProgram(t, a, "small", endpoints[0], 20000)

	got, err := a.Pick(context.Background(), bandOf(
		makeQueue("big", 1, now),
		makeQueue("small", 1, now),
	))
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "small", got.FlowKey().ID)
}

// Class dominates size: a large in-trajectory program still precedes a small
// new one.
func TestPick_ClassOutranksFootprint(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")
	now := time.Now()

	seedProgram(t, a, "big-in-trajectory", endpoints[0], 500000)

	got, err := a.Pick(context.Background(), bandOf(
		makeQueue("big-in-trajectory", 1, now),
		makeQueue("small-new", 1, now),
	))
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "big-in-trajectory", got.FlowKey().ID)
}

func TestPick_TieBreaksOnOldestHead(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")
	now := time.Now()

	seedProgram(t, a, "older", endpoints[0], 1000)
	seedProgram(t, a, "newer", endpoints[0], 1000)

	got, err := a.Pick(context.Background(), bandOf(
		makeQueue("older", 1, now.Add(-30*time.Second)),
		makeQueue("newer", 1, now),
	))
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "older", got.FlowKey().ID, "equal class and footprint serve FIFO")
}

func TestPick_StarvationGuardPromotesLongWaiter(t *testing.T) {
	cfg := testConfig()
	cfg.HeadWaitStarvationMs = 5000
	a := newTestAgent(cfg)
	endpoints := newTestEndpoints("pod1")
	now := time.Now()

	// A large program waiting past the bound outranks a small fresh one that
	// would otherwise win on footprint.
	seedProgram(t, a, "starved-big", endpoints[0], 500000)
	seedProgram(t, a, "small", endpoints[0], 1000)

	got, err := a.Pick(context.Background(), bandOf(
		makeQueue("starved-big", 1, now.Add(-30*time.Second)),
		makeQueue("small", 1, now),
	))
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "starved-big", got.FlowKey().ID)
}

func TestPick_StarvationGuardDisabled(t *testing.T) {
	cfg := testConfig()
	cfg.HeadWaitStarvationMs = 0
	a := newTestAgent(cfg)
	endpoints := newTestEndpoints("pod1")
	now := time.Now()

	seedProgram(t, a, "starved-big", endpoints[0], 500000)
	seedProgram(t, a, "small", endpoints[0], 1000)

	got, err := a.Pick(context.Background(), bandOf(
		makeQueue("starved-big", 1, now.Add(-10*time.Minute)),
		makeQueue("small", 1, now),
	))
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "small", got.FlowKey().ID, "disabling the guard reproduces upstream starvation")
}

func TestPick_SkipsEmptyQueues(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")
	now := time.Now()

	seedProgram(t, a, "active", endpoints[0], 1000)

	got, err := a.Pick(context.Background(), bandOf(
		makeQueue("empty", 0, now),
		makeQueue("active", 1, now),
	))
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "active", got.FlowKey().ID)

	got, err = a.Pick(context.Background(), bandOf(makeQueue("empty", 0, now)))
	require.NoError(t, err)
	assert.Nil(t, got, "no eligible queue")
}

// A pod with no admission room still dispatches turns of admitted programs:
// their footprint is already counted and finishing them frees capacity.
func TestPick_ReasoningBypassesFullPod(t *testing.T) {
	a := newTestAgent(testConfig())
	sched := newTestEndpoints("pod1")
	seedProgram(t, a, "veteran", sched[0], 900) // fills the 900-token admission room
	primeFitView(a, "pod1")

	q := makeQueue("veteran", 1, time.Now())
	picked, err := a.Pick(context.Background(), bandOf(q))
	require.NoError(t, err)
	assert.Same(t, q, picked)
}

func TestPick_NewHeldWhenNoFit(t *testing.T) {
	a := newTestAgent(testConfig())
	sched := newTestEndpoints("pod1")
	seedProgram(t, a, "veteran", sched[0], 900)
	primeFitView(a, "pod1")

	q := makeQueueWithBytes("newbie", 1, time.Now(), 400) // estimates 100 tokens
	picked, err := a.Pick(context.Background(), bandOf(q))
	require.NoError(t, err)
	assert.Nil(t, picked, "a new program that does not fit any pod is held")
}

func TestPick_NewAdmissionReservesRoom(t *testing.T) {
	a := newTestAgent(testConfig())
	sched := newTestEndpoints("pod1")
	seedProgram(t, a, "veteran", sched[0], 500) // 400 tokens of admission room left
	primeFitView(a, "pod1")

	first := makeQueueWithBytes("first", 1, time.Now(), 400) // estimates 100 tokens
	picked, err := a.Pick(context.Background(), bandOf(first))
	require.NoError(t, err)
	require.Same(t, first, picked)

	a.table.mu.Lock()
	_, reserved := a.table.pending["first"]
	a.table.mu.Unlock()
	assert.True(t, reserved, "an admitted new program holds a reservation until PreRequest binds it")

	// 350 estimated tokens no longer fit: 400 room minus the 100 reserved.
	second := makeQueueWithBytes("second", 1, time.Now(), 1400)
	picked, err = a.Pick(context.Background(), bandOf(second))
	require.NoError(t, err)
	assert.Nil(t, picked, "reservations count against admission room until the program binds")
}

func TestPick_ReservationClearedOnBind(t *testing.T) {
	a := newTestAgent(testConfig())
	sched := newTestEndpoints("pod1")
	primeFitView(a, "pod1")

	q := makeQueueWithBytes("newbie", 1, time.Now(), 400)
	picked, err := a.Pick(context.Background(), bandOf(q))
	require.NoError(t, err)
	require.Same(t, q, picked)

	req := newRequest("newbie", 400)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(sched[0])))

	a.table.mu.Lock()
	_, reserved := a.table.pending["newbie"]
	a.table.mu.Unlock()
	assert.False(t, reserved)
}

func TestPick_StarvingNewBypassesFit(t *testing.T) {
	cfg := testConfig()
	cfg.HeadWaitStarvationMs = 100
	a := newTestAgent(cfg)
	sched := newTestEndpoints("pod1")
	seedProgram(t, a, "veteran", sched[0], 900)
	primeFitView(a, "pod1")

	q := makeQueueWithBytes("newbie", 1, time.Now().Add(-time.Second), 400)
	picked, err := a.Pick(context.Background(), bandOf(q))
	require.NoError(t, err)
	assert.Same(t, q, picked, "a head past the starvation bound is force-admitted regardless of fit")
}

func TestPick_NoFitViewFailsOpen(t *testing.T) {
	a := newTestAgent(testConfig())

	q := makeQueueWithBytes("newbie", 1, time.Now(), 400)
	picked, err := a.Pick(context.Background(), bandOf(q))
	require.NoError(t, err)
	assert.Same(t, q, picked, "without a fit view only class ordering applies")
}
