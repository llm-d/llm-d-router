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

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// seed runs one 400-byte turn for the session on pod-a and completes it with
// the given committed total.
func seed(t *testing.T, a *ThunderAgent, id string, committed int) {
	t.Helper()
	ep := schedEndpoint("pod-a", 0, 0)
	req := startTurn(t, a, id, ep, 400)
	a.ResponseBody(context.Background(), req, endOfStream(committed, 100), nil)
}

// A new session is admitted only when a pod has room for its estimate.
func TestNewSessionHeldWhenNoRoom(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", 900) // 900 of 1000 used
	primeFitView(a, dlEndpoint("pod-a"))

	big := makeQueue("s2", time.Now(), 2000) // estimate 500 > room 100
	require.Nil(t, pick(t, a, big))

	small := makeQueue("s3", time.Now(), 320) // estimate 80 <= room 100
	require.Equal(t, small, pick(t, a, small))
}

// A reasoning session's next turn dispatches even when nothing fits.
func TestReasoningBypassesFit(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", 950)
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s1", time.Now(), 4000)
	require.Equal(t, q, pick(t, a, q))
}

// The pause sweep pauses the smallest idle session first and stops at the
// ceiling.
func TestPauseSweepPausesSmallestIdleFirst(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", 400)
	seed(t, a, "s2", 300)
	seed(t, a, "s3", 500) // 1200 > 1000

	primeFitView(a, dlEndpoint("pod-a"))
	require.True(t, isPaused(a, "s2"))
	require.False(t, isPaused(a, "s1"))
	require.False(t, isPaused(a, "s3"))
	require.Equal(t, float64(900), endpointTokens(a, "default/pod-a"))
}

// The pause sweep runs on the first Saturation call and then at most once
// per pauseSweepSeconds, for all endpoints together.
func TestPauseSweepInterval(t *testing.T) {
	cfg := testConfig()
	cfg.PauseSweepSeconds = 5
	a := newTestAgent(cfg)
	seed(t, a, "s1", 400)
	seed(t, a, "s2", 300)
	primeFitView(a, dlEndpoint("pod-a")) // sweeps; 700 fits, nothing paused

	seed(t, a, "s3", 500) // 1200 > 1000
	a.mgr.mu.Lock()
	a.mgr.endpoints["default/pod-a"].updatedAt = time.Now().Add(-2 * endpointStaleAfter)
	a.mgr.mu.Unlock()
	primeFitView(a, dlEndpoint("pod-a"))
	require.False(t, isPaused(a, "s2"), "no sweep within the interval")
	s, _ := sessionOf(a, "s1")
	require.NotNil(t, s.endpoint, "the ledger is refreshed between sweeps")

	a.mgr.mu.Lock()
	a.mgr.lastSweep = a.mgr.lastSweep.Add(-a.pauseSweep)
	a.mgr.mu.Unlock()
	primeFitView(a, dlEndpoint("pod-a"))
	require.True(t, isPaused(a, "s2"))
}

// A paused session resumes only on its own pod, even when another pod has
// room (strict origin affinity); it goes ahead of new sessions once its pod
// has room again.
func TestPausedResumesOnlyOnItsOwnPod(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", 600)
	seed(t, a, "s2", 500) // pod-a over: sweep pauses s2 (smallest)
	primeFitView(a, dlEndpoint("pod-a"), dlEndpoint("pod-b"))
	require.True(t, isPaused(a, "s2"))

	// pod-b is empty, but s2 waits for pod-a: room on a is 400 < 500.
	q2 := makeQueue("s2", time.Now(), 2000)
	require.Nil(t, pick(t, a, q2))

	// s1 idles past the TTL and is evicted (the only release path); pod-a
	// has room again; s2 beats a fitting newcomer.
	a.mgr.mu.Lock()
	a.mgr.sessions["s1"].lastActivity = time.Now().Add(-2 * a.mgr.ttl)
	a.mgr.mu.Unlock()
	forceMaintenance(a)
	primeFitView(a, dlEndpoint("pod-a"), dlEndpoint("pod-b"))

	qNew := makeQueue("s9", time.Now().Add(-time.Minute), 320)
	require.Equal(t, q2, pick(t, a, q2, qNew))

	// PreRequest confirms the resume.
	req := newRequest("s2", 2000)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(schedEndpoint("pod-a", 0, 0))))
	require.False(t, isPaused(a, "s2"))
}

// An admission reservation blocks a second admit into the same room until
// it is confirmed or expires.
func TestReservationPreventsDoubleAdmit(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", 900) // room 100
	primeFitView(a, dlEndpoint("pod-a"))

	q2 := makeQueue("s2", time.Now(), 320) // estimate 80
	q3 := makeQueue("s3", time.Now(), 320)
	require.Equal(t, q2, pick(t, a, q2, q3))
	// 80 of the 100 are reserved for s2; s3 no longer fits.
	require.Nil(t, pick(t, a, q3))

	// Maintenance before the reservation expires keeps it: the idle TTL
	// does not apply to a session that has never dispatched.
	forceMaintenance(a)
	primeFitView(a, dlEndpoint("pod-a"))
	_, ok := sessionOf(a, "s2")
	require.True(t, ok)
	require.Nil(t, pick(t, a, q3))

	// The reservation expires unconfirmed; the room frees and s3 fits.
	a.mgr.mu.Lock()
	a.mgr.sessions["s2"].reservedUntil = time.Now().Add(-time.Second)
	a.mgr.mu.Unlock()
	require.Equal(t, q3, pick(t, a, q3))
}

// The starvation backstop force-admits a head past the deadline, fit or not.
func TestStarvationForcesAdmission(t *testing.T) {
	cfg := testConfig()
	cfg.HeadWaitStarvationMs = 1000
	a := newTestAgent(cfg)
	seed(t, a, "s1", 1000)
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s2", time.Now().Add(-2*time.Second), 4000)
	require.Equal(t, q, pick(t, a, q))
}

// When a pod stops being reported, its sessions re-enter as new.
func TestPodVanishUnbinds(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", 300)
	a.mgr.mu.Lock()
	a.mgr.endpoints["default/pod-a"].updatedAt = time.Now().Add(-2 * endpointStaleAfter)
	a.mgr.mu.Unlock()
	primeFitView(a, dlEndpoint("pod-b"))

	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Nil(t, s.endpoint)
}

// Without a fit view (the plugin not wired as the saturation detector) the
// gate fails open and only class ordering remains.
func TestFailsOpenWithoutFitView(t *testing.T) {
	a := newTestAgent(testConfig())
	q := makeQueue("s1", time.Now(), 4000000)
	require.Equal(t, q, pick(t, a, q))
}

func pick(t *testing.T, a *ThunderAgent, queues ...fwkfc.FlowQueueAccessor) fwkfc.FlowQueueAccessor {
	t.Helper()
	got, err := a.Pick(context.Background(), bandOf(queues...))
	require.NoError(t, err)
	return got
}
