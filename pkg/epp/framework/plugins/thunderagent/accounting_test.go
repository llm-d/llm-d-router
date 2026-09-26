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

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func TestInflightAndCommittedAccounting(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1", "pod2")
	pod1 := endpoints[0].GetMetadata().ID.String()

	// Program A sends a request routed to pod1. Body of 2000 bytes at the
	// initial 4.0 bytes-per-token ratio estimates 500 tokens in flight.
	req := newRequest("program-a", 2000)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[0])))

	a.table.mu.Lock()
	loads := a.table.podLoads(time.Now())
	dispatched := a.table.programs["program-a"].dispatchCount
	a.table.mu.Unlock()
	assert.InDelta(t, 500.0, loads[pod1].tokens, 0.0001)
	assert.Equal(t, int64(1), dispatched)

	// The response reports 800 total tokens: the estimate is removed and the
	// committed footprint replaces it.
	a.ResponseBody(context.Background(), req, endOfStream(800, 500), nil)

	a.table.mu.Lock()
	loads = a.table.podLoads(time.Now())
	a.table.mu.Unlock()
	assert.InDelta(t, 800.0, loads[pod1].tokens, 0.0001)

	// The next request of the same program replaces, not accumulates, the
	// committed footprint.
	req2 := newRequest("program-a", 4000)
	require.NoError(t, a.PreRequest(context.Background(), req2, schedulingResultFor(endpoints[0])))
	a.ResponseBody(context.Background(), req2, endOfStream(900, 850), nil)

	a.table.mu.Lock()
	loads = a.table.podLoads(time.Now())
	a.table.mu.Unlock()
	assert.InDelta(t, 900.0, loads[pod1].tokens, 0.0001)
}

func TestAbortedRequestReleasesEstimate(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	req := newRequest("program-a", 2000)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[0])))

	// Abort: end of stream with no usage. The 500-token estimate becomes the
	// committed footprint because nothing better is known.
	a.ResponseBody(context.Background(), req, &fwkrc.Response{EndOfStream: true}, nil)

	a.table.mu.Lock()
	st := a.table.programs["program-a"]
	a.table.mu.Unlock()
	require.NotNil(t, st)
	assert.Equal(t, int64(0), st.inflightTokens)
	assert.Equal(t, int64(500), st.committedTokens)
}

func TestActingDecay(t *testing.T) {
	cfg := testConfig()
	cfg.ActingHalfLifeSeconds = 10
	a := newTestAgent(cfg)
	endpoints := newTestEndpoints("pod1")
	pod1 := endpoints[0].GetMetadata().ID.String()

	seedProgram(t, a, "program-a", endpoints[0], 800)

	// One half-life since the last response: the 800 committed tokens count as 400.
	a.table.mu.Lock()
	a.table.programs["program-a"].lastResponseAt = time.Now().Add(-10 * time.Second)
	loads := a.table.podLoads(time.Now())
	a.table.mu.Unlock()
	assert.InDelta(t, 400.0, loads[pod1].tokens, 10.0)
}

func TestAnonymousRequestsAreNotTracked(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	for _, id := range []string{"", metadata.DefaultFairnessID} {
		req := newRequest(id, 2000)
		require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[0])))
	}

	a.table.mu.Lock()
	tracked := len(a.table.programs)
	a.table.mu.Unlock()
	assert.Equal(t, 0, tracked)
}

func TestEstimatorRefinement(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	// 8000 bytes for 1000 prompt tokens is an 8.0 sample; with momentum 0.8
	// the ratio moves from 4.0 to 0.8*4.0 + 0.2*8.0 = 4.8.
	req := newRequest("program-a", 8000)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[0])))
	a.ResponseBody(context.Background(), req, endOfStream(1200, 1000), nil)

	a.table.mu.Lock()
	ratio := a.table.bytesPerToken
	a.table.mu.Unlock()
	assert.InDelta(t, 4.8, ratio, 0.0001)
}

func TestRebindMovesProgram(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1", "pod2")

	seedProgram(t, a, "program-a", endpoints[0], 300)

	// The next turn is scheduled onto pod2 (e.g. pod1 was filtered out): the
	// binding and the whole footprint follow.
	req := newRequest("program-a", 0)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[1])))

	a.table.mu.Lock()
	loads := a.table.podLoads(time.Now())
	a.table.mu.Unlock()
	assert.Zero(t, loads[endpoints[0].GetMetadata().ID.String()].tokens)
	assert.InDelta(t, 300.0, loads[endpoints[1].GetMetadata().ID.String()].tokens, 0.0001)
}

func TestEviction(t *testing.T) {
	cfg := testConfig()
	cfg.EvictionTTLSeconds = 1
	a := newTestAgent(cfg)
	endpoints := newTestEndpoints("pod1")

	// Idle program: evictable once past the TTL.
	seedProgram(t, a, "program-idle", endpoints[0], 50)

	// In-flight program: kept regardless of age.
	reqBusy := newRequest("program-busy", 100)
	require.NoError(t, a.PreRequest(context.Background(), reqBusy, schedulingResultFor(endpoints[0])))

	a.evictIdle(time.Now().Add(2 * time.Second))

	a.table.mu.Lock()
	_, idleKept := a.table.programs["program-idle"]
	_, busyKept := a.table.programs["program-busy"]
	a.table.mu.Unlock()
	assert.False(t, idleKept, "idle program past TTL should be evicted")
	assert.True(t, busyKept, "in-flight program should be kept")
}
