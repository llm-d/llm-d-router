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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

func TestSessionFinalReleasesProgram(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	seedProgram(t, a, "program-a", endpoints[0], 800)

	// The session's last turn carries the final marker; when its response
	// completes, the program's capacity is freed immediately instead of
	// waiting for the eviction TTL.
	req := newRequest("program-a", 0)
	req.Headers = map[string]string{"x-session-final": "true"}
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[0])))

	a.table.mu.Lock()
	_, tracked := a.table.programs["program-a"]
	a.table.mu.Unlock()
	assert.True(t, tracked, "the final turn itself is still accounted")

	a.ResponseBody(context.Background(), req, endOfStream(900, 800), nil)

	a.table.mu.Lock()
	_, tracked = a.table.programs["program-a"]
	a.table.mu.Unlock()
	assert.False(t, tracked, "session-final release frees the program at end of stream")
}

func TestSessionFinalOnFirstTurn(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	// A standalone release request (final header on a session's only turn)
	// flows through as a tiny turn and leaves no state behind.
	req := newRequest("program-a", 10)
	req.Headers = map[string]string{"x-session-final": "1"}
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[0])))
	a.ResponseBody(context.Background(), req, endOfStream(5, 2), nil)

	a.table.mu.Lock()
	tracked := len(a.table.programs)
	a.table.mu.Unlock()
	assert.Equal(t, 0, tracked)
}

func TestSessionFinalUnknownProgramIsNoOp(t *testing.T) {
	a := newTestAgent(testConfig())

	// A final response for a program that was never scheduled here must not
	// create state or panic.
	req := newRequest("never-seen", 0)
	req.Headers = map[string]string{"x-session-final": "true"}
	a.ResponseBody(context.Background(), req, &fwkrc.Response{EndOfStream: true}, nil)

	a.table.mu.Lock()
	tracked := len(a.table.programs)
	a.table.mu.Unlock()
	assert.Equal(t, 0, tracked)
}

func TestSessionFinalWaitsForInflightTurns(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	final := newRequest("program-a", 100)
	final.Headers = map[string]string{"x-session-final": "true"}
	require.NoError(t, a.PreRequest(context.Background(), final, schedulingResultFor(endpoints[0])))

	// A concurrent turn of the same program is still in flight when the final
	// turn completes: release waits so the in-flight bookkeeping still finds
	// its state.
	other := newRequest("program-a", 100)
	require.NoError(t, a.PreRequest(context.Background(), other, schedulingResultFor(endpoints[0])))

	a.ResponseBody(context.Background(), final, endOfStream(50, 25), nil)
	a.table.mu.Lock()
	_, tracked := a.table.programs["program-a"]
	a.table.mu.Unlock()
	assert.True(t, tracked, "release is deferred while another turn is in flight")

	a.ResponseBody(context.Background(), other, endOfStream(60, 25), nil)
	a.table.mu.Lock()
	_, tracked = a.table.programs["program-a"]
	a.table.mu.Unlock()
	assert.False(t, tracked, "the last completed turn performs the release")
}

func TestParentSessionRecorded(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	req := newRequest("subagent-1", 0)
	req.Headers = map[string]string{"x-parent-session-id": "root-42"}
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[0])))

	a.table.mu.Lock()
	parent := a.table.programs["subagent-1"].parentID
	a.table.mu.Unlock()
	assert.Equal(t, "root-42", parent)
}
