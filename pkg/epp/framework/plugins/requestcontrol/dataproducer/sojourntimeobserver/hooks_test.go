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

package sojourntimeobserver

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const testProfile = "default"

func newObserver(t *testing.T) *Observer {
	t.Helper()
	observer, err := NewObserver("sojourn-observer", DefaultConfig)
	require.NoError(t, err)
	return observer
}

func newSchedEndpoint(id string) fwksched.Endpoint {
	attr := fwkdl.NewAttributes()
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: id, Namespace: "default"}}
	return fwksched.NewEndpoint(meta, nil, attr)
}

func newRequest(id string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{RequestID: id}
}

func resultFor(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: testProfile,
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			testProfile: {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

// TestPreRequest_RecordsDispatch verifies PreRequest inserts an in-flight
// entry for the primary target endpoint.
func TestPreRequest_RecordsDispatch(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	ep := newSchedEndpoint("a")

	require.NoError(t, p.PreRequest(ctx, newRequest("req-1"), resultFor(ep)))

	entries := p.InFlightRequestsFor("default/a")
	require.Len(t, entries, 1)
	assert.False(t, entries[0].DispatchedAt.IsZero(), "dispatchedAt must be stamped")
	assert.True(t, entries[0].FirstChunkAt.IsZero(), "firstChunkAt is zero before first chunk arrives")
}

// TestPreRequest_ToleratesNilInputs verifies PreRequest is safe against
// missing inputs — failing to record an observation is never a reason to
// reject a request.
func TestPreRequest_ToleratesNilInputs(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)

	// nil request
	require.NoError(t, p.PreRequest(ctx, nil, resultFor(newSchedEndpoint("a"))))
	// nil result
	require.NoError(t, p.PreRequest(ctx, newRequest("req-1"), nil))
	// empty request ID
	require.NoError(t, p.PreRequest(ctx, &fwksched.InferenceRequest{}, resultFor(newSchedEndpoint("a"))))
	// no target endpoint
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: testProfile,
		ProfileResults:     map[string]*fwksched.ProfileRunResult{testProfile: {}},
	}
	require.NoError(t, p.PreRequest(ctx, newRequest("req-1"), result))

	assert.Empty(t, p.InFlightRequestsFor("default/a"))
}

// TestResponseBody_StreamingLifecycle exercises the streaming path:
// PreRequest -> ResponseBody(StartOfStream=true) -> ResponseBody(EndOfStream=true).
// After the terminal chunk, the endpoint's in-flight index is empty, and both
// digests have accumulated one sample each.
func TestResponseBody_StreamingLifecycle(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	ep := newSchedEndpoint("a")

	require.NoError(t, p.PreRequest(ctx, newRequest("req-1"), resultFor(ep)))
	// Small sleep so ttft is measurably > 0.
	time.Sleep(2 * time.Millisecond)
	p.ResponseBody(ctx, newRequest("req-1"), &requestcontrol.Response{StartOfStream: true}, nil)
	time.Sleep(2 * time.Millisecond)
	p.ResponseBody(ctx, newRequest("req-1"), &requestcontrol.Response{EndOfStream: true}, nil)

	// After end-of-stream, the request is cleared from the in-flight index.
	assert.Empty(t, p.InFlightRequestsFor("default/a"), "in-flight index cleared after end-of-stream")

	// Both digests each carry exactly one sample.
	state := p.stateForOrCreate("default/a")
	state.mu.Lock()
	defer state.mu.Unlock()
	assert.Equal(t, uint64(1), state.ttft.Count(), "one TTFT sample")
	assert.Equal(t, uint64(1), state.decode.Count(), "one decode sample")
}

// TestResponseBody_NonStreamingLifecycle exercises the single-chunk path where
// StartOfStream and EndOfStream both arrive on the same event. TTFT sample
// non-zero; decode sample is 0 (or a small floor near 0).
func TestResponseBody_NonStreamingLifecycle(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	ep := newSchedEndpoint("a")

	require.NoError(t, p.PreRequest(ctx, newRequest("req-1"), resultFor(ep)))
	time.Sleep(2 * time.Millisecond)
	p.ResponseBody(ctx, newRequest("req-1"), &requestcontrol.Response{StartOfStream: true, EndOfStream: true}, nil)

	assert.Empty(t, p.InFlightRequestsFor("default/a"))

	state := p.stateForOrCreate("default/a")
	state.mu.Lock()
	defer state.mu.Unlock()
	assert.Equal(t, uint64(1), state.ttft.Count(), "one TTFT sample")
	assert.Equal(t, uint64(1), state.decode.Count(), "one decode sample; digest correctly records the zero")
}

// TestInFlightRequestsFor_MultipleEndpoints verifies the fleet-wide index
// filters correctly by endpoint.
func TestInFlightRequestsFor_MultipleEndpoints(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	epA := newSchedEndpoint("a")
	epB := newSchedEndpoint("b")

	require.NoError(t, p.PreRequest(ctx, newRequest("req-a1"), resultFor(epA)))
	require.NoError(t, p.PreRequest(ctx, newRequest("req-a2"), resultFor(epA)))
	require.NoError(t, p.PreRequest(ctx, newRequest("req-b1"), resultFor(epB)))

	assert.Len(t, p.InFlightRequestsFor("default/a"), 2)
	assert.Len(t, p.InFlightRequestsFor("default/b"), 1)
	assert.Empty(t, p.InFlightRequestsFor("default/nowhere"))
}

// TestInFlightRequestsFor_FirstChunkStamped verifies that after the first
// chunk arrives, the InFlightRequest entry for that request has FirstChunkAt
// set, while other in-flight requests on the same endpoint still show it as
// zero.
func TestInFlightRequestsFor_FirstChunkStamped(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	ep := newSchedEndpoint("a")

	require.NoError(t, p.PreRequest(ctx, newRequest("req-1"), resultFor(ep)))
	require.NoError(t, p.PreRequest(ctx, newRequest("req-2"), resultFor(ep)))
	// First chunk for req-1 only.
	p.ResponseBody(ctx, newRequest("req-1"), &requestcontrol.Response{StartOfStream: true}, nil)

	entries := p.InFlightRequestsFor("default/a")
	require.Len(t, entries, 2)
	stampedCount := 0
	for _, e := range entries {
		if !e.FirstChunkAt.IsZero() {
			stampedCount++
		}
	}
	assert.Equal(t, 1, stampedCount, "exactly one entry should have FirstChunkAt stamped")
}

// TestResponseBody_TerminalWithoutStart handles the edge case where an
// EndOfStream chunk arrives without a prior StartOfStream. The observer must
// not crash and must not emit a nonsense decode sample.
func TestResponseBody_TerminalWithoutStart(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	ep := newSchedEndpoint("a")

	require.NoError(t, p.PreRequest(ctx, newRequest("req-1"), resultFor(ep)))
	// EndOfStream without prior StartOfStream.
	p.ResponseBody(ctx, newRequest("req-1"), &requestcontrol.Response{EndOfStream: true}, nil)

	// The observer clears the in-flight entry (defense against orphans).
	assert.Empty(t, p.InFlightRequestsFor("default/a"))

	// No decode sample should have been emitted (firstChunkAt was zero).
	state := p.stateForOrCreate("default/a")
	state.mu.Lock()
	defer state.mu.Unlock()
	assert.Equal(t, uint64(0), state.decode.Count(), "no decode sample without a first chunk")
}

// TestExtract_DeleteClearsInflight verifies endpoint delete purges both the
// digest state and the in-flight index for that endpoint.
func TestExtract_DeleteClearsInflight(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)

	// Datalayer-shape endpoint for Extract's event, plus a scheduling-shape
	// endpoint of the same ID for PreRequest.
	dataEp := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Name: "a", Namespace: "default"},
	}, nil)
	schedEp := newSchedEndpoint("a")

	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: dataEp}))
	require.NoError(t, p.PreRequest(ctx, newRequest("req-1"), resultFor(schedEp)))
	require.Len(t, p.InFlightRequestsFor("default/a"), 1)

	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: dataEp}))
	assert.Empty(t, p.InFlightRequestsFor("default/a"), "delete must purge the in-flight index")
}
