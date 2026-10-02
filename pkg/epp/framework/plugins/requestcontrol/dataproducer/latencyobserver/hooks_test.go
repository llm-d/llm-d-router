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

package latencyobserver

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const testProfile = "default"

// newSchedEndpoint builds a scheduling endpoint named id, optionally carrying a
// live in-flight load attribute.
func newSchedEndpoint(id string, inflight *int64) fwksched.Endpoint {
	attr := fwkdl.NewAttributes()
	if inflight != nil {
		attr.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: *inflight})
	}
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: id, Namespace: "default"}}
	return fwksched.NewEndpoint(meta, nil, attr)
}

func inflightPtr(v int64) *int64 { return &v }

func newRequest() *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{RequestID: "req-1"}
}

func resultFor(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: testProfile,
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			testProfile: {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func TestProduce(t *testing.T) {
	ctx := context.Background()

	t.Run("captures every candidate's live in-flight load", func(t *testing.T) {
		p := newObserver(t)
		endpoints := []fwksched.Endpoint{newSchedEndpoint("a", inflightPtr(7)), newSchedEndpoint("b", nil)}

		require.NoError(t, p.Produce(ctx, newRequest(), endpoints))

		assert.Equal(t, int64(7), p.pinnedInflight("req-1", "default/a"))
		assert.Zero(t, p.pinnedInflight("req-1", "default/b"), "a missing attribute reads as zero")
	})

	// The reading rides in PluginState, not on the endpoint, so it survives a
	// profile handler that rebuilds endpoints with empty attribute maps -- which
	// data-parallel and disaggregated both do.
	t.Run("the pin survives a rebuilt endpoint", func(t *testing.T) {
		p := newObserver(t)
		require.NoError(t, p.Produce(ctx, newRequest(), []fwksched.Endpoint{newSchedEndpoint("a", inflightPtr(5))}))

		rebuilt := newSchedEndpoint("a", nil) // same ID, no attributes
		require.NoError(t, p.PreRequest(ctx, newRequest(), resultFor(rebuilt)))

		dispatch, err := fwkplugin.ReadPluginStateKey[*dispatchInfo](p.PluginState, "req-1", dispatchStateKey)
		require.NoError(t, err)
		assert.Equal(t, int64(5), dispatch.inflight)
	})

	t.Run("tolerates nil endpoints and requests", func(t *testing.T) {
		p := newObserver(t)
		require.NoError(t, p.Produce(ctx, newRequest(), []fwksched.Endpoint{nil}))
		require.NoError(t, p.Produce(ctx, nil, []fwksched.Endpoint{newSchedEndpoint("a", inflightPtr(1))}))
	})
}

func TestPreRequest(t *testing.T) {
	ctx := context.Background()

	t.Run("records only the winning endpoint", func(t *testing.T) {
		p := newObserver(t)
		winner, loser := newSchedEndpoint("winner", inflightPtr(4)), newSchedEndpoint("loser", inflightPtr(11))

		require.NoError(t, p.Produce(ctx, newRequest(), []fwksched.Endpoint{winner, loser}))
		require.NoError(t, p.PreRequest(ctx, newRequest(), resultFor(winner)))

		dispatch, err := fwkplugin.ReadPluginStateKey[*dispatchInfo](p.PluginState, "req-1", dispatchStateKey)
		require.NoError(t, err)
		assert.Equal(t, "default/winner", dispatch.endpointID)
		assert.Equal(t, int64(4), dispatch.inflight)
		assert.False(t, dispatch.dispatchedAt.IsZero())
	})

	t.Run("writes nothing when there is no usable dispatch", func(t *testing.T) {
		tests := map[string]*fwksched.SchedulingResult{
			"nil result":                 nil,
			"primary picked no endpoint": {PrimaryProfileName: testProfile, ProfileResults: map[string]*fwksched.ProfileRunResult{testProfile: {}}},
		}
		for name, result := range tests {
			t.Run(name, func(t *testing.T) {
				p := newObserver(t)
				// Never an error: a rejected request is worse than a lost
				// observation, so PreRequest reports nothing upstream.
				require.NoError(t, p.PreRequest(ctx, newRequest(), result))

				_, err := fwkplugin.ReadPluginStateKey[*dispatchInfo](p.PluginState, "req-1", dispatchStateKey)
				assert.Error(t, err)
			})
		}
	})
}

func TestResponseBody(t *testing.T) {
	ctx := context.Background()

	// dispatch primes a request as if Produce and PreRequest had run.
	dispatch := func(p *Observer, at time.Time) {
		p.PluginState.Write("req-1", dispatchStateKey, &dispatchInfo{
			endpointID: testEndpointID, inflight: 6, dispatchedAt: at,
		})
	}

	t.Run("the first chunk of a stream becomes a TTFT observation", func(t *testing.T) {
		p := newObserver(t)
		dispatch(p, time.Now().Add(-250*time.Millisecond))

		p.ResponseBody(ctx, newRequest(), &requestcontrol.Response{StartOfStream: true}, nil)

		state := p.stateFor(testEndpointID)
		state.mu.Lock()
		defer state.mu.Unlock()
		assert.Equal(t, int64(1), state.observations)
		assert.Equal(t, int64(6), state.lastInflightAtDispatch)
		assert.InDelta(t, 0.25, state.lastTTFT, 0.15)
	})

	// StartOfStream and EndOfStream together: the whole body arrived at once, so
	// its latency is end-to-end, not a TTFT. Recording it would push both the
	// floor and the load anchors far above the truth.
	t.Run("a single-chunk response is discarded", func(t *testing.T) {
		p := newObserver(t)
		dispatch(p, time.Now().Add(-2*time.Second))

		p.ResponseBody(ctx, newRequest(), &requestcontrol.Response{StartOfStream: true, EndOfStream: true}, nil)

		state := p.stateFor(testEndpointID)
		state.mu.Lock()
		defer state.mu.Unlock()
		assert.Zero(t, state.observations, "e2e latency must not be recorded as a TTFT")
	})

	t.Run("middle chunks are ignored and end of stream releases the record", func(t *testing.T) {
		p := newObserver(t)
		dispatch(p, time.Now())

		p.ResponseBody(ctx, newRequest(), &requestcontrol.Response{}, nil)
		assert.Zero(t, p.stateFor(testEndpointID).observations)

		p.ResponseBody(ctx, newRequest(), &requestcontrol.Response{StartOfStream: true}, nil)
		p.ResponseBody(ctx, newRequest(), &requestcontrol.Response{EndOfStream: true}, nil)

		_, err := fwkplugin.ReadPluginStateKey[*dispatchInfo](p.PluginState, "req-1", dispatchStateKey)
		assert.Error(t, err, "dispatch record must not outlive the request")
	})
}
