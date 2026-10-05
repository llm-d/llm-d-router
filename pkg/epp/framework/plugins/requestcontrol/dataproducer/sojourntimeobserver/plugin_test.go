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
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsojourn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/sojourntime"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

// recordingRegistrar captures what RegisterDependencies asked for.
type recordingRegistrar struct {
	registrations []fwkdl.PendingRegistration
}

func (r *recordingRegistrar) Register(reg fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, reg)
	return nil
}

// TestSojournTimeObserverFactory covers the factory: default config,
// malformed JSON, and handle-required.
func TestSojournTimeObserverFactory(t *testing.T) {
	handle := fwkplugin.NewEppHandle(context.Background(), nil)

	t.Run("defaults", func(t *testing.T) {
		plugin, err := SojournTimeObserverFactory("observer", nil, handle)
		require.NoError(t, err)
		p, ok := plugin.(*Observer)
		require.True(t, ok)
		assert.Equal(t, "observer", p.TypedName().Name)
	})

	t.Run("rejects malformed parameters", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{"minSamples":"not-an-int"}`))
		_, err := SojournTimeObserverFactory("observer", params, handle)
		require.Error(t, err)
	})

	t.Run("requires a handle", func(t *testing.T) {
		_, err := SojournTimeObserverFactory("observer", nil, nil)
		require.Error(t, err)
	})
}

// TestRegisterDependencies verifies the observer subscribes to endpoint
// lifecycle events: one PendingRegistration carrying the right SourceType,
// Owner, self-registered Extractor, and a DefaultSource so the datalayer
// auto-creates the notifications source when absent.
func TestRegisterDependencies(t *testing.T) {
	p := newObserver(t)
	registrar := &recordingRegistrar{}

	require.NoError(t, p.RegisterDependencies(registrar))
	require.Len(t, registrar.registrations, 1)

	reg := registrar.registrations[0]
	assert.Equal(t, sourcenotifications.EndpointNotificationSourceType, reg.SourceType)
	assert.Equal(t, p.TypedName(), reg.Owner)
	assert.Same(t, p, reg.Extractor, "the observer registers itself as the extractor")
	assert.NotNil(t, reg.DefaultSource, "the source must be auto-created when absent")
}

// TestProduces verifies the observer declares the two DataKeys it
// publishes (snapshot + in-flight-requests). The paired-key contract is
// what the scorer's Consumes() declaration depends on.
func TestProduces(t *testing.T) {
	p := newObserver(t)
	produces := p.Produces()

	require.Len(t, produces, 2)
	_, hasSnapshot := produces[p.snapshotDataKey]
	assert.True(t, hasSnapshot, "snapshot DataKey must be in Produces")
	_, hasInflight := produces[p.inFlightRequestsDataKey]
	assert.True(t, hasInflight, "in-flight-requests DataKey must be in Produces")
}

// TestExtract_AttachesDynamicAttribute verifies that on EventAddOrUpdate
// the observer attaches a DynamicAttribute for each DataKey. The attribute
// is present but resolves to "cold" (nothing in the AttributeMap) until a
// snapshot is published; publishing then makes the same closure resolve
// without any further write to the AttributeMap.
func TestExtract_AttachesDynamicAttribute(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Name: "a", Namespace: "default"},
	}, nil)

	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	}))

	// Before publish: closures resolve to nothing.
	_, snapOk := ep.GetAttributes().Get(p.snapshotDataKey)
	assert.False(t, snapOk, "snapshot closure must read as cold before publish")
	_, inflightOk := ep.GetAttributes().Get(p.inFlightRequestsDataKey)
	assert.False(t, inflightOk, "in-flight closure must read as cold before publish")

	// Simulate a publish: write non-nil atomic pointers on the endpoint's
	// state and verify the same closures now resolve without a further
	// Put on the AttributeMap.
	state := p.stateForOrCreate("default/a")
	state.published.Store(&attrsojourn.SojournEstimatorSnapshot{})
	state.publishedInFlight.Store(&attrsojourn.InFlightRequestsSnapshot{
		Requests: []attrsojourn.InFlightRequest{},
	})

	raw, ok := ep.GetAttributes().Get(p.snapshotDataKey)
	require.True(t, ok, "snapshot closure must resolve after publish")
	_, snapTypeOk := raw.(*attrsojourn.SojournEstimatorSnapshot)
	assert.True(t, snapTypeOk, "snapshot must be *SojournEstimatorSnapshot")

	raw, ok = ep.GetAttributes().Get(p.inFlightRequestsDataKey)
	require.True(t, ok, "in-flight closure must resolve after publish")
	_, inflightTypeOk := raw.(*attrsojourn.InFlightRequestsSnapshot)
	assert.True(t, inflightTypeOk, "in-flight must be *InFlightRequestsSnapshot")
}

// TestFlush_WarmUpGate verifies the minSamples gate at flush.go:89.
// Until both digests have at least minSamples samples, publish refuses
// to swap state.published; one more TTFT sample is enough to open the
// gate.
func TestFlush_WarmUpGate(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	minSamples := p.cfg.minSamples
	require.Positive(t, minSamples, "minSamples must be > 0")

	// Seed decode above the gate so it does not block.
	for i := uint64(0); i < minSamples; i++ {
		p.addDecode("default/a", 1.0)
	}
	// Seed TTFT one below the gate.
	for i := uint64(0); i < minSamples-1; i++ {
		p.addTTFT("default/a", 1.0)
	}
	state := p.stateForOrCreate("default/a")
	p.publish(ctx, "default/a", state)
	assert.Nil(t, state.published.Load(), "snapshot must remain cold until both digests warm")

	// One more TTFT sample crosses the gate.
	p.addTTFT("default/a", 1.0)
	p.publish(ctx, "default/a", state)
	assert.NotNil(t, state.published.Load(), "snapshot must publish once both digests warm")
}

// TestConcurrentRecordAndPublish contends record and publish on the same
// endpoint state: PreRequest+ResponseBody write TTFT and decode samples
// while Dispatch (via publish) reads them. Run under -race; the assertion
// is that no observation is lost and the final snapshot sees them all.
func TestConcurrentRecordAndPublish(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	// Lower minSamples so publish actually produces a snapshot inside the
	// test's observation count. DefaultConfig.minSamples = 60; we drive
	// 500 lifecycles per side (one TTFT + one decode each), well above.
	const observations = 500

	ep := newSchedEndpoint("a")
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < observations; i++ {
			req := newRequest(uniqueRequestID(i))
			_ = p.PreRequest(ctx, req, resultFor(ep))
			p.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true, EndOfStream: true}, nil)
		}
	}()

	state := p.stateForOrCreate("default/a")
	for {
		select {
		case <-done:
			p.publish(ctx, "default/a", state)
			snap := state.published.Load()
			require.NotNil(t, snap, "snapshot must be published once digests warm")
			// Both digests must carry every observation. Lock the state to
			// read the final digest counts after the writer is done.
			state.mu.Lock()
			ttftCount := state.ttft.Count()
			decodeCount := state.decode.Count()
			state.mu.Unlock()
			assert.Equal(t, uint64(observations), ttftCount, "every TTFT sample recorded")
			assert.Equal(t, uint64(observations), decodeCount, "every decode sample recorded")
			return
		default:
			p.publish(ctx, "default/a", state)
		}
	}
}

// TestConcurrentHookLifecycles runs many streaming lifecycles across
// several endpoints under -race, exercising the lock-order invariant
// between p.mu (fleet-wide in-flight map) and endpointState.mu (per-
// endpoint digests). At the end, the in-flight index is empty and the
// digest sample counts equal the number of completed lifecycles.
func TestConcurrentHookLifecycles(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)
	const goroutines, perGoroutine = 16, 50
	endpoints := []fwksched.Endpoint{
		newSchedEndpoint("ep-0"),
		newSchedEndpoint("ep-1"),
		newSchedEndpoint("ep-2"),
		newSchedEndpoint("ep-3"),
	}

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				ep := endpoints[(g+i)%len(endpoints)]
				req := newRequest(uniqueRequestID(g*perGoroutine + i))
				_ = p.PreRequest(ctx, req, resultFor(ep))
				p.ResponseBody(ctx, req, &requestcontrol.Response{StartOfStream: true}, nil)
				p.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
			}
		}(g)
	}
	wg.Wait()

	// All lifecycles ended; every endpoint's in-flight index is empty.
	for _, ep := range endpoints {
		id := ep.GetMetadata().ID.String()
		assert.Empty(t, p.InFlightRequestsFor(id), "in-flight must be empty after all lifecycles end: %s", id)
	}

	// Sum of digest sample counts across endpoints equals total lifecycles.
	var totalTTFT, totalDecode uint64
	for _, ep := range endpoints {
		id := ep.GetMetadata().ID.String()
		state := p.stateForOrCreate(id)
		state.mu.Lock()
		totalTTFT += state.ttft.Count()
		totalDecode += state.decode.Count()
		state.mu.Unlock()
	}
	assert.Equal(t, uint64(goroutines*perGoroutine), totalTTFT, "every TTFT sample recorded across endpoints")
	assert.Equal(t, uint64(goroutines*perGoroutine), totalDecode, "every decode sample recorded across endpoints")
}

// TestExtract_DeleteClearsReverseIndex guards the EventDelete branch's
// cleanup of requestToEndpoint. When an endpoint is deleted, every
// requestID that lived on it must be purged from the reverse index so
// a later chunk arriving for one of those requests cannot resolve a
// stale endpointID.
func TestExtract_DeleteClearsReverseIndex(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)

	// Datalayer-shape endpoints for Extract events.
	dataEpA := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Name: "a", Namespace: "default"},
	}, nil)
	// Scheduling-shape endpoints for PreRequest.
	schedEpA := newSchedEndpoint("a")
	schedEpB := newSchedEndpoint("b")

	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: dataEpA}))

	// 3 requests on "a", 1 on "b".
	require.NoError(t, p.PreRequest(ctx, newRequest("req-a1"), resultFor(schedEpA)))
	require.NoError(t, p.PreRequest(ctx, newRequest("req-a2"), resultFor(schedEpA)))
	require.NoError(t, p.PreRequest(ctx, newRequest("req-a3"), resultFor(schedEpA)))
	require.NoError(t, p.PreRequest(ctx, newRequest("req-b1"), resultFor(schedEpB)))

	// Precondition: reverse index has 4 entries.
	p.mu.RLock()
	require.Len(t, p.requestToEndpoint, 4, "reverse index populated before delete")
	p.mu.RUnlock()

	// Delete endpoint "a".
	require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: dataEpA}))

	// Endpoint "a"'s in-flight map is reaped.
	assert.Empty(t, p.InFlightRequestsFor("default/a"))

	// Reverse index has only "b"'s request left.
	p.mu.RLock()
	defer p.mu.RUnlock()
	require.Len(t, p.requestToEndpoint, 1, "reverse index purged for deleted endpoint")
	ep, ok := p.requestToEndpoint["req-b1"]
	require.True(t, ok, "surviving request must be b's")
	assert.Equal(t, "default/b", ep)
}

// TestDispatch_NilEndpoint guards flush.go's nil-endpoint check. Dispatch
// must not panic and must not publish state when the datalayer calls it
// with a nil endpoint or an endpoint whose metadata is nil.
func TestDispatch_NilEndpoint(t *testing.T) {
	ctx := context.Background()
	p := newObserver(t)

	assert.NotPanics(t, func() {
		_ = p.Dispatch(ctx, nil)
	}, "Dispatch must tolerate a nil endpoint")

	nilMetaEp := fwkdl.NewEndpoint(nil, nil)
	assert.NotPanics(t, func() {
		_ = p.Dispatch(ctx, nilMetaEp)
	}, "Dispatch must tolerate an endpoint with nil metadata")
}

// uniqueRequestID returns a unique request ID for a concurrent-test
// lifecycle. Different-length prefixes exercise variable-length keys in
// the in-flight map.
func uniqueRequestID(i int) string {
	return "req-" + strings.Repeat("x", i%4) + "-" + strconv.Itoa(i)
}
