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
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

// newDataEndpoint builds a datalayer endpoint, as the lifecycle source delivers.
func newDataEndpoint() fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Name: "a", Namespace: "default"},
	}, nil)
}

func newObserver(t *testing.T) *Observer {
	t.Helper()
	return newObserverWithConfig(t, DefaultConfig)
}

func newObserverWithConfig(t *testing.T, cfg Config) *Observer {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	observer, err := NewObserver(ctx, "ttft-observer", cfg)
	require.NoError(t, err)
	return observer
}

// recordingRegistrar captures what RegisterDependencies asked for.
type recordingRegistrar struct {
	registrations []fwkdl.PendingRegistration
}

func (r *recordingRegistrar) Register(reg fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, reg)
	return nil
}

func TestLatencyObserverFactory(t *testing.T) {
	handle := fwkplugin.NewEppHandle(context.Background(), nil)

	t.Run("defaults", func(t *testing.T) {
		plugin, err := LatencyObserverFactory("observer", nil, handle)
		require.NoError(t, err)
		p, ok := plugin.(*Observer)
		require.True(t, ok)

		assert.Equal(t, LatencyObserverProducerType, p.TypedName().Type)
		assert.Equal(t, "observer", p.TypedName().Name)
		assert.NotNil(t, p.PluginState)
		// The published key carries this instance's name, so two observers do
		// not collide on the same endpoint.
		assert.Contains(t, p.percentilesDataKey.String(), "observer")
	})

	t.Run("selects a named in-flight producer", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{"inFlightLoadProducerName":"load-b"}`))
		plugin, err := LatencyObserverFactory("observer", params, handle)
		require.NoError(t, err)
		assert.Contains(t, plugin.(*Observer).inFlightLoadDataKey.String(), "load-b")
	})

	t.Run("rejects malformed parameters", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{"inFlightLoadProducerName":5}`))
		_, err := LatencyObserverFactory("observer", params, handle)
		require.Error(t, err)
	})

	t.Run("requires a handle", func(t *testing.T) {
		_, err := LatencyObserverFactory("observer", nil, nil)
		require.Error(t, err)
	})
}

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

func TestExtractEndpointLifecycle(t *testing.T) {
	ctx := context.Background()

	t.Run("add attaches an attribute that reads cold until a snapshot exists", func(t *testing.T) {
		p := newObserver(t)
		endpoint := newDataEndpoint()

		require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{
			Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint,
		}))

		// The attribute is present, but resolves to nothing: no snapshot yet.
		_, ok := endpoint.GetAttributes().Get(p.percentilesDataKey)
		assert.False(t, ok, "an endpoint with no snapshot must read as cold")

		// Publishing makes the same closure resolve, with no further write to
		// the attribute map.
		p.stateFor("default/a").published.Store(&attrlatency.TTFTPercentiles{FloorTTFT: 0.2, Observations: 50, CalibrationThreshold: 10})

		raw, ok := endpoint.GetAttributes().Get(p.percentilesDataKey)
		require.True(t, ok)
		snapshot, ok := raw.(*attrlatency.TTFTPercentiles)
		require.True(t, ok)
		assert.InDelta(t, 0.2, snapshot.Floor(), 1e-9)
	})

	t.Run("the attribute tracks later snapshots", func(t *testing.T) {
		p := newObserver(t)
		endpoint := newDataEndpoint()
		require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint}))

		state := p.stateFor("default/a")
		state.published.Store(&attrlatency.TTFTPercentiles{FloorTTFT: 0.2, Observations: 50, CalibrationThreshold: 10})
		state.published.Store(&attrlatency.TTFTPercentiles{FloorTTFT: 0.4, Observations: 50, CalibrationThreshold: 10})

		raw, ok := endpoint.GetAttributes().Get(p.percentilesDataKey)
		require.True(t, ok)
		assert.InDelta(t, 0.4, raw.(*attrlatency.TTFTPercentiles).Floor(), 1e-9)
	})

	t.Run("delete drops the endpoint's state", func(t *testing.T) {
		p := newObserver(t)
		endpoint := newDataEndpoint()
		require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint}))
		p.record("default/a", 0.5, 3, time.Now())
		require.Len(t, p.snapshotDebugState().Endpoints, 1)

		require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: endpoint}))
		assert.Empty(t, p.snapshotDebugState().Endpoints)
	})

	t.Run("tolerates events without an endpoint", func(t *testing.T) {
		p := newObserver(t)
		require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate}))
		require.NoError(t, p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete}))
	})
}

// Concurrent callers must share one state instance and lose no observations.
// Run under -race to also catch unsynchronised access.
func TestConcurrentRecording(t *testing.T) {
	p := newObserver(t)

	const goroutines, perGoroutine = 16, 50
	var wg sync.WaitGroup
	seen := make([]*endpointState, goroutines)
	for i := range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			seen[i] = p.stateFor("default/a")
			for range perGoroutine {
				p.record("default/a", 0.5, 3, time.Now())
			}
		}()
	}
	wg.Wait()

	for i := 1; i < goroutines; i++ {
		assert.Same(t, seen[0], seen[i], "every caller must get the same state instance")
	}
	state := p.stateFor("default/a")
	state.mu.Lock()
	defer state.mu.Unlock()
	assert.Equal(t, int64(goroutines*perGoroutine), state.observations)
}

// record and publish contend for the same endpoint's lock: requests append while
// the datalayer recomputes. Run under -race; the assertion is that no
// observation is lost and the last published snapshot sees all of them.
func TestConcurrentRecordAndPublish(t *testing.T) {
	p := newObserver(t)
	ctx := context.Background()
	const observations = 500

	done := make(chan struct{})
	go func() {
		defer close(done)
		for range observations {
			p.record("default/a", 0.5, 3, time.Now())
		}
	}()

	for {
		select {
		case <-done:
			p.publish(ctx, "default/a", p.stateFor("default/a"), time.Now())
			snapshot := p.stateFor("default/a").published.Load()
			require.NotNil(t, snapshot)
			assert.Equal(t, int64(observations), snapshot.Observations)
			return
		default:
			p.publish(ctx, "default/a", p.stateFor("default/a"), time.Now())
		}
	}
}

func TestDumpState(t *testing.T) {
	p := newObserver(t)
	p.record("default/busy", 0.5, 3, time.Now())
	p.record("default/busy", 0.6, 4, time.Now())
	p.record("default/quiet", 0.2, 0, time.Now())

	raw, err := p.DumpState()
	require.NoError(t, err)

	var dump observerDebugState
	require.NoError(t, json.Unmarshal(raw, &dump))

	assert.Equal(t, 2, dump.TotalEndpoints)
	assert.False(t, dump.Truncated)
	require.Len(t, dump.Endpoints, 2)

	// Sorted by observation count, busiest first.
	assert.Equal(t, "default/busy", dump.Endpoints[0].Endpoint)
	assert.Equal(t, int64(2), dump.Endpoints[0].Observations)
	assert.InDelta(t, 0.6, dump.Endpoints[0].LastTTFTSeconds, 1e-9)
	assert.Equal(t, int64(4), dump.Endpoints[0].LastInflightAtDispatch)

	assert.Equal(t, "default/quiet", dump.Endpoints[1].Endpoint)
	assert.Equal(t, int64(1), dump.Endpoints[1].Observations)
}
