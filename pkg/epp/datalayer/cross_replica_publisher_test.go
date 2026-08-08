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

package datalayer

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

type fakeCloneable struct{ id string }

func (f fakeCloneable) Clone() fwkdl.Cloneable { return f }

type setCall struct {
	key        fwkdl.StateKey
	endpointID string
	value      any
}

type fakeSyncer struct {
	mu   sync.Mutex
	sets []setCall
}

func (s *fakeSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-syncer", Name: "fake-syncer"}
}

func (s *fakeSyncer) Set(_ context.Context, key fwkdl.StateKey, endpointID string, value any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sets = append(s.sets, setCall{key: key, endpointID: endpointID, value: value})
	return nil
}

func (s *fakeSyncer) Get(context.Context, fwkdl.StateKey, string, func([]any) any) (any, bool, error) {
	return nil, false, nil
}

func (s *fakeSyncer) Delete(context.Context, fwkdl.StateKey, string) error { return nil }

// fakeContributor is a Plugin + CrossReplicaContributor whose supplied value
// echoes the endpoint ID, so tests can assert routing to the right key.
type fakeContributor struct {
	key          fwkdl.StateKey
	syncDisabled bool
}

func (c fakeContributor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-contributor", Name: string(c.key)}
}

func (c fakeContributor) CrossReplicaState() fwkdl.CrossReplicaSpec {
	return fwkdl.CrossReplicaSpec{
		StateKey:     c.key,
		SyncDisabled: c.syncDisabled,
		Supply: func(id string) func() fwkdl.Cloneable {
			return func() fwkdl.Cloneable { return fakeCloneable{id: id} }
		},
	}
}

func testEndpoint(namespace, name string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Namespace: namespace, Name: name},
	}, nil)
}

func extractorMapWith(contributors ...fwkdl.CrossReplicaContributor) *extractorMap {
	em := newExtractorMap()
	for _, c := range contributors {
		em.Append("src", c.(fwkplugin.Plugin))
	}
	return em
}

func TestCrossReplicaPublisher_PublishesForEndpoint(t *testing.T) {
	syncer := &fakeSyncer{}
	pub := &crossReplicaPublisher{
		syncer:       syncer,
		contributors: []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
	}

	require.NoError(t, pub.Dispatch(context.Background(), testEndpoint("ns", "ep-a")))

	require.Len(t, syncer.sets, 1)
	assert.Equal(t, fwkdl.StateKey("inflight:test"), syncer.sets[0].key)
	assert.Equal(t, "ns/ep-a", syncer.sets[0].endpointID)
	assert.Equal(t, fakeCloneable{id: "ns/ep-a"}, syncer.sets[0].value)
}

func TestCrossReplicaPublisher_SkipsSyncDisabled(t *testing.T) {
	em := extractorMapWith(
		fakeContributor{key: "enabled"},
		fakeContributor{key: "disabled", syncDisabled: true},
	)

	pub := newCrossReplicaPublisher(&fakeSyncer{}, em, 0)
	require.NotNil(t, pub)
	require.Len(t, pub.contributors, 1)
	assert.Equal(t, fwkdl.StateKey("enabled"), pub.contributors[0].CrossReplicaState().StateKey)
	assert.Equal(t, defaultCrossReplicaSyncInterval, pub.Interval(), "zero interval falls back to default")
}

func TestCrossReplicaPublisher_ConfiguredInterval(t *testing.T) {
	em := extractorMapWith(fakeContributor{key: "enabled"})
	pub := newCrossReplicaPublisher(&fakeSyncer{}, em, 500*time.Millisecond)
	require.NotNil(t, pub)
	assert.Equal(t, 500*time.Millisecond, pub.Interval())
}
