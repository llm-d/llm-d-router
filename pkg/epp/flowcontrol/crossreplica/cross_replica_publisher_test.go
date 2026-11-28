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

package crossreplica

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/clock"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	key100 = "flowcontrol-priorityband-demand:ns/pool:100"
	key0   = "flowcontrol-priorityband-demand:ns/pool:0"
)

type fakeBandRegistry struct {
	mu    sync.Mutex
	bands map[int]contracts.CapacityDimension
}

func (r *fakeBandRegistry) AllOrderedPriorityLevels() []int {
	r.mu.Lock()
	defer r.mu.Unlock()
	priorities := make([]int, 0, len(r.bands))
	for priority := range r.bands {
		priorities = append(priorities, priority)
	}
	slices.Sort(priorities)
	slices.Reverse(priorities)
	return priorities
}

func (r *fakeBandRegistry) CapacitySnapshot(priority int) (contracts.CapacitySnapshot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	band, ok := r.bands[priority]
	if !ok {
		return contracts.CapacitySnapshot{}, contracts.ErrPriorityBandNotFound
	}
	return contracts.CapacitySnapshot{Band: band}, nil
}

func (r *fakeBandRegistry) removeBand(priority int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.bands, priority)
}

func twoBands() *fakeBandRegistry {
	return &fakeBandRegistry{bands: map[int]contracts.CapacityDimension{
		100: {Len: 3, ByteSize: 300},
		0:   {},
	}}
}

// fakeDemandSyncer records writes keyed like RedisSyncer: StateKey + ":" + id.
type fakeDemandSyncer struct {
	mu      sync.Mutex
	sets    map[string]any
	batches [][]string
}

func newFakeDemandSyncer() *fakeDemandSyncer {
	return &fakeDemandSyncer{sets: map[string]any{}}
}

func (s *fakeDemandSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-syncer", Name: "fake-syncer"}
}

func (s *fakeDemandSyncer) Set(_ context.Context, spec fwkdl.CrossReplicaSpec, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sets[string(spec.StateKey)+":"+id] = spec.Read(id)
	return nil
}

func (s *fakeDemandSyncer) Get(context.Context, fwkdl.CrossReplicaSpec, string) (any, bool, error) {
	return nil, false, nil
}

func (s *fakeDemandSyncer) Delete(context.Context, fwkdl.StateKey, string) error {
	return errors.New("not implemented")
}

func (s *fakeDemandSyncer) GetOrSet(context.Context, fwkdl.StateKey, string, any) (any, bool, error) {
	return nil, false, errors.New("not implemented")
}

func (s *fakeDemandSyncer) setKeys() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	keys := make([]string, 0, len(s.sets))
	for k := range s.sets {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

type fakeBatchDemandSyncer struct {
	*fakeDemandSyncer
}

func (s *fakeBatchDemandSyncer) SetBatch(ctx context.Context, spec fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
	s.mu.Lock()
	s.batches = append(s.batches, slices.Clone(ids))
	s.mu.Unlock()
	for _, id := range ids {
		_ = s.Set(ctx, spec, id)
	}
	return nil
}

func newTestPublisher(syncer fwkdl.CrossReplicaSyncer, registry bandOccupancy, interval time.Duration) *Publisher {
	return NewPublisher(syncer, registry, "ns/pool", interval, time.Second, clock.RealClock{})
}

func TestPublisher_PublishesEveryBandIncludingIdle(t *testing.T) {
	t.Parallel()
	syncer := newFakeDemandSyncer()
	p := newTestPublisher(syncer, twoBands(), time.Hour)

	p.publishAll(context.Background())

	assert.Equal(t, []string{key0, key100}, syncer.setKeys())
	assert.Equal(t, &bandDemand{Requests: 3, Bytes: 300}, syncer.sets[key100])
	assert.Equal(t, &bandDemand{}, syncer.sets[key0], "an idle band publishes zero")
}

func TestPublisher_UsesBatchWhenSupported(t *testing.T) {
	t.Parallel()
	syncer := &fakeBatchDemandSyncer{fakeDemandSyncer: newFakeDemandSyncer()}
	p := newTestPublisher(syncer, twoBands(), time.Hour)

	p.publishAll(context.Background())

	require.Len(t, syncer.batches, 1)
	assert.Equal(t, []string{"100", "0"}, syncer.batches[0])
}

func TestPublisher_StopsPublishingRemovedBands(t *testing.T) {
	t.Parallel()
	syncer := newFakeDemandSyncer()
	registry := twoBands()
	p := newTestPublisher(syncer, registry, time.Hour)

	registry.removeBand(0)
	p.publishAll(context.Background())

	assert.Equal(t, []string{key100}, syncer.setKeys())
}

func TestPublisher_PublishesEverySource(t *testing.T) {
	t.Parallel()
	syncer := newFakeDemandSyncer()
	other := &crossReplicaSource{
		spec: fwkdl.CrossReplicaSpec{
			StateKey:  "other",
			Read:      func(string) fwkdl.Cloneable { return &bandDemand{} },
			Aggregate: aggregateBandDemand,
		},
		ids: func() []string { return []string{"a", "b"} },
	}
	p := newPublisher(syncer, []*crossReplicaSource{newPriorityBandDemandSource(twoBands(), "ns/pool"), other},
		time.Hour, time.Second, clock.RealClock{})

	p.publishAll(context.Background())

	assert.Equal(t, []string{key0, key100, "other:a", "other:b"}, syncer.setKeys())
}

func TestPublisher_NonPositiveDurationsUseDefaults(t *testing.T) {
	t.Parallel()
	p := NewPublisher(newFakeDemandSyncer(), twoBands(), "ns/pool", 0, -time.Second, clock.RealClock{})

	assert.Equal(t, 200*time.Millisecond, p.interval)
	assert.Equal(t, time.Second, p.publishTimeout)
}

func TestPublisher_AggregateSumsReplicas(t *testing.T) {
	t.Parallel()
	got := aggregateBandDemand([]any{&bandDemand{Requests: 3, Bytes: 300}, &bandDemand{Requests: 5, Bytes: 500}, "ignored"})

	assert.Equal(t, &bandDemand{Requests: 8, Bytes: 800}, got)
}
