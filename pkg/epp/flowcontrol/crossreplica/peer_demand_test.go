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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	clocktesting "k8s.io/utils/clock/testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// peerAggregateSyncer answers Get as the syncers do: the live local value combined with a cached peer aggregate.
type peerAggregateSyncer struct {
	*fakeDemandSyncer
	mu     sync.Mutex
	peer   map[string]*bandDemand
	getErr error
}

func (s *peerAggregateSyncer) Get(_ context.Context, spec fwkdl.CrossReplicaSpec, id string) (any, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.getErr != nil {
		return nil, false, s.getErr
	}
	local := spec.Read(id)
	peer, ok := s.peer[id]
	if !ok {
		return local, true, nil
	}
	return spec.Aggregate([]any{local, peer}), true, nil
}

func newPeerTestPublisher(t *testing.T) (*Publisher, *peerAggregateSyncer, *clocktesting.FakeClock) {
	t.Helper()
	syncer := &peerAggregateSyncer{fakeDemandSyncer: newFakeDemandSyncer(), peer: map[string]*bandDemand{}}
	clk := clocktesting.NewFakeClock(time.Unix(0, 0))
	return NewPublisher(syncer, twoBands(), "ns/pool", 20*time.Millisecond, time.Second, clk), syncer, clk
}

func TestPeerDemand_ReportsPeersOnly(t *testing.T) {
	t.Parallel()
	p, syncer, _ := newPeerTestPublisher(t)
	syncer.peer["100"] = &bandDemand{Requests: 5, Bytes: 500}
	view := p.PeerDemand()

	_, ok := view.QueuedRequests(100)
	assert.False(t, ok, "no view before the first refresh")

	require.NoError(t, p.view.refresh(context.Background()))
	queued, ok := view.QueuedRequests(100)
	assert.True(t, ok)
	assert.Equal(t, uint64(5), queued, "the local 3 queued requests are not peer demand")
	queued, ok = view.QueuedRequests(0)
	assert.True(t, ok)
	assert.Zero(t, queued)
}

func TestPeerDemand_AgesOutWithoutRefresh(t *testing.T) {
	t.Parallel()
	p, syncer, clk := newPeerTestPublisher(t)
	syncer.peer["100"] = &bandDemand{Requests: 5}
	view := p.PeerDemand()
	require.NoError(t, p.view.refresh(context.Background()))

	clk.Step(peerDemandMaxAgeIntervals * 20 * time.Millisecond)
	_, ok := view.QueuedRequests(100)
	assert.True(t, ok, "current up to the max age")

	clk.Step(time.Millisecond)
	_, ok = view.QueuedRequests(100)
	assert.False(t, ok, "older than the max age")
}

func TestPeerDemand_ReadErrorKeepsThePreviousSnapshot(t *testing.T) {
	t.Parallel()
	p, syncer, clk := newPeerTestPublisher(t)
	syncer.peer["100"] = &bandDemand{Requests: 5}
	view := p.PeerDemand()
	require.NoError(t, p.view.refresh(context.Background()))

	syncer.mu.Lock()
	syncer.getErr = errors.New("unavailable")
	syncer.mu.Unlock()
	clk.Step(20 * time.Millisecond)
	assert.Error(t, p.view.refresh(context.Background()))
	queued, ok := view.QueuedRequests(100)
	assert.True(t, ok)
	assert.Equal(t, uint64(5), queued)

	clk.Step(peerDemandMaxAgeIntervals * 20 * time.Millisecond)
	_, ok = view.QueuedRequests(100)
	assert.False(t, ok)
}

func TestPeerDemand_SameViewOnEveryCall(t *testing.T) {
	t.Parallel()
	p, _, _ := newPeerTestPublisher(t)
	assert.Same(t, p.PeerDemand(), p.PeerDemand())
}
