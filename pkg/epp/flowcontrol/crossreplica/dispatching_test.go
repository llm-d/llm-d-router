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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/clock"
)

func TestDispatching_CountsUntilRemoved(t *testing.T) {
	t.Parallel()
	d := NewDispatching()
	d.Add("a", 100)
	d.Add("a", 100)
	d.Add("b", 100)
	d.Add("c", 0)
	assert.Equal(t, uint64(2), d.Count(100), "a repeated request ID counts once")
	assert.Equal(t, uint64(1), d.Count(0))

	d.Remove("a")
	d.Remove("a")
	d.Remove("unknown")
	assert.Equal(t, uint64(1), d.Count(100))
	var none *Dispatching
	assert.Zero(t, none.Count(100))
}

func TestPublisher_PublishesDispatchingWithTheQueue(t *testing.T) {
	t.Parallel()
	syncer := newFakeDemandSyncer()
	p := NewPublisher(syncer, twoBands(), "ns/pool", time.Hour, time.Second, clock.RealClock{})
	p.Dispatching().Add("r1", 100)
	p.Dispatching().Add("r2", 0)

	p.publishAll(context.Background())

	assert.Equal(t, &bandDemand{Requests: 3, Bytes: 300, Dispatching: 1}, syncer.sets[key100])
	assert.Equal(t, &bandDemand{Dispatching: 1}, syncer.sets[key0])
}

func TestPeerDemand_CountsPeerDispatchingAsDemand(t *testing.T) {
	t.Parallel()
	p, syncer, _ := newPeerTestPublisher(t)
	syncer.peer["100"] = &bandDemand{Dispatching: 2}
	view := p.PeerDemand()

	require.NoError(t, p.view.refresh(context.Background()))
	queued, ok := view.QueuedRequests(100)
	assert.True(t, ok)
	assert.Equal(t, uint64(2), queued, "a peer band whose requests are on the way to an endpoint is not empty")
}
