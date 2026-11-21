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

package p2p

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/clock"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/crossreplica"
)

// bands is a flow registry stand-in holding queued requests per priority band.
type bands struct {
	mu     sync.Mutex
	queued map[int]uint64
}

func (b *bands) set(priority int, queued uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.queued[priority] = queued
}

func (b *bands) AllOrderedPriorityLevels() []int { return []int{80, 0} }

func (b *bands) CapacitySnapshot(priority int) (contracts.CapacitySnapshot, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return contracts.CapacitySnapshot{Band: contracts.CapacityDimension{Len: b.queued[priority]}}, nil
}

func TestBandDemandReachesPeerAndExcludesLocal(t *testing.T) {
	a := newReplica(t, "a", time.Second, 100*time.Millisecond)
	b := newReplica(t, "b", time.Second, 100*time.Millisecond)
	mesh(a, b)
	bandsA := &bands{queued: map[int]uint64{80: 0, 0: 0}}
	bandsB := &bands{queued: map[int]uint64{80: 0, 0: 0}}
	pubA := crossreplica.NewPublisher(a.syncer, bandsA, "ns/pool", 10*time.Millisecond, time.Second, clock.RealClock{})
	pubB := crossreplica.NewPublisher(b.syncer, bandsB, "ns/pool", 10*time.Millisecond, time.Second, clock.RealClock{})
	viewA, viewB := pubA.PeerDemand(), pubB.PeerDemand()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go pubA.Run(ctx)
	go pubB.Run(ctx)

	bandsA.set(80, 3)
	bandsB.set(0, 7)
	queued := func(view interface {
		QueuedRequests(int) (uint64, bool)
	}, priority int) uint64 {
		n, ok := view.QueuedRequests(priority)
		if !ok {
			return 1 << 63
		}
		return n
	}
	require.Eventually(t, func() bool {
		return queued(viewB, 80) == 3 && queued(viewA, 0) == 7
	}, eventually, tick, "each replica sees the other's queued band")
	require.Zero(t, queued(viewA, 80), "a replica's own queue is not peer demand")
	require.Zero(t, queued(viewB, 0), "a replica's own queue is not peer demand")

	bandsA.set(80, 0)
	require.Eventually(t, func() bool { return queued(viewB, 80) == 0 }, eventually, tick,
		"a drained band reaches the peer")
}
