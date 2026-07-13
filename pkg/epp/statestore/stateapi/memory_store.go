/*
Copyright 2026 The Kubernetes Authors.

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

package stateapi

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// Config controls the in-memory Store's behavior.
type Config struct {
	// GlobalMaxConcurrency is the fleet-wide concurrency cap per flow key.
	// Zero or negative means unlimited (every Admit succeeds).
	GlobalMaxConcurrency int64
	// LeaseTTL bounds how long a concurrency lease may be held before the
	// background sweep reclaims it as leaked (e.g. the reserving replica
	// crashed or dropped its Release call). Defaults to 5 minutes if zero.
	LeaseTTL time.Duration
	// SweepInterval controls how often the reclaim sweep runs. Defaults to
	// LeaseTTL/4 if zero.
	SweepInterval time.Duration
}

func (c Config) withDefaults() Config {
	if c.LeaseTTL <= 0 {
		c.LeaseTTL = 5 * time.Minute
	}
	if c.SweepInterval <= 0 {
		c.SweepInterval = c.LeaseTTL / 4
	}
	return c
}

// memoryStore is the in-memory Store implementation used by this feasibility
// spike. A future Redis/PVC-backed Store can replace this without touching
// the gRPC server (server.go) or any client code, since both depend only on
// the Store interface.
type memoryStore struct {
	*memoryInflightStore
	*memoryPrefixStore
	*memoryConcurrencyStore
}

var _ Store = (*memoryStore)(nil)

// NewMemoryStore returns an in-memory Store. The returned store's background
// concurrency-lease sweep goroutine stops when ctx is cancelled.
func NewMemoryStore(ctx context.Context, cfg Config) Store {
	cfg = cfg.withDefaults()
	return &memoryStore{
		memoryInflightStore:    newMemoryInflightStore(),
		memoryPrefixStore:      newMemoryPrefixStore(),
		memoryConcurrencyStore: newMemoryConcurrencyStore(ctx, cfg),
	}
}

// --- Inflight ---

type inflightCounter struct {
	requests atomic.Int64
	tokens   atomic.Int64
}

type inflightReservation struct {
	// tokens is swapped to 0 by whichever Release call wins the race, so a
	// duplicate/retried Release is a no-op rather than a double decrement.
	tokens atomic.Int64
}

type memoryInflightStore struct {
	mu           sync.RWMutex
	counters     map[string]*inflightCounter // endpointID -> counter
	reservations sync.Map                    // "requestID|endpointID" -> *inflightReservation
}

func newMemoryInflightStore() *memoryInflightStore {
	return &memoryInflightStore{counters: make(map[string]*inflightCounter)}
}

func (s *memoryInflightStore) getOrCreateCounter(endpointID string) *inflightCounter {
	s.mu.RLock()
	c, ok := s.counters[endpointID]
	s.mu.RUnlock()
	if ok {
		return c
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if c, ok = s.counters[endpointID]; ok {
		return c
	}
	c = &inflightCounter{}
	s.counters[endpointID] = c
	return c
}

func (s *memoryInflightStore) SnapshotBatch(endpointIDs []string) map[string]InflightSnapshot {
	result := make(map[string]InflightSnapshot, len(endpointIDs))
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, id := range endpointIDs {
		c, ok := s.counters[id]
		if !ok {
			continue
		}
		result[id] = InflightSnapshot{Requests: c.requests.Load(), Tokens: c.tokens.Load()}
	}
	return result
}

func (s *memoryInflightStore) Reserve(requestID, endpointID string, estimatedTokens int64) {
	key := requestID + "|" + endpointID
	reservation := &inflightReservation{}
	reservation.tokens.Store(estimatedTokens)
	if _, loaded := s.reservations.LoadOrStore(key, reservation); loaded {
		// Already reserved (retried call): no-op, matches Store's idempotency contract.
		return
	}

	c := s.getOrCreateCounter(endpointID)
	c.requests.Add(1)
	c.tokens.Add(estimatedTokens)
}

func (s *memoryInflightStore) Release(requestID, endpointID string, _ int64) {
	key := requestID + "|" + endpointID
	val, ok := s.reservations.LoadAndDelete(key)
	if !ok {
		return
	}
	reservation := val.(*inflightReservation)
	tokens := reservation.tokens.Swap(0)

	s.mu.RLock()
	c, ok := s.counters[endpointID]
	s.mu.RUnlock()
	if !ok {
		return
	}
	decrementClamped(&c.requests, 1)
	decrementClamped(&c.tokens, tokens)
}

func (s *memoryInflightStore) DeleteEndpoint(endpointID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.counters, endpointID)
}

// decrementClamped subtracts delta from counter with a hard floor at zero,
// mirroring inflightload's decrementClamped (producer.go) so behavior is
// consistent between a stateless replica's local shadow and the stateful
// EPP's fleet-wide aggregate.
func decrementClamped(counter *atomic.Int64, delta int64) {
	for {
		current := counter.Load()
		if current <= 0 {
			return
		}
		next := current - delta
		if next < 0 {
			next = 0
		}
		if counter.CompareAndSwap(current, next) {
			return
		}
	}
}

// --- Prefix ---

type memoryPrefixStore struct {
	mu    sync.RWMutex
	index map[uint64]map[string]struct{} // hash -> set of endpoint IDs
}

func newMemoryPrefixStore() *memoryPrefixStore {
	return &memoryPrefixStore{index: make(map[uint64]map[string]struct{})}
}

func (s *memoryPrefixStore) Match(hash uint64) []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	pods, ok := s.index[hash]
	if !ok {
		return nil
	}
	result := make([]string, 0, len(pods))
	for id := range pods {
		result = append(result, id)
	}
	return result
}

func (s *memoryPrefixStore) Commit(_, endpointID string, hashes []uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, h := range hashes {
		pods, ok := s.index[h]
		if !ok {
			pods = make(map[string]struct{})
			s.index[h] = pods
		}
		pods[endpointID] = struct{}{}
	}
}

func (s *memoryPrefixStore) RemoveEndpoint(endpointID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for hash, pods := range s.index {
		delete(pods, endpointID)
		if len(pods) == 0 {
			delete(s.index, hash)
		}
	}
}

// --- Concurrency lease ---

type lease struct {
	key       FlowKey
	timestamp time.Time
}

type memoryConcurrencyStore struct {
	mu        sync.Mutex
	counters  map[FlowKey]*atomic.Int64
	leases    map[string]lease // requestID -> lease
	globalMax int64
	leaseTTL  time.Duration
	now       func() time.Time
}

func newMemoryConcurrencyStore(ctx context.Context, cfg Config) *memoryConcurrencyStore {
	cfg = cfg.withDefaults()
	s := &memoryConcurrencyStore{
		counters:  make(map[FlowKey]*atomic.Int64),
		leases:    make(map[string]lease),
		globalMax: cfg.GlobalMaxConcurrency,
		leaseTTL:  cfg.LeaseTTL,
		now:       time.Now,
	}
	go s.runSweep(ctx, cfg.SweepInterval)
	return s
}

func (s *memoryConcurrencyStore) counterFor(key FlowKey) *atomic.Int64 {
	c, ok := s.counters[key]
	if !ok {
		c = &atomic.Int64{}
		s.counters[key] = c
	}
	return c
}

func (s *memoryConcurrencyStore) Admit(requestID string, key FlowKey) ConcurrencyOutcome {
	s.mu.Lock()
	defer s.mu.Unlock()

	if existing, ok := s.leases[requestID]; ok {
		// Idempotent retry of an already-admitted request.
		_ = existing
		return ConcurrencyOutcomeAdmitted
	}

	counter := s.counterFor(key)
	if s.globalMax > 0 && counter.Load() >= s.globalMax {
		return ConcurrencyOutcomeRejected
	}
	counter.Add(1)
	s.leases[requestID] = lease{key: key, timestamp: s.now()}
	return ConcurrencyOutcomeAdmitted
}

func (s *memoryConcurrencyStore) ReleaseConcurrency(requestID string, _ FlowKey) {
	s.mu.Lock()
	defer s.mu.Unlock()

	l, ok := s.leases[requestID]
	if !ok {
		return
	}
	delete(s.leases, requestID)
	if counter, ok := s.counters[l.key]; ok {
		decrementClamped(counter, 1)
	}
}

// runSweep periodically reclaims leases that were never released (a crashed
// or unresponsive stateless replica), so a leak cannot permanently shrink
// effective fleet capacity over the course of a long-running measurement.
func (s *memoryConcurrencyStore) runSweep(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.reclaimExpiredLeases()
		}
	}
}

func (s *memoryConcurrencyStore) reclaimExpiredLeases() {
	s.mu.Lock()
	defer s.mu.Unlock()

	cutoff := s.now().Add(-s.leaseTTL)
	for requestID, l := range s.leases {
		if l.timestamp.After(cutoff) {
			continue
		}
		delete(s.leases, requestID)
		if counter, ok := s.counters[l.key]; ok {
			decrementClamped(counter, 1)
		}
	}
}
