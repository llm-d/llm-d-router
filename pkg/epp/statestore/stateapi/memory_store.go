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
	"hash/fnv"
	"sync"
	"sync/atomic"
	"time"
)

// shardCount is the number of independent lock domains used by the prefix
// index and the concurrency-lease store. Every stateless replica's remote
// calls land on this one stateful EPP process, so a single global mutex per
// store serializes all of them regardless of which hash/endpoint/requestID is
// involved. 32 is a reasonable default, not tuned against real traffic shape.
const shardCount = 32

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

// stringShardIndex hashes s (fnv-1a, cheap and non-cryptographic — this is a
// shard-assignment function, not a security boundary) into [0, shardCount).
func stringShardIndex(s string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(s)) // fnv.Write never errors
	return int(h.Sum32() % shardCount)
}

// --- Inflight ---
//
// Not sharded: getOrCreateCounter already only takes the map mutex on the
// cold path (first touch of a given endpoint); the hot path (Reserve/Release
// on an already-known endpoint) is lock-free atomics. Endpoint cardinality is
// also naturally low (one entry per backend pod, not per request), unlike the
// prefix and concurrency-lease stores below, whose keys are per-request-hash
// and per-request-lease respectively.

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
//
// Sharded by block hash: every request's PreRequest commits its matched
// hashes here, so this is on the hot path for every one of N stateless
// replicas. A single global mutex serialized all of them against each other
// regardless of which hash/endpoint was involved; sharding lets writes to
// unrelated hashes proceed concurrently.

type prefixShard struct {
	mu    sync.RWMutex
	index map[uint64]map[string]struct{} // hash -> set of endpoint IDs
}

type memoryPrefixStore struct {
	shards [shardCount]*prefixShard
}

func newMemoryPrefixStore() *memoryPrefixStore {
	s := &memoryPrefixStore{}
	for i := range s.shards {
		s.shards[i] = &prefixShard{index: make(map[uint64]map[string]struct{})}
	}
	return s
}

func (s *memoryPrefixStore) shardFor(hash uint64) *prefixShard {
	return s.shards[hash%uint64(shardCount)]
}

func (s *memoryPrefixStore) Match(hash uint64) []string {
	shard := s.shardFor(hash)
	shard.mu.RLock()
	defer shard.mu.RUnlock()
	pods, ok := shard.index[hash]
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
	// Group by shard first so a single Commit call (typically many hashes for
	// one prompt) takes each shard's mutex at most once instead of once per
	// hash.
	byShard := make(map[int][]uint64, len(hashes))
	for _, h := range hashes {
		idx := int(h % uint64(shardCount))
		byShard[idx] = append(byShard[idx], h)
	}
	for idx, hs := range byShard {
		shard := s.shards[idx]
		shard.mu.Lock()
		for _, h := range hs {
			pods, ok := shard.index[h]
			if !ok {
				pods = make(map[string]struct{})
				shard.index[h] = pods
			}
			pods[endpointID] = struct{}{}
		}
		shard.mu.Unlock()
	}
}

func (s *memoryPrefixStore) RemoveEndpoint(endpointID string) {
	// Rare (pod-delete events only, not per-request), so sequential per-shard
	// locking is fine — no need to avoid locking every shard here.
	for _, shard := range s.shards {
		shard.mu.Lock()
		for hash, pods := range shard.index {
			delete(pods, endpointID)
			if len(pods) == 0 {
				delete(shard.index, hash)
			}
		}
		shard.mu.Unlock()
	}
}

// --- Concurrency lease ---
//
// Leases are sharded by requestID (one entry per in-flight request — high
// cardinality, hot path); counters are kept in one small map (one entry per
// distinct FlowKey — low cardinality by nature, and mutated via atomics once
// resolved, matching memoryInflightStore's counter pattern) rather than
// sharded themselves, since sharding a handful of keys buys nothing.

type lease struct {
	key       FlowKey
	timestamp time.Time
}

type leaseShard struct {
	mu     sync.Mutex
	leases map[string]lease // requestID -> lease
}

type memoryConcurrencyStore struct {
	counterMu sync.RWMutex
	counters  map[FlowKey]*atomic.Int64

	leaseShards [shardCount]*leaseShard

	globalMax int64
	leaseTTL  time.Duration
	now       func() time.Time
}

func newMemoryConcurrencyStore(ctx context.Context, cfg Config) *memoryConcurrencyStore {
	cfg = cfg.withDefaults()
	s := &memoryConcurrencyStore{
		counters:  make(map[FlowKey]*atomic.Int64),
		globalMax: cfg.GlobalMaxConcurrency,
		leaseTTL:  cfg.LeaseTTL,
		now:       time.Now,
	}
	for i := range s.leaseShards {
		s.leaseShards[i] = &leaseShard{leases: make(map[string]lease)}
	}
	go s.runSweep(ctx, cfg.SweepInterval)
	return s
}

func (s *memoryConcurrencyStore) leaseShardFor(requestID string) *leaseShard {
	return s.leaseShards[stringShardIndex(requestID)]
}

func (s *memoryConcurrencyStore) counterFor(key FlowKey) *atomic.Int64 {
	s.counterMu.RLock()
	c, ok := s.counters[key]
	s.counterMu.RUnlock()
	if ok {
		return c
	}

	s.counterMu.Lock()
	defer s.counterMu.Unlock()
	if c, ok = s.counters[key]; ok {
		return c
	}
	c = &atomic.Int64{}
	s.counters[key] = c
	return c
}

// tryAdmit atomically increments counter unless doing so would exceed max (max
// <= 0 means unlimited), returning whether the increment happened.
//
// This must be a CAS loop, not "Load, branch, Add": Admit's shard lock is
// keyed by requestID, but counter is the FlowKey-wide total shared by every
// requestID for that key, so two concurrent Admits for the same key landing
// on different shards would otherwise both pass a Load-based check before
// either Add lands, overshooting max.
func tryAdmit(counter *atomic.Int64, max int64) bool {
	for {
		current := counter.Load()
		if max > 0 && current >= max {
			return false
		}
		if counter.CompareAndSwap(current, current+1) {
			return true
		}
	}
}

func (s *memoryConcurrencyStore) Admit(requestID string, key FlowKey) ConcurrencyOutcome {
	shard := s.leaseShardFor(requestID)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	if _, ok := shard.leases[requestID]; ok {
		// Idempotent retry of an already-admitted request.
		return ConcurrencyOutcomeAdmitted
	}

	if !tryAdmit(s.counterFor(key), s.globalMax) {
		return ConcurrencyOutcomeRejected
	}
	shard.leases[requestID] = lease{key: key, timestamp: s.now()}
	return ConcurrencyOutcomeAdmitted
}

func (s *memoryConcurrencyStore) ReleaseConcurrency(requestID string, _ FlowKey) {
	shard := s.leaseShardFor(requestID)
	shard.mu.Lock()
	l, ok := shard.leases[requestID]
	if ok {
		delete(shard.leases, requestID)
	}
	shard.mu.Unlock()
	if !ok {
		return
	}
	decrementClamped(s.counterFor(l.key), 1)
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
	cutoff := s.now().Add(-s.leaseTTL)
	for _, shard := range s.leaseShards {
		shard.mu.Lock()
		for requestID, l := range shard.leases {
			if l.timestamp.After(cutoff) {
				continue
			}
			delete(shard.leases, requestID)
			decrementClamped(s.counterFor(l.key), 1)
		}
		shard.mu.Unlock()
	}
}
