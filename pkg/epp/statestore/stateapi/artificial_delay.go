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

import "time"

// artificialDelayStore decorates a Store with a fixed delay before every
// call. This exists solely for the performance-test harness: on a
// single-node kind cluster, stateless<->stateful traffic is loopback (sub-
// millisecond), which understates the real cross-pod network hop that any
// cost/benefit measurement needs to account for. It is not a general
// resilience or throttling mechanism and must not be used outside test
// modeling.
type artificialDelayStore struct {
	Store
	delay time.Duration
}

// NewArtificialDelayStore returns a Store that sleeps for delay before
// delegating every call to the given Store. A non-positive delay returns the
// given Store unwrapped.
func NewArtificialDelayStore(store Store, delay time.Duration) Store {
	if delay <= 0 {
		return store
	}
	return &artificialDelayStore{Store: store, delay: delay}
}

func (s *artificialDelayStore) SnapshotBatch(endpointIDs []string) map[string]InflightSnapshot {
	time.Sleep(s.delay)
	return s.Store.SnapshotBatch(endpointIDs)
}

func (s *artificialDelayStore) Reserve(requestID, endpointID string, estimatedTokens int64) {
	time.Sleep(s.delay)
	s.Store.Reserve(requestID, endpointID, estimatedTokens)
}

func (s *artificialDelayStore) Release(requestID, endpointID string, estimatedTokens int64) {
	time.Sleep(s.delay)
	s.Store.Release(requestID, endpointID, estimatedTokens)
}

func (s *artificialDelayStore) DeleteEndpoint(endpointID string) {
	time.Sleep(s.delay)
	s.Store.DeleteEndpoint(endpointID)
}

func (s *artificialDelayStore) Match(hash uint64) []string {
	time.Sleep(s.delay)
	return s.Store.Match(hash)
}

func (s *artificialDelayStore) Commit(requestID, endpointID string, hashes []uint64) {
	time.Sleep(s.delay)
	s.Store.Commit(requestID, endpointID, hashes)
}

func (s *artificialDelayStore) RemoveEndpoint(endpointID string) {
	time.Sleep(s.delay)
	s.Store.RemoveEndpoint(endpointID)
}

func (s *artificialDelayStore) Admit(requestID string, key FlowKey) ConcurrencyOutcome {
	time.Sleep(s.delay)
	return s.Store.Admit(requestID, key)
}

func (s *artificialDelayStore) ReleaseConcurrency(requestID string, key FlowKey) {
	time.Sleep(s.delay)
	s.Store.ReleaseConcurrency(requestID, key)
}
