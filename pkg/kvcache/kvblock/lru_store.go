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

package kvblock

import (
	"sync"

	"github.com/hashicorp/golang-lru/v2/simplelru"
)

// lruStore is the request-key LRU behind one lock the index owns, so a read
// path can refresh the recency of every key it visited under a single
// acquisition instead of one per key.
type lruStore struct {
	mu  sync.RWMutex
	lru *simplelru.LRU[BlockHash, *PodCache]
}

func newLRUStore(size int) (*lruStore, error) {
	lru, err := simplelru.NewLRU[BlockHash, *PodCache](size, nil)
	if err != nil {
		return nil, err
	}
	return &lruStore{lru: lru}, nil
}

// Get returns the key's cache and marks it most recently used.
func (s *lruStore) Get(key BlockHash) (*PodCache, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lru.Get(key)
}

// Peek returns the key's cache without touching recency.
func (s *lruStore) Peek(key BlockHash) (*PodCache, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lru.Peek(key)
}

// Promote marks keys most recently used in order, so the last key ends up
// the most recent, under one acquisition. Absent keys are skipped.
func (s *lruStore) Promote(keys []BlockHash) {
	if len(keys) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		s.lru.Get(key)
	}
}

func (s *lruStore) Add(key BlockHash, value *PodCache) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lru.Add(key, value)
}

// ContainsOrAdd reports whether key is present and, when it is not, adds
// value.
func (s *lruStore) ContainsOrAdd(key BlockHash, value *PodCache) (contains, evicted bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lru.Contains(key) {
		return true, false
	}
	return false, s.lru.Add(key, value)
}

func (s *lruStore) Remove(key BlockHash) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lru.Remove(key)
}

func (s *lruStore) Keys() []BlockHash {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lru.Keys()
}

func (s *lruStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lru.Len()
}
