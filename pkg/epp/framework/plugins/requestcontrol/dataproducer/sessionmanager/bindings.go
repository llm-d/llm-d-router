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

package sessionmanager

import (
	"container/list"
	"sync"
	"time"
)

type requestBinding struct {
	sessionTag string
	modelName  string
	endpoint   string
	createdAt  time.Time
	boundAt    time.Time
	observed   bool
}

type bindingEntry struct {
	binding requestBinding
	element *list.Element
}

type bindingStore struct {
	mu       sync.Mutex
	items    map[string]*bindingEntry
	order    *list.List
	resetAt  map[string]time.Time
	ttl      time.Duration
	capacity int
	now      func() time.Time
}

func newBindingStore(ttl time.Duration, capacity int) *bindingStore {
	return &bindingStore{
		items:    make(map[string]*bindingEntry),
		order:    list.New(),
		resetAt:  make(map[string]time.Time),
		ttl:      ttl,
		capacity: capacity,
		now:      time.Now,
	}
}

func (s *bindingStore) put(stamp, sessionTag, modelName string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	s.removeExpiredLocked(now)
	if existing, ok := s.items[stamp]; ok {
		s.removeLocked(stamp, existing)
	}
	if len(s.items) >= s.capacity {
		s.removeOldestLocked()
	}
	element := s.order.PushBack(stamp)
	s.items[stamp] = &bindingEntry{
		binding: requestBinding{sessionTag: sessionTag, modelName: modelName, createdAt: now},
		element: element,
	}
}

func (s *bindingStore) bindEndpoint(stamp, endpoint string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	entry, ok := s.items[stamp]
	if !ok || !entry.binding.createdAt.Add(s.ttl).After(now) {
		if ok {
			s.removeLocked(stamp, entry)
		}
		return false
	}
	entry.binding.endpoint = endpoint
	entry.binding.boundAt = now
	return true
}

func (s *bindingStore) observe(stamp, modelName, endpoint string) (bool, bool, bool, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	entry, ok := s.items[stamp]
	if !ok {
		return false, false, false, false
	}
	if !entry.binding.createdAt.Add(s.ttl).After(now) {
		s.removeLocked(stamp, entry)
		return false, false, false, false
	}
	if entry.binding.modelName != modelName || entry.binding.endpoint != endpoint {
		return true, false, true, false
	}
	if resetAt, ok := s.resetAt[endpoint]; ok && !entry.binding.boundAt.After(resetAt) {
		return true, false, false, true
	}
	duplicate := entry.binding.observed
	entry.binding.observed = true
	return true, duplicate, false, false
}

func (s *bindingStore) resetEndpoint(endpoint string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resetAt[endpoint] = s.now()
}

func (s *bindingStore) len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.removeExpiredLocked(s.now())
	return len(s.items)
}

func (s *bindingStore) removeExpiredLocked(now time.Time) {
	for s.order.Len() > 0 {
		element := s.order.Front()
		stamp := element.Value.(string)
		entry := s.items[stamp]
		if entry.binding.createdAt.Add(s.ttl).After(now) {
			return
		}
		s.removeLocked(stamp, entry)
	}
}

func (s *bindingStore) removeOldestLocked() {
	element := s.order.Front()
	if element == nil {
		return
	}
	stamp := element.Value.(string)
	s.removeLocked(stamp, s.items[stamp])
}

func (s *bindingStore) removeLocked(stamp string, entry *bindingEntry) {
	delete(s.items, stamp)
	s.order.Remove(entry.element)
}
