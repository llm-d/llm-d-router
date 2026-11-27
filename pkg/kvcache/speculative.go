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

package kvcache

import (
	"context"
	"sort"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// SpeculativeIndex records routing decisions as hints: per request block key,
// the pods a request carrying that block was routed to and when each hint
// expires. The matcher reads it next to the KV-block index and scores a hint
// as a speculative entry, so a request sharing a prefix with one routed
// moments earlier matches that pod before the engine's KV-events confirm the
// blocks. Hints are held outside kvblock.Index: they take no room in a key's
// pod cache, expire without an index write, and are local to the process.
//
// Hints live in two generations. Add writes the current one. Run retires it
// once per TTL and drops the one retired before it, whose hints have all
// expired by then, so reclaiming memory does not scan the hints.
type SpeculativeIndex struct {
	ttl time.Duration

	mu                sync.RWMutex
	current, previous map[kvblock.BlockHash][]speculativeHint
}

// speculativeHint is one pod's hint for a block.
type speculativeHint struct {
	pod       string
	expiresAt time.Time
}

// NewSpeculativeIndex builds an index whose hints live for ttl.
func NewSpeculativeIndex(ttl time.Duration) *SpeculativeIndex {
	return &SpeculativeIndex{
		ttl:     ttl,
		current: make(map[kvblock.BlockHash][]speculativeHint),
	}
}

// Add hints keys, the leading block keys of a prompt, as held by pod until the
// TTL elapses. A pod already hinted for a key has its hint renewed.
//
// A block key hashes its block and every block before it, so a pod hinted for
// a key was hinted for the keys leading to it by the same Add or a later one.
// Its live hints along any prompt are therefore a leading run of the prompt's
// keys, which Match relies on.
func (s *SpeculativeIndex) Add(keys []kvblock.BlockHash, pod string) {
	if len(keys) == 0 {
		return
	}
	expiresAt := time.Now().Add(s.ttl)

	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		hints := s.current[key]
		renewed := false
		for i := range hints {
			if hints[i].pod == pod {
				hints[i].expiresAt = expiresAt
				renewed = true
				break
			}
		}
		if !renewed {
			s.current[key] = append(hints, speculativeHint{pod: pod, expiresAt: expiresAt})
		}
	}
}

// Match returns, per pod, the number of leading keys it is hinted for.
// Candidates are the pods hinted for the first key, restricted to podFilter
// when it is non-empty, and a pod's chain ends at the first key it is not
// hinted for. The result is nil when no candidate exists.
func (s *SpeculativeIndex) Match(keys []kvblock.BlockHash, podFilter sets.Set[string]) map[string]int {
	if len(keys) == 0 {
		return nil
	}
	now := time.Now()

	s.mu.RLock()
	defer s.mu.RUnlock()
	var chains map[string]int
	for _, generation := range [...]map[kvblock.BlockHash][]speculativeHint{s.current, s.previous} {
		for _, h := range generation[keys[0]] {
			if !h.expiresAt.After(now) || (podFilter.Len() > 0 && !podFilter.Has(h.pod)) {
				continue
			}
			if _, matched := chains[h.pod]; matched {
				continue
			}
			if chains == nil {
				chains = make(map[string]int)
			}
			// The pod's hinted keys are a leading run, so bisection finds
			// the first key past it.
			chains[h.pod] = 1 + sort.Search(len(keys)-1, func(i int) bool {
				return !s.hinted(keys[i+1], h.pod, now)
			})
		}
	}
	return chains
}

// hinted reports whether pod holds a live hint for key. The caller holds mu.
func (s *SpeculativeIndex) hinted(key kvblock.BlockHash, pod string, now time.Time) bool {
	for _, generation := range [...]map[kvblock.BlockHash][]speculativeHint{s.current, s.previous} {
		for _, h := range generation[key] {
			if h.pod == pod && h.expiresAt.After(now) {
				return true
			}
		}
	}
	return false
}

// rotate retires the current generation and drops the previous one. The new
// generation is sized like the one it replaces, so steady traffic does not
// regrow the map after every rotation.
func (s *SpeculativeIndex) rotate() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.previous = s.current
	s.current = make(map[kvblock.BlockHash][]speculativeHint, len(s.previous))
}

// Run rotates the generations once per TTL until ctx is done. Rotations are
// at least a TTL apart, so a generation is dropped only after the last hint
// written to it has expired.
func (s *SpeculativeIndex) Run(ctx context.Context) {
	timer := time.NewTimer(s.ttl)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			s.rotate()
			timer.Reset(s.ttl)
		}
	}
}

// Len reports the number of block keys held in the two generations. A key
// hinted in both counts twice, and an expired hint counts until its
// generation is dropped.
func (s *SpeculativeIndex) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.current) + len(s.previous)
}
