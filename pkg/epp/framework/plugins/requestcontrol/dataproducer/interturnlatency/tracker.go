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

package interturnlatency

import (
	"sync"
	"time"
)

// sessionTracker records the latest activity timestamp per session so that a
// request's arrival yields the idle gap since the session's previous turn.
type sessionTracker struct {
	mu         sync.Mutex
	lastActive map[string]time.Time

	// maxSessions is a soft cap on tracked sessions: exceeding it triggers a
	// sweep of entries idle longer than maxIdle. New sessions are always
	// admitted, so the map can transiently exceed the cap when every entry is
	// active.
	maxSessions int
	// maxIdle bounds a usable gap observation; a session quiet for longer is
	// treated as a new session.
	maxIdle time.Duration
}

func newSessionTracker(maxSessions int, maxIdle time.Duration) *sessionTracker {
	return &sessionTracker{
		lastActive:  make(map[string]time.Time),
		maxSessions: maxSessions,
		maxIdle:     maxIdle,
	}
}

// observe returns the elapsed time since the session's last recorded activity
// and records now as the latest activity. ok is false for the first sighting
// of a session and when the gap exceeds maxIdle.
func (t *sessionTracker) observe(id string, now time.Time) (gap time.Duration, ok bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	last, seen := t.lastActive[id]
	if !seen && len(t.lastActive) >= t.maxSessions {
		t.sweepLocked(now)
	}
	t.lastActive[id] = now
	if !seen {
		return 0, false
	}
	gap = now.Sub(last)
	if gap <= 0 || gap > t.maxIdle {
		return 0, false
	}
	return gap, true
}

// touch records activity for the session without producing a gap observation.
func (t *sessionTracker) touch(id string, now time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, seen := t.lastActive[id]; !seen && len(t.lastActive) >= t.maxSessions {
		t.sweepLocked(now)
	}
	t.lastActive[id] = now
}

// size returns the number of tracked sessions.
func (t *sessionTracker) size() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.lastActive)
}

// sweepLocked drops sessions idle longer than maxIdle. Callers must hold t.mu.
func (t *sessionTracker) sweepLocked(now time.Time) {
	for id, last := range t.lastActive {
		if now.Sub(last) > t.maxIdle {
			delete(t.lastActive, id)
		}
	}
}
