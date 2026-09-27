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

package sessionstate

import (
	"sync"
	"time"
)

type sessionRecord struct {
	state       SessionState
	firstSeenAt time.Time
}

// SessionStateRegistry stores session history and owns its concurrency rules.
// Its zero value is ready for use.
type SessionStateRegistry struct {
	mu       sync.Mutex
	sessions map[string]*sessionRecord
}

// GetState returns the state observed before the current request and marks the
// session as seen at the current time.
func (r *SessionStateRegistry) GetState(identity string) SessionState {
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	record := r.getOrCreate(identity, now)
	state := record.state
	state.Duration = now.Sub(record.firstSeenAt)
	record.state.LastSeenAt = now
	return state
}

// RecordDispatch records one request dispatched for the session.
func (r *SessionStateRegistry) RecordDispatch(identity string) {
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	record := r.getOrCreate(identity, now)
	record.state.TurnsTaken++
	record.state.InFlightRequests++
}

// RecordResponse records the end of a dispatched request's response lifecycle.
// Only naturally completed responses contribute completion and token totals.
func (r *SessionStateRegistry) RecordResponse(identity string, naturallyCompleted bool, inputTokens, outputTokens int64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	record, exists := r.sessions[identity]
	if !exists || record.state.InFlightRequests == 0 {
		return
	}
	record.state.InFlightRequests--
	if !naturallyCompleted {
		return
	}
	record.state.CompletedRequests++
	record.state.TotalInputTokens += inputTokens
	record.state.TotalOutputTokens += outputTokens
}

// EvictIdle removes sessions that have been idle longer than ttl. Sessions
// with in-flight requests are retained until those requests end.
func (r *SessionStateRegistry) EvictIdle(now time.Time, ttl time.Duration) {
	if ttl <= 0 {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	for identity, record := range r.sessions {
		if record.state.InFlightRequests > 0 {
			continue
		}
		if now.Sub(record.state.LastSeenAt) > ttl {
			delete(r.sessions, identity)
		}
	}
}

func (r *SessionStateRegistry) getOrCreate(identity string, now time.Time) *sessionRecord {
	if r.sessions == nil {
		r.sessions = make(map[string]*sessionRecord)
	}
	if record, exists := r.sessions[identity]; exists {
		return record
	}
	record := &sessionRecord{
		state: SessionState{
			LastSeenAt: now,
		},
		firstSeenAt: now,
	}
	r.sessions[identity] = record
	return record
}
