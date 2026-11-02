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
	"container/list"
	"sync"
	"sync/atomic"
	"time"
)

type sessionRecord struct {
	identity string
	// state is guarded by SessionStateRegistry.mu.
	state       SessionState
	firstSeenAt time.Time
	// idleElem is the record's entry in SessionStateRegistry.idle, nil while
	// the session has a pending or in-flight request. Guarded by
	// SessionStateRegistry.mu.
	idleElem *list.Element
	// pendingRequests counts requests observed at RequestHeader that are not
	// yet dispatched or abandoned, such as requests waiting in a flow-control
	// queue. Guarded by SessionStateRegistry.mu.
	pendingRequests int64
	// published is a copy of state replaced after every write, so readers on
	// the request path do not take the registry lock.
	published atomic.Pointer[SessionState]
}

func (record *sessionRecord) publishLocked() {
	state := record.state
	record.published.Store(&state)
}

// observation is one request's view of its session. Counters and token totals
// are read live from the record; LastSeenAt and Duration are fixed when the
// request is observed.
type observation struct {
	record     *sessionRecord
	observedAt time.Time
	prevSeenAt time.Time
}

func (o observation) state() SessionState {
	state := *o.record.published.Load()
	state.LastSeenAt = o.prevSeenAt
	state.Duration = o.observedAt.Sub(o.record.firstSeenAt)
	return state
}

// SessionStateRegistry stores session history and owns its concurrency rules.
// Its zero value is ready for use and tracks any number of sessions.
type SessionStateRegistry struct {
	mu sync.Mutex
	// maxSessions bounds the number of tracked sessions; zero means unbounded.
	// Set before first use.
	maxSessions int
	sessions    map[string]*sessionRecord
	// idle holds the sessions without pending or in-flight requests, most
	// recently active first.
	idle list.List
	// pending maps the ID of a request observed at RequestHeader to its
	// session record until the request is dispatched or abandoned.
	pending map[string]*sessionRecord
}

// observe marks the session as seen at the current time and returns the
// request's view of it. A non-empty requestID marks the request as pending,
// which keeps the session from being evicted until RecordDispatch or
// ReleasePending is called with the same ID.
func (r *SessionStateRegistry) observe(identity, requestID string) observation {
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	record := r.getOrCreate(identity, now)
	if requestID != "" {
		// A request ID observed twice moves to the newer record, so each
		// record counts exactly the pending entries that point to it.
		if detached := r.detachPendingLocked(requestID); detached != nil && detached != record {
			r.refreshIdleLocked(detached)
		}
		if r.pending == nil {
			r.pending = make(map[string]*sessionRecord)
		}
		r.pending[requestID] = record
		record.pendingRequests++
	}
	prevSeenAt := record.state.LastSeenAt
	record.state.LastSeenAt = now
	record.publishLocked()
	r.refreshIdleLocked(record)
	return observation{record: record, observedAt: now, prevSeenAt: prevSeenAt}
}

// GetState returns the state observed before the current request and marks the
// session as seen at the current time.
func (r *SessionStateRegistry) GetState(identity string) SessionState {
	return r.observe(identity, "").state()
}

// RecordDispatch records one request dispatched for the session and ends the
// pending state of requestID, if any.
func (r *SessionStateRegistry) RecordDispatch(identity, requestID string) {
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	detached := r.detachPendingLocked(requestID)
	record := r.getOrCreate(identity, now)
	if detached != nil && detached != record {
		r.refreshIdleLocked(detached)
	}
	record.state.TurnsTaken++
	record.state.InFlightRequests++
	record.publishLocked()
	r.refreshIdleLocked(record)
}

// TurnUsage is the token usage reported by one response.
type TurnUsage struct {
	InputTokens  int64
	OutputTokens int64
	CachedTokens int64
}

// RecordResponse records the end of a dispatched request's response lifecycle.
// Only naturally completed responses contribute completion and token totals.
func (r *SessionStateRegistry) RecordResponse(identity string, naturallyCompleted bool, usage TurnUsage) {
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	record, exists := r.sessions[identity]
	if !exists || record.state.InFlightRequests == 0 {
		return
	}
	record.state.InFlightRequests--
	record.state.LastResponseAt = now
	if naturallyCompleted {
		record.state.CompletedRequests++
		record.state.TotalInputTokens += usage.InputTokens
		record.state.TotalOutputTokens += usage.OutputTokens
		record.state.TotalCachedTokens += usage.CachedTokens
	}
	record.publishLocked()
	r.refreshIdleLocked(record)
}

// ReleasePending ends the pending state of a request that leaves without being
// dispatched. It is safe to call for an unknown or already released ID.
func (r *SessionStateRegistry) ReleasePending(requestID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if record := r.detachPendingLocked(requestID); record != nil {
		r.refreshIdleLocked(record)
	}
}

// detachPendingLocked ends the pending state of requestID and returns its
// record, or nil for an empty, unknown or already released ID. The record is
// left off the idle list; the caller refreshes it once the request's own
// counters are settled, so a record whose request moves straight to dispatch
// never lands on the list between the two counts.
func (r *SessionStateRegistry) detachPendingLocked(requestID string) *sessionRecord {
	if requestID == "" {
		return nil
	}
	record, ok := r.pending[requestID]
	if !ok {
		return nil
	}
	delete(r.pending, requestID)
	record.pendingRequests--
	return record
}

// EvictIdle removes sessions whose latest request or response is older than
// ttl. Sessions with pending or in-flight requests are retained until those
// requests are abandoned or end.
func (r *SessionStateRegistry) EvictIdle(now time.Time, ttl time.Duration) {
	if ttl <= 0 {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	for _, record := range r.sessions {
		if record.state.InFlightRequests > 0 || record.pendingRequests > 0 {
			continue
		}
		lastActive := record.state.LastSeenAt
		if record.state.LastResponseAt.After(lastActive) {
			lastActive = record.state.LastResponseAt
		}
		if now.Sub(lastActive) > ttl {
			r.removeLocked(record)
		}
	}
}

// refreshIdleLocked moves an idle session to the front of the idle list and
// takes a session with a pending or in-flight request off it, so the bound
// never evicts a session a request still refers to.
func (r *SessionStateRegistry) refreshIdleLocked(record *sessionRecord) {
	// With no bound, nothing reads the idle list.
	if r.maxSessions <= 0 {
		return
	}
	if record.pendingRequests > 0 || record.state.InFlightRequests > 0 {
		if record.idleElem != nil {
			r.idle.Remove(record.idleElem)
			record.idleElem = nil
		}
		return
	}
	if record.idleElem != nil {
		r.idle.MoveToFront(record.idleElem)
		return
	}
	record.idleElem = r.idle.PushFront(record)
}

func (r *SessionStateRegistry) removeLocked(record *sessionRecord) {
	if record.idleElem != nil {
		r.idle.Remove(record.idleElem)
		record.idleElem = nil
	}
	delete(r.sessions, record.identity)
}

func (r *SessionStateRegistry) getOrCreate(identity string, now time.Time) *sessionRecord {
	if r.sessions == nil {
		r.sessions = make(map[string]*sessionRecord)
	}
	if record, exists := r.sessions[identity]; exists {
		return record
	}
	// At the bound, idle sessions are evicted, least recently active first.
	// When every tracked session has a pending or in-flight request, the new
	// session is added above the bound.
	for r.maxSessions > 0 && len(r.sessions) >= r.maxSessions {
		oldest := r.idle.Back()
		if oldest == nil {
			break
		}
		r.removeLocked(oldest.Value.(*sessionRecord))
	}
	record := &sessionRecord{
		identity: identity,
		state: SessionState{
			LastSeenAt: now,
		},
		firstSeenAt: now,
	}
	record.publishLocked()
	r.sessions[identity] = record
	return record
}
