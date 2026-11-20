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
	"sync/atomic"
	"time"
)

type sessionRecord struct {
	// state is guarded by SessionStateRegistry.mu.
	state       SessionState
	firstSeenAt time.Time
	// pendingRequests counts requests observed at RequestHeader that are not
	// yet dispatched or abandoned, such as requests waiting in a flow-control
	// queue. Guarded by SessionStateRegistry.mu.
	pendingRequests int64
	// unnamedInflight tracks estimates charged by direct callers that pass an
	// empty requestID. Guarded by SessionStateRegistry.mu.
	unnamedInflight []int64
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

type inflightEntry struct {
	record   *sessionRecord
	estimate int64
}

// SessionStateRegistry stores session history and owns its concurrency rules.
// Its zero value is ready for use.
type SessionStateRegistry struct {
	mu       sync.Mutex
	sessions map[string]*sessionRecord
	// pending maps the ID of a request observed at RequestHeader to its
	// session record until the request is dispatched or abandoned.
	pending map[string]*sessionRecord
	// inflight maps a dispatched requestID to its owning record and charged
	// prompt token estimate until RecordResponse settles it.
	inflight map[string]inflightEntry
	// byEndpoint indexes sessions by their currently bound endpoint ID
	// (endpointID -> identity -> record).
	byEndpoint map[string]map[string]*sessionRecord
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
		r.releasePendingLocked(requestID)
		if r.pending == nil {
			r.pending = make(map[string]*sessionRecord)
		}
		r.pending[requestID] = record
		record.pendingRequests++
	}
	prevSeenAt := record.state.LastSeenAt
	record.state.LastSeenAt = now
	record.publishLocked()
	return observation{record: record, observedAt: now, prevSeenAt: prevSeenAt}
}

// GetState returns the state observed before the current request and marks the
// session as seen at the current time.
func (r *SessionStateRegistry) GetState(identity string) SessionState {
	return r.observe(identity, "").state()
}

// RecordDispatch records one request dispatched for the session, binds or moves
// its endpoint when non-empty, charges promptEstimate into InFlightTokens, and
// ends the pending state of requestID, if any.
func (r *SessionStateRegistry) RecordDispatch(identity, requestID, endpoint string, promptEstimate int64) {
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	r.releasePendingLocked(requestID)
	if requestID != "" {
		if _, exists := r.inflight[requestID]; exists {
			return
		}
	}
	record := r.getOrCreate(identity, now)
	if promptEstimate < 0 {
		promptEstimate = 0
	}
	if requestID == "" {
		record.unnamedInflight = append(record.unnamedInflight, promptEstimate)
	} else {
		if r.inflight == nil {
			r.inflight = make(map[string]inflightEntry)
		}
		r.inflight[requestID] = inflightEntry{record: record, estimate: promptEstimate}
	}
	if endpoint != "" && record.state.Endpoint != endpoint {
		r.moveEndpointLocked(identity, record, endpoint)
	}
	record.state.TurnsTaken++
	record.state.InFlightRequests++
	record.state.InFlightTokens += promptEstimate
	record.publishLocked()
}

// TurnUsage is the token usage reported by one response.
type TurnUsage struct {
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
	CachedTokens int64
}

// RecordResponse records the end of a dispatched request's response lifecycle.
// Charged in-flight estimates are deducted regardless of termination cause;
// only naturally completed responses contribute completion and cumulative token
// totals.
func (r *SessionStateRegistry) RecordResponse(identity, requestID string, naturallyCompleted bool, usage TurnUsage) {
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	record, exists := r.sessions[identity]
	if !exists || record.state.InFlightRequests == 0 {
		return
	}

	var estimate int64
	if requestID != "" {
		entry, ok := r.inflight[requestID]
		if !ok || entry.record != record {
			return
		}
		estimate = entry.estimate
		delete(r.inflight, requestID)
	} else {
		if len(record.unnamedInflight) == 0 {
			return
		}
		estimate = record.unnamedInflight[0]
		if len(record.unnamedInflight) == 1 {
			record.unnamedInflight = nil
		} else {
			record.unnamedInflight = record.unnamedInflight[1:]
		}
	}

	record.state.InFlightRequests--
	record.state.InFlightTokens = max(record.state.InFlightTokens-estimate, 0)
	if record.state.InFlightRequests == 0 {
		record.state.InFlightTokens = 0
	}
	record.state.LastResponseAt = now

	switch {
	case usage.TotalTokens > 0:
		record.state.ContextTokens = usage.TotalTokens
	case naturallyCompleted && usage.InputTokens+usage.OutputTokens > 0:
		record.state.ContextTokens = usage.InputTokens + usage.OutputTokens
	case estimate > record.state.ContextTokens:
		record.state.ContextTokens = estimate
	}

	if naturallyCompleted {
		record.state.CompletedRequests++
		record.state.TotalInputTokens += usage.InputTokens
		record.state.TotalOutputTokens += usage.OutputTokens
		record.state.TotalCachedTokens += usage.CachedTokens
	}
	record.publishLocked()
}

// RemoveEndpoint clears endpoint bindings for all sessions currently bound to
// endpoint while retaining their session history and context footprint.
func (r *SessionStateRegistry) RemoveEndpoint(endpoint string) {
	if endpoint == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	records := r.byEndpoint[endpoint]
	for _, record := range records {
		record.state.Endpoint = ""
		record.publishLocked()
	}
	delete(r.byEndpoint, endpoint)
}

// EndpointSessions returns a fresh slice of SessionState values for sessions
// currently bound to endpoint.
func (r *SessionStateRegistry) EndpointSessions(endpoint string) EndpointSessions {
	r.mu.Lock()
	defer r.mu.Unlock()

	records := r.byEndpoint[endpoint]
	if len(records) == 0 {
		return EndpointSessions{}
	}
	now := time.Now()
	out := make(EndpointSessions, 0, len(records))
	for _, record := range records {
		state := record.state
		state.Duration = max(now.Sub(record.firstSeenAt), 0)
		out = append(out, state)
	}
	return out
}

// ReleasePending ends the pending state of a request that leaves without being
// dispatched. It is safe to call for an unknown or already released ID.
func (r *SessionStateRegistry) ReleasePending(requestID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.releasePendingLocked(requestID)
}

func (r *SessionStateRegistry) releasePendingLocked(requestID string) {
	if requestID == "" {
		return
	}
	if record, ok := r.pending[requestID]; ok {
		delete(r.pending, requestID)
		record.pendingRequests--
	}
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
	for identity, record := range r.sessions {
		if record.state.InFlightRequests > 0 || record.pendingRequests > 0 {
			continue
		}
		lastActive := record.state.LastSeenAt
		if record.state.LastResponseAt.After(lastActive) {
			lastActive = record.state.LastResponseAt
		}
		if now.Sub(lastActive) > ttl {
			r.removeFromEndpointLocked(identity, record.state.Endpoint)
			delete(r.sessions, identity)
		}
	}
}

func (r *SessionStateRegistry) moveEndpointLocked(identity string, record *sessionRecord, endpoint string) {
	r.removeFromEndpointLocked(identity, record.state.Endpoint)
	if r.byEndpoint == nil {
		r.byEndpoint = make(map[string]map[string]*sessionRecord)
	}
	if r.byEndpoint[endpoint] == nil {
		r.byEndpoint[endpoint] = make(map[string]*sessionRecord)
	}
	r.byEndpoint[endpoint][identity] = record
	record.state.Endpoint = endpoint
}

func (r *SessionStateRegistry) removeFromEndpointLocked(identity, endpoint string) {
	if endpoint == "" {
		return
	}
	records := r.byEndpoint[endpoint]
	if records == nil {
		return
	}
	delete(records, identity)
	if len(records) == 0 {
		delete(r.byEndpoint, endpoint)
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
			SessionID:  identity,
			LastSeenAt: now,
		},
		firstSeenAt: now,
	}
	record.publishLocked()
	r.sessions[identity] = record
	return record
}
