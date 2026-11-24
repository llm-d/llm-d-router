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

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

type sessionRecord struct {
	// state is guarded by SessionStateRegistry.mu.
	state       SessionState
	firstSeenAt time.Time
	// pendingRequests counts requests observed at RequestHeader that are not
	// yet dispatched or abandoned, such as requests waiting in a flow-control
	// queue. Guarded by SessionStateRegistry.mu.
	pendingRequests int64
	// inflight maps a dispatched requestID to its charged prompt token estimate
	// until RecordResponse settles it. Guarded by SessionStateRegistry.mu.
	inflight map[string]int64
	// published is replaced after every write so request-path and endpoint-view
	// lookups do not take the registry lock.
	published atomic.Pointer[SessionState]

	// Endpoint bucket accounting state, guarded by SessionStateRegistry.mu.
	bucket          *endpointBucket
	accountedTokens int64
}

func (record *sessionRecord) syncInflightLocked() {
	var total int64
	for _, estimate := range record.inflight {
		total += estimate
	}
	record.state.InFlightRequests = int64(len(record.inflight))
	record.state.InFlightTokens = total
}

func (record *sessionRecord) publishLocked() {
	state := record.state
	record.published.Store(&state)
}

type endpointBucket struct {
	registry    *SessionStateRegistry
	id          string
	records     map[string]*sessionRecord // guarded by registry.mu
	byID        sync.Map                  // sessionID -> *sessionRecord
	totalTokens atomic.Int64
	cached      atomic.Pointer[[]SessionState]
	alive       atomic.Bool
}

func newEndpointBucket(registry *SessionStateRegistry, id string) *endpointBucket {
	b := &endpointBucket{
		registry: registry,
		id:       id,
	}
	b.alive.Store(true)
	return b
}

func (b *endpointBucket) session(id string) (SessionState, bool) {
	if !b.alive.Load() {
		return SessionState{}, false
	}
	val, ok := b.byID.Load(id)
	if !ok {
		return SessionState{}, false
	}
	rec := val.(*sessionRecord)
	pub := rec.published.Load()
	if pub == nil || !b.alive.Load() || pub.Endpoint != b.id {
		return SessionState{}, false
	}
	state := *pub
	state.Duration = max(time.Since(rec.firstSeenAt), 0)
	return state, true
}

func (b *endpointBucket) sessions() []SessionState {
	if !b.alive.Load() {
		return nil
	}
	if cached := b.cached.Load(); cached != nil {
		return *cached
	}
	b.registry.mu.Lock()
	defer b.registry.mu.Unlock()
	if !b.alive.Load() {
		return nil
	}
	if cached := b.cached.Load(); cached != nil {
		return *cached
	}
	out := b.snapshotLocked()
	b.cached.Store(&out)
	return out
}

func (b *endpointBucket) snapshotLocked() []SessionState {
	if len(b.records) == 0 {
		return []SessionState{}
	}
	now := time.Now()
	out := make([]SessionState, 0, len(b.records))
	for _, record := range b.records {
		state := record.state
		state.Duration = max(now.Sub(record.firstSeenAt), 0)
		out = append(out, state)
	}
	return out
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
// Its zero value is ready for use.
type SessionStateRegistry struct {
	mu       sync.Mutex
	sessions map[string]*sessionRecord
	// pending maps the ID of a request observed at RequestHeader to its
	// session record until the request is dispatched or abandoned.
	pending map[string]*sessionRecord
	// byEndpoint indexes sessions and aggregates by bound endpoint ID.
	byEndpoint map[string]*endpointBucket
	// registeredEndpoints, when non-nil, restricts bindings to currently
	// registered live endpoints so a late dispatch after endpoint deletion
	// cannot create a zombie binding.
	registeredEndpoints map[string]fwkdl.Endpoint
}

// EnableEndpointTracking activates live endpoint registration checks so
// RecordDispatch binds only to currently registered endpoints.
func (r *SessionStateRegistry) EnableEndpointTracking() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.registeredEndpoints == nil {
		r.registeredEndpoints = make(map[string]fwkdl.Endpoint)
	}
}

// RegisterEndpoint records ep as the active endpoint for endpoint and returns
// its EndpointSessions view. Replacing a previous Endpoint instance under the
// same ID unbinds sessions from the old instance.
func (r *SessionStateRegistry) RegisterEndpoint(endpoint string, ep fwkdl.Endpoint) EndpointSessions {
	if endpoint == "" || ep == nil {
		return EndpointSessions{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.registeredEndpoints == nil {
		r.registeredEndpoints = make(map[string]fwkdl.Endpoint)
	}
	if registered, ok := r.registeredEndpoints[endpoint]; !ok || registered != ep {
		r.removeEndpointLocked(endpoint)
	}
	r.registeredEndpoints[endpoint] = ep
	return EndpointSessions{bucket: r.ensureBucketLocked(endpoint)}
}

// UnregisterEndpoint removes ep if it is still the active endpoint for
// endpoint, unbinding its sessions. Returns false when ep was already replaced
// by a newer Endpoint object.
func (r *SessionStateRegistry) UnregisterEndpoint(endpoint string, ep fwkdl.Endpoint) bool {
	if endpoint == "" {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if registered, ok := r.registeredEndpoints[endpoint]; ok && ep != nil && registered != ep {
		return false
	}
	delete(r.registeredEndpoints, endpoint)
	r.removeEndpointLocked(endpoint)
	return true
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
// its endpoint when non-empty and registered, charges promptEstimate into
// InFlightTokens, and ends the pending state of requestID, if any. Calls with
// an empty identity or requestID are ignored.
func (r *SessionStateRegistry) RecordDispatch(identity, requestID, endpoint string, promptEstimate int64) {
	if identity == "" || requestID == "" {
		return
	}
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	r.releasePendingLocked(requestID)
	record := r.getOrCreate(identity, now)
	if _, exists := record.inflight[requestID]; exists {
		return
	}
	if record.inflight == nil {
		record.inflight = make(map[string]int64)
	}
	record.inflight[requestID] = max(promptEstimate, 0)
	if endpoint != "" && (r.registeredEndpoints == nil || r.registeredEndpoints[endpoint] != nil) {
		record.state.Endpoint = endpoint
	}
	record.state.TurnsTaken++
	record.syncInflightLocked()
	r.syncRecordLocked(identity, record)
}

// TurnUsage is the token usage reported by one response.
type TurnUsage struct {
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
	CachedTokens int64
}

// RecordResponse records the end of a dispatched request's response lifecycle.
// Charged in-flight estimates are settled regardless of termination cause; only
// naturally completed responses contribute completion and token totals. Calls
// with an empty identity or requestID, or for an untracked requestID, are
// ignored.
func (r *SessionStateRegistry) RecordResponse(identity, requestID string, naturallyCompleted bool, usage TurnUsage) {
	if identity == "" || requestID == "" {
		return
	}
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()

	record, exists := r.sessions[identity]
	if !exists {
		return
	}
	estimate, ok := record.inflight[requestID]
	if !ok {
		return
	}
	delete(record.inflight, requestID)
	record.syncInflightLocked()
	record.state.LastResponseAt = now

	switch {
	case naturallyCompleted && usage.TotalTokens > 0:
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
	r.syncRecordLocked(identity, record)
}

func (r *SessionStateRegistry) removeEndpointLocked(endpoint string) {
	bucket := r.byEndpoint[endpoint]
	if bucket == nil {
		return
	}
	bucket.alive.Store(false)
	delete(r.byEndpoint, endpoint)
	for identity, record := range bucket.records {
		record.state.Endpoint = ""
		r.syncRecordLocked(identity, record)
	}
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
		if len(record.inflight) > 0 || record.pendingRequests > 0 {
			continue
		}
		lastActive := record.state.LastSeenAt
		if record.state.LastResponseAt.After(lastActive) {
			lastActive = record.state.LastResponseAt
		}
		if now.Sub(lastActive) > ttl {
			record.state.Endpoint = ""
			r.syncRecordLocked(identity, record)
			delete(r.sessions, identity)
		}
	}
}

func (r *SessionStateRegistry) ensureBucketLocked(endpoint string) *endpointBucket {
	if r.byEndpoint == nil {
		r.byEndpoint = make(map[string]*endpointBucket)
	}
	if bucket := r.byEndpoint[endpoint]; bucket != nil {
		return bucket
	}
	bucket := newEndpointBucket(r, endpoint)
	r.byEndpoint[endpoint] = bucket
	return bucket
}

func (r *SessionStateRegistry) syncRecordLocked(identity string, record *sessionRecord) {
	record.publishLocked()

	var target *endpointBucket
	if record.state.Endpoint != "" {
		target = r.ensureBucketLocked(record.state.Endpoint)
	}

	if old := record.bucket; old != nil && old != target {
		delete(old.records, identity)
		old.byID.Delete(identity)
		old.totalTokens.Add(-record.accountedTokens)
		old.cached.Store(nil)
		if len(old.records) == 0 && (r.registeredEndpoints == nil || r.registeredEndpoints[old.id] == nil) && r.byEndpoint[old.id] == old {
			old.alive.Store(false)
			delete(r.byEndpoint, old.id)
		}
		record.bucket = nil
		record.accountedTokens = 0
	}

	if target == nil {
		return
	}

	if record.bucket == nil {
		if target.records == nil {
			target.records = make(map[string]*sessionRecord)
		}
		target.records[identity] = record
		target.byID.Store(identity, record)
		record.bucket = target
	}

	newTokens := record.state.ContextSize()
	if newTokens != record.accountedTokens {
		target.totalTokens.Add(newTokens - record.accountedTokens)
		record.accountedTokens = newTokens
	}
	target.cached.Store(nil)
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
