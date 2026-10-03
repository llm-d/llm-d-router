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

package thunderagent

import (
	"math"
	"sort"
	"sync"
	"time"
)

// bytesPerToken converts a request body size to an estimated prompt token
// count.
const bytesPerToken = 4.0

// maintenanceInterval bounds how often the full-table maintenance (TTL and
// reservation expiry, stale endpoint cleanup) runs.
const maintenanceInterval = time.Second

// endpointStaleAfter drops a pod entry that holds no sessions and has not been
// seen for this long.
const endpointStaleAfter = 5 * time.Second

// reservationTTL bounds an admission reservation whose session never reached
// PreRequest, such as a request cancelled between dispatch and binding.
// Counting a cancelled request as occupancy would be phantom load.
const reservationTTL = 5 * time.Second

// sessionClass ranks a waiting session for admission. Lower dispatches
// first.
type sessionClass int

const (
	// classReasoning is a session already admitted to a pod: bound to it,
	// at least one turn dispatched, not paused. Its footprint is already
	// counted and finishing its trajectory is what frees capacity, so its
	// turns always dispatch.
	classReasoning sessionClass = iota
	// classPaused was pushed out by the sweep in saturation loop: its next turn
	// must fit its own pod again, but it outranks never-admitted sessions.
	classPaused
	// classNew is not admitted to any pod: it never dispatched, or its pod
	// left the pool. Admitted only when a pod has room.
	classNew
)

func (c sessionClass) String() string {
	switch c {
	case classReasoning:
		return "reasoning"
	case classPaused:
		return "paused"
	default:
		return "new"
	}
}

// session is one agent trajectory, identified by the request FairnessID.
// All fields are guarded by sessionManager.mu.
type session struct {
	// endpoint the session is bound to; nil before the first dispatch and
	// after the endpoint leaves the pool.
	endpoint *endpointState
	// committedTokens is usage.total_tokens of the last completed turn.
	committedTokens int64
	// inflightTokens is the estimate of the turn currently being processed.
	// A session is assumed to have at most one request in flight.
	inflightTokens int64
	lastResponseAt time.Time
	lastActivity   time.Time
	turnCount      int64
	// paused is set by the pause sweep: the session stops counting against
	// its endpoint, and its next turn must pass the fit check again before
	// it dispatches.
	paused bool
	// reserved marks a session Pick has admitted but PreRequest has not yet
	// bound.
	reserved       bool
	reservedTokens float64
	reservedUntil  time.Time
}

// undecayed is the session's KV footprint in tokens.
func (s *session) undecayed() float64 {
	if f := float64(s.inflightTokens); f > float64(s.committedTokens) {
		return f
	}
	return float64(s.committedTokens)
}

// decayed is the admission view of the footprint: an idle session's
// committed tokens decay with the configured half-life, because the engine
// gradually evicts its KV blocks while it waits on a tool. Sessions with a
// turn in flight count in full.
func (s *session) decayed(now time.Time, halfLife time.Duration) float64 {
	u := s.undecayed()
	if s.inflightTokens > 0 || halfLife <= 0 || s.lastResponseAt.IsZero() {
		return u
	}
	elapsed := now.Sub(s.lastResponseAt)
	if elapsed <= 0 {
		return u
	}
	return float64(s.committedTokens) * math.Exp2(-float64(elapsed)/float64(halfLife))
}

// class ranks the session for admission. The order of the checks matters: a
// session whose pod left the pool re-enters as new even if it was paused.
func (s *session) class() sessionClass {
	if s.endpoint == nil || s.turnCount == 0 {
		return classNew
	}
	if s.paused {
		return classPaused
	}
	return classReasoning
}

// footprints is the accounting rule: an unexpired reservation counts its
// reserved size, a paused session counts nothing, everything else counts
// its two views.
func (s *session) footprints(now time.Time, halfLife time.Duration) (undecayed, decayed float64) {
	switch {
	case s.reserved:
		if now.Before(s.reservedUntil) {
			return s.reservedTokens, s.reservedTokens
		}
		return 0, 0
	case s.paused:
		return 0, 0
	}
	return s.undecayed(), s.decayed(now, halfLife)
}

// endpointState is the plugin's own record of one endpoint.
type endpointState struct {
	id       string
	capacity float64
	// sessions are the sessions bound to this pod.
	sessions map[string]*session
	// updatedAt is the last time this pod was seen.
	updatedAt time.Time
}

// occupancy sums the endpoint's working set in both views under the
// accounting rule.
func (p *endpointState) occupancy(now time.Time, halfLife time.Duration) (undecayed, decayed float64) {
	for _, s := range p.sessions {
		u, d := s.footprints(now, halfLife)
		undecayed += u
		decayed += d
	}
	return undecayed, decayed
}

// pauseSmallest pauses the endpoint's idle sessions smallest first until
// the undecayed working set drops back to the ceiling.
func (p *endpointState) pauseSmallest(tokens, ceiling float64) int {
	type candidate struct {
		s         *session
		footprint float64
	}
	var idle []candidate
	for _, s := range p.sessions {
		if s.paused || s.reserved || s.inflightTokens > 0 {
			continue
		}
		idle = append(idle, candidate{s, s.undecayed()})
	}
	sort.Slice(idle, func(i, j int) bool { return idle[i].footprint < idle[j].footprint })
	paused := 0
	for _, c := range idle {
		if tokens <= ceiling {
			break
		}
		c.s.paused = true
		tokens -= c.footprint
		paused++
	}
	return paused
}

// sessionManager is the ledger shared by all of the thunder agent plugin's hooks/
type sessionManager struct {
	// mu guards everything below, including all session and endpointState
	// fields. The Locked suffix and the session / endpointState helpers all
	// assume the caller holds it.
	mu        sync.Mutex
	sessions  map[string]*session
	endpoints map[string]*endpointState

	ttl             time.Duration
	halfLife        time.Duration
	lastMaintenance time.Time
	lastSweep       time.Time
}

func newSessionManager(cfg Config) *sessionManager {
	return &sessionManager{
		sessions:  make(map[string]*session),
		endpoints: make(map[string]*endpointState),
		ttl:       time.Duration(cfg.EvictionTTLSeconds * float64(time.Second)),
		halfLife:  time.Duration(cfg.IdleDecayHalfLifeSeconds * float64(time.Second)),
	}
}

// ensureEndpointLocked returns the ledger entry for an endpoint, creating
// it on first sight and refreshing its capacity.
func (m *sessionManager) ensureEndpointLocked(id string, capacity float64, now time.Time) *endpointState {
	p, ok := m.endpoints[id]
	if !ok {
		p = &endpointState{id: id, sessions: make(map[string]*session)}
		m.endpoints[id] = p
	}
	p.capacity = capacity
	p.updatedAt = now
	return p
}

// bindLocked returns the session for id bound to the given endpoint,
// creating the session on first sight and moving it if it was bound
// elsewhere.
func (m *sessionManager) bindLocked(id string, ep *endpointState) *session {
	s, ok := m.sessions[id]
	if !ok {
		s = &session{}
		m.sessions[id] = s
	}
	if s.endpoint != ep {
		if s.endpoint != nil {
			delete(s.endpoint.sessions, id)
		}
		s.endpoint = ep
		ep.sessions[id] = s
	}
	return s
}

// removeLocked drops a session from the ledger and from its endpoint.
func (m *sessionManager) removeLocked(id string) {
	s, ok := m.sessions[id]
	if !ok {
		return
	}
	if s.endpoint != nil {
		delete(s.endpoint.sessions, id)
	}
	delete(m.sessions, id)
}

// estimateTokens converts a request body size to a token estimate.
func estimateTokens(sizeBytes int) int64 {
	if sizeBytes <= 0 {
		return 0
	}
	return int64(float64(sizeBytes) / bytesPerToken)
}

// gaugeSnapshot is the ledger state the metrics collector reports.
type gaugeSnapshot struct {
	endpoints             map[string]endpointGauge
	running, idle, paused int
}

type endpointGauge struct {
	undecayed float64
	decayed   float64
	capacity  float64
}

// maintainLocked is the housekeeping pass, run at most once per
// maintenanceInterval by whichever hook holds the lock: expire unconfirmed
// reservations, drop sessions idle past the TTL and drop empty stale
// endpoints.
func (m *sessionManager) maintainLocked(now time.Time) {
	if now.Sub(m.lastMaintenance) < maintenanceInterval {
		return
	}
	m.lastMaintenance = now

	for id, s := range m.sessions {
		// Keep live reservations (a new session has no activity for the TTL to
		// see); on expiry, drop the reservation and any session never dispatched.
		if s.reserved {
			if now.Before(s.reservedUntil) {
				continue
			}
			s.reserved = false
			s.reservedTokens = 0
			if s.turnCount == 0 {
				m.removeLocked(id)
				continue
			}
		}
		if s.inflightTokens > 0 {
			continue
		}
		if now.Sub(s.lastActivity) <= m.ttl {
			continue
		}
		m.removeLocked(id)
	}
	for id, p := range m.endpoints {
		if len(p.sessions) == 0 && now.Sub(p.updatedAt) > endpointStaleAfter {
			delete(m.endpoints, id)
		}
	}
}

// snapshot runs due maintenance, so a scrape keeps the ledger current without
// traffic, and returns the values to report.
func (m *sessionManager) snapshot(now time.Time) gaugeSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.maintainLocked(now)

	snap := gaugeSnapshot{endpoints: make(map[string]endpointGauge, len(m.endpoints))}
	for _, s := range m.sessions {
		switch {
		case s.paused:
			snap.paused++
		case s.inflightTokens > 0:
			snap.running++
		default:
			snap.idle++
		}
	}
	for id, p := range m.endpoints {
		u, d := p.occupancy(now, m.halfLife)
		snap.endpoints[id] = endpointGauge{undecayed: u, decayed: d, capacity: p.capacity}
	}
	return snap
}
